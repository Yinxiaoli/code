#include "structured_svm_multiclass.h"

/**
 * @file structured_svm_multiclass.cpp
 * @brief Simple example of how to use this structured SVM package: implements routines for cost-sensitive multiclass SVM training and classification
 */



/**
 * @example structured_svm_multiclass.cpp
 *
 * This is an example of how to use the structured learning API to implement a custom structured learner.  This
 * example implements a multiclass SVM learner and classification with custom loss function
 *
 * Example usage:
 * - Train using a fixed dataset without running in server mode, outputting the learned model to learned_model.txt.  classes.txt
 *    defines the confusion cost between classes and is in the format of MulticlassStructuredSVM::Load, and train.txt is in the
 *    format of MulticlassStructuredSVM::LoadDataset
\htmlonly <div style="padding: 0.5em 1em; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; background-color: #eaeafa;">
$ examples/bin/release_static/structured_svm_multiclass.out -p classes.txt -d train.txt -o learned_model.txt
</div> \endhtmlonly
 * - Evaluate performance on a testset:
\htmlonly <div style="padding: 0.5em 1em; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; background-color: #eaeafa;">
$ examples/bin/release_static/structured_svm_multiclass.out -p learned_model.txt -t test.txt test.txt.predictions
</div> \endhtmlonly
 *
 */


StructuredLabel *MulticlassStructuredSVM::NewStructuredLabel(StructuredData *x) { return new MulticlassStructuredLabel(x); }

StructuredData *MulticlassStructuredSVM::NewStructuredData() { return new MulticlassStructuredData; }


MulticlassStructuredSVM::MulticlassStructuredSVM() {
  params.eps = .01;
  params.C = 5000.0;
  params.canScaleW = true;
  params.mergeSamples = false;
  params.method = SPO_DUAL_MULTI_SAMPLE_UPDATE_WITH_CACHE;

  // Normally, load the full dataset into memory.  Add the command line option "-D no_dataset" to 
  // handle the case when the dataset is too big to fit in memory, reading the dataset from disk on the fly.
  params.memoryMode = MEM_KEEP_DATASET_IN_MEMORY; 

  // For most structured SVM problems, set allSamplesOrthogonal=false.  Setting allSamplesOrthogonal=true
  // assumes all samples returned by ImportanceSample() have feature products with dot product 0 to each
  // other and the ground truth feature vector.  This results in faster optimization.
  params.allSamplesOrthogonal = true;

  num_classes = 0;
  num_features = 0;
  classConfusionCosts = NULL;

  alphas = NULL;
  alphas_alloc_size = 0;
  cached_samples = (SVM_cached_sample**)malloc(sizeof(SVM_cached_sample*)*(params.num_thr+10));
  memset(cached_samples, 0, sizeof(SVM_cached_sample*)*(params.num_thr+10));
  cached_num_samples = (int*)malloc(sizeof(int)*(params.num_thr+10));
  memset(cached_num_samples, 0, sizeof(int)*(params.num_thr+10));
  alphasFile = NULL;
  
  for(int i = 0; i < 1000; i++)
    lines[i] = NULL;
}

MulticlassStructuredSVM::~MulticlassStructuredSVM() {
  if(classConfusionCosts) 
    free(classConfusionCosts);
  if(alphas) {
    for(int i = 0; i < alphas_alloc_size; i++)
      if(alphas[i]) 
	delete [] alphas[i];
    free(alphas);
  }
  for(int i = 0; i < 1000; i++)
    if(lines[i])
      delete [] lines[i];
  if(alphasFile)
    fclose(alphasFile);
}

double MulticlassStructuredSVM::Inference(StructuredData *x, StructuredLabel *ybar, SparseVector *w,
					  StructuredLabel *y_partial, StructuredLabel *y_gt, double w_scale) {
  int bestclass=-1, first=1;
  double score,bestscore=-1;

  MulticlassStructuredLabel *m_ybar = (MulticlassStructuredLabel*)ybar;

  // Loop through every possible class y and compute its score <w,Psi(x,y)>
  for(int class_id = 1; class_id <= num_classes; class_id++) {
    // By default, compute ybar = max_y <w,Psi(x,y)>, but it y_partial is non-null,
    // only consider labels that agree with y_partial 
    m_ybar->class_id=class_id;
    score = w->dot(Psi(x, ybar))*w_scale;
   
    // If y_gt is non-null, compute ybar = max_y <w,Psi(x,y)>+Loss(y_gt,y) 
    if(y_gt)  
      score += Loss(y_gt, ybar);

    if(score > bestscore || first) {
      bestscore=score;
      bestclass=class_id;
      first=0;
    }
  }
  m_ybar->class_id = bestclass;

  return bestscore;
}



double MulticlassStructuredSVM::ImportanceSample(StructuredData *x, SparseVector *w, StructuredLabel *y_gt, 
						 struct _SVM_cached_sample_set *set, double w_scale) {
  int first=1, best_ind = -1;
  double score,bestscore=0;
  int cl = ((MulticlassStructuredLabel*)y_gt)->class_id;

  // Helper function to create one sample per class
  CreateSamples(set, (MulticlassStructuredData*)x, (MulticlassStructuredLabel*)y_gt);

  for(int i = 0, j = 0; i < num_classes; i++) {
    if(i+1 == ((MulticlassStructuredLabel*)y_gt)->class_id) continue;
    SVM_cached_sample *sample = &set->samples[j++];
    score = w->dot(*sample->psi)*w_scale + sample->loss;
    sample->slack = score-set->score_gt;
    sample->dot_w = (sample->slack - sample->loss)*sum_w_scale;
    
    if(score > bestscore || first) {
      bestscore=score;
      first=0;
      cl = i+1;
      best_ind = j-1;
    }
  }
  set->dot_sum_w_psi_gt = set->score_gt*sum_w_scale;
  SVM_cached_sample tmp = set->samples[0];
  set->samples[0] = set->samples[best_ind]; 
  set->samples[best_ind] = tmp;
  set->slack_before = bestscore-set->score_gt;

  return bestscore;
}


SparseVector MulticlassStructuredSVM::Psi(StructuredData *x, StructuredLabel *y) {
  // The dimensionality of Psi(x,y) is num_featuresXnum_classes, by concatenating
  // num_features features for each class. The entries for Psi are equal to x->psi for 
  // the true class y and 0 for all classes other than y.  
  MulticlassStructuredData *m_x = (MulticlassStructuredData*)x;
  MulticlassStructuredLabel *m_y = (MulticlassStructuredLabel*)y;

  if(!m_x->psi && (params.memoryMode == MEM_KEEP_NOTHING_IN_MEMORY || params.memoryMode == MEM_KEEP_DUAL_VARIABLES_IN_MEMORY)) {
    // Handle the case where the dataset is too big to fit in memory
    if(FSEEK(m_x->fins[omp_get_thread_num()], m_x->seek_pos, SEEK_SET) || LoadExample(m_x, NULL, false, lines[omp_get_thread_num()])) 
      fprintf(stderr, "Failed to load example in Psi()\n");
  }

  return m_x->psi->shift((m_y->class_id-1)*num_features);
}


double MulticlassStructuredSVM::Loss(StructuredLabel *y_gt, StructuredLabel *y_pred) {
  // Computes the loss of prediction y_pred against the correct label y_gt. 
  MulticlassStructuredLabel *m_y_gt = (MulticlassStructuredLabel*)y_gt;
  MulticlassStructuredLabel *m_y_pred = (MulticlassStructuredLabel*)y_pred;
  if(classConfusionCosts)
    return classConfusionCosts[m_y_gt->class_id][m_y_pred->class_id];
  else 
    return m_y_gt->class_id == m_y_pred->class_id ? 0 : 1;
}

Json::Value MulticlassStructuredSVM::Save() {
  Json::Value root;
  
  root["version"] = VERSION;
  root["Num Classes"] = num_classes;
  root["Num Features"] = num_features;

  Json::Value c;
  int n = 0;
  for(int i = 1; i <= num_classes; i++) {
    for(int j = 1; j <= num_classes; j++) {
      if((!classConfusionCosts && i != j) || (classConfusionCosts && classConfusionCosts[i][j])) {
	Json::Value o;
	o["c_gt"] = i;
	o["c_pred"] = j;
	o["loss"] = classConfusionCosts ? classConfusionCosts[i][j] : 1;
	c[n++] = o;
      }
    }
  }
  root["Class Confusion Costs"] = c;
    
  return root;
}

void MulticlassStructuredSVM::CreateSamples(struct _SVM_cached_sample_set *set, MulticlassStructuredData *x, MulticlassStructuredLabel *y_gt) {
  if(!set->num_samples) {
    if(params.memoryMode == MEM_KEEP_EVERYTHING_IN_MEMORY) {
      // Keep full datatset and generic structured SVM memory caches in memory
      // Add a sample set that includes all classes 
      for(int i = 0; i < num_classes; i++) {
	if(i+1 == y_gt->class_id) continue;
	MulticlassStructuredLabel *ybar = (MulticlassStructuredLabel*)NewStructuredLabel(x);
	ybar->class_id = i+1;
	SVM_cached_sample *sample = SVM_cached_sample_set_add_sample(set, ybar);
	
	// Optionally set these things, so they don't have to be computed later
	sample->psi = Psi(x, ybar).ptr();
	sample->loss = Loss(y_gt, ybar);
	sample->sqr = 2*set->psi_gt_sqr;   // <set->psi_gt-sample->psi,set->psi_gt-sample->psi>
	sample->dot_psi_gt_psi = 0;  // <set->psi_gt,sample->psi>
      }
      set->samples = (SVM_cached_sample*)realloc(set->samples, sizeof(SVM_cached_sample)*(set->num_samples+1));
    } else {

      // For better memory-efficiency don't store various memory buffers associated with a structured 
      // SVM sample sets
      if(cache_old_examples) {
	if(set->i+1 > alphas_alloc_size) {
	  int old = alphas_alloc_size;
	  alphas_alloc_size = my_max(set->i+1,n);
	  alphas = (double**)realloc(alphas, sizeof(double*)*alphas_alloc_size);
	  for(int i = old; i < alphas_alloc_size; i++)
	    alphas[i] = NULL;
	}
	if(!alphas[set->i]) {
	  alphas[set->i] = new double[num_classes];
	  for(int i = 0; i < num_classes; i++)
	    alphas[set->i][i] = 0;
	}
      }

      int tid = omp_get_thread_num();
      if(params.memoryMode == MEM_KEEP_NOTHING_IN_MEMORY) {
	// Doesn't keep the dataset or dual variables in memory, and instead re-reads them from disk each 
	// time we sequentially process an example. 
	Lock(); 
	if(FSEEK(alphasFile, set->i*(long)num_classes*sizeof(double), SEEK_SET) ||
	   (int)fread(alphas[set->i], sizeof(double), num_classes, alphasFile) != num_classes)
	  fprintf(stderr, "Failed to load example alphas %d\n", set->i);
	Unlock();
      }
    
      if(cached_samples[tid]) {
	// Instead of dynamically allocating memory, use memory that is specifically allocated for
	// this thread
	assert(cached_num_samples[tid] == num_classes-1);
	set->samples = cached_samples[tid];
	set->num_samples = 0;
      }
    
      for(int i = 0; i < num_classes; i++) {
	if(i+1 == y_gt->class_id) continue;
	SVM_cached_sample *sample;
	if(!cached_samples[tid]) {
	  MulticlassStructuredLabel *ybar = (MulticlassStructuredLabel*)NewStructuredLabel(x);
	  ybar->class_id = i+1;
	  sample = SVM_cached_sample_set_add_sample(set, ybar);
	  sample->psi = Psi(x, ybar).ptr();
	} else {
	  sample = &set->samples[set->num_samples++];
	  sample->psi->set_shift(((MulticlassStructuredData*)x)->psi, i*num_features);
	}
	((MulticlassStructuredLabel*)sample->ybar)->class_id = i+1;
	((MulticlassStructuredLabel*)sample->ybar)->x = x;
	sample->slack = sample->dot_w = sample->dot_psi_gt_psi = 0;
	sample->loss = Loss(y_gt, sample->ybar);
	sample->sqr = 2*set->psi_gt_sqr;   // <set->psi_gt-sample->psi,set->psi_gt-sample->psi>
	sample->alpha = cache_old_examples ? alphas[set->i][i] : 0;
      }
    }
  }
}

void MulticlassStructuredSVM::OnFinishedIteration(StructuredExample *ex) {
  if(params.memoryMode != MEM_KEEP_EVERYTHING_IN_MEMORY) {
    // For better memory-efficiency don't store various memory buffers associated with a sample set, and
    // instead just share an array of dual variables alpha
    struct _SVM_cached_sample_set *set = ex->set;
    if(set) {
      if(cache_old_examples)
	for(int i = 0; i < set->num_samples; i++) 
	  alphas[set->i][((MulticlassStructuredLabel*)set->samples[i].ybar)->class_id-1] = set->samples[i].alpha;
      int tid = omp_get_thread_num();
      assert(cached_samples[tid] == set->samples || !cached_samples[tid]);
      cached_samples[tid] = set->samples;
      cached_num_samples[tid] = set->num_samples;
      set->samples = NULL;
      set->num_samples = 0;
    }
    if(params.memoryMode ==  MEM_KEEP_DUAL_VARIABLES_IN_MEMORY || params.memoryMode == MEM_KEEP_NOTHING_IN_MEMORY) {
      MulticlassStructuredData *x = (MulticlassStructuredData*)ex->x;
      if(x->psi) {
	delete x->psi;
	x->psi = NULL;
      }
      if(set && set->psi_gt) {
	delete set->psi_gt;
	set->psi_gt = NULL;
      }
    }

    if(params.memoryMode == MEM_KEEP_NOTHING_IN_MEMORY && alphas) {
      // Doesn't keep the dataset or dual variables in memory, and instead saves and re-reads them from disk each 
      // time we sequentially process an example.
      Lock(); 
      if(FSEEK(alphasFile, set->i*(long)num_classes*sizeof(double), SEEK_SET) ||
	 (int)fwrite(alphas[set->i], sizeof(double), num_classes, alphasFile) != num_classes)
	fprintf(stderr, "Failed to write example alphas %d\n", set->i);
      Unlock();
    }
  }
}



void MulticlassStructuredSVM::SetClassConfusionCosts(double **c) {
  classConfusionCosts = Create2DArray<double>(num_classes+1, num_classes+1);
  for(int i = 0; i < num_classes; i++)
    memcpy(classConfusionCosts[i], c[i], sizeof(double)*(num_classes+1));
}

bool MulticlassStructuredSVM::Load(const Json::Value &root) {
  fprintf(stdout, "loading parameters\n");
  if(strcmp(root.get("version", "").asString().c_str(), VERSION)) {
    fprintf(stderr, "Version of parameter file does not match version of the software"); 
    return false;
  }
  num_classes = root.get("Num Classes",0).asInt();
  num_features = root.get("Num Features",0).asInt();
  
  sizePsi = num_features*num_classes;

  if(root.isMember("Class Confusion Costs") && root["Class Confusion Costs"].isArray()) {
    classConfusionCosts = (double**)malloc((num_classes+1)*(sizeof(double*)+(num_classes+1)*sizeof(double)));
    double *ptr = (double*)(classConfusionCosts+(num_classes+1));
    for(int i = 0; i <= num_classes; i++, ptr += (num_classes+1)) {
      classConfusionCosts[i] = ptr;
      for(int j = 0; j <= num_classes; j++)
	classConfusionCosts[i][j] = 0;
    }
    Json::Value a = root["Class Confusion Costs"];
    for(int i = 0; i < (int)a.size(); i++) {
      int c_gt = a[i].get("c_gt",-1).asInt();
      int c_pred = a[i].get("c_pred",-1).asInt();
      double l = a[i].get("loss",0).asDouble();
      if(c_gt > num_classes || c_pred > num_classes || c_gt <= 0 || c_pred <= 0) {
	fprintf(stderr, "Error reading Class Confusion Costs\n");
	return false;
      }
      classConfusionCosts[c_gt][c_pred] = l;
    }
  }
  params.max_samples = num_classes-1;

  return true;
}


int g_start, g_end;
int MulticlassStructuredSVM::LoadExample(MulticlassStructuredData *x, MulticlassStructuredLabel *y, bool lazyLoad, char* &line) {
  int failed = 0;
  SparseVector *psi = x->psi = new SparseVector;
  int class_id = -1;
  FILE *fin = x->fins[omp_get_thread_num()];

  if(!line) 
    line = new char[10000000];
  if(x->readBinary) {
    failed = (!fread(&class_id, sizeof(int), 1, fin) || 
	      (!lazyLoad && !psi->read(fin)) || (lazyLoad && !psi->read_skip(fin))) ? 1 : 0;
  } else {
    // SVM-light format (the simplest form of it), assume each example is a line containing a class id followed by %d:%f pairs 
    if(!fgets(line, 99999999, fin) || strlen(line) <= 1) 
      failed = 1;
    chomp(line);
    if(!failed)
      failed = (!sscanf(line, "%d", &class_id) || !psi->from_string(strstr(line, " ")+1, lazyLoad)) ? 2 : 0;
  }   
  if(!failed && class_id <= 0) {
    fprintf(stderr, "Invalid class %d\n", class_id);
    failed = 2;
  } 
  if(y) y->class_id = class_id;
  return failed;
}

// Ordinarily, one need not override this function and should use the default StructuredSVM::LoadDataset() function
// instead.  We override because we want to import SVM^light format files
StructuredDataset *MulticlassStructuredSVM::LoadDataset(const char *fname, bool getLock) {
  if(params.debugLevel > 0) fprintf(stderr, "Reading dataset %s...", fname);
  
  if(getLock) Lock();

  bool detectNumFeatures = 1;
  bool detectNumClasses = num_classes == 0;

  bool readBinary = !strcmp(GetFileExtension(fname), "bin");
  bool lazyLoad = params.memoryMode == MEM_KEEP_NOTHING_IN_MEMORY || params.memoryMode == MEM_KEEP_DUAL_VARIABLES_IN_MEMORY;
  FILE *fin = fopen(fname, readBinary ? "rb" : "r");


  if(!fin) {
    fprintf(stderr, "Couldn't open dataset file %s\n", fname);
    if(getLock) Unlock();
    return NULL;
  }
  MulticlassStructuredDataset *d = new MulticlassStructuredDataset();

  int numThreads = params.num_thr;
  if(!params.runMultiThreaded) numThreads = 1;
  else if(params.runMultiThreaded > 1) numThreads = params.runMultiThreaded;
  d->fins = new FILE*[numThreads];
  d->fins[omp_get_thread_num()] = fin;
  d->numThreads = numThreads;
  if(lazyLoad) {
    for(int i = 0; i < numThreads; i++)
      if(i != omp_get_thread_num())
	d->fins[i] = fopen(fname, readBinary ? "rb" : "r");
  }

  
  while(1) {
    StructuredExample *ex = new StructuredExample;
    ex->x = NewStructuredData();
    ex->y = NewStructuredLabel(ex->x);
    MulticlassStructuredData *x = (MulticlassStructuredData*)ex->x;

    x->seek_pos = FTELL(fin);  
    x->readBinary = readBinary;
    x->fins = d->fins;
    
    int failed;
    if((failed=LoadExample(x, (MulticlassStructuredLabel*)ex->y, lazyLoad, lines[0])) > 0) { 
      delete ex;
      if(!d->num_examples || failed == 2) {
	fprintf(stderr, "Error parsing dataset example %d %s\n", d->num_examples, failed == 2 ? lines[0] : NULL);
	delete d;
      }
      break;
    }

    if(detectNumFeatures) 
      num_features = my_max(num_features, x->psi->Length());
    if(detectNumClasses) {
      num_classes = my_max(num_classes, ((MulticlassStructuredLabel*)ex->y)->class_id);
      params.max_samples = num_classes-1;
    }

    if(lazyLoad) {
      delete x->psi;
      x->psi = NULL;
    } 

    d->AddExample(ex);

  }
  if(!lazyLoad) {
    fclose(fin);
    delete [] d->fins;
    d->fins = NULL;
  }

  if(getLock) Unlock();

  if(detectNumFeatures || detectNumClasses)
    sizePsi = num_features*num_classes;

  char alphaName[1000];
  sprintf(alphaName, "%s.alphas", fname);
  if(!alphasFile && params.memoryMode == MEM_KEEP_NOTHING_IN_MEMORY) {
    double zero = 0;
    alphasFile = fopen(alphaName, "wb");
    for(int i = 0; i < num_classes*d->num_examples; i++)
      fwrite(&zero, sizeof(double), 1, alphasFile);
    fclose(alphasFile);
    alphasFile = fopen(alphaName, "r+b");
  }

  if(params.debugLevel > 0) fprintf(stderr, "done\nNum classes=%d\n", num_classes);

  return d;
}

// Ordinarily, one need not override this function and should use the default StructuredSVM::SaveDataset() function
// instead.  We override because we want to import SVM^light format files
bool MulticlassStructuredSVM::SaveDataset(StructuredDataset *d, const char *fname, int start_from) {
  if(params.debugLevel > 0 && start_from == 0) fprintf(stderr, "Saving dataset %s...", fname);

  Lock();

  bool writeBinary = !strcmp(GetFileExtension(fname), "bin");
  FILE *fout = fopen(fname, writeBinary ? (start_from>0 ? "ab" : "wb") : (start_from>0 ? "a" : "w"));
  if(!fout) {
    fprintf(stderr, "Couldn't open dataset file %s for writing\n", fname);
    Unlock();
    return false;
  }

  for(int i = start_from; i < d->num_examples; i++) {
    if(writeBinary) {
      fwrite(&((MulticlassStructuredLabel*)d->examples[i]->y)->class_id, sizeof(int), 1, fout);
      ((MulticlassStructuredData*)d->examples[i]->x)->psi->write(fout);
    } else {
      // SVM-light format (the simplest form of it), assume each example is a line containing a class id followed by %d:%f pairs 
      char *data = ((MulticlassStructuredData*)d->examples[i]->x)->psi->to_string();
      fprintf(fout, "%d %s\n", ((MulticlassStructuredLabel*)d->examples[i]->y)->class_id, data);
      free(data);
    }
  }
  fclose(fout);
  Unlock();

  if(params.debugLevel > 0 && start_from == 0) fprintf(stderr, "done\n");

  return true;
}


#ifndef NO_SERVER

#include "online_interactive_server.h"

/** 
 * @brief Run the structured learning server
 *
 * To run as a standalone structured learning algorithm with a fixed dataset file, run something like
 *   Train: ./online_interactive_server -p data/params.txt -d data/train.dat -o data/learned_model.txt
 *   Test:  ./online_interactive_server -i data/learned_model.txt -t data/test.dat data/predictions.dat
 *
 * where 
 *   data/params.txt is in the format of StructuredSVM::Load()
 *   data/train.dat is a training set in the format of  StructuredSVM::LoadDataset() (the default implementation 
 *       reads each training example as a line "y x", where y is in the format of StructuredLabel::read()
 *       and x is in the format of StructuredData::read()
 *   data/learned_model.txt is the file where the learned model is written
 *   data/test.dat is a testset in the format of  StructuredSVM::LoadDataset() 
 *   data/predictions.txt is the file where predictions for all labels are written, where each line
 *      corresponds to a test example in the format 
 *          "y_predicted y_ground_truth loss score_prediction score_ground_truth"
 *
 *
 *
 * To run as a server that trains in online fashion, allowing a client to interactively classify examples
 * and add new training examples, run something like
 *   ./online_interactive_server -P 8086 -p data/params.txt -d data/initial_train.txt
 *
 * where data/initial_train.dat is optional and 8086 is the port in which the serve listens on.
 *
 * See StructuredLearnerRpc for info on the network protocol
 *
 **/ 
int main(int argc, const char **argv) {
  StructuredLearnerRpc v(new MulticlassStructuredSVM);
  v.main(argc, argv);
}

#endif
