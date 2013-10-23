#include "structured_svm.h"
#include "visualizationTools.h"


void StructuredSVM::VisualizeDataset(StructuredDataset *dataset, const char *htmlDir, int max_examples) {
  Lock();
  CreateDirectoryIfNecessary(htmlDir);
  int nc = NumHTMLColumns();
  char fname[1000];  sprintf(fname, "%s/index.html", htmlDir);
  FILE *fout = fopen(fname, "w");
  if(!fout) { fprintf(stderr, "Could not open %s for writing\n", fname); return; }
  fprintf(fout, "<html><table>\n");
  for(int i = 0; i < dataset->num_examples && (max_examples < 0 || i < max_examples); i++) {
    char *htmlStr = VisualizeExample(htmlDir, dataset->examples[i]);
    if(i%nc == 0) {
      if(i) fprintf(fout, "</tr>\n");
      fprintf(fout, "<tr>\n");
    }
    fprintf(fout, "<td>%s</td>\n", htmlStr);
    free(htmlStr);
  }
  fprintf(fout, "</tr></table></html>\n");
  fclose(fout);
  Unlock();
}


char *StructuredSVM::VisualizeExample(const char *htmlDir, StructuredExample *ex, const char *extraInfo) {
  Json::StyledWriter writer;
  std::string str = writer.write(ex->y->save(this));
  char *retval = (char*)malloc(strlen(str.c_str())+1);
  strcpy(retval, str.c_str());
  return retval;
}


void StructuredDataset::MakeGallery(const char *fname, const char *title, const char *header) {
  const char **fnames = (const char**)malloc(sizeof(char*)*num_examples*4); 
  memset(fnames, 0, sizeof(char*)*num_examples*4); 
  const char **thumbs = fnames + num_examples;
  const char **descriptions = thumbs + num_examples;
  
  int numFnames = 0, numThumbs = 0, numDescriptions = 0;
  for(int i = 0; i < num_examples; i++) {
    ExampleVisualization *v = examples[i]->visualization;
    if(v && v->fname) {
      if(v->thumb) { assert(numThumbs == numFnames); thumbs[numThumbs++] = v->thumb; }
      if(v->description) { assert(numDescriptions == numFnames); descriptions[numDescriptions++] = v->description; }
      fnames[numFnames++] = v->fname;
    }
  }

  BuildGallery(fnames, numFnames, fname, title, header, numThumbs ? thumbs : NULL, numDescriptions ? descriptions : NULL);
  free(fnames);
}


void StructuredExample::AddExampleVisualization(const char *fname, const char *thumb, const char *description, double loss) {
  if(visualization) free(visualization);
  visualization = AllocateExampleVisualization(fname, thumb, description, loss);
}




SVM_cached_sample_set *SVM_cached_sample_set_thread_safe_copy(SVM_cached_sample_set *set) {
  SVM_cached_sample_set *retval = (SVM_cached_sample_set*)malloc(sizeof(SVM_cached_sample_set));
  *retval = *set;
  if(set->u_i) 
    retval->u_i = new SparseVector(*set->u_i);
  retval->samples = (SVM_cached_sample*)malloc(sizeof(SVM_cached_sample)*(set->num_samples));
  memcpy(retval->samples, set->samples, sizeof(SVM_cached_sample)*(set->num_samples)); 
  return retval;
}

void free_SVM_cached_sample_set_thread_safe_copy(SVM_cached_sample_set *set) {
  if(set->u_i) 
    delete set->u_i;
  if(set->samples) 
    free(set->samples);
  free(set);
}

/*
void StructuredSVM::EstimateOutOfSampleError() {
  int allocSize = 1000;
  bool *removed_examples_map = (bool*)malloc(sizeof(bool)*allocSize);
  int *removed_examples = (int*)malloc(sizeof(bool)*allocSize);
  int iterAllocSize = 0;
  StructuredSVM estimator(*this);
  
  numOutOfSampleIters = 0;
  while(!finished) {
    int batch_size = my_max(1, my_min(100, (int)(.05*n)));
    int num_iters = my_max(1, my_min(1000, n));
    if(examples_by_iteration_number[0] >= 0) {
      Lock();
      if(examples_by_iteration_number[0] >= 0) {
	
      }
      Unlock();
    }

    if(numOutOfSampleIters + num_iters >= iterAllocSize) {
      iterAllocSize = 100 + (int)(iterAllocSize*1.1);
      out_of_sample_hinge_error = (float*)realloc(out_of_sample_hinge_error, sizeof(float)*iterAllocSize);
      out_of_sample_loss = (float*)realloc(out_of_sample_loss, sizeof(float)*iterAllocSize);
    }
    if(n >= allocSize) {
      int s = allocSize;
      allocSize = 100 + (int)(allocSize*1.1);
      removed_examples_map = (bool*)realloc(removed_examples_map, sizeof(bool)*allocSize);
      removed_examples = (int*)realloc(removed_examples, sizeof(int)*allocSize);
      memset(removed_examples_map+s, 0, sizeof(bool)*(allocSize-s));
      memset(removed_examples+s, 0, sizeof(int)*(allocSize-s));
    }
    estimator.EstimateOutOfSampleError(this, batch_size, num_iters, removed_examples_map, removed_examples, 
				       out_of_sample_hinge_error+numOutOfSampleIters, out_of_sample_loss+numOutOfSampleIters);
    numOutOfSampleIters += num_iters;
  }

  free(removed_examples_map);
  free(removed_examples);
}

void StructuredSVM::EstimateOutOfSampleError(StructuredSVM *svm, int batch_size, int num_iters, bool *removed_examples_map, int *removed_examples, 
					     float *out_of_sample_hinge_error, float *out_of_sample_loss) {
  svm->Lock();
  sum_w = svm->sum_w->Copy();
  sum_dual = svm->sum_dual;
  sum_alpha_loss = svm->sum_alpha_loss;
  sum_w_sqr = svm->sum_w_sqr;
  sum_w_scale = svm->sum_w_scale;
  regularization_error = svm->regularization_error;
  n = svm->n;
  svm->Unlock();

  // Remove batch_size examples from the training set, by setting their dual parameters to zero
  for(int i = 0; i < batch_size; i++) {
    svm->Lock();
    int j = rand()%n;
    if(trainset->examples[j]->set->lock)
      continue;
    trainset->examples[j]->set->lock = true;
    removed_examples[i] = j;
    removed_examples_map[j] = true;

    SVM_cached_sample_set *set = SVM_cached_sample_set_thread_safe_copy(trainset->examples[j]->set);
    svm->Unlock();

    RemoveExample(set);
    free_SVM_cached_sample_set_thread_safe_copy(set);
    trainset->examples[j]->set->lock = false;
  }

  // Optimize the current model weights to adjust themselves for the removed examples
  for(int k = 0; k < num_iters; k++) {
    svm->Lock();
    int i = rand() % n;
    if(removed_examples_map[i] || trainset->examples[i]->set->lock) {
      svm->Unlock();
      continue;  // exclude optimizing over examples we just removed
    }
    trainset->examples[i]->set->lock = true;
      
    SVM_cached_sample_set *set = SVM_cached_sample_set_thread_safe_copy(trainset->examples[i]->set);
    svm->Unlock();

    MultiSampleUpdate(set, svm->trainset->examples[set->i], 1);
    free_SVM_cached_sample_set_thread_safe_copy(set);
    trainset->examples[i]->set->lock = false;
  }
  
  // Measure error on the removed examples
  SparseVector *w = GetCurrentWeights(false);
  for(int i = 0; i < batch_size; i++) {
    int j = removed_examples[i];
    removed_examples_map[j] = false;

    if(bruteForce) {
      svm->Lock();
      StructuredExample *ex = CopyExample(svm->trainset->examples[j]->x, svm->trainset->examples[j]->y);
      svm->Unlock();
      double score_gt = w->dot(Psi(svm->trainset->examples[j]->x, svm->trainset->examples[j]->y));
      double score = Inference(ex->x, ybar, w, NULL, NULL, 1);
      double loss = Loss(ex->y, ybar);
      double slack = Inference(ex->x, ybar, w, NULL, ex->y, 1) - score_gt;
      out_of_sample_hinge_error[i] = slack;
      out_of_sample_loss[i] = loss;
      delete ex;
    } else {
      svm->Lock();
      SVM_cached_sample_set *set = trainset->examples[j]->set;
      double score_gt = w->dot(set->psi_gt); 
      double max_slack = 0, loss = 0, max_score = 0;
      for(int k = 0; k < set->num_samples; k++) {
	double score = sum_w->dot(*set->samples[k].psi)/sum_w_scale;
	double slack = score-score_gt + set->samples[k].loss;
	max_slack = my_max(max_slack, slack);
	if(score > max_score) {
	  max_score = score;
	  loss = set->samples[k].loss;
	}
      }
      svm->Unlock();

      out_of_sample_hinge_error[i] = max_slack;
      out_of_sample_loss[i] = loss;
    }
  }
  delete w;
}


void StructuredSVM::ComputeSuggestedLabels() {
  while(!finished) {
      if(numThreads!=1 || !canScaleW) delete w;
      w = numThreads==1 && canScaleW ? sum_w : GetCurrentWeights(false);
      Lock(); UpdateWeights(set, -1); Unlock();
      double score_best = Inference(ex->x, set->ybar, w, NULL, NULL, w_scale);
      set->slackSuggest = score_best - (set->psi_gt ? w->dot(*set->psi_gt)*w_scale : 0);
  }
}

*/
