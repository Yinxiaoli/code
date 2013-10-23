#include "structured_svm.h"
#include "structured_svm_train_choose_example.h"

#ifdef HAVE_SVM_STRUCT
extern StructuredSVM *g_learner;
int main_train (int argc, char* argv[]);
#endif



void StructuredSVM::TrainCuttingPlane(const char *modelout, bool saveFull, const char *initial_sample_set)  {
#ifdef HAVE_SVM_STRUCT
  g_learner = this;
  char C_str[1000], eps_str[1000], modelfile[1000];
  sprintf(C_str, "%lf", params.C);
  sprintf(eps_str, "%lf", params.eps);
  if(modelout) strcpy(modelfile, modelout);
  char* argv[11] = {(char*)"./svm_struct_learn.out", (char*)"-c", C_str, (char*)"-e", eps_str, 
                    (char*)"-M", modelfile, (char*)"-b", (char*)"1", (char*)"-w", (char*)"4" };
  start_time = get_runtime();
  main_train(params.method == SPO_CUTTING_PLANE_1SLACK ? 11 : (modelout ? 7 : 5), argv);
#else
  fprintf(stderr, "ERROR: to train using SVM^struct, you must define -DHAVE_SVM_STRUCT in the Makefile and link the SVM^struct library\n");
#endif
}


void StructuredSVM::TrainMineHardNegatives(const char *modelout, bool saveFull, const char *initial_sample_set, 
					   int numMineHardNegativesRound, int numHardNegativesPerExample)  {
  if(params.debugLevel > 2) params.debugLevel--;

  // Pre-extract a training set of samples
  int iter = 0;
  while(iter < numMineHardNegativesRound) {
    long old_n = n;
    StructuredDataset *old_trainset = trainset;
    char sample_set_name[1000];
    
    sprintf(sample_set_name, "%s.samples.%d", modelout, iter);
    params.maxIters = my_min(params.maxIters,1000000);

    // Extract a binary sample set, or load a cached sample set from disk if it exists
    double elapsedTime = GetElapsedTime();
    if(!FileExists(sample_set_name)) {
      ExtractSampleSet(numHardNegativesPerExample, iter > 0);
      chooser->SaveCachedExamples(sample_set_name);
    } else if(params.method == SPO_MAP_TO_BINARY_MINE_HARD_NEGATIVES) {
      int i = iter;
      while(FileExists(sample_set_name)) 
        sprintf(sample_set_name, "%s.samples.%d", modelout, ++i);
      iter = i-1;
      sprintf(sample_set_name, "%s.samples.%d", modelout, iter);
      fprintf(stderr, "Loading %s...\n", sample_set_name);
      chooser->LoadCachedExamples(sample_set_name);
    }
    double elapsedTime2 = GetElapsedTime();
    if(iter) 
      start_time += elapsedTime2-elapsedTime;
    old_n = n;
    old_trainset = trainset;
    
    ConvertCachedExamplesToBinaryTrainingSet();

    // Train a binary classifier
    TrainStructured(modelout, saveFull, initial_sample_set);

    // Restore the original structured training set
    for(int i = 0; i < n; i++) {
      if(trainset->examples[i]->set) {
        trainset->examples[i]->set->samples = NULL;
        trainset->examples[i]->set->num_samples = 0;
        trainset->examples[i]->set->psi_gt = NULL;
        free_SVM_cached_sample_set(trainset->examples[i]->set);
        trainset->examples[i]->set = NULL;
      }
    }
    trainset = old_trainset;
    n = old_n;

    iter++;
  }
}

void StructuredSVM::TrainBinary(const char *modelout, bool saveFull, const char *initial_sample_set)  {
  ConvertCachedExamplesToBinaryTrainingSet();
  TrainStructured(modelout, saveFull, initial_sample_set);
  if(modelout)
    DumpModelIfNecessary(modelout, true, true);
}

void StructuredSVM::ExtractSampleSet(int num_per_negative, bool augment) {
  StructuredDataset *train = GetTrainset(); 

  params.max_samples = num_per_negative;

  Lock();
  if(augment) {
    for(int i = 0; i < n; i++) 
      if(trainset->examples[i]->set || trainset->examples[i]->cache_fname) 
	CondenseSamples(chooser->BringCacheSetIntoMemory(i, nextUpdateInd, false));
  }
  Unlock();

  SparseVector *w = sum_w ? GetCurrentWeights(false) : NULL;
  int num = 0;

#pragma omp parallel for
  for(int i = 0; i < train->num_examples; i++) {
    chooser->BringCacheSetIntoMemory(i, nextUpdateInd, false);    
	num++;
    assert(trainset->examples[i]->set);
    trainset->examples[i]->set->score_gt = sum_w ? sum_w->dot(*trainset->examples[i]->set->psi_gt)/sum_w_scale : 0; 
    fprintf(stderr, "Extracting sample %d of %d...\n", num, train->num_examples);

    // Mine hard negative examples.  If the model is uninitialized, this instead draws random
    // negative examples.  Features are extracted for each sample
    ImportanceSample(train->examples[i]->x, w, train->examples[i]->y, trainset->examples[i]->set, 1);
    SVM_cached_sample_set_compute_features(trainset->examples[i]->set, train->examples[i]);
    OnFinishedIteration(train->examples[i]);
  }

  n = train->num_examples;
  SetSumWScale(params.lambda*n);
  delete w;
}

void StructuredSVM::CondenseSamples(SVM_cached_sample_set *set) {
  if(!set->u_i && !set->alpha && set->num_samples && params.method != SPO_MAP_TO_BINARY && 
     params.method != SPO_MAP_TO_BINARY_MINE_HARD_NEGATIVES && set->samples[0].psi) {
    set->alpha = set->samples[0].alpha;	
    set->u_i = set->samples[0].psi;
    set->samples[0].psi = NULL;
    *set->u_i *= set->samples[0].alpha;	
    set->D_i = set->samples[0].alpha*set->samples[0].loss;
    set->dot_u_psi_gt = set->samples[0].alpha*set->samples[0].dot_psi_gt_psi;
    set->u_i_sqr = SQR(set->samples[0].alpha)*set->samples[0].sqr;
  }

  if(set->num_samples > params.maxCachedSamplesPerExample) {
    qsort(set->samples, set->num_samples, sizeof(SVM_cached_sample), SVM_cached_sample_alpha_cmp);

    for(int i = params.maxCachedSamplesPerExample; i < set->num_samples; i++) {
      if(!set->samples[i].alpha)
        free_SVM_cached_sample(&set->samples[i]);
      else {
        if(set->samples[i].psi)
          delete set->samples[i].psi;
        set->samples[i].psi = NULL;
        set->evicted_samples = (SVM_cached_sample*)realloc(set->evicted_samples, sizeof(SVM_cached_sample)*(set->num_evicted_samples+1));
        set->evicted_samples[set->num_evicted_samples++] = set->samples[i];
      }
    }
    set->num_samples = params.maxCachedSamplesPerExample;
  }
}	  

void StructuredSVM::ConvertCachedExamplesToBinaryTrainingSet() {
  long num = 0;
  StructuredDataset *binary_trainset = new StructuredDataset;

  for(int i = 0; i < n; i++) {
    chooser->BringCacheSetIntoMemory(i, nextUpdateInd, false);
    for(int j = 0; j < trainset->examples[i]->set->num_samples; j++) {
      binary_trainset->examples[num]->set = new_SVM_cached_sample_set(num, trainset->examples[i]->set->psi_gt);
      *binary_trainset->examples[num]->set = *trainset->examples[i]->set;
      binary_trainset->examples[num]->set->i = num;
      binary_trainset->examples[num]->set->num_samples = 1;
      binary_trainset->examples[num++]->set->samples = trainset->examples[i]->set->samples+j;
      binary_trainset->AddExample(CopyExample(trainset->examples[i]->x, trainset->examples[i]->y, trainset->examples[i]->y_latent));
    }
  }
  
  trainset = binary_trainset;
  n = num;
  SetSumWScale(params.lambda*n);
}



void StructuredSVM::InferLatentValues(StructuredExample *ex, SparseVector *w, double w_scale) {
  if(!ex->y)
    ex->y = NewStructuredLabel(ex->x);
  Inference(ex->x, ex->y, w, ex->y_latent ? ex->y_latent : ex->y, NULL, w_scale);
  Lock();
  OnRelabeledExample(ex);
  Unlock();
  OnFinishedIteration(ex);
}


void StructuredSVM::InferLatentValues(StructuredDataset *d) {
  SparseVector *w = GetCurrentWeights(false);
  double sum_dual_before = sum_dual;

  if(params.debugLevel)
    fprintf(stderr, "Inferring latent values...\n");

#pragma omp parallel for
  for(int i = 0; i < d->num_examples; i++) {
    InferLatentValues(d->examples[i], w);
  }
  if(params.debugLevel)
    fprintf(stderr, "done (%f->%f)\n", (float)sum_dual_before, (float)sum_dual);

  sum_w_sqr = sum_w->dot(*sum_w);
  regularization_error = sum_w_sqr/SQR(sum_w_scale)*params.lambda/2;
  sum_dual = -sum_w_sqr/(2*sum_w_scale) + sum_alpha_loss;
}

