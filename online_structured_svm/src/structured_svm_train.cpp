#include "structured_svm.h"
#include "structured_svm_train_statistics.h"
#include "structured_svm_train_choose_example.h"


void StructuredSVM::Train(const char *modelout, bool saveFull, const char *initial_sample_set)  {
  runForever = saveFull && params.method != SPO_MAP_TO_BINARY_MINE_HARD_NEGATIVES;
  if(!trainset) trainset = new StructuredDataset;

  finished = false;
  params.lambda = 1.0/params.C;

  // If useFixedSampleSet=true, never call Inference() or ImportanceSample() to concentrate on 
  // different samples.  Instead, use a fixed set of of feature vectors
  useFixedSampleSet = params.method == SPO_MAP_TO_BINARY || 
                      params.method == SPO_MAP_TO_BINARY_MINE_HARD_NEGATIVES || 
                      params.method == SPO_FIXED_SAMPLE_SET;

  // If cache_old_examples=true, then cache features for extracted samples, such that we can
  // optimize with respect to extracted samples in future iterations before calling 
  // Inference() or ImportanceSample() again
  cache_old_examples = params.method == SPO_DUAL_UPDATE_WITH_CACHE ||
                       params.method == SPO_DUAL_MULTI_SAMPLE_UPDATE_WITH_CACHE || 
                       useFixedSampleSet;

  // If isMultiSample=true, then call ImportanceSample() instead of Inference(), such that
  // we extract a set of samples in each iteration instead of just one
  isMultiSample = params.method == SPO_DUAL_MULTI_SAMPLE_UPDATE ||
                  params.method == SPO_DUAL_MULTI_SAMPLE_UPDATE_WITH_CACHE ||
                  params.method == SPO_FIXED_SAMPLE_SET;


  // Begin with a pre-extracted set of samples (e.g., cached features from a previous training session)
  if(initial_sample_set) {
    chooser->LoadCachedExamples(initial_sample_set);
    SetSumWScale(cache_old_examples ? params.lambda*n :params.lambda*t);
  }

  if(params.method == SPO_CUTTING_PLANE || params.method == SPO_CUTTING_PLANE_1SLACK) 
    TrainCuttingPlane(modelout, saveFull, initial_sample_set); // Run SVM^struct instead of our own optimizer
  else if(params.method == SPO_MAP_TO_BINARY)
    TrainBinary(modelout, saveFull, initial_sample_set);
  else if(params.method == SPO_MAP_TO_BINARY_MINE_HARD_NEGATIVES) 
    TrainMineHardNegatives(modelout, saveFull, initial_sample_set, 100, params.maxCachedSamplesPerExample);
  else
    TrainStructured(modelout, saveFull, initial_sample_set);

  fprintf(stderr, "Training finished in %lf seconds\n", GetElapsedTime()); 

  if(modelout)
    Save(modelout, saveFull);
}

void StructuredSVM::TrainStructured(const char *modelout, bool saveFull, const char *initial_sample_set) {
  int numThreads = TrainInit(modelout, initial_sample_set);
  bool allocW = !(/*numThreads==1 &&*/ params.canScaleW);

#ifdef WIN32
  // Weird thing happens in release mode on windows, where some variables don't get shared.  Having a meaningless
  // shared loop before hand seems to fix it
  #pragma omp parallel num_threads(numThreads) 
  {
    int i = 0;
    while(i < 100) {
      i++;
      this->finished = false;
    }
  }
#endif

  #pragma omp parallel num_threads(numThreads)
  {
    int tid = omp_get_thread_num();
    if(tid > 0 || !cache_old_examples || numThreads<=1 || !params.updateFromCacheThread) {
      // Worker threads, continuously call ybar=Inference or set=ImportanceSample and then use ybar to 
      // update the current weights

      while(!finished) {
	TrainOptimizeOverOneExample(allocW);
    
	// Optionally periodically save the learned model to disk
	if(modelout && params.dumpModelStartTime)
	  DumpModelIfNecessary(modelout, false, true);
      }
    } else  {

      // Occurs only if updateFromCacheThread==true
      // Optimization thread, each iteration randomly selects a previous cached example ybar_i (earlier calls
      // to finding the most violated constraint) and then makes an update to w with respect to that sample.  
      // Can be useful if the operation of finding the most violated constraint is much more expensive than 
      // the operation of updating w
      while(!finished && t <= numThreads) // wait for at least one first constraint
        usleep(100);
      TrainOptimizeOverPreExtractedSamples();
    }

    fprintf(stderr, "Finishing thread %d\n", tid);
  }
  fprintf(stderr, "Finishing training\n");
}


void StructuredSVM::TrainOptimizeOverOneExample(bool allocW) {
  StructuredExample *ex;
  SparseVector *w;
  double w_scale;
  int i;
  SVM_cached_sample_set *set;
  VFLOAT score = 0;
  double score_loss = 0;

  // Choose a training example 'i' to process
  if(!TrainPrepareExample(i, ex, set, w, w_scale, allocW))
    return;
    
  // Compute a feature vector or set of feature vectors that will be used to update the model weights
  TrainComputeExampleSubGradientOrSampleSet(i, ex, set, w, w_scale, allocW, score, score_loss);

  Lock();
  TrainUpdateWeights(set, -1);  // Use the sample(s) for training example i to update the model weights
  stats->UpdateStatistics(set, t-1, sum_dual, regularization_error, cache_old_examples);   // Update estimates of training and model error
  chooser->UpdateExampleIterationQueue(set, t-1);  // Maintain a queue used to select which example to process next
  
  if(t > 1000 && i % n == n-1)  // avoid numerical drifting problems
    RecomputeWeights(false);
    
  stats->CheckConvergence(finished, hasConverged, runForever, cache_old_examples, sum_dual, regularization_error);           // Check if optimization is finished

  if(cache_old_examples) 
    trainset->examples[i]->set = set;
  
  if(params.debugLevel > 2) {
    printf("Example %d: m=%d slack=%f->%f score=%f score_gt=%f loss=%f alpha=%f num_samples=%d\n",
	   i, GetCurrentEpoch(), (VFLOAT)set->slack_before, (VFLOAT)set->slack_after, (VFLOAT)(score), 
	   (VFLOAT)(set->score_gt), (VFLOAT)set->loss, (VFLOAT)set->alpha, set->num_samples);
  }

  // Cleanup
  if(allocW) 
    delete w;
  if(!cache_old_examples) {
    chooser->RemoveCacheSetFromMemory(set);
    set = NULL;
  } else if(!useFixedSampleSet && params.mergeSamples) 
    CondenseSamples(set);
    
  if(set)
    set->lock = false;
  
  Unlock();

  OnFinishedIteration(ex);  // called because the API user might want to free temporary memory caches
  if(t % n == n-1) 
    OnFinishedPassThroughTrainset();   
}




 bool StructuredSVM::TrainPrepareExample(int &i, StructuredExample *&ex, SVM_cached_sample_set *&set, 
					 SparseVector *&w, double &w_scale, bool allocW) {
   Lock();
   w_scale = !allocW ? (t ? 1.0/sum_w_scale : 1) : 1;
   i = chooser->ChooseNextExample(true);
   if(i < 0 || (trainset->examples[i]->set && trainset->examples[i]->set->lock)) {
     Unlock();
     usleep(100000);
     return false;
   }
   ex = trainset->examples[i];
   set = chooser->BringCacheSetIntoMemory(i, nextUpdateInd, true);

   if(cache_old_examples && !useFixedSampleSet && params.numCacheUpdatesPerIteration && t) {
     // Optimize weights with respect to previously cached samples for this example, 
     // to ensure returned samples are as independent as possible
     if(chooser->GetExNumIters()[i])  
       TrainUpdateFromCache(false, &numCacheIters, i);
     
     // Optimize weights with respect to cached samples that are randomly selected
     if(!params.updateFromCacheThread) {  
       for(int k = 0; k < params.numCacheUpdatesPerIteration; k++) {
	 if(nextUpdateInd >= n || nextUpdateInd >= t)
	   nextUpdateInd = 0;
	 TrainUpdateFromCache(false, &numCacheIters, nextUpdateInd);
	 if(nextUpdateInd >= 0) nextUpdateInd++;
       }
     }
   }
   w = !allocW ? sum_w : GetCurrentWeights(false);
   set->score_gt = set->psi_gt ? w->dot(*set->psi_gt)*w_scale : 0;   // <w_t,psi(x_i,y_i)>
   Unlock();

   return true;
 }


 void StructuredSVM::TrainComputeExampleSubGradientOrSampleSet(int &i, StructuredExample *&ex, 
							       SVM_cached_sample_set *&set, 
							       SparseVector *&w, double &w_scale,
							       bool allocW, double &score,
							       double &score_loss) {
   if(!useFixedSampleSet) {
     if(!isMultiSample) {
       // Find the most violated label ybar = max_y <w,Psi(x_i,y)>+Loss(y_i,y)
       StructuredLabel *ybar = NewStructuredLabel(ex->x);
       score_loss = Inference(ex->x, ybar, w, NULL, ex->y, w_scale);
       SVM_cached_sample *sample = SVM_cached_sample_set_add_sample(set, ybar);
       sample->slack = score_loss - set->score_gt;
     } else {
       // Extract a set of samples with non-zero slack.  Should include the most violated label ybar
       score_loss = ImportanceSample(ex->x, w, ex->y, set, w_scale);
     }

     SVM_cached_sample_set_compute_features(set, ex);
     set->loss = set->num_samples ? set->samples[0].loss : 0;
     if(set->ybar) delete set->ybar;
     set->ybar = NULL;
     score = score_loss-set->loss;      // <w_t,psi(x_i,y)>
     set->slack_before = score_loss - set->score_gt;

     set->ybar = NewStructuredLabel(ex->x);
     if(set->num_samples && set->samples[0].ybar) {
       Json::Value yy = set->samples[0].ybar->save(this);
       set->ybar->load(yy, this);
     }
   } else {
     // Optimize over pre-extracted sample set, instead of dynamically choosing new samples using 
     //Inference() or ImportanceSample()
     set->slack_before = -1000000;
     for(int j = 0; j < set->num_samples; j++) {
       set->samples[j].dot_w = sum_w->dot(*set->samples[j].psi) - set->score_gt*sum_w_scale;
       set->samples[j].slack = set->samples[j].dot_w/sum_w_scale + set->samples[j].loss;
       if(set->samples[j].slack > set->slack_before) {
	 set->slack_before = set->samples[j].slack;
	 set->loss = set->samples[j].loss;
	 score_loss = set->slack_before + set->score_gt;
	 score = score_loss-set->loss;
       }
     }
   }
   set->numIters++;
   set->sumSlack += set->slack_before;
 }
 

void StructuredSVM::TrainOptimizeOverPreExtractedSamples() {
  while(!finished) {
    usleep(1);
    Lock();
    if(params.numCacheUpdatesPerIteration) {
      if(nextUpdateInd >= n || nextUpdateInd >= t)
        nextUpdateInd = 0;
      TrainUpdateFromCache(false, &numCacheIters, nextUpdateInd);
      if(nextUpdateInd >= 0) nextUpdateInd++;
      stats->CheckConvergence(finished, hasConverged, runForever, cache_old_examples, sum_dual, regularization_error); 
    }
    chooser->PrefetchCache(nextUpdateInd);
    Unlock();
  }
}

void StructuredSVM::TrainUpdateFromCache(bool lock, int *num, int i) {
  // Choose a label from a random iteration (a label ybar from a previous worker thread iteration), 
  // and optimize its dual parameters
  if(!n) return;
  if(lock) Lock();
  if(i < 0) i = rand()%(n);
  if((!trainset->examples[i]->set && !trainset->examples[i]->cache_fname) || 
     (trainset->examples[i]->set && trainset->examples[i]->set->lock)) {
    if(lock) Unlock();
    return;
  }
  SVM_cached_sample_set *set = chooser->BringCacheSetIntoMemory(i, nextUpdateInd, false); 
  set->score_gt = sum_w->dot(*set->psi_gt)/sum_w_scale;   // <w_t,psi(x_i,y_i)>
  for(int j = 0; j < set->num_samples; j++) 
    set->samples[j].dot_w = sum_w ? sum_w->dot(*set->samples[j].psi) - set->score_gt*sum_w_scale : 0;
  TrainUpdateWeights(set, i);

  if(num) {
    *num++;
    if(*num >= t*50) {
      // Sanity check: occasionally recompute weights from dual parameters.  Could help avoid drifting 
      // due to numerical precision errors
      RecomputeWeights(true);
      *num = 0;
    }
  }
  if(lock) Unlock();
}

// Initialize data structures for training
int StructuredSVM::TrainInit(const char *modelout, const char *initial_sample_set) {
  nextUpdateInd = -1;//0;
  numCacheIters = 0;

  int numThreads = params.num_thr;
  if(!params.runMultiThreaded) numThreads = 1;
  else if(params.runMultiThreaded > 1) numThreads = params.runMultiThreaded;
  if(params.updateFromCacheThread && cache_old_examples) numThreads++;
  //if(numThreads < 2 && cache_old_examples) numThreads = 2;

  SetTrainset(trainset);
  
  if(!sum_w) {
    sum_w = new SparseVector;
    sum_w->make_non_sparse(true, sizePsi);
  }

  chooser->Init(modelout);

  if(cache_old_examples) {
    n = trainset->num_examples;
    SetSumWScale(params.lambda*n);
    if(initial_sample_set || AddInitialSamples())
      OptimizeAllConstraints(300);
  }

  /* some training information */
  if(params.debugLevel > 0) {
    char mstr[1000];  OptimizationMethodToString(params.method, mstr);
    char mem_str[1000];  MemoryModeToString(params.memoryMode, mem_str);
    printf("Number of threads=%d\n", numThreads);
    printf("Optimization method=%s\n", mstr);
    printf("Memory method=%s\n", mem_str);
    printf("Regularization constant (lambda): %.8lg\n", params.lambda);
    printf("Approximation factor (epsilon): %.8lg\n", params.eps);
    printf("Number of training examples (n): %ld\n", (long)trainset->num_examples);
    printf("Feature space dimension (sizePsi): %d\n", sizePsi); fflush(stdout);
  }

  return numThreads;
}

void StructuredSVM::TrainUpdateWeights(SVM_cached_sample_set *ex, int iterInd) {
  bool bound_w = params.method != SPO_SGD; 
  
  for(int j = 0; j < ex->num_samples; j++)
    if(ex->samples[j].loss > params.maxLoss) 
      params.maxLoss = ex->samples[j].loss;

  if(chooser->GetExNumIters()[ex->i] == 0) {
    if(n < trainset->num_examples) {
      n++;
      if(cache_old_examples) 
        SetSumWScale(params.lambda*n);
    }
  }

  // When we increase t, w^{t+1} = t/(t+1) * w^{t}, causing a change in the regularization error and dual
  if(iterInd == -1) {
    t++;
    if(!cache_old_examples) 
      SetSumWScale(params.lambda*t);
  }

 
  if(!isMultiSample && !ex->u_i) {
    // Update the model by taking a step in the direction of the subgradient v=ex->psi_gt-ex->sample[0].psi
    SingleSampleUpdate(ex, params.method != SPO_SGD && params.method != SPO_SGD_PEGASOS);
  } else {
    // Take a more complicated update step with respect to multiple samples, instead of just the
    // sub-gradient
    MultiSampleUpdate(ex, trainset->examples[ex->i], iterInd >= 0 ? 1 : params.numMultiSampleIterations);
  }

  if(bound_w) {
    // Project sum_w onto the L2 ball, such that ||w||^2 <= 1/lambda
    // This is the projection step used by a Pegasos-like update
    if(regularization_error > .5*params.maxLoss) {
      // scale w by s = (1/sqrt(lambda))/|w| = sqrt(1/lambda/w^2) = sqrt(1/(2*regularization_error))
      double s = sqrt(params.maxLoss / (2*regularization_error));
      *sum_w *= s;
      sum_w_sqr = 1.0/params.lambda*SQR(sum_w_scale);
      sum_alpha_loss *= s;
      sum_dual = -sum_w_sqr/(2*sum_w_scale) + sum_alpha_loss;
      //sum_dual = s*sum_dual + (1-s)*t/(2*s);
      regularization_error = .5*params.maxLoss;
    }
  }
}


void StructuredSVM::DumpModelIfNecessary(const char *modelfile, bool force, bool getLock) {
  // Periodically dump the current learned model to disk, in geometrically increasing time intervals
  if(getLock) Lock();
  double elapsedTime = GetElapsedTime();
  if((params.dumpModelStartTime && elapsedTime >= params.dumpModelStartTime*pow(params.dumpModelFactor,params.numModelDumps)) || force) {
    double tm_beg = get_runtime();
    char modelfile_times[1000], modelfile_dump[1000];
    sprintf(modelfile_times, "%s.times", modelfile);
    sprintf(modelfile_dump, "%s.%d", modelfile, params.numModelDumps);
    Save(modelfile_dump, false, false);
    double loss = 0, svm_err = 0;
    if(validationfile)
      loss = Test(validationfile, NULL, NULL, &svm_err, false);

    FILE *fout = fopen(modelfile_times, params.numModelDumps ? "a" : "w");
    fprintf(fout, "%d %lf %d %lf %lf %lf\n", params.numModelDumps, elapsedTime, (int)t, 
	    sum_dual/(cache_old_examples ? n : t), loss, svm_err);
    fclose(fout);
    while(elapsedTime >= params.dumpModelStartTime*pow(params.dumpModelFactor,params.numModelDumps)) 
      params.numModelDumps++;
    double tm_end = get_runtime();
    start_time += tm_end-tm_beg;
  }
  if(getLock) Unlock();
}

// Sanity check: recompute weights from dual parameters.  Could avoid drifting due to numerical precision errors
void StructuredSVM::RecomputeWeights(bool full) {
  int i;
  if(full) {
    if(cache_old_examples || useFixedSampleSet) {
      SparseVector *sum_w_new = new SparseVector;
      sum_w_new->make_non_sparse(true, sizePsi);
      sum_alpha_loss = 0;
      for(i = 0; i < n; i++) {
	if(trainset->examples[i]->set || trainset->examples[i]->cache_fname) {
	  SVM_cached_sample_set *set = chooser->BringCacheSetIntoMemory(i, nextUpdateInd, false); 
	  if(set) {
	    if(set->alpha) 
	      *sum_w_new += set->psi_gt->mult_scalar(set->alpha);
	    if(set->u_i) {
	      *sum_w_new -= *set->u_i;
	      sum_alpha_loss += set->D_i;
	    } else {
	      for(int j = 0; j < set->num_samples; j++) {
		if(set->samples[j].alpha)
		  *sum_w_new -= set->samples[j].psi->mult_scalar(set->samples[j].alpha);
		sum_alpha_loss += set->samples[j].alpha * set->samples[j].loss;
	      }
	    }
	  }
	}
      }
      //assert(sum_sqr_diff_ss(sum_w, sum_w_new) < .001*sum_w_scale);
      delete sum_w;
      sum_w = sum_w_new;
	  /*
      sum_iter_error = 0;
      for(i = 0; i < t; i++) 
	sum_iter_error += iter_errors_by_t[i];
	*/
    }
  }
  /*
  sum_iter_error_window = 0;
  for(i = my_max(0,t-window); i < t; i++) 
    sum_iter_error_window += iter_errors_by_t[i];
  assert(sum_iter_error_window >= 0);
  */
  sum_w_sqr = sum_w->dot(*sum_w);
  regularization_error = sum_w_sqr/SQR(sum_w_scale)*params.lambda/2;
  sum_dual = -sum_w_sqr/(2*sum_w_scale) + sum_alpha_loss;
}

void StructuredSVM::OptimizeAllConstraints(int num_iter) {
  double iter_dual = 0;
  int i = 0;
  while((num_iter && i < num_iter) || (!num_iter && sum_dual/n-iter_dual/(i+1) > params.eps)) {
    for(int j = 0; j < n; j++)
      if(trainset->examples[j]->set || trainset->examples[j]->cache_fname)
	TrainUpdateWeights(chooser->BringCacheSetIntoMemory(j, nextUpdateInd, false), j);
    RecomputeWeights(true);
    iter_dual += sum_dual/n;
    fprintf(stderr, "OptimizeAllConstraints i=%d dual=%lf e=%lf\n", i, sum_dual/n, sum_dual/n-iter_dual/(i+2));
    i++;
  }
}

void StructuredSVM::SetSumWScale(double sum_w_scale_new) {
  sum_dual += .5*((sum_w_scale ? 1/sum_w_scale : 0) - 1/sum_w_scale_new)*sum_w_sqr;
  sum_w_scale = sum_w_scale_new;
  regularization_error = (sum_w_sqr/SQR(sum_w_scale))*params.lambda/2;
}

void StructuredSVM::SetLambda(double l, int num_iter) {
  Lock();
  params.lambda = l;
  params.C = (VFLOAT)(1.0/params.lambda);
  if(n) SetSumWScale(cache_old_examples ? params.lambda*n :params.lambda*t);
  if(sizePsi && t && num_iter) {
    RecomputeWeights();
    OptimizeAllConstraints(num_iter);
  }
  Unlock();
}


void StructuredSVM::SetTrainset(StructuredDataset *t) { 
  trainset=t; 
  exampleIdsToIndices = (int*)realloc(exampleIdsToIndices, sizeof(int)*(t->num_examples));
  exampleIndicesToIds = (int*)realloc(exampleIndicesToIds, sizeof(int)*(t->num_examples));
  numExampleIds = 0;
  for(int i = 0; i < t->num_examples; i++) {
    exampleIdsToIndices[numExampleIds] = i;
    exampleIndicesToIds[i] = numExampleIds++;
  }
  
  chooser->Reset();

  regularization_error = 0;
  sum_dual = 0;
  sum_alpha_loss = 0;
  sum_w_sqr = 0;
  this->t = 0;
  if(sum_w) delete sum_w;
  sum_w = NULL;
}

void StructuredSVM::SingleSampleUpdate(SVM_cached_sample_set *set, bool useSmartStepSize) {
  assert(set->num_samples == 1);
  if(params.runMultiThreaded)
    set->samples[0].dot_w = sum_w->dot(*set->samples[0].psi) - set->score_gt*sum_w_scale;

  // SPO_SGD and SPO_SGD_PEGASOS: take a step of size -1/(lambda*t) in the direction of the
  // sub-gradient psi(ybar,x)-psi(y_i,x), which corresponds to setting dalpha=1
  double dalpha = 1;
  SVM_cached_sample *s = &set->samples[0];
  SparseVector dpsi = *s->psi - *set->psi_gt;
  double dot = s->dot_w, d_sum_w_sqr;

  // SPO_DUAL_UPDATE and SPO_DUAL_UPDATE_WITH_CACHE: take a step in the direction of the 
  // sub-gradient psi(ybar,x)-psi(y_i,x), where the chosen step size maximizes the dual objective
  if(useSmartStepSize) 
    dalpha = (dot + s->loss*(sum_w_scale)) / my_max(s->sqr,.0000000001);
    
  // SPO_DUAL_UPDATE and SPO_DUAL_UPDATE_WITH_CACHE: ensure 0 <= alpha <= 1
  dalpha = my_min(1-s->alpha, my_max(-s->alpha,dalpha));

  if(dalpha != 0) {
    // Take a step of size dalpha the direction of the sub-gradient psi(ybar,x)-psi(y_i,x)
    // (lambda*t)*w_t = (lambda*(t-1))*w_{t-1} - dalpha*(psi(ybar,x)-psi(y_i,x))
    *sum_w -= dpsi.mult_scalar(dalpha);
      
    // Keep track of the change in the dual objective, regularization_error, and w^2
    s->alpha += dalpha;
    d_sum_w_sqr = -2*dalpha*dot + SQR(dalpha)*(s->sqr);
    sum_dual += -d_sum_w_sqr/(2*sum_w_scale) + dalpha*s->loss;
    sum_alpha_loss += dalpha*s->loss;
    sum_w_sqr += d_sum_w_sqr;
    regularization_error = (sum_w_sqr/SQR(sum_w_scale))*params.lambda/2;
    set->alpha = s->alpha;
    set->loss = s->loss;
    set->slack_after = dot/(sum_w_scale)+s->loss - dalpha*(s->sqr)/(sum_w_scale);
    assert(!isnan(sum_w_sqr) && !isnan(sum_dual));
  }
}



VFLOAT StructuredSVM::Test(const char *testfile, const char *predictionsFile, const char *htmlDir, 
			   double *svm_err, bool getLock) {
  StructuredDataset *testset = LoadDataset(testfile, getLock);
  VFLOAT v = Test(testset, predictionsFile, htmlDir, svm_err, getLock);
  delete testset;
  return v;
}

VFLOAT StructuredSVM::Test(StructuredDataset *testset, const char *predictionsFile, const char *htmlDir, 
			   double *svm_err, bool getLock) {
  SparseVector *w = GetCurrentWeights(getLock);
  w->make_non_sparse(true, sizePsi);
  double sum_slack = 0;

  if(getLock) Lock();
  int nc = NumHTMLColumns();
  double sum_los = 0;
  char **strs;
  int num = 0;
  omp_lock_t l_lock;
  omp_init_lock(&l_lock);

  FILE *htmlOut = NULL;
  if(htmlDir) {
    char fname[1000];  sprintf(fname, "%s/index.html", htmlDir);
    htmlOut = fopen(fname, "w");
    fprintf(htmlOut, "<html><table>\n");
  }

  if(params.debugLevel > 0) fprintf(stderr, "Evaluating testset...\n");
  if(predictionsFile)
    strs = (char**)malloc(sizeof(char*)*testset->num_examples);

  int numThreads = params.num_thr;
  if(!params.runMultiThreaded) numThreads = 1;
  else if(params.runMultiThreaded > 1) numThreads = params.runMultiThreaded;
  
  // Getting strange problems in Visual C++ Release mode.  num and sum_los
  // aren't getting shared in the main loop below unless I run this loop first
#pragma omp parallel num_threads(numThreads) 
  for(int i = 0; i < 100; i++) {
    num++;
    sum_los++;
    sum_slack++;
  }
  num = 0;
  sum_los = sum_slack = 0;

  //omp_set_nested(2);
#pragma omp parallel for num_threads(numThreads) 
  for(int i = 0; i < testset->num_examples; i++) {
    StructuredLabel *y = NewStructuredLabel(testset->examples[i]->x);
    double score = Inference(testset->examples[i]->x, y, w);
    double los = Loss(testset->examples[i]->y, y);
    double score_gt = w->dot(Psi(testset->examples[i]->x, testset->examples[i]->y)); 
    double slack = score-score_gt+los;

    if(predictionsFile) {
      Json::Value o, pred, gt;
      pred["y"] = y->save(this);
      pred["score"] = score;
      gt["y"] = testset->examples[i]->y->save(this);
      gt["score"] = score_gt;
      o["predicted"] = pred;
      o["ground_truth"] = gt;
      o["loss"] = los;
      Json::FastWriter writer;
      char tmp[100000];
      strcpy(tmp, writer.write(o).c_str());
      //fprintf(stderr, "%s\n", tmp);
      strs[i] = StringCopy(tmp);
    }
    delete y;

    omp_set_lock(&l_lock);
    sum_los += los;
    sum_slack += slack;
    num++;
    if(htmlOut) {
      char *htmlStr = VisualizeExample(htmlDir, testset->examples[i]); 
      if(i%nc == 0) {
	if(i) fprintf(htmlOut, "</tr>\n");
	fprintf(htmlOut, "<tr>\n");
      }
      fprintf(htmlOut, "<td>%s</td>\n", htmlStr);
      free(htmlStr);
    }
    fprintf(stderr, "After %d examples: ave_loss=%f, ave_slack=%f\n",  num, (float)(sum_los/num), (float)(sum_slack/num));
    omp_unset_lock(&l_lock);

    OnFinishedIteration(testset->examples[i]);
  }
  if(htmlOut) {
    fprintf(htmlOut, "</tr></table></html>\n");
    fclose(htmlOut);
  }
  double svm_error = (float)(sum_slack/testset->num_examples) + w->dot(*w)*params.lambda/2;
  printf("Average loss was %f, slack=%f, svm_err=%f\n", (float)(sum_los/testset->num_examples), 
	 (float)(sum_slack/testset->num_examples), svm_error);

  omp_destroy_lock(&l_lock);
  if(predictionsFile) {
    FILE *fout = fopen(predictionsFile, "w");
    for(int i = 0; i < testset->num_examples; i++) {
      fprintf(fout, "%s", strs[i]);
      free(strs[i]);
    }
    fclose(fout);
    free(strs);
  }

  if(getLock) Unlock();

  if(svm_err) *svm_err = svm_error;

  return (sum_los/testset->num_examples);
}
