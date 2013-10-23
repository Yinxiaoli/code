#include "structured_svm.h"
#include "structured_svm_train_statistics.h"
#include "structured_svm_train_choose_example.h"
#include "util.h"

StructuredSVMStatistics::StructuredSVMStatistics(StructuredSVM *svm) {
  this->svm = svm;
  alloc_t = 0;
  alloc_n = 0;
  window = 1000;
  
  sum_iter_error = 0;
  sum_iter_error_window = 0;
  numIterCombine = 1;
  iter_buff_size = 5012;

  generalization_errors_by_n = NULL;
  generalization_errors_by_t = iter_errors_by_t = sum_dual_by_t = regularization_errors_by_t = 
    losses_by_t = elapsed_time_by_t = NULL;
  iter_examples = NULL;
  sum_generalization_error = sum_generalization_error_window = 0;
}

StructuredSVMStatistics::~StructuredSVMStatistics() {
  if(iter_examples) free(iter_examples); 

  if(generalization_errors_by_n) free(generalization_errors_by_n);
  if(generalization_errors_by_t) free(generalization_errors_by_t);
  if(iter_errors_by_t) free(iter_errors_by_t);
  if(sum_dual_by_t) free(sum_dual_by_t);
  if(regularization_errors_by_t) free(regularization_errors_by_t);
  if(losses_by_t) free(losses_by_t);
  if(elapsed_time_by_t) free(elapsed_time_by_t);
}

bool StructuredSVMStatistics::Save(FILE *fout) {
  StructuredDataset *trainset = svm->GetTrainset();
  long t = svm->GetNumIterations();
  long n = svm->GetNumExamples();
  bool b = (fwrite(&sum_iter_error, sizeof(double), 1, fout) &&
	    fwrite(&sum_iter_error_window, sizeof(double), 1, fout) &&
	    fwrite(&sum_generalization_error, sizeof(double), 1, fout) &&
	    fwrite(&sum_generalization_error_window, sizeof(double), 1, fout) &&
	    fwrite(&window, sizeof(int), 1, fout) &&
	    fwrite(&numIterCombine, sizeof(int), 1, fout) &&
	    fwrite(&iter_buff_size, sizeof(int), 1, fout));
  
  if(iter_examples) b &= fwrite(iter_examples, sizeof(long), t, fout) > 0;
  if(iter_errors_by_t) b &= (fwrite(iter_errors_by_t, sizeof(double), t, fout) > 0);
  if(generalization_errors_by_n) b &= (fwrite(generalization_errors_by_n, sizeof(double), n, fout) > 0);

  if(generalization_errors_by_t) b &= (fwrite(generalization_errors_by_t, sizeof(double), iter_buff_size, fout) > 0);
  if(sum_dual_by_t) b &= (fwrite(sum_dual_by_t, sizeof(double), iter_buff_size, fout) > 0);
  if(regularization_errors_by_t) b &= (fwrite(regularization_errors_by_t, sizeof(double), iter_buff_size, fout) > 0);
  if(losses_by_t) b &= (fwrite(losses_by_t, sizeof(double), iter_buff_size, fout) > 0);
  if(elapsed_time_by_t) b &= (fwrite(elapsed_time_by_t, sizeof(double), iter_buff_size, fout) > 0);

  return b;
}

bool StructuredSVMStatistics::Load(FILE *fin) {
  long t = svm->GetNumIterations();
  long n = svm->GetNumExamples();
  bool b = (fread(&sum_iter_error, sizeof(double), 1, fin) &&
	    fread(&sum_iter_error_window, sizeof(double), 1, fin) &&
	    fread(&sum_generalization_error, sizeof(double), 1, fin) &&
	    fread(&sum_generalization_error_window, sizeof(double), 1, fin) &&
	    fread(&window, sizeof(int), 1, fin) &&
	    fread(&numIterCombine, sizeof(int), 1, fin) &&
	    fread(&iter_buff_size, sizeof(int), 1, fin));
  if(!b) 
    return false;

  alloc_n = (n+1);
  alloc_t = t;
  iter_examples = (long*)realloc(iter_examples, sizeof(long)*alloc_t);
  generalization_errors_by_n = (double*)realloc(generalization_errors_by_n, sizeof(double)*alloc_n);
  iter_errors_by_t = (double*)realloc(iter_errors_by_t, sizeof(double)*alloc_t);

  generalization_errors_by_t = (double*)realloc(generalization_errors_by_t, sizeof(double)*iter_buff_size);
  sum_dual_by_t = (double*)realloc(sum_dual_by_t, sizeof(double)*iter_buff_size);
  regularization_errors_by_t = (double*)realloc(regularization_errors_by_t, sizeof(double)*iter_buff_size);
  losses_by_t = (double*)realloc(losses_by_t, sizeof(double)*iter_buff_size);
  elapsed_time_by_t = (double*)realloc(elapsed_time_by_t, sizeof(double)*iter_buff_size);
  
  b &= (fread(iter_examples, sizeof(long), t, fin) > 0);
  b &= (fread(iter_errors_by_t, sizeof(double), t, fin) > 0);
  b &= (fread(generalization_errors_by_n, sizeof(double), n, fin) > 0);

  b &= (fread(generalization_errors_by_t, sizeof(double), iter_buff_size, fin) > 0);
  b &= (fread(sum_dual_by_t, sizeof(double), iter_buff_size, fin) > 0);
  b &= (fread(regularization_errors_by_t, sizeof(double), iter_buff_size, fin) > 0);
  b &= (fread(losses_by_t, sizeof(double), iter_buff_size, fin) > 0);
  b &= (fread(elapsed_time_by_t, sizeof(double), iter_buff_size, fin) > 0);

  return b;
}


// Book-keeping stuff, for estimating generalization error, optimization error, and regret when
// a new example ex->i is processed in iteration t=iter
void StructuredSVMStatistics::UpdateStatistics(SVM_cached_sample_set *ex, int iter, double sum_dual, double regularization_error, bool cache_old_examples) {
  double e = ex->slack_before;  // the slack at iteration iter 
  double elapsedTime = svm->GetElapsedTime();
  StructuredSVMTrainParams *params = svm->GetTrainParams();
  long t = svm->GetNumIterations();
  long n = svm->GetNumExamples();
  int *ex_num_iters = svm->GetChooser()->GetExNumIters();

  // Allocate memory buffers
  if(t+1 > alloc_t) {
    alloc_t = (int)(alloc_t*1.1)+10;
    iter_examples = (long*)realloc(iter_examples, sizeof(long)*alloc_t);
    iter_errors_by_t = (double*)realloc(iter_errors_by_t, sizeof(double)*alloc_t);
  }
  if(!sum_dual_by_t){
    sum_dual_by_t = (double*)realloc(sum_dual_by_t, sizeof(double)*iter_buff_size);
    generalization_errors_by_t = (double*)realloc(generalization_errors_by_t, sizeof(double)*iter_buff_size);
    regularization_errors_by_t = (double*)realloc(regularization_errors_by_t, sizeof(double)*iter_buff_size);
    losses_by_t = (double*)realloc(losses_by_t, sizeof(double)*iter_buff_size);
    elapsed_time_by_t = (double*)realloc(elapsed_time_by_t, sizeof(double)*iter_buff_size);
  }

  if(n+2 > alloc_n) {
    int j = alloc_n;
    alloc_n = (int)(alloc_n*1.1 + 10);
    generalization_errors_by_n = (double*)realloc(generalization_errors_by_n, sizeof(double)*alloc_n);
    for(int j = 0; j < alloc_n; j++)
      generalization_errors_by_n[j] = 0; 
  }
  
  if(ex_num_iters[ex->i] == 0) {
    // If this is the first time ex was processed, update estimate of the generalization error
    sum_generalization_error += my_max(e,0); // Stores the sum of the (online estimate of) test error
    generalization_errors_by_n[ex->i] = my_max(e,0)+regularization_error;  
    if(ex->i >= window) sum_generalization_error_window -= generalization_errors_by_n[ex->i-window];
    sum_generalization_error_window += generalization_errors_by_n[ex->i];
  }

  sum_iter_error += my_max(e,0);   // Stores the sum of the (online estimate of) training error
  iter_examples[iter] = ex->i;  

  // Combine (average) stats for numIterCombine consecutive iterations into the same memory entry
  int iter2 = iter/numIterCombine;
  if(iter2 >= iter_buff_size) {
    for(int i = 0, j = 0; i < iter_buff_size; i+=2, j++) {
      sum_dual_by_t[j] = (sum_dual_by_t[i] + sum_dual_by_t[i+1])/2;
      generalization_errors_by_t[j] = (generalization_errors_by_t[i] + generalization_errors_by_t[i+1])/2;
      regularization_errors_by_t[j] = (regularization_errors_by_t[i] + regularization_errors_by_t[i+1])/2;
      losses_by_t[j] = (losses_by_t[i] + losses_by_t[i+1])/2;
      elapsed_time_by_t[j] = (elapsed_time_by_t[i] + elapsed_time_by_t[i+1])/2;
    }
    numIterCombine *= 2;  // double numIterCombine, such that the total memory usage never exceeds iter_buff_size
    iter2 = iter/numIterCombine;
  }
  double f = 1.0/numIterCombine;
  if(iter % numIterCombine == 0) {
    sum_dual_by_t[iter2] = generalization_errors_by_t[iter2] = regularization_errors_by_t[iter2] = 
      losses_by_t[iter2] = elapsed_time_by_t[iter2] = 0;
  }

  iter_errors_by_t[iter] = my_max(e,0);  // Stores the slack (training error) in each iteration

  sum_dual_by_t[iter2] = f*sum_dual;  // Stores the value of the dual objective, which lowerbounds training error
  generalization_errors_by_t[iter2] = ex_num_iters[ex->i] == 0 ? f*generalization_errors_by_n[ex->i] : 
   f*sum_generalization_error_window/my_min(n,window);  
  regularization_errors_by_t[iter2] = f*regularization_error;
  losses_by_t[iter2] = ex->num_samples ? f*ex->samples[0].loss : 0;  // Stores the loss (not including the slack)
  elapsed_time_by_t[iter2] = (double)svm->GetElapsedTime();
  

  // Error measured over last set of examples/iterations of size window
  int curr_window_t = my_min(t+1,window), curr_window_n = my_min(n,window);
  if(cache_old_examples) curr_window_t = my_min(curr_window_t,n);
  assert(sum_iter_error_window >= 0);
  if(iter >= curr_window_t) 
    sum_iter_error_window += iter_errors_by_t[iter] - iter_errors_by_t[iter-curr_window_t];
  else
    sum_iter_error_window += iter_errors_by_t[iter];
  
  assert(sum_iter_error_window >= 0);
  
  double nn = (double)(!svm->IsOnlineObjective() ? n : t);
    
  if(params->debugLevel > 2 || (params->debugLevel > 1 && iter%10000==9999))
    printf("Last %d iters: Average Training Error=%f (Model error=%f, Optimization error=%f, Regularization error=%f), t=%d\n",
	   (int)curr_window_t, (float)(sum_iter_error_window/curr_window_t)+(float)regularization_error, 
	   (float)(sum_dual/nn), 
	   (float)((sum_iter_error_window)/curr_window_t)-sum_dual/nn+regularization_error, (float)regularization_error, (int)t);
}


void StructuredSVMStatistics::CheckConvergence(bool &finished, bool &hasConverged, bool runForever, bool cache_old_examples, double sum_dual, double regularization_error) {
  long n = svm->GetNumExamples();
  long t = svm->GetNumIterations();
  int M = svm->GetCurrentEpoch();
  StructuredSVMTrainParams *params = svm->GetTrainParams();
  int curr_window_t = my_min(t+1,window), curr_window_n = my_min(n,window);
  if(cache_old_examples) curr_window_t = my_min(curr_window_t,n);
  double nn = (double)(cache_old_examples ? n : t);
  double eps_empirical_measured = (sum_iter_error_window) / curr_window_t - sum_dual/nn + regularization_error;
  double eps_generalization_measured = sum_generalization_error_window / curr_window_n - sum_dual/nn;

  if(!hasConverged && ((params->eps && eps_empirical_measured < params->eps && !finished && 
			(!runForever || !cache_old_examples)) || t > params->maxIters)) {
    if(t > curr_window_t) {
      if(!runForever)
	finished = true;
      if(params->debugLevel > 0 && !hasConverged) {
	printf("%s at t=%d: epsilon_measured=%f\n", 
	       runForever ? "Convergence of empirical error detected" : "Finishing", 
	       (int)t, (float)eps_empirical_measured);
	printf("Last %d iters: Average Training Error=%f (Model error=%f, Optimization error=%f, Regularization error=%f)\n",
	       (int)curr_window_t, (float)(sum_iter_error_window/curr_window_t)+(float)regularization_error, 
	       (float)(sum_dual/nn), 
	       (float)((sum_iter_error_window)/curr_window_t)-sum_dual/nn+regularization_error, (float)regularization_error);
      }
      if(!runForever && M) {
	finished = true;
      }
    }
  } else if(0 && !finished && n > window && eps_generalization_measured < params->eps && !runForever) {
    printf("Convergence of generalization error detected at t=%d: epsilon_measured=%f\n", 
	   (int)n, (float)eps_generalization_measured);
    finished = true;
  }
} 

void StructuredSVMStatistics::RemoveExample(int ind) {
  StructuredDataset *trainset = svm->GetTrainset();
  long n = svm->GetNumExamples();
  long t = svm->GetNumIterations();
  if(svm->GetChooser()->GetExNumIters()[ind] != 0) 
    sum_generalization_error -= generalization_errors_by_n[ind];
  sum_generalization_error_window = 0;
  for(int i = 0; i < window && i < n; i++) {
    sum_generalization_error_window += generalization_errors_by_n[iter_examples[t-1-i]];
    if(iter_examples[t-1-i] == trainset->num_examples-1)
      iter_examples[t-1-i] = ind;
  }
  generalization_errors_by_n[ind] = generalization_errors_by_n[trainset->num_examples-1];
}



double *ComputeWindowAverageArray(double *a, int n, int window, double *a2=NULL, float s2=0, double *a3=NULL, float s3=0) {
  double cumSum = 0;
  int w = 0, i;
  double *retval = (double*)malloc(sizeof(double)*n);

  for(i = 0; i < n; i++) {
    if(w < window) w++;
    else cumSum -= a[i-window];
    cumSum += a[i];
    retval[i] = cumSum/w;
  }
  if(a2) {
    cumSum = 0;
    w = 0;
    for(i = 0; i < n; i++) {
      if(w < window) w++;
      else cumSum -= a2[i-window];
      cumSum += a2[i];
      retval[i] += cumSum/w*s2;
    }
  }
  if(a3) {
    cumSum = 0;
    w = 0;
    for(i = 0; i < n; i++) {
      if(w < window) w++;
      else cumSum -= a3[i-window];
      cumSum += a3[i];
      retval[i] += cumSum/w*s3;
    }
  }
  return retval;
}



void StructuredSVMStatistics::GetStatisticsByExample(int ave, long *nn, double **gen_err_buff, double **emp_err_buff, double **model_err_buff, double **reg_err_buff, double **loss_err_buff) {
  svm->Lock();
  long n = svm->GetNumExamples();
  long t = svm->GetNumIterations();
  double *gen_errors_by_n = (double*)malloc(sizeof(double)*n*5);
  double *emp_errors_by_n = gen_errors_by_n+n;
  double *mod_errors_by_n = emp_errors_by_n+n;
  double *reg_errors_by_n = mod_errors_by_n+n;
  double *loss_by_n = reg_errors_by_n+n;
  int num = 0;
  double tt = !svm->IsOnlineObjective() ? n : t;
  for(int i = 0; i < t; i++) {
    if(svm->GetChooser()->GetExNumIters()[iter_examples[i]] == i) {
      gen_errors_by_n[num] = generalization_errors_by_t[i];
      emp_errors_by_n[num] = iter_errors_by_t[i]+regularization_errors_by_t[i];
      mod_errors_by_n[num] = sum_dual_by_t[i]/my_min(i+1,tt);
      reg_errors_by_n[num] = regularization_errors_by_t[i];
      loss_by_n[num] = losses_by_t[i]/t;
      num++;
    }
  }
  *nn = num;
  if(gen_err_buff) *gen_err_buff = ComputeWindowAverageArray(gen_errors_by_n, num, ave);
  if(emp_err_buff) *emp_err_buff = ComputeWindowAverageArray(emp_errors_by_n, num, ave); 
  if(model_err_buff) *model_err_buff = ComputeWindowAverageArray(mod_errors_by_n, num, ave);
  if(reg_err_buff) *reg_err_buff = ComputeWindowAverageArray(reg_errors_by_n, num, ave);
  if(loss_err_buff) *loss_err_buff = ComputeWindowAverageArray(loss_by_n, num, ave);
  free(gen_errors_by_n);
  svm->Unlock();
}

void StructuredSVMStatistics::GetStatisticsByIteration(int ave, long *tt, double **gen_err_buff, double **emp_err_buff, double **model_err_buff,
					     double **reg_err_buff, double **loss_err_buff, double **time_buff) {
  svm->Lock();
  long n = svm->GetNumExamples();
  long t = svm->GetNumIterations();
  *tt = t;
  double nn = !svm->IsOnlineObjective() ? n : t;
  double *ave_dual_by_t = (double*)malloc(sizeof(double)*t);
  for(int i = 0; i < t; i++) ave_dual_by_t[i] = sum_dual_by_t[i]/my_min(i+1,nn);
  if(gen_err_buff) *gen_err_buff = ComputeWindowAverageArray(generalization_errors_by_t, t, ave);
  if(emp_err_buff) *emp_err_buff = ComputeWindowAverageArray(iter_errors_by_t, t, ave, regularization_errors_by_t, 1); 
  if(model_err_buff) *model_err_buff = ComputeWindowAverageArray(ave_dual_by_t, t, ave); 
  if(reg_err_buff) *reg_err_buff = ComputeWindowAverageArray(regularization_errors_by_t, t, ave);
  if(loss_err_buff) *loss_err_buff = ComputeWindowAverageArray(losses_by_t, t, ave);
  if(time_buff) { *time_buff = (double*)malloc(sizeof(double)*(t+1)); memcpy(*time_buff, elapsed_time_by_t, sizeof(double)*t); }
  free(ave_dual_by_t);
  svm->Unlock();
}


