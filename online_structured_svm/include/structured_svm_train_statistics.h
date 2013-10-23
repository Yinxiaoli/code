#ifndef __STRUCTURED_SVM_TRAIN_STATISTICS_H
#define __STRUCTURED_SVM_TRAIN_STATISTICS_H

#include "structured_svm.h"

class StructuredSVMStatistics {
  int alloc_t, alloc_n;
  int window;  /**< default window size used for computing statistics for decomposing the test error */

  double sum_generalization_error;  /**< Total error measured on new examples as we stream them in and before processing them. */
  double sum_iter_error;            /**< Total error measured on unseen labels.  This is different than sum_generalization_error if we have 
				      processed each example more than once (it is effectively the sum loss associated with each call 
				      to Inference(x_i, ybar_i, w_t, NULL, y_i) */
  double sum_iter_error_window, sum_generalization_error_window;
  double *generalization_errors_by_n;
  double *generalization_errors_by_t, *iter_errors_by_t, *sum_dual_by_t, 
    *regularization_errors_by_t, *losses_by_t, *elapsed_time_by_t;
  long *iter_examples;
  StructuredSVM *svm;
  int numIterCombine, iter_buff_size;

 public:
  StructuredSVMStatistics(StructuredSVM *svm);
  ~StructuredSVMStatistics();
  bool Save(FILE *fout);
  bool Load(FILE *fin);
  
  void UpdateStatistics(struct _SVM_cached_sample_set *set, int iter, double sum_dual, double regularization_error, bool cache_old_examples);

  /**
   * @brief Get statistics useful for plotting the progression of different types of 
   * training error as a function of training computation time
   */
  void GetStatisticsByIteration(int ave, long *tt, double **gen_err_buff, double **emp_err_buff, double **model_err_buff, 
				double **reg_err_buff, double **loss_buff, double **time_buff);

  /**
   * @brief Get statistics useful for plotting the progression of different types of 
   * training error as a function of number of training examples
   */
  void GetStatisticsByExample(int ave, long *nn, double **gen_err_buff, double **emp_err_buff, double **model_err_buff, 
			      double **reg_err_buff, double **loss_buff);

  void SetWindow(int w) { window = w; }
  void CheckConvergence(bool &finished, bool &hasConverged, bool runForever, bool cache_old_examples, double sum_dual, double regularization_error);
  void RemoveExample(int ind);
};

#endif
