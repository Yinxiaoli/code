#ifndef __STRUCTURED_SVM_CHOOSE_EXAMPLE_H
#define __STRUCTURED_SVM_CHOOSE_EXAMPLE_H

#include "structured_svm.h"

class StructuredSVMExampleChooser {
  StructuredSVM *svm;
  int alloc_n;
  int *ex_num_iters;  /**< an array of size sample.n, where each entry stores the number of iterations each training 
			 example has been processed */ 
  int *ex_first_iter; /**<  an array of size sample.n, where each entry stores the iteration (value of t) when each 
			 training example was first iterated over; */
  long curr;  /**< the next example to process */ 
  int *examples_by_iteration_number; /**< An array of size M, where each entry stores an index to the start of a 
					linked list (implemented in examples_by_iteration_next_ind).  Each linked 
					list stores the indices of all examples that have been iterated over i times */
  int *examples_by_iteration_next_ind;  /**< An array of size sample.n.  For each example, contains the index of the 
					   next example in a linked list of examples that have been iterated over the 
					   same number of times as this example */ 
  int *examples_by_iteration_prev_ind;  /**< An array of size sample.n.  For each example, contains the index of the 
					   previous example in a linked list of examples that have been iterated over 
					   the same number of times as this example */ 
  int currMinIterByExample;
  struct _SVM_cached_sample_set *cache_sets_in_memory_queue;
  int num_cache_sets_in_memory, max_cache_sets_in_memory;
  bool savingCachedExamples, prefetching;
  int M;   
  bool createdCacheDir;

  char cacheDir[1000];

public:
  StructuredSVMExampleChooser(StructuredSVM *svm);
  ~StructuredSVMExampleChooser();
  void Init(const char *modelout=NULL);
  int ChooseNextExample(bool remove=true);
  void RemoveExampleFromQueue(int i);
  void MarkAsInMemory(struct _SVM_cached_sample_set *set);
  void PrefetchCache(int nextUpdateInd);
  void RemoveCacheSetFromMemory(struct _SVM_cached_sample_set *r);
  struct _SVM_cached_sample_set *BringCacheSetIntoMemory(int i, int nextUpdateInd, bool lock);
  void UpdateExampleIterationQueue(struct _SVM_cached_sample_set *set, int iter);
  void CreateTrainingExampleQueues(int ind);
  int *GetExNumIters() { return ex_num_iters; }
  bool Save(FILE *fout);
  bool Load(FILE *fin);
  int GetCurrentEpoch() { return M; }
  void Reset();
  void RemoveExample(int ind);
  void SaveCachedExamples(const char *output_name, bool saveFull=true);
  void LoadCachedExamples(const char *fname, bool loadFull=true);

 private:
  void ShuffleExamples(int iter);
};

#endif
