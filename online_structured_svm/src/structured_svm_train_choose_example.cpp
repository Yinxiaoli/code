#include "structured_svm.h"
#include "structured_svm_train_choose_example.h"
#include "util.h"

#define CACHE_DIR "cache"

void write_SVM_cached_sample_set(struct _SVM_cached_sample_set *s, FILE *fout, StructuredSVM *svm, bool fullWrite);


StructuredSVMExampleChooser::StructuredSVMExampleChooser(StructuredSVM *s) {
  svm = s;
  alloc_n = 0;
  examples_by_iteration_number = NULL;
  examples_by_iteration_next_ind = NULL;
  examples_by_iteration_prev_ind = NULL;
  ex_num_iters = NULL;
  ex_first_iter = NULL;
  savingCachedExamples = false;
  prefetching = false;
  M = 0;
  createdCacheDir = false;

  cache_sets_in_memory_queue = NULL;
  num_cache_sets_in_memory = 0;
  max_cache_sets_in_memory = 100000000;
  
  curr = 0;
  currMinIterByExample = 0;
}  

StructuredSVMExampleChooser::~StructuredSVMExampleChooser() {
  if(ex_num_iters) free(ex_num_iters); 
  if(ex_first_iter) free(ex_first_iter); 
  if(examples_by_iteration_number) free(examples_by_iteration_number);
  if(examples_by_iteration_next_ind) free(examples_by_iteration_next_ind);
  if(examples_by_iteration_prev_ind) free(examples_by_iteration_prev_ind);
}

bool StructuredSVMExampleChooser::Save(FILE *fout) {
  StructuredDataset *trainset = svm->GetTrainset();
  bool b = (fwrite(&M, sizeof(int), 1, fout) &&
        fwrite(&curr, sizeof(long), 1, fout) &&
	    fwrite(&currMinIterByExample, sizeof(int), 1, fout) &&
	    fwrite(ex_num_iters, sizeof(int), trainset->num_examples, fout) &&
	    fwrite(ex_first_iter, sizeof(int), trainset->num_examples, fout));

  if(examples_by_iteration_number) b &= fwrite(examples_by_iteration_number, sizeof(int), M+1, fout) > 0;
  if(examples_by_iteration_next_ind) b &= fwrite(examples_by_iteration_next_ind, sizeof(int), trainset->num_examples, fout) > 0;
  if(examples_by_iteration_prev_ind) b &= fwrite(examples_by_iteration_prev_ind, sizeof(int), trainset->num_examples, fout) > 0;

  return b;
}

bool StructuredSVMExampleChooser::Load(FILE *fin) {
  StructuredDataset *trainset = svm->GetTrainset();
  alloc_n = (trainset->num_examples+1);
  ex_num_iters = (int*)realloc(ex_num_iters, sizeof(int)*alloc_n);
  bool b = (fread(&M, sizeof(int), 1, fin) &&
        fread(&curr, sizeof(long), 1, fin) &&
	    fread(&currMinIterByExample, sizeof(long), 1, fin) &&
	    fread(ex_num_iters, sizeof(int), trainset->num_examples, fin) &&
	    fread(ex_first_iter, sizeof(int), trainset->num_examples, fin));
  if(!b) 
    return false;

  examples_by_iteration_number = (int*)realloc(examples_by_iteration_number, sizeof(int)*(M+2));
  examples_by_iteration_next_ind = (int*)realloc(examples_by_iteration_next_ind, sizeof(int)*alloc_n);
  examples_by_iteration_prev_ind = (int*)realloc(examples_by_iteration_prev_ind, sizeof(int)*alloc_n);
  examples_by_iteration_number[M+1] = -1;

  b &= (fread(examples_by_iteration_number, sizeof(int), M+1, fin) > 0);
  b &= (fread(examples_by_iteration_next_ind, sizeof(int), trainset->num_examples, fin) > 0);
  b &= (fread(examples_by_iteration_prev_ind, sizeof(int), trainset->num_examples, fin) > 0);

  return b;
}


void StructuredSVMExampleChooser::Init(const char *modelout) {
  if(modelout)
    sprintf(cacheDir, "%s.cache", modelout);
  else
    strcpy(cacheDir, CACHE_DIR);
  
}

void StructuredSVMExampleChooser::Reset() {
  M = 0;
  examples_by_iteration_number = (int*)realloc(examples_by_iteration_number, sizeof(int)*(M+2));
  for(int i = 0; i <= M+1; i++)
    examples_by_iteration_number[i] = -1;
  for(int i = 0; i < svm->GetTrainset()->num_examples; i++) 
    CreateTrainingExampleQueues(i); 
}

/*
 * The rules for choosing which example to process next are (from highest to lowest precedence):
 *   1) Never allow different threads to process the same example at the same time
 *   2) Prefer processing an example that has been iterated over at least once but less than minItersBeforeNewExample
 *      iterations before going onto a new example (allows better estimation of generalization and optimization errors)
 *   3) Otherwise, prefer processing an example that has been iterated over the fewest times
 */
int StructuredSVMExampleChooser::ChooseNextExample(bool remove) {
  if(!svm->GetTrainset()->num_examples || savingCachedExamples) return -1;
  
  // Choose the examples in order, selecting the one that has been processed the fewest number of times
  int it = currMinIterByExample;
  while(it <= M && examples_by_iteration_number[it] < 0)
    it++;
  if(svm->HasConverged() && it == M)
    return -1;
  if(it > M || examples_by_iteration_number[it] < 0)
    return -1;

  // Remove (temporarily) the selected example, such that no other threads can process it at the same time
  int retval = examples_by_iteration_number[it];
  if(remove) {
    assert(ex_num_iters[retval] == it);
    RemoveExampleFromQueue(retval);
  }
  
  return retval;
}

void LinkedListRemove(SVM_cached_sample_set **head, SVM_cached_sample_set *r) {
  if(r == *head) *head = r->next == r ? NULL : r->next;
  r->next->prev = r->prev;
  r->prev->next = r->next;
  r->next = r->prev = NULL;
}

void LinkedListAdd(SVM_cached_sample_set **head, SVM_cached_sample_set *a) {
  if(!*head) {
    *head = a;
    a->prev = a->next = a;
  } else {
    a->prev = (*head)->prev;
    a->next = *head;
    (*head)->prev->next = a;
    (*head)->prev = a;
  }
}

int LinkedListPrint(const char *name, SVM_cached_sample_set *head) {
  int n = 0;
  SVM_cached_sample_set *curr = head;
  printf("%s:", name);
  while(curr) {
    n++;
    printf(" %d", curr->i);
    curr = curr->next;
    if(curr == head) 
      break;
  }
  printf("\n");

  return n;
}

void StructuredSVMExampleChooser::MarkAsInMemory(SVM_cached_sample_set *set) {
  if(set->next)
    LinkedListRemove(&cache_sets_in_memory_queue, set);
  LinkedListAdd(&cache_sets_in_memory_queue, set);
  set->inMemory = true;
}

void StructuredSVMExampleChooser::PrefetchCache(int nextUpdateInd) {
  int ind = nextUpdateInd;
  if(prefetching || nextUpdateInd < 0) 
    return;
  prefetching = true;

  StructuredSVMTrainParams *params = svm->GetTrainParams();
  StructuredDataset *trainset = svm->GetTrainset();
  long n = svm->GetNumExamples(), t = svm->GetNumIterations();
  int num_evict = 0, num_prefetch = 0;
  int *prefetch=NULL, *evict=NULL;
  SVM_cached_sample_set **prefetch_sets=NULL;

  int next_ex = ChooseNextExample(false);

  // Check which examples will need to be brought into memory soon, and subsequently, which examples
  // will need to be evicted from memory
  for(int k = 0; k < params->numCacheUpdatesPerIteration+(next_ex >= 0 ? params->num_thr*3 : 0) && k < n && k < t; k++) {
    int i = k < params->numCacheUpdatesPerIteration ? (ind+k)%my_min(t,n) : (next_ex+k-params->numCacheUpdatesPerIteration)%my_min(t,n);
    if(!trainset->examples[i]->set && trainset->examples[i]->cache_fname) {
      prefetch = (int*)realloc(prefetch, (num_prefetch+1)*sizeof(int));
      prefetch_sets = (SVM_cached_sample_set**)realloc(prefetch_sets, (num_prefetch+1)*sizeof(SVM_cached_sample_set*));
      prefetch[num_prefetch++] = i;
      while(num_cache_sets_in_memory+num_prefetch-num_evict > max_cache_sets_in_memory) {
	SVM_cached_sample_set *r = cache_sets_in_memory_queue;
	assert(r);
	int e = ind+params->numCacheUpdatesPerIteration;
	int e2 = next_ex+params->num_thr*3;
	while(r->i == i || r->lock || (r->i >= ind && r->i < e) || 
	      (r->i < ind && r->i < e%my_min(t,n) && e >= my_min(t,n)) || 
		  (next_ex >= 0 && r->i >= next_ex && r->i < e2) || 
	      (next_ex >= 0 && r->i < next_ex && r->i < e2%my_min(t,n) && e2 >= my_min(t,n))) {
	  r = r->next;
	  assert(r != cache_sets_in_memory_queue);
	}
	r->lock = true;
	evict = (int*)realloc(evict, (num_evict+1)*sizeof(int));
	evict[num_evict++] = r->i;
      }
    }
  }
  if(!num_evict && !num_prefetch) {
    prefetching = false;
    return;
  }
  svm->Unlock();

  // Prefetch memory for examples that need to be brought into memory
  int i;
  for(i = 0; i < num_prefetch; i++) {
    //fprintf(stderr, "PREFETCH %d\n", prefetch[i]);
    FILE *fin = fopen(trainset->examples[prefetch[i]]->cache_fname, "rb");
    assert(fin);
    prefetch_sets[i] = read_SVM_cached_sample_set(fin, svm, trainset->examples[prefetch[i]]->x, true);
    fclose(fin);
  }

  // Write out examples that need to be evicted
  for(i = 0; i < num_evict; i++) {
    char fname[1000];
    if(!createdCacheDir) {
      CreateDirectoryIfNecessary(cacheDir);
      createdCacheDir = true;
    }
    sprintf(fname, "%s/%d", cacheDir, evict[i]);
    //fprintf(stderr, "PREFETCH EVICT %d\n", evict[i]);
    FILE *fout = fopen(fname, "wb");
    assert(fout);
    assert(trainset->examples[evict[i]]->set);
    write_SVM_cached_sample_set(trainset->examples[evict[i]]->set, fout, svm, true);
    fclose(fout);
  }

  // Since prefetching and eviction was done without locking other threads, check if any of the prefetching was redundant with
  // other threads and record changes
  svm->Lock();
  for(i = 0; i < num_prefetch; i++) {
    if(!trainset->examples[prefetch[i]]->set) {
      //fprintf(stderr, "*PREFETCH %d\n", prefetch[i]);
      trainset->examples[prefetch[i]]->set = prefetch_sets[i];
      MarkAsInMemory(prefetch_sets[i]);
      num_cache_sets_in_memory++;
    } else {
      free_SVM_cached_sample_set(prefetch_sets[i]);  // another thread already prefetched this one
    }
  }
  for(i = 0; i < num_evict; i++) {
    //fprintf(stderr, "*PREFETCH EVICT %d\n", evict[i]);
    SVM_cached_sample_set *r = trainset->examples[evict[i]]->set;
    assert(r);
    char fname[1000];
    sprintf(fname, "%s/%d", cacheDir, evict[i]);
    if(!createdCacheDir) {
      CreateDirectoryIfNecessary(cacheDir);
      createdCacheDir = true;
    }
    if(!trainset->examples[r->i]->cache_fname)
      trainset->examples[r->i]->cache_fname = StringCopy(fname);
    RemoveCacheSetFromMemory(r);
  }

  if(prefetch) free(prefetch);
  if(evict) free(evict);
  if(prefetch_sets) free(prefetch_sets);
  prefetching = false;
}

void StructuredSVMExampleChooser::CreateTrainingExampleQueues(int ind) {
  if(svm->GetTrainset()->num_examples+1 > alloc_n) {
    alloc_n = (int)(alloc_n*1.1 + 10);
    ex_num_iters = (int*)realloc(ex_num_iters, sizeof(int)*alloc_n);
    ex_first_iter = (int*)realloc(ex_first_iter, sizeof(int)*alloc_n);
    examples_by_iteration_next_ind = (int*)realloc(examples_by_iteration_next_ind, sizeof(int)*alloc_n);
    examples_by_iteration_prev_ind = (int*)realloc(examples_by_iteration_prev_ind, sizeof(int)*alloc_n);
  }
  ex_num_iters[ind] = 0;
  ex_first_iter[ind] = -1;

  // Add this example to the front of the list of examples that have been iterated over 0 times
  if(examples_by_iteration_number[0] >= 0) {
    examples_by_iteration_prev_ind[ind] = examples_by_iteration_prev_ind[examples_by_iteration_number[0]];
    examples_by_iteration_next_ind[ind] = examples_by_iteration_number[0];
    examples_by_iteration_next_ind[examples_by_iteration_prev_ind[examples_by_iteration_number[0]]] = ind;
    examples_by_iteration_prev_ind[examples_by_iteration_number[0]] = ind;
  } else {
    examples_by_iteration_prev_ind[ind] = examples_by_iteration_next_ind[ind] = ind;
    examples_by_iteration_number[0] = ind;
  }
  this->currMinIterByExample = 0;
}

void StructuredSVMExampleChooser::RemoveCacheSetFromMemory(SVM_cached_sample_set *r) {
  if(r->inMemory) {
    LinkedListRemove(&cache_sets_in_memory_queue, r);
    svm->GetTrainset()->examples[r->i]->set = NULL;
    r->inMemory = false;
    num_cache_sets_in_memory--;
  }
  free_SVM_cached_sample_set(r);
}

SVM_cached_sample_set *StructuredSVMExampleChooser::BringCacheSetIntoMemory(int i, int nextUpdateInd, bool lock) {
  StructuredDataset *trainset = svm->GetTrainset();
  StructuredSVMTrainParams *params = svm->GetTrainParams();
  SVM_cached_sample_set *set = trainset->examples[i]->set;
  long n = svm->GetNumExamples(), t = svm->GetNumIterations();
  int ind = nextUpdateInd;
  if(!set) {
    if(trainset->examples[i]->cache_fname) {
      // Fetch a cached sample set from disk
      //fprintf(stderr, "UNEVICT %d\n", i);
      FILE *fin = fopen(trainset->examples[i]->cache_fname, "rb");
      assert(fin);
      set = trainset->examples[i]->set = read_SVM_cached_sample_set(fin, svm, trainset->examples[i]->x, true);
      fclose(fin);
    } else {
      // Create a new cached sample set
      set = new_SVM_cached_sample_set(i,svm->Psi(trainset->examples[i]->x, trainset->examples[i]->y).ptr());
      if(set->psi_gt) set->psi_gt_sqr = set->psi_gt->dot(*set->psi_gt);
    }
    num_cache_sets_in_memory++;
  } else if(!set->psi_gt) {
    set->psi_gt = svm->Psi(trainset->examples[i]->x, trainset->examples[i]->y).ptr();
  }
  MarkAsInMemory(set);
  if(lock) set->lock = true;

  if(num_cache_sets_in_memory > max_cache_sets_in_memory) {
    // Evict a cached sample set from disk
    char fname[1000];
    SVM_cached_sample_set *r = cache_sets_in_memory_queue;
    assert(r);
    int next_ex = ChooseNextExample(false);
    int e = ind+params->numCacheUpdatesPerIteration;
    int e2 = next_ex+params->num_thr*3;
	while(r->i == i || r->lock || (r->i >= ind && r->i < e) || 
	      (r->i < ind && r->i < e%my_min(t,n) && e >= my_min(t,n)) || 
		  (next_ex >= 0 && r->i >= next_ex && r->i < e2) || 
	      (next_ex >= 0 && r->i < next_ex && r->i < e2%my_min(t,n) && e2 >= my_min(t,n))) {
      r = r->next;
      assert(r != cache_sets_in_memory_queue);
    }
    sprintf(fname, "%s/%d", cacheDir, r->i);
    //fprintf(stderr, "EVICT %d\n", r->i);
    if(!trainset->examples[r->i]->cache_fname)
      trainset->examples[r->i]->cache_fname = StringCopy(fname);
    FILE *fout = fopen(trainset->examples[r->i]->cache_fname, "wb");
    assert(fout);
    write_SVM_cached_sample_set(r, fout, svm, true);
    fclose(fout);

    RemoveCacheSetFromMemory(r);
  }

  //assert(LinkedListPrint("IN MEMORY", cache_sets_in_memory_queue) == num_cache_sets_in_memory);

  return set;
}

void StructuredSVMExampleChooser::UpdateExampleIterationQueue(SVM_cached_sample_set *ex, int iter) {
  if(!ex_num_iters[ex->i])
    ex_first_iter[ex->i] = iter;
  ex_num_iters[ex->i]++;

  if(ex_num_iters[ex->i] > M) {
    if(M) 
      ShuffleExamples(ex_num_iters[ex->i]-1);
    assert(M == ex_num_iters[ex->i]-1);
    M++;
    
    examples_by_iteration_number = (int*)realloc(examples_by_iteration_number, sizeof(int)*(M+2));
    examples_by_iteration_number[M] = -1;
  }
  
  // Add this example back to the queue of examples to be processed
  int r = examples_by_iteration_number[ex_num_iters[ex->i]];
  if(r >= 0) {
    examples_by_iteration_prev_ind[ex->i] = examples_by_iteration_prev_ind[r];
    examples_by_iteration_next_ind[ex->i] = r;
    examples_by_iteration_next_ind[examples_by_iteration_prev_ind[r]] = ex->i;
    examples_by_iteration_prev_ind[r] = ex->i;
  } else {
    examples_by_iteration_prev_ind[ex->i] = examples_by_iteration_next_ind[ex->i] = ex->i;
    examples_by_iteration_number[ex_num_iters[ex->i]] = ex->i;
  }
}

void StructuredSVMExampleChooser::ShuffleExamples(int iter) {
  // Randomly shuffle the order of the queue of examples that have been iterated over 'iter' times
  

  // Count the applicable number of examples 
  int num = 0;
  int r = examples_by_iteration_number[iter];
  int i = r;
  if(r == -1) return;
  do {
    i = examples_by_iteration_next_ind[i];
    num++;
  } while(i != r);  

  // Compute a random permutation of the applicable examples
  int *inds = (int*)malloc(sizeof(int)*num);
  int *perm = RandPerm(num);
  int j = 0;
  i = r;
  r = examples_by_iteration_number[iter];
  do {
    inds[perm[j++]] = i;
    i = examples_by_iteration_next_ind[i];
  } while(i != r);

  // Set the next and prev pointers for the shuffled queue
  for(int i = 1; i < num-1; i++) {
    examples_by_iteration_prev_ind[inds[i]] = inds[i-1];
    examples_by_iteration_next_ind[inds[i]] = inds[i+1];
  }
  examples_by_iteration_prev_ind[inds[0]] = inds[num-1];
  examples_by_iteration_next_ind[inds[num-1]] = inds[0];
  if(num > 1) {
    examples_by_iteration_next_ind[inds[0]] = inds[1];
    examples_by_iteration_prev_ind[inds[num-1]] = inds[num-2];
  }
  examples_by_iteration_number[iter] = inds[0];
  

  free(perm);
  free(inds);
}

void StructuredSVMExampleChooser::RemoveExampleFromQueue(int i) {
  int it = ex_num_iters[i];
  if(examples_by_iteration_next_ind[i] == i) {
    examples_by_iteration_number[it] = -1;
  } else {
    examples_by_iteration_number[it] = examples_by_iteration_next_ind[i];
    examples_by_iteration_prev_ind[examples_by_iteration_next_ind[i]] = examples_by_iteration_prev_ind[i];
    examples_by_iteration_next_ind[examples_by_iteration_prev_ind[i]] = examples_by_iteration_next_ind[i];
  }
  examples_by_iteration_next_ind[i] = examples_by_iteration_prev_ind[i] = -1;
}

void StructuredSVMExampleChooser::RemoveExample(int ind) {
  StructuredDataset *trainset = svm->GetTrainset();
  RemoveExampleFromQueue(ind);
  ex_num_iters[ind] = ex_num_iters[trainset->num_examples-1];
  ex_first_iter[ind] = ex_first_iter[trainset->num_examples-1];
  examples_by_iteration_next_ind[ind] = examples_by_iteration_next_ind[trainset->num_examples-1];
  examples_by_iteration_prev_ind[ind] = examples_by_iteration_prev_ind[trainset->num_examples-1];
}


void StructuredSVMExampleChooser::SaveCachedExamples(const char *output_name, bool saveFull) {
  int i;
  StructuredDataset *trainset = svm->GetTrainset();
  long n = svm->GetNumExamples();

  savingCachedExamples = true;
  for(i = 0; i < n; i++) {
    while(trainset->examples[i]->set && trainset->examples[i]->set->lock) {
      svm->Unlock();
      usleep(100000);
      svm->Lock();
    }
  }
  
  FILE *fout = fopen(output_name, "wb");
  fwrite(&n, sizeof(long), 1, fout);
  for(int i = 0; i < n; i++) {
    bool b = (trainset->examples[i]->set || trainset->examples[i]->cache_fname) ? true : false;
    BringCacheSetIntoMemory(i, 0, false);
    fwrite(&b, sizeof(bool), 1, fout);
    if(b)
      write_SVM_cached_sample_set(trainset->examples[i]->set, fout, svm, saveFull);
  }
  fclose(fout);

  savingCachedExamples = false;
}


void StructuredSVMExampleChooser::LoadCachedExamples(const char *fname, bool loadFull) {
  int i;
  StructuredDataset *trainset = svm->GetTrainset();
  long n = svm->GetNumExamples();

  for(i = 0; i < n; i++) {
    if(trainset->examples[i]->set) {
      free_SVM_cached_sample_set(trainset->examples[i]->set);
      trainset->examples[i]->set = NULL;
    }
  }
  FILE *fin = fopen(fname, "rb");
  if(fin) {
    long n2;
    int b2 = fread(&n2, sizeof(long), 1, fin) > 0; assert(b2);
    if(!n) n = n2;
    assert(n2 == n);
    for(i = 0; i < n; i++) {
      bool b;
      b2 = fread(&b, sizeof(bool), 1, fin) > 0;  assert(b2);
      if(b) {
	trainset->examples[i]->set = read_SVM_cached_sample_set(fin, svm, trainset->examples[i]->x, loadFull);
	//svm->SVM_cached_sample_set_compute_features(trainset->examples[i]->set, trainset->examples[i]);
	svm->OnFinishedIteration(trainset->examples[i]);
	MarkAsInMemory(trainset->examples[i]->set);
	num_cache_sets_in_memory++;
      }
    }
    fclose(fin);
  }
}
