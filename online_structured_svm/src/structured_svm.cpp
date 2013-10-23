#include "structured_svm.h"
#include "structured_svm_train_statistics.h"
#include "structured_svm_train_choose_example.h"
#include "visualizationTools.h"

void InitTrainParams(StructuredSVMTrainParams *params);



StructuredSVM::StructuredSVM() {
  omp_init_lock(&my_lock);

  InitTrainParams(&params);
  stats = new StructuredSVMStatistics(this);
  chooser = new StructuredSVMExampleChooser(this);
  
  sizePsi = 0;
  t = 0;
  sum_w = NULL;
  u_i_buff = NULL;

  trainfile = modelfile = NULL; 
  numCacheIters = 0;

  numExampleIds = 0;
  exampleIdsToIndices = NULL;
  exampleIndicesToIds = NULL;

  base_time = 0;
  runForever = false;
  hasConverged = false;
  n = 0;
  finished = false;
  nextUpdateInd = 0;
  useFixedSampleSet = false;

  isMultiSample = false;

  validationfile = NULL;

  trainset = NULL;

  regularization_error = 0;
  sum_dual = 0;
  sum_alpha_loss = 0;
  sum_w_sqr = 0;
  sum_w_scale = 1;
}

StructuredSVM::~StructuredSVM() {
  if(trainset) delete trainset;
  if(trainfile) free(trainfile);
  if(modelfile) free(modelfile);
  if(validationfile) free(validationfile);

  if(exampleIdsToIndices) free(exampleIdsToIndices);
  if(exampleIndicesToIds) free(exampleIndicesToIds);

  if(sum_w) delete sum_w;
  if(u_i_buff) free(u_i_buff);

  omp_destroy_lock(&my_lock);
}


StructuredDataset *StructuredSVM::LoadDataset(const char *fname, bool getLock) {
  if(params.debugLevel > 0) fprintf(stderr, "Reading dataset %s...", fname);
  
  if(getLock) Lock();

  FILE *fin = fopen(fname, "r");
  if(!fin) {
    fprintf(stderr, "Couldn't open dataset file %s\n", fname);
    Unlock();
    return NULL;
  }

  StructuredDataset *d = new StructuredDataset();
  char *line = new char[1000000];
  
  Json::Reader reader;
  while(fgets(line, 999999, fin) && strlen(line) > 1) {
    chomp(line);
    Json::Value r;
    if(!reader.parse(line, r)) {
      fprintf(stderr, "Error parsing dataset example %s\n", line);
      delete d;
      fclose(fin);
      return NULL;
    }
    StructuredExample *ex = new StructuredExample;
    ex->x = NewStructuredData();
    ex->y = NewStructuredLabel(ex->x);
    if(r.isMember("y_latent")) ex->y_latent = NewStructuredLabel(ex->x);
    if(!r.isMember("x") || !r.isMember("y") || 
       !ex->x->load(r["x"], this) || !ex->y->load(r["y"], this) ||
       (r.isMember("y_latent") && !ex->y_latent->load(r["y_latent"], this))) { 
      fprintf(stderr, "Error parsing values for dataset example %s\n", line);
      delete ex;
      delete d;
      fclose(fin);
      return NULL;
    }
    d->AddExample(ex);
  }
  fclose(fin);
  if(getLock) Unlock();
  delete [] line;

  if(params.debugLevel > 0) fprintf(stderr, "done\n");

  return d;
}



bool StructuredSVM::SaveDataset(StructuredDataset *d, const char *fname, int start_from) {
  if(params.debugLevel > 0 && start_from == 0) fprintf(stderr, "Saving dataset %s...", fname);

  Lock();

  FILE *fout = fopen(fname, start_from>0 ? "a" : "w");
  if(!fout) {
    fprintf(stderr, "Couldn't open dataset file %s for writing\n", fname);
    Unlock();
    return false;
  }

  Json::FastWriter writer;

  char data[100000];
  for(int i = start_from; i < d->num_examples; i++) {
    Json::Value o;
    o["x"] = d->examples[i]->x->save(this);
    o["y"] = d->examples[i]->y->save(this);
    if(d->examples[i]->y_latent) o["y_latent"] =d->examples[i]->y_latent->save(this);
    strcpy(data, writer.write(o).c_str());
    chomp(data);
    fprintf(fout, "%s\n", data);
  }
  fclose(fout);
  Unlock();

  if(params.debugLevel > 0 && start_from == 0) fprintf(stderr, "done\n");

  return true;
}

bool StructuredSVM::Save(const char *fname, bool saveFull, bool getLock) {
  if(getLock) Lock();
  Json::Value root;
  if(modelfile) free(modelfile);
  modelfile = StringCopy(fname);
  if(sum_w) root["Sum w"] = sum_w->save();
  root["Regularization (C)"] = params.C;
  root["Training accuracy (epsilon)"] = params.eps;
  root["T"] = (int)t;
  if(trainfile) root["Training Set"] = trainfile;
  root["Custom"] = Save();
  if(saveFull) {
    char full_name[1000];
    sprintf(full_name, "%s.online", fname);
    root["Online Data"] = full_name;
    if(!SaveOnlineData(full_name)) { Unlock(); return false; }
  }

  Json::StyledWriter writer;
  FILE *fout = fopen(fname, "w");
  if(!fout) { fprintf(stderr, "Couldn't open %s for writing\n", fname); Unlock(); return false; }
  fprintf(fout, "%s", writer.write(root).c_str());
  fclose(fout);
  if(getLock) Unlock();

  return true;
}

bool StructuredSVM::Load(const char *fname, bool loadFull) {
  Lock();
  if(modelfile) free(modelfile);
  modelfile = StringCopy(fname);

  char *str = ReadStringFile(fname);
  if(!str) { fprintf(stderr, "Couldn't open %s for reading\n", fname); Unlock(); return false; }

  if(trainfile) free(trainfile);
  trainfile = NULL;

  Json::Reader reader;
  Json::Value root;
  if(!reader.parse(str, root)) { 
    fprintf(stderr, "Couldn't read JSON file %s\n", fname); Unlock(); return false; 
  }

  if(root.isMember("Sum w")) { 
    if(!sum_w) sum_w = new SparseVector; 
    sum_w->load(root["Sum w"]); 
  }

  if(root.isMember("Regularization (C)")) {
    params.C = root.get("Regularization (C)", 0).asDouble();
    params.lambda = 1/params.C;
  }
  t = root.get("T", 0).asInt();
  if(root.isMember("Training Set")) { 
    char str[1000]; strcpy(str, root.get("Training Set", "").asString().c_str()); trainfile = StringCopy(str); 
  }
  if(!Load(root["Custom"])) { Unlock(); return false; }
     
  Unlock();

  if(loadFull && !root.isMember("Online Data")) {
    
    fprintf(stderr, "Can't load full data from %s\n", fname); //Unlock(); return false; 
  } else if(loadFull) {
    char str[1000]; strcpy(str, root["Online Data"].asString().c_str());
    if(!LoadOnlineData(str)) { return false; }
  }
  return true;
}



StructuredExample::StructuredExample() { 
  x = NULL; 
  y = NULL; 
  y_latent = NULL;
  set = NULL;
  cache_fname = NULL;
  visualization = NULL;
}

StructuredExample::~StructuredExample() {
  if(x) delete x;
  if(y) delete y;
  if(y_latent) delete y_latent;
  if(set) free_SVM_cached_sample_set(set);
  if(cache_fname) free(cache_fname);
  if(visualization) free(visualization);
}

StructuredDataset::StructuredDataset() { 
  examples = NULL; 
  num_examples = 0; 
}

StructuredDataset::~StructuredDataset() {
  if(examples) {
    for(int i = 0; i < num_examples; i++)
      delete examples[i];
    free(examples);
  }
}

void StructuredDataset::AddExample(StructuredExample *e) {
  examples = (StructuredExample**)realloc(examples, sizeof(StructuredExample*)*(num_examples+1));
  examples[num_examples++] = e;
}


void StructuredDataset::Randomize() {
  int *perm = RandPerm(num_examples);
  StructuredExample **examples_new = (StructuredExample**)malloc(sizeof(StructuredExample*)*(num_examples));
  for(int i = 0; i < num_examples; i++) 
    examples_new[i] = examples[perm[i]];
  free(examples);
  examples = examples_new;
  free(perm);
}




StructuredExample *StructuredSVM::CopyExample(StructuredData *x, StructuredLabel *y, StructuredLabel *y_latent) {
  StructuredExample *copy = new StructuredExample();
  copy->x = NewStructuredData();
  copy->y = NewStructuredLabel(copy->x);
  Json::Value xx = x->save(this);
  bool b = copy->x->load(xx, this);
  assert(b);
  Json::Value yy = y->save(this);
  b = copy->y->load(yy, this);
  assert(b);
  if(y_latent) {
    copy->y_latent = NewStructuredLabel(copy->x);
    Json::Value yy_latent = y_latent->save(this);
    b = copy->y_latent->load(yy_latent, this);
    assert(b);
  }

  return copy;
}



bool StructuredSVM::SaveTrainingSet(int start_from) {
  if(!trainfile && modelfile) {
    char tmp[1000];
    sprintf(tmp, "%s.train", modelfile);
    trainfile = StringCopy(tmp);
  }
  if(trainfile) return SaveDataset(trainset, trainfile, start_from);

  return false;
}

bool StructuredSVM::SaveOnlineData(const char *fname) {
  FILE *fout = fopen(fname, "wb");
  if(!fout) {
    fprintf(stderr, "Error saving online data to %s, open failed\n", fname);
    return false;
  }

  long tm = (long)GetElapsedTime();
  bool b = (fwrite(&regularization_error, sizeof(double), 1, fout) &&
         fwrite(&tm, 1, sizeof(long), fout) &&
         fwrite(&n, sizeof(int), 1, fout) &&
         fwrite(&sum_dual, sizeof(double), 1, fout) &&
         fwrite(&sum_alpha_loss, sizeof(double), 1, fout) &&
         fwrite(&sum_w_sqr, sizeof(double), 1, fout) &&
         fwrite(&sum_w_scale, sizeof(double), 1, fout) && 
         fwrite(&nextUpdateInd, sizeof(int), 1, fout));
  if(!b) { 
    fprintf(stderr, "Error saving online data to %s\n", fname);
    return false; 
  }
  if(!chooser->Save(fout)) { 
    fprintf(stderr, "Error saving example choosing buffers to %s\n", fname);
    return false; 
  }
  if(!stats->Save(fout)) { 
    fprintf(stderr, "Error saving training statistics to %s\n", fname);
    return false; 
  }

  fclose(fout);

  char cache_name[1000];  
  sprintf(cache_name, "%s.cache", fname);
  chooser->SaveCachedExamples(cache_name);

  return true;
}

bool StructuredSVM::LoadOnlineData(const char *fname) {
  FILE *fin = fopen(fname, "rb");
  if(!fin) {
    fprintf(stderr, "Error loading online data from %s, open failed\n", fname);
    return false;
  }

  if(trainfile)
    assert((trainset=LoadDataset(trainfile)) != NULL);

  bool b = (fread(&regularization_error, sizeof(double), 1, fin) &&
         fread(&base_time, 1, sizeof(double), fin) &&
         fread(&n, sizeof(int), 1, fin) &&
         fread(&sum_dual, sizeof(double), 1, fin) &&
         fread(&sum_alpha_loss, sizeof(double), 1, fin) &&
         fread(&sum_w_sqr, sizeof(double), 1, fin) &&
         fread(&sum_w_scale, sizeof(double), 1, fin) &&
         fread(&nextUpdateInd, sizeof(long), 1, fin));
  if(!b) { 
    fprintf(stderr, "Error loading online data from %s\n", fname);
    return false; 
  }
  if(!chooser->Load(fin)) { 
    fprintf(stderr, "Error loading example choosing buffers from %s\n", fname);
    return false; 
  }
  if(!stats->Load(fin)) { 
    fprintf(stderr, "Error loading training statistics from %s\n", fname);
    return false; 
  }
  fclose(fin);

  char cache_name[1000];  
  sprintf(cache_name, "%s.cache", fname);
  chooser->LoadCachedExamples(cache_name);

  return true;
}

void StructuredSVM::LoadTrainset(const char *fname) {
  if(trainfile) free(trainfile);
  trainfile = StringCopy(fname);

  trainset = LoadDataset(fname);
  if(!trainset) {
    fprintf(stderr, "Failed to load trainset %s\n", fname);
    assert(trainset != NULL);
    exit(0);
  }
}


SparseVector *StructuredSVM::GetCurrentWeights(bool lock) {
  if(lock) Lock();
  SparseVector *retval = sum_w->mult_scalar(sum_w_scale ? 1.0/(sum_w_scale) : 0, NULL).ptr();
  if(lock) Unlock();
  return retval;
}

int StructuredSVM::GetCurrentEpoch() { return chooser->GetCurrentEpoch(); }

void OptimizationMethodToString(StructuredPredictionOptimizationMethod method, char *str) {
  switch(method) {
  case SPO_CUTTING_PLANE: strcpy(str, "cutting_plane"); break;
  case SPO_CUTTING_PLANE_1SLACK: strcpy(str, "cutting_plane_1slack"); break;
  case SPO_SGD: strcpy(str, "SGD"); break;
  case SPO_SGD_PEGASOS: strcpy(str, "SGD_pegasos"); break;
  case SPO_DUAL_UPDATE: strcpy(str, "online_dual_ascent"); break;
  case SPO_DUAL_UPDATE_WITH_CACHE: strcpy(str, "online_dual_ascent_with_cache"); break;
  case SPO_DUAL_MULTI_SAMPLE_UPDATE: strcpy(str, "multi_sample"); break;
  case SPO_DUAL_MULTI_SAMPLE_UPDATE_WITH_CACHE: strcpy(str, "multi_sample_with_cache"); break;
  case SPO_MAP_TO_BINARY: strcpy(str, "binary_classification"); break;
  case SPO_MAP_TO_BINARY_MINE_HARD_NEGATIVES: strcpy(str, "mine_hard_negatives"); break;
  case SPO_FIXED_SAMPLE_SET: strcpy(str, "fixed_sample_set"); break;
  default:  strcpy(str, "unknown"); break;
  }
}

StructuredPredictionOptimizationMethod OptimizationMethodFromString(const char *str) {
  if(!strcmp(str, "cutting_plane")) return SPO_CUTTING_PLANE; 
  else if(!strcmp(str, "cutting_plane_1slack")) return SPO_CUTTING_PLANE_1SLACK; 
  else if(!strcmp(str, "SGD")) return SPO_SGD; 
  else if(!strcmp(str, "SGD_pegasos")) return SPO_SGD_PEGASOS; 
  else if(!strcmp(str, "online_dual_update")) return SPO_DUAL_UPDATE; 
  else if(!strcmp(str, "online_dual_ascent")) return SPO_DUAL_UPDATE; 
  else if(!strcmp(str, "online_dual_ascent_with_cache")) return SPO_DUAL_UPDATE_WITH_CACHE; 
  else if(!strcmp(str, "multi_sample")) return SPO_DUAL_MULTI_SAMPLE_UPDATE; 
  else if(!strcmp(str, "multi_sample_with_cache")) return SPO_DUAL_MULTI_SAMPLE_UPDATE_WITH_CACHE; 
  else if(!strcmp(str, "binary_classification")) return SPO_MAP_TO_BINARY; 
  else if(!strcmp(str, "mine_hard_negatives")) return SPO_MAP_TO_BINARY_MINE_HARD_NEGATIVES; 
  else if(!strcmp(str, "fixed_sample_set")) return SPO_FIXED_SAMPLE_SET; 
  else return (StructuredPredictionOptimizationMethod)-1;
}

void MemoryModeToString(MemoryMode method, char *str) {
  switch(method) {
  case MEM_KEEP_DATASET_IN_MEMORY:
    strcpy(str, "dataset"); break;
  case MEM_KEEP_DUAL_VARIABLES_IN_MEMORY: 
    strcpy(str, "no_dataset"); break;
  case MEM_KEEP_NOTHING_IN_MEMORY:
    strcpy(str, "nothing"); break;
  case MEM_KEEP_EVERYTHING_IN_MEMORY:
  default:
    strcpy(str, "everything"); break;
  }
}
MemoryMode MemoryModeFromString(const char *str) {
  if(!strcmp(str, "dataset")) return MEM_KEEP_DATASET_IN_MEMORY; 
  else if(!strcmp(str, "no_dataset")) return MEM_KEEP_DUAL_VARIABLES_IN_MEMORY; 
  else if(!strcmp(str, "nothing")) return MEM_KEEP_NOTHING_IN_MEMORY; 
  else if(!strcmp(str, "everything")) return MEM_KEEP_EVERYTHING_IN_MEMORY; 
  else return (MemoryMode)-1;
}

void InitTrainParams(StructuredSVMTrainParams *params) {
  params->eps = 0;
  params->C = 0;
  params->debugLevel = 2;
  params->lambda = params->C ? 1/params->C : 0;
  params->method = SPO_DUAL_MULTI_SAMPLE_UPDATE_WITH_CACHE;
  params->maxLoss = 0;
  params->canScaleW = false;
  params->mergeSamples = true;
  params->randomize = true;

  params->memoryMode = MEM_KEEP_EVERYTHING_IN_MEMORY;

  params->dumpModelStartTime = 0;
  params->dumpModelFactor = 0;
  params->numModelDumps = 0;

  params->trainLatent = 0;

  params->runMultiThreaded = 0;
  params->maxIters = 1000000000;

  params->maxCachedSamplesPerExample = 0;
  params->max_samples = 50;
  params->updateFromCacheThread = false;
  params->numCacheUpdatesPerIteration = 0;
  params->numMultiSampleIterations = 10;
  params->allSamplesOrthogonal = false;

  params->num_thr = omp_get_num_procs();
#ifdef MAX_THREADS
  if(params->num_thr > MAX_THREADS) params->num_thr = MAX_THREADS;
#endif
}
