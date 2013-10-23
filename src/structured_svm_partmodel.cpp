#include "structured_svm_partmodel.h"
#include "structured_svm_multi_object.h"
#include "pose.h"
#include "spatialModel.h"
#include "dataset.h"
#include "structured_svm_train_choose_example.h"

//#define DATASET_TOO_BIG_FOR_MEMORY

PartLocalizedStructuredSVM::PartLocalizedStructuredSVM() : StructuredSVM() {
  classes = NULL;
  params.canScaleW = false;
}

StructuredLabel *PartLocalizedStructuredSVM::NewStructuredLabel(StructuredData *x) { 
  return new PartLocalizedStructuredLabelWithUserResponses(x); 
}

StructuredData *PartLocalizedStructuredSVM::NewStructuredData() { 
  return new PartLocalizedStructuredData; 
}


bool PartLocalizedStructuredSVM::Load(const Json::Value &root) {
  char cname[1000];
  if(classes) delete classes;
  classes = new Classes();
  strcpy(cname, root.get("classes", "").asString().c_str());
  if(!strlen(cname) && !classes->Load(root)) return false;
  else if(!classes->Load(cname)) return false;
  sizePsi=classes->NumWeights(true,false); 
  return true;
}



PartModelStructuredSVM::PartModelStructuredSVM() : PartLocalizedStructuredSVM() {
  partLosses = NULL;
  params.debugLevel = 3;
  params.maxLoss = 1;
  params.numCacheUpdatesPerIteration = 50;
  params.max_samples = 50;
  params.maxCachedSamplesPerExample = 20;
  params.numMultiSampleIterations = 20;
  params.runMultiThreaded = true;//false;
  params.method = SPO_DUAL_MULTI_SAMPLE_UPDATE_WITH_CACHE;   
  params.updateFromCacheThread = true;

  useSoftPartLocations = true;
#ifdef DATASET_TOO_BIG_FOR_MEMORY
  max_cache_sets_in_memory = (max_samples+2)*num_thr*2+50;//(int)(1.0e9 / (sizePsi*12));   // 1GB cache
#else
  params.maxCachedSamplesPerExample = 5;
#endif
  multiSampleMethod = SAMPLE_ALL_POSES;
}

PartModelStructuredSVM::~PartModelStructuredSVM() {
  if(partLosses) 
    free(partLosses);
}
double PartModelStructuredSVM::Inference(StructuredData *x, StructuredLabel *ybar, SparseVector *w, 
					 StructuredLabel *y_partial, StructuredLabel *y_gt, double w_scale) {
  ImageProcess *process = ((PartLocalizedStructuredData*)x)->GetProcess(classes);
  PartLocation *y_gt_locs = y_gt ? ((PartLocalizedStructuredLabel*)y_gt)->GetPartLocations() : NULL;
  PartLocation *y_partial_locs = y_partial ? ((PartLocalizedStructuredLabel*)y_partial)->GetPartLocations() : NULL;
  PartLocalizedStructuredLabel *m_ybar = (PartLocalizedStructuredLabel*)ybar;
  float *ww = w->get_non_sparse<float>(sizePsi);
  process->SetCustomWeights(ww, true, false);

  if(m_ybar->part_locations)
    delete [] m_ybar->part_locations;
  m_ybar->part_locations = NULL;

  //process->Clear(true, true);

  // Ensure that the predicted part locations are consistent with y_partial
  if(y_partial) {
    if(((PartLocalizedStructuredLabelWithUserResponses*)y_partial)->NumUsers()) 
      process->RestrictLocationsToAgreeWithAtLeastOneUser((PartLocalizedStructuredLabelWithUserResponses*)y_partial);
    else {
      for(int i = 0; i < process->GetClasses()->NumParts(); i++) {
        if(!y_partial_locs[i].IsLatent()) {
          if(useSoftPartLocations) process->GetPartInst(i)->SetClickPoint(&y_partial_locs[i]);
          else process->GetPartInst(i)->SetLocation(&y_partial_locs[i], false);
        }
      }
    }
  }


  float s = 0;
  if(y_gt_locs && process->GetClasses()->GetDetectionLossMethod() == LOSS_DETECTION) {
    // If we are training an object detector and don't care about part locations, we return the
    // the not present prediction for positive examples (images where the object is present), and the part locations 
    // with the highest detection score for negative examples (background images where the object isn't present).
    // This is simpler to implement than trying to deal with multiple objects in the same image
    m_ybar->part_locations = NULL;
    fprintf(stderr, "*");
    s = (float)(w->dot(Psi(x, m_ybar)) + Loss(y_gt, ybar));
  } else {
    // Otherwise, find the highest scoring set of part locations
    if(y_gt_locs) process->SetLossImages(y_gt_locs, partLosses);
    process->UseLoss(y_gt_locs != NULL);
    s = process->Detect();
    if(y_gt && !y_gt_locs) s += (float)params.maxLoss;
    //double s2 = w->dot(Psi(x, m_ybar)) + Loss(y_gt, ybar);
    //if(s > s2)
    m_ybar->part_locations = process->ExtractPartLocations();
  }
  if(y_gt) ((PartLocalizedStructuredLabel*)y_gt)->x = (PartLocalizedStructuredData*)x;

  
  //SparseVector ybar_psi = Psi(x, ybar);
  //SparseVector y_psi = Psi(x, y_gt);
  //process->Debug(ww, ybar_psi.get_non_sparse<float>(sizePsi), m_ybar->part_locations, 
  //		 true, false, true, false, y_psi.get_non_sparse<float>(sizePsi));
  

  free(ww);

  return s;
}

#define PAD 20

bool PartModelStructuredSVM::AddInitialSamples() {
  if(classes->GetDetectionLossMethod() == LOSS_DETECTION) {
#pragma omp parallel for
    for(int i = 0; i < trainset->num_examples; i++) {
      PartLocalizedStructuredLabel *y_gt = (PartLocalizedStructuredLabel*)trainset->examples[i]->y;
      PartLocation *y_gt_locs = y_gt ? ((PartLocalizedStructuredLabel*)y_gt)->GetPartLocations() : NULL;
      if(y_gt_locs) {
	SVM_cached_sample_set *set = trainset->examples[i]->set = chooser->BringCacheSetIntoMemory(i, nextUpdateInd, false);
	PartLocalizedStructuredLabel *ybar = (PartLocalizedStructuredLabel*)NewStructuredLabel(trainset->examples[i]->x);
	ybar->part_locations = NULL;
	SVM_cached_sample_set_add_sample(set, ybar);
	SVM_cached_sample_set_compute_features(set, trainset->examples[set->i]);
      }
      OnFinishedIteration(trainset->examples[i]);
    }
    return true;
  }
  return false;
}

extern float g_score2, *g_w2;
double PartModelStructuredSVM::ImportanceSample(StructuredData *x, SparseVector *w, StructuredLabel *y_gt, struct _SVM_cached_sample_set *set, double w_scale) {
  ImageProcess *process = ((PartLocalizedStructuredData*)x)->GetProcess(classes);
  PartLocation *y_gt_locs = y_gt ? ((PartLocalizedStructuredLabel*)y_gt)->GetPartLocations() : NULL;
  float *ww = w ? w->get_non_sparse<float>(sizePsi) : NULL;
  double s = 0;
  int i;
  if(ww) process->SetCustomWeights(ww, true, false);
  FeatureOptions *feat = process->Features();
  int image_width = feat->GetImage()->width, image_height = feat->GetImage()->height;
  PartLocation loc(process->GetRootPart(), image_width, image_height);
  
  
  g_score2 = 0; g_w2 = ww;

  if(y_gt) {
    PartLocation *locs = ((PartLocalizedStructuredLabel*)y_gt)->part_locations;
    if(locs)
      for(int j = 0; j < classes->NumParts(); j++) 
	locs[j].SetFeat(feat); 
  }

  for(int i = 0; i < set->num_samples; i++) {
    PartLocation *locs = ((PartLocalizedStructuredLabel*)set->samples[i].ybar)->part_locations;
    if(locs)
      for(int j = 0; j < classes->NumParts(); j++) 
	locs[j].SetFeat(feat); 
    set->samples[i].slack = w->dot(set->samples[i].psi ? *set->samples[i].psi : Psi(x, set->samples[i].ybar)) + 
      Loss(y_gt, set->samples[i].ybar) - set->score_gt;
  }

  PartLocalizedStructuredLabel *ybar;
  if(y_gt_locs && process->GetClasses()->GetDetectionLossMethod() == LOSS_DETECTION) {
    // LOSS_DETECTION is a particular loss function where it is assumed that we have separated
    // the dataset into positive examples (e.g., a particular location of an object that should have high score)
    // and negative images (e.g., an image such that all possible bounding boxes or assignments to part locations)
    // have low score.  In this case, for positive examples we can avoid having to run our object detector,
    // and just return the not present prediction.  LOSS_DETECTION corresponds to penalizing the number of false
    // negatives plus the number of false positives, where the number of false positives per image is capped at 1.
    if(!set->num_samples) {
      ybar = (PartLocalizedStructuredLabel*)NewStructuredLabel(x);
      ybar->part_locations = NULL;
      SVM_cached_sample_set_add_sample(set, ybar);
    } else
      ybar = (PartLocalizedStructuredLabel*)set->samples[0].ybar;
    fprintf(stderr, "*");
    if(w) s = w->dot(Psi(x, ybar)) + Loss(y_gt, ybar);
  } else {
    // Otherwise, run our object detector, then apply non-maximal suppression to extract
    // a sparse subset of samples that have 1) currently have high slack, and 2) are likely to
    // have feature vectors that are not completely correlated.  
    int wi, he;
    process->Features()->GetDetectionImageSize(&wi, &he, 0, 0);
    if(y_gt_locs) process->SetLossImages(y_gt_locs, partLosses);
    process->UseLoss(y_gt_locs != NULL);
    SparseVector *wc = NULL;
    if(w) {
      int id = process->GetRootPart()->Id();

      s = process->Detect();
      if(y_gt && !y_gt_locs) s += params.maxLoss;
      
      int numFound;
      PartLocation **locs = NULL;
      if(multiSampleMethod == SAMPLE_ALL_POSES)
	locs = process->GetRootPart()->GreedyNonMaximalSuppression(params.max_samples, set->score_gt - (y_gt && !y_gt_locs ? params.maxLoss : 0), &numFound, 2, 2, 2*M_PI, false, true);
      else if(multiSampleMethod == SAMPLE_BY_BOUNDING_BOX)
	locs = process->GetRootPart()->GreedyNonMaximalSuppressionByBoundingBox(params.max_samples, set->score_gt - (y_gt && !y_gt_locs ? params.maxLoss : 0), &numFound); 
      if((multiSampleMethod != SAMPLE_ALL_POSES && multiSampleMethod != SAMPLE_BY_BOUNDING_BOX) || !numFound) {
	numFound = 1;
	locs = new PartLocation*[params.max_samples];
        locs[0] = PartLocation::NewPartLocations(classes, image_width, image_height, feat, false);
	process->ExtractPartLocations(locs[0]);
	locs[0][id].SetScore((float)s);

	if(multiSampleMethod == SAMPLE_RANDOMLY) {
	  for(int k = 1; k < params.max_samples; k++) {
	    locs[k] = PartLocation::NewPartLocations(classes, image_width, image_height, feat, false);
	    locs[k][id] = process->GetRootPart()->DrawRandomPartLocation();
	    process->GetRootPart()->ExtractPartLocations(locs[k], &locs[k][id], NULL);
	  }
	  numFound = params.max_samples;
	}
      }
      

      for(i = 0; i < numFound; i++) {
	if(y_gt && !y_gt_locs) locs[i][id].SetScore(locs[i][id].GetScore() + (float)params.maxLoss);
	PartLocalizedStructuredLabel *ybar_c = (PartLocalizedStructuredLabel*)NewStructuredLabel(x);
	ybar_c->part_locations = (PartLocation*)locs[i];
	SVM_cached_sample *sample = SVM_cached_sample_set_add_sample(set, ybar_c);
	sample->slack = locs[i][id].GetScore()- set->score_gt;
        if(i == 0) {
	  ybar = ybar_c;
	  SVM_cached_sample tmp = set->samples[0];
	  set->samples[0] = *sample;
	  *sample = tmp;
        }
      }

      if(locs) delete [] locs;
      
    } else {
      for(i = 0; i < params.max_samples; i++) {
	PartLocalizedStructuredLabel *ybar_c = (PartLocalizedStructuredLabel*)NewStructuredLabel(x);
	if(i == 0) ybar = ybar_c;
	ybar_c->part_locations = PartLocation::NewPartLocations(classes, wi, he, process->Features(), false);
	process->GetRootPart()->DrawRandomPartLocationsWithoutDetector(ybar_c->part_locations, NULL, PAD);
	SVM_cached_sample_set_add_sample(set, ybar_c);
      }
    }
    if(wc) delete wc;
  }

  //qsort(set->samples, set->num_samples, sizeof(SVM_cached_sample), SVM_cached_sample_cmp);

  if(w && 0) {
    double los = Loss(y_gt, ybar);
    float *tmp = Psi(x, ybar).get_non_sparse<float>(sizePsi);
    double s2 = los;	  
    for(int i = 0; i < sizePsi; i++) s2 += tmp[i]*ww[i];
    free(tmp);

    double s3 = 0, s4 = 0, s5 = 0;
    float *tmp2 = Psi(x, y_gt).get_non_sparse<float>(sizePsi);
    float *tmp3 = set->psi_gt->get_non_sparse<float>(sizePsi);
    double *tmp4 = set->psi_gt->get_non_sparse<double>(sizePsi);
    double *w4 = w->get_non_sparse<double>(sizePsi);
    float *w5 = w->get_non_sparse<float>(sizePsi);
    for(int j = 0; j < sizePsi; j++) {
      s3 += tmp2[j]*ww[j];
      s4 += tmp4[j]*w4[j];
      s5 += tmp2[j]*w5[j];
      assert(tmp2[j] == tmp3[j]);
      assert(my_abs(s3 - s4) < .001);
      assert(my_abs(s3 - s5) < .001);
    }
    assert(my_abs(s3 - set->score_gt) < .001);
    free(tmp2);
    free(tmp3);

#ifdef EXTRA_DEBUG
    process->SanityCheckDynamicProgramming(y_gt_locs);
#endif


    if(my_abs(s - s2) > .001 || s < set->score_gt) {
      SparseVector y_psi = Psi(x, y_gt);
      fprintf(stderr, "\n");
      SparseVector ybar_psi = Psi(x, ybar);
      process->Debug(ww, ybar_psi.get_non_sparse<float>(sizePsi), ((PartLocalizedStructuredLabel*)ybar)->part_locations, 
		     true, false, true, false, y_psi.get_non_sparse<float>(sizePsi));
      process->Debug(ww, y_psi.get_non_sparse<float>(sizePsi), y_gt_locs, true, false, true, false);
      assert(0);
    }
  }

  free(ww);
  return s;
}
  


SparseVector PartModelStructuredSVM::Psi(StructuredData *x, StructuredLabel *y) {
  ImageProcess *process = ((PartLocalizedStructuredData*)x)->GetProcess(classes);
  PartLocation *locs = ((PartLocalizedStructuredLabel*)y)->GetPartLocations();
  VFLOAT *tmp_features = (VFLOAT*)malloc((sizePsi+2)*sizeof(double));
  FeatureOptions *feat = process->Features();

  if(locs) {
    for(int j = 0; j < process->GetClasses()->NumParts(); j++) locs[j].SetFeat(feat);
  }

  /*if(!locs) {
    for(int i = 0; i < sizePsi; i++)
      tmp_features[i] = 0;
  } else {*/
    int n = process->GetFeatures(tmp_features, locs, NULL, true, false);
    assert(n == sizePsi);
    //}

  SparseVector retval(tmp_features, sizePsi);
  free(tmp_features);
  return retval;
}

void PartModelStructuredSVM::OnFinishedIteration(StructuredExample *ex) {
  // Since memory can be a scarce resource, free all memory associated with this example after
  // a training iteration finishes.  This means features will have to be recomputed the next
  // time we iterate over this example, but ensures that memory usage associated with training 
  // doesn't depend on the size of the training set
  PartLocalizedStructuredData *xx = (PartLocalizedStructuredData*)ex->x;
  if(xx->process) {
    delete xx->process; 
    xx->process = NULL;
  }
}

double PartModelStructuredSVM::Loss(StructuredLabel *y_gt, StructuredLabel *y_pred) {
  // Computes the loss of prediction y_pred against the correct label y_gt
  ImageProcess *process = ((PartLocalizedStructuredData*)((PartLocalizedStructuredLabel*)y_gt)->x)->GetProcess(classes);
  PartLocation *y_gt_locs = ((PartLocalizedStructuredLabel*)y_gt)->GetPartLocations();
  PartLocation *y_pred_locs = ((PartLocalizedStructuredLabel*)y_pred)->GetPartLocations();
  if(y_gt_locs) process->SetLossImages(y_gt_locs, partLosses);
  process->UseLoss(y_gt_locs != NULL);
  if(!y_gt_locs) return y_pred_locs ? params.maxLoss : 0;
  else if(!y_pred_locs) return y_gt_locs ? params.maxLoss: 0;
  else return process->ComputeLoss(y_pred_locs);
}




Json::Value PartModelStructuredSVM::Save() {
  Json::Value root;
  
  char fname[1000];
  sprintf(fname, "%s.classes", modelfile);
  root["classes"] = fname;

  SparseVector *w = GetCurrentWeights(false);
  float *ww = w->get_non_sparse<float>(GetSizePsi());
  classes->SetWeights(ww, true, false); 
  classes->Save(fname);
  free(ww);
  delete w;

  Json::Value c;
  if(partLosses) {
    for(int i = 0; i < classes->NumParts(); i++)
      c[i] = partLosses[i];
    root["partLosses"] = c;
  }
  
  return root;
}

bool PartModelStructuredSVM::Load(const Json::Value &root) {
  if(!PartLocalizedStructuredSVM::Load(root)) 
    return false;

  // Read array of part losses
  if(root.isMember("partLosses")) {
    if((int)root["partLosses"].size() != classes->NumParts()) {
      fprintf(stderr, "Unexpected size of partLosses\n");
      return false;
    }
    partLosses = (double*)malloc(classes->NumParts()*sizeof(double));
    for(int i = 0; i < classes->NumParts(); i++)
      partLosses[i] = root["partLosses"][i].asDouble();
  }
  
  return true;
}

							  
char *PartModelStructuredSVM::VisualizeExample(const char *htmlDir, StructuredExample *ex, const char *extraInfo) { 
  PartLocation *y_gt_locs = ex->y ? ((PartLocalizedStructuredLabel*)ex->y)->GetPartLocations() : NULL;
  ImageProcess *process = ((PartLocalizedStructuredData*)ex->y->GetData())->GetProcess(classes);
  IplImage *img = cvCloneImage(process->Image());
  bool drawParts = classes->NumParts() > 1;
  char fname[1000], html[10000], name[1000];
  PartLocalizedStructuredLabel *y = ex->set ? (PartLocalizedStructuredLabel*)ex->set->ybar : NULL;

  if(ex->set) {
    for(int i = 0; i < ex->set->num_samples; i++)
      if(((PartLocalizedStructuredLabel*)ex->set->samples[i].ybar)->GetPartLocations())
	process->Draw(img, ((PartLocalizedStructuredLabel*)ex->set->samples[i].ybar)->GetPartLocations(), CV_RGB(0,0,200), drawParts, false, 
		      drawParts, !drawParts, false, -1, drawParts);
  }
  
  if(y_gt_locs)
    process->Draw(img, y_gt_locs, CV_RGB(0,255,0), drawParts, false, 
		  drawParts, !drawParts, false, -1, drawParts);
  if(y && y->GetPartLocations())
    process->Draw(img, y->GetPartLocations(), CV_RGB(255,0,0), drawParts, false, 
		  drawParts, !drawParts, false, -1, drawParts);
  if(extraInfo) { 
    strcpy(name, extraInfo); 
  } else {
    strcpy(name, process->Features()->Name());
    StripFileExtension(name); StringReplaceChar(name, '/', '_'); StringReplaceChar(name, '\\', '_'); 
    if(name[0] == '.' && name[1] == '.') name[0] = fname[1] = '_';
  }
  if(htmlDir) sprintf(fname, "%s/%s.jpg", htmlDir, name);
  else strcpy(fname, name);
  cvSaveImage(fname, img);
  cvReleaseImage(&img);
  process->Clear();
  
  if(htmlDir) {
    sprintf(html, "<img src=\"%s.jpg\">", name);
    if(ex->set && ex->set->num_samples) 
      sprintf(html+strlen(html), "<center><br>score=%f, score_gt=%f, loss=%f, alpha=%f", 
	    (float)ex->set->samples[0].slack+ex->set->score_gt-ex->set->samples[0].loss, 
	    (float)ex->set->score_gt, (float)ex->set->samples[0].loss, (float)ex->set->alpha);
    else if(ex->set) sprintf(html+strlen(html), "<center><br>slack=%f, score_gt=%f, alpha=%f", 
	    (float)ex->set->slack_before, (float)ex->set->score_gt, (float)ex->set->alpha);
    return StringCopy(html);
  } else
    return NULL;
}

void PartModelStructuredSVM::OnFinishedPassThroughTrainset() { 
  SparseVector *w = GetCurrentWeights(false);
  float *ww = w->get_non_sparse<float>(GetSizePsi());
  classes->SetWeights(ww, true, false); 
  free(ww);
  delete w;
}
