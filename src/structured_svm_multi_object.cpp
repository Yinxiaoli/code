#include "structured_svm_partmodel.h"
#include "structured_svm_multi_object.h"
#include "dataset.h"

StructuredLabel *MultiObjectStructuredSVM::NewStructuredLabel(StructuredData *x) { 
  return new MultiObjectLabelWithUserResponses(x); 
}

StructuredDataset *MultiObjectStructuredSVM::ConvertToSingleObjectDataset(StructuredDataset *d) {
  StructuredDataset *d_single = new StructuredDataset;
  for(int i = 0; i < d->num_examples; i++) {
    MultiObjectLabel *objects = (MultiObjectLabel*)d->examples[i]->y;
    MultiObjectLabel *objects_latent = (MultiObjectLabel*)d->examples[i]->y_latent;
    for(int j = objects->NumObjects() ? 0 : -1; j < objects->NumObjects(); j++) {
      StructuredExample *ex = new StructuredExample;
      ex->x = NewStructuredData();
      Json::Value xx = d->examples[i]->x->save(this);
      ex->x->load(xx, this); 
      ex->y = new PartLocalizedStructuredLabel(ex->x);
      if(j >= 0) {
        Json::Value yy = objects->GetObject(j)->save(this);
        ex->y->load(yy, this);
      }
      if(objects_latent) {
	ex->y_latent = new PartLocalizedStructuredLabel(ex->x);
	if(j >= 0) {
	  Json::Value yy = objects_latent->GetObject(j)->save(this);
	  ex->y_latent->load(yy, this);
	}
      }
      d_single->AddExample(ex);
    }
  }
  return d_single;
}

double MultiObjectStructuredSVM::Inference(StructuredData *x, StructuredLabel *ybar, SparseVector *w, 
					   StructuredLabel *y_partial, StructuredLabel *y_gt, double w_scale) {
  int i;
  ImageProcess *process = ((PartLocalizedStructuredData*)x)->GetProcess(classes);
  FeatureOptions *feat = process->Features();
  int image_width = process->Image()->width, image_height = process->Image()->height;
  MultiObjectLabelWithUserResponses *m_ybar = (MultiObjectLabelWithUserResponses*)ybar;
  MultiObjectLabelWithUserResponses *m_y_partial = (MultiObjectLabelWithUserResponses*)y_partial;
  MultiObjectLabelWithUserResponses *m_y_gt = (MultiObjectLabelWithUserResponses*)y_gt;
  double score_gt = y_gt ? w->dot(Psi(x, y_gt)) : 0;
  float *ww = w->get_non_sparse<float>(sizePsi);
  process->SetCustomWeights(ww, true, false);

  m_ybar->Clear();

  double maxPartLoss = 0;
  for(i = 0; i < process->GetClasses()->NumParts(); i++) 
    maxPartLoss += partLosses ? partLosses[i] : 1;

  float s = 0;
  if(y_gt) process->SetMultiObjectLoss(m_y_gt, partLosses);
  process->UseLoss(y_gt && m_y_gt->NumObjects());

  if(y_partial) {
    // Ensure that the predicted part locations are consistent with y_partial.  Assume y_partial tells us how many
    // objects there are in the image, and possibly some information about the location of each object
    if(m_y_gt)
      assert(m_y_partial->NumObjects() == m_y_gt->NumObjects());
    for(int i = 0; i < m_y_partial->NumObjects(); i++) {
      if(m_y_partial->GetObject(i)->NumUsers()) 
	process->RestrictLocationsToAgreeWithAtLeastOneUser(m_y_partial->GetObject(i));
      else {
	PartLocation *locs = m_y_partial->GetObject(i)->GetPartLocations();
	for(int j = 0; j < process->GetClasses()->NumParts(); j++) {
	  process->GetPartInst(j)->Clear();
	  if(!locs[j].IsLatent()) {
	    if(useSoftPartLocations) process->GetPartInst(j)->SetClickPoint(&locs[j], false);
	    else process->GetPartInst(j)->SetLocation(&locs[j], false, false);
	  }
	}
      }
      s += process->Detect();
      PartLocalizedStructuredLabel *o = new PartLocalizedStructuredLabel(x);
      PartLocation *part_locations = PartLocation::NewPartLocations(classes, image_width, image_height, feat, false);
      process->ExtractPartLocations(part_locations);
      o->SetPartLocations(part_locations);
      m_ybar->AddObject(o);
    }
    return s;
  }

  s = process->Detect();

  // Extract all non-overlapping object locations with positive detection score
  int wi, he;
  process->Features()->GetDetectionImageSize(&wi, &he, 0, 0);
  int num_boxes;
  CvRectScore *boxes = process->GetRootPart()->GetBoundingBoxes(&num_boxes, true, true, false);
  int num_boxes_nms = NonMaximalSuppression(boxes, num_boxes, DEFAULT_OVERLAP, wi, he);
  for(i = 0; i < num_boxes_nms && i < max_objects_per_image; i++) {
    if(y_gt && !m_y_gt->NumObjects()) 
      boxes[i].score += maxPartLoss;
    if(boxes[i].score - score_gt > 0) {
      PartLocalizedStructuredLabel *o = new PartLocalizedStructuredLabel(x);
      PartLocation *part_locations = (PartLocation*)boxes[i].data;
      if(!part_locations) {
        PartLocation loc(process->GetRootPart(), image_width, image_height);
	loc.SetDetectionLocation(boxes[i].det_x, boxes[i].det_y, boxes[i].scale_ind, boxes[i].rot_ind, LATENT, LATENT, LATENT);
	loc.SetScore(boxes[i].score - score_gt);
	part_locations = PartLocation::NewPartLocations(classes, image_width, image_height, feat, false);
	process->GetRootPart()->ExtractPartLocations(part_locations, &loc);
      }
      o->SetPartLocations(part_locations);
      boxes[i].data = NULL;
      m_ybar->AddObject(o);
      s += boxes[i].score;
    }
  }

  for(i = 0; i < num_boxes; i++)
    if(boxes[i].data)
      delete [] (PartLocation*)boxes[i].data;
  free(boxes);

  free(ww);

  if(y_gt) {
    double s2 = w->dot(Psi(x, ybar))+Loss(y_gt,ybar);
    assert(s2 == s);
  }
  return s;
}

double MultiObjectStructuredSVM::ImportanceSample(StructuredData *x, SparseVector *w, StructuredLabel *y_gt, struct _SVM_cached_sample_set *set, double w_scale) {
  ImageProcess *process = ((PartLocalizedStructuredData*)x)->GetProcess(classes);
  FeatureOptions *feat = process->Features();
  int i;
  
  // Update samples maintained from a previous iteration
  for(i = 0; i < set->num_samples; i++) {
    MultiObjectLabel *o = (MultiObjectLabel*)set->samples[i].ybar;
    for(int j = 0; j < o->NumObjects(); j++) {
      PartLocation *locs = o->GetObject(j)->GetPartLocations();
      for(int k = 0; k < classes->NumParts(); k++) 
	locs[j].SetFeat(feat); 
      set->samples[i].slack = w->dot(Psi(x, set->samples[i].ybar)) + Loss(y_gt, set->samples[i].ybar) - set->score_gt;
    }
  }

  // Get the set of all objects with non-zero slack
  MultiObjectLabel *ybar = (MultiObjectLabel*)NewStructuredLabel(x);
  double s = Inference(x, ybar, w, NULL, y_gt, w_scale);
  SVM_cached_sample *sample = SVM_cached_sample_set_add_sample(set, ybar);
  sample->slack = s - set->score_gt;

  // Add a sample for each object in isolation
  int r_id = process->GetRootPart()->Model()->Id();
  for(i = 0; i < ybar->NumObjects() && i < params.max_samples; i++) {
    MultiObjectLabel *o = (MultiObjectLabel*)NewStructuredLabel(x);
    o->AddObject(ybar->GetObject(i));
    SVM_cached_sample *sample = SVM_cached_sample_set_add_sample(set, o);
    sample->slack = ybar->GetObject(i)->GetPartLocations()[r_id].GetScore();
  }

  qsort(set->samples, set->num_samples, sizeof(SVM_cached_sample), SVM_cached_sample_cmp);

  return s;
}
  


SparseVector MultiObjectStructuredSVM::Psi(StructuredData *x, StructuredLabel *y) {
  int i;
  ImageProcess *process = ((PartLocalizedStructuredData*)x)->GetProcess(classes);
  MultiObjectLabel *m_y = (MultiObjectLabel*)y;
  float *tmp_features = (float*)malloc((sizePsi+2)*sizeof(float));
  VFLOAT *tmp_features2 = (VFLOAT*)malloc((sizePsi+2)*sizeof(double));
  for(i = 0; i <= sizePsi; i++)
    tmp_features[i] = 0;
  
  for(i = 0; i < m_y->NumObjects(); i++) {
    int n = process->UpdatePartFeatures(tmp_features, m_y->GetObject(i)->GetPartLocations());
    assert(n == sizePsi);
  }

  for(i = 0; i < sizePsi; i++)
    tmp_features2[i] = tmp_features[i];

  SparseVector retval(tmp_features2, sizePsi);
  free(tmp_features);
  return retval;
}


double MultiObjectStructuredSVM::Loss(StructuredLabel *y_gt, StructuredLabel *y_pred) {
  // Computes the loss of prediction y_pred against the correct label y_gt
  MultiObjectLabel *m_y_gt = (MultiObjectLabel*)y_gt;
  MultiObjectLabel *m_y_pred = (MultiObjectLabel*)y_pred;
  ImageProcess *process = ((PartLocalizedStructuredData*)m_y_gt->GetData())->GetProcess(classes);
  process->SetMultiObjectLoss(m_y_gt, partLosses);
  process->UseLoss(true);

  float loss = 0;
  for(int i = 0; i < m_y_pred->NumObjects(); i++) 
    loss += process->ComputeLoss(m_y_pred->GetObject(i)->GetPartLocations());
  
  return loss;
}



							  
char *MultiObjectStructuredSVM::VisualizeExample(const char *htmlDir, StructuredExample *ex, const char *extraInfo) { 
  ImageProcess *process = ((PartLocalizedStructuredData*)ex->y->GetData())->GetProcess(classes);
  IplImage *img = cvCloneImage(process->Image());
  bool drawParts = classes->NumParts() > 1;
  char fname[1000], html[10000], name[1000];

  if(ex->set) {
    for(int i = 0; i < ex->set->num_samples && i < 1; i++) {
      MultiObjectLabel *y = (MultiObjectLabel*)ex->set->samples[i].ybar;
      for(int j = 0; j < y->NumObjects(); j++)
	process->Draw(img, y->GetObject(j)->GetPartLocations(), CV_RGB(0,0,200), drawParts, false, 
		      drawParts, !drawParts, false, -1, drawParts);
    }
  }
  
  MultiObjectLabel *y = (MultiObjectLabel*)ex->y;
  if(y)
    for(int j = 0; j < y->NumObjects(); j++)
      process->Draw(img, y->GetObject(j)->GetPartLocations(), CV_RGB(0,255,0), drawParts, false, 
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


Json::Value MultiObjectStructuredSVM::Save() {
  Json::Value v = PartModelStructuredSVM::Save();
  v["max_objects_per_image"] = max_objects_per_image;
  return v;
}


bool MultiObjectStructuredSVM::Load(const Json::Value &root) {
  max_objects_per_image = root.get("max_objects_per_image", 50).asInt();
  return PartModelStructuredSVM::Load(root);
}
