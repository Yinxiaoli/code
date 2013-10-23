/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#include "imageProcess.h"
#include "pose.h"
#include "part.h"
#include "attribute.h"
#include "entropy.h"
#include "spatialModel.h"
#include "dataset.h"
#include "structured_svm_multi_object.h"


#define GENERATE_HEAT_MAPS

ImageProcess::ImageProcess(Classes *c, const char *name, InferenceMethod m, bool bidirectional, bool computeClick, bool parallelize) {
  classes = c;
  
  char fname[1000];
  if(name) {
    imgName = StringCopy(name);
    strcpy(fname, name);
    StripFileExtension(fname); StringReplaceChar(fname, '/', '_'); StringReplaceChar(fname, '\\', '_'); 
    if(fname[0] == '.' && fname[1] == '.') fname[0] = fname[1] = '_';
    name = fname;
  }

  feat = new FeatureOptions(imgName, classes->GetFeatureParams(), name, classes);
  feat->SetImageScale(classes->GetImageScale());

  scaleOrientationMethod = classes->GetScaleOrientationMethod();
  inferenceMethod = m;

  useLoss = false;

  computeClickParts = computeClick;
  computeBidirectional = bidirectional;
  partInstances = NULL;
  attributeInstances = NULL;
  partClickInstances = NULL;

  custom_weights = NULL;
  attribute_scores = NULL;
  part_scores = NULL;
  part_click_scores = NULL;

  CreateInstances();
  ResolveLinks();

  nthreads=1;
#if USE_OPENMP
  nthreads = omp_get_num_procs();
#endif
  use_within_image_parallelization = parallelize;
}

ImageProcess::~ImageProcess() {
  int i;
 
  //fprintf(stderr, "Free process %s %x\n", imgName, this);

  if(imgName) free(imgName);

  if(partInstances) {
    for(i = 0; i < classes->NumParts(); i++)
      delete partInstances[i];
    free(partInstances);
  }
  if(partClickInstances) {
    for(i = 0; i < classes->NumClickParts(); i++)
      delete partClickInstances[i];
    free(partClickInstances);
  }
  if(attributeInstances) {
    for(i = 0; i < classes->NumAttributes(); i++)
      delete attributeInstances[i];
    free(attributeInstances);
  } 
  if(feat) {
    feat->Clear(true);
    delete feat;
  }

  if(part_scores) free(part_scores);
  if(part_click_scores) free(part_click_scores);
}

void ImageProcess::ResolveLinks() {
  int i;

  for(i = 0; i < classes->NumParts(); i++)
    partInstances[i]->ResolveLinks();
  for(i = 0; i < classes->NumParts(); i++)
    partInstances[i]->ResolveParentLinks();

  for(i = 0; i < classes->NumClickParts(); i++) {
    partClickInstances[i]->ResolveLinks();
    partClickInstances[i]->GetParent()->SetClickPart(partClickInstances[i]);
  }
  for(i = 0; i < classes->NumClickParts(); i++) 
    partClickInstances[i]->ResolveParentLinks();
}

void ImageProcess::CreateInstances() {
  int i;

  partInstances = (ObjectPartInstance**)malloc(sizeof(ObjectPartInstance*)*classes->NumParts());
  for(i = 0; i < classes->NumParts(); i++) {
    partInstances[i] = new ObjectPartInstance(classes->GetPart(i), this);
  }

  attributeInstances = (AttributeInstance**)malloc(sizeof(AttributeInstance*)*classes->NumAttributes());
  for(i = 0; i < classes->NumAttributes(); i++)
    attributeInstances[i] = new AttributeInstance(classes->GetAttribute(i), this);

  
  partClickInstances = (ObjectPartInstance**)malloc(sizeof(ObjectPartInstance*)*classes->NumClickParts());
  for(i = 0; i < classes->NumClickParts(); i++)
    partClickInstances[i] = new ObjectPartInstance(classes->GetClickPart(i), this);
}

void ImageProcess::DetectAttributes() {
  for(int i = 0; i < classes->NumAttributes(); i++) {
    feat->ReleaseResponses(&attribute_scores[i]);
    attribute_scores[i] = attributeInstances[i]->Detect();
  }
}

void ImageProcess::SetLossImages(PartLocation *locs, double *partLosses) {
  for(int i = 0; i < classes->NumParts(); i++) {
    //partInstances[i]->Detect(false);
    partInstances[i]->SetLoss(locs ? &locs[i] : NULL, (partLosses ? (float)partLosses[i] : 1));
  }
}
void ImageProcess::UseLoss(bool u) {
  useLoss = u;
  for(int i = 0; i < classes->NumParts(); i++)
    partInstances[i]->UseLoss(u);
}
float ImageProcess::ComputeLoss(PartLocation *pred_locs) {
  float loss = 0;
  for(int i = 0; i < classes->NumParts(); i++) {
    float l = partInstances[i]->GetLoss(&pred_locs[i]);
    assert(l >= 0);
    loss += l;
  }
  return loss;
}

void ImageProcess::SetMultiObjectLoss(MultiObjectLabel *y, double *partLosses) {
  SetLossImages(y->NumObjects() ? y->GetObject(0)->GetPartLocations() : NULL, partLosses);
  for(int o = 1; o < y->NumObjects(); o++) 
    for(int i = 0; i < classes->NumParts(); i++) 
      for(int j = 0; j < classes->GetPart(i)->NumPoses(); j++) 
	partInstances[i]->GetPose(j)->ComputeLoss(y->GetObject(o)->GetPartLocations()+i, true);
}

float ImageProcess::GetMultiObjectLoss(MultiObjectLabel *y) {
  float loss = 0;
  for(int o = 0; o < y->NumObjects(); o++) {
    for(int i = 0; i < classes->NumParts(); i++) {
      for(int j = 0; j < classes->GetPart(i)->NumPoses(); j++) {
	loss += partInstances[i]->GetPose(j)->GetLoss(y->GetObject(o)->GetPartLocations()+i);
      }
    }
  }
  return loss;
}

PartLocation *ImageProcess::ExtractPartLocations(PartLocation *locs) {
  if(!locs) locs = PartLocation::NewPartLocations(classes, feat->GetImage()->width, feat->GetImage()->height, feat, false);
  for(int i = 0; i < classes->NumParts(); i++)
    if(partInstances[i]->Model()->GetParent() == NULL)
      partInstances[i]->ExtractPartLocations(locs);
  return locs;
}

int ImageProcess::UpdatePartFeatures(float *f, PartLocation *locs) {
  float *curr = f;
  int i;
  float static_feat[10000];
    
  if(!locs) return classes->NumWeights(true, false);

  for(i = 0; i < classes->NumParts(); i++) {
    int n = partInstances[i]->GetStaticFeatures(static_feat, &locs[i]);
    for(int j = 0; j < n; j++) 
      curr[j] += static_feat[j];
    curr += n;
  }

  int *poseOffsets, *spatialOffsets;
  int n = classes->GetWeightOffsets(&poseOffsets, &spatialOffsets);
  for(i = 0; i < classes->NumParts(); i++) 
    partInstances[i]->UpdateSpatialFeatures(curr, locs, spatialOffsets);
  for(i = 0; i < classes->NumParts(); i++) 
    partInstances[i]->UpdatePoseFeatures(curr, locs, poseOffsets);
  curr += n;

  return curr-f;
}

// Get features around specified part locations
int ImageProcess::GetLocalizedFeatures(float *f, PartLocation *locs, FeatureWindow *featureWindows, int numWindows, int *partInds) {
  float *fptr = f;
  for(int i = 0; i < classes->NumParts(); i++) {
    if(partInds)
      partInds[i] = fptr-f;
    fptr += partInstances[i]->GetLocalizedFeatures(fptr, &locs[i]);
  }
  return fptr-f;
}


// Get features from the full image, possibly in a spatial pyramid
int ImageProcess::GetImageFeatures(float *f, FeatureWindow *featureWindows, int numWindows) {
  PartLocation l;
  float *fptr = f;
  int width = feat->GetImage()->width, height = feat->GetImage()->height;
  l.Init(classes, width, height, feat);
  if(!featureWindows) {
    featureWindows = classes->FeatureWindows();
    numWindows = classes->NumFeatureWindows();
  }

  for(int i = 0; i < numWindows; i++) {
    FeatureWindow *fw = &featureWindows[i];
    SlidingWindowFeature *fe = feat->Feature(fw->name);
    int fscale = fw->scale, w = fw->w, h = fw->h;
    double scale = feat->Scale(fscale);
    if(fw->w > 0 && fw->h > 0) {
      // For features like HOG, compute the closest scale such that a wXh template fills the image
      double x_scale = width/(double)fw->w/classes->SpatialGranularity()/scale;
      double y_scale = height/(double)fw->h/classes->SpatialGranularity()/scale;
      fscale = classes->ScaleInd(my_min(x_scale, y_scale));
      scale = feat->Scale(fscale);
    } else {
      feat->GetDetectionImageSize(&w, &h, fscale, 0);
      if(fw->w < 0) w = w*2/my_abs(fw->w);
      if(fw->h < 0) h = h*2/my_abs(fw->h);
    }
    double x = width/scale*(fw->dx/(float)my_abs(fw->w)+.5);
    double y = height/scale*(fw->dy/(float)my_abs(fw->h)+.5);
    l.SetImageLocation(x, y, scale, 0, NULL);
    
    fptr += fe->GetFeaturesAtLocation(fptr, w, h, fscale, &l, false);
  }
  //for(int i = 0; i < fptr-f; i++)
  //  assert(f[i] <= 1);
  return fptr-f;
}

extern float g_score2, *g_w2;
int ImageProcess::GetFeatures(float *f, PartLocation *locs, float *has_attribute, bool getPartFeatures, bool getAttributeFeatures) {
  int i, numWeights = classes->NumWeights(getPartFeatures, getAttributeFeatures);
  float *curr = f;

  g_score2 = 0;

  for(i = 0; i < numWeights; i++)
    f[i] = 0;

  if(getPartFeatures) 
    curr += UpdatePartFeatures(f, locs);
  if(getAttributeFeatures) {
    for(i = 0; i < classes->NumAttributes(); i++) {
      if(attributeInstances[i]->Model()->NumFeatureTypes()) {
	assert(attributeInstances[i]->Model()->Part());
	curr += attributeInstances[i]->GetFeatures(curr, locs+attributeInstances[i]->Model()->Part()->Id(), has_attribute);
      }
    }
  }
  /*if(getClassFeatures) {
    for(i = 0; i < classes->NumClasses(); i++) 
      curr += classInstances[i]->GetFeatures(curr, locs);
  }*/
  assert((int)(curr-f) == numWeights);
  return numWeights;
}

int ImageProcess::GetFeatures(double *f, PartLocation *locs, float *has_attribute, bool getPartFeatures, bool getAttributeFeatures) {
  float *tmp = (float*)malloc(sizeof(float)*classes->NumWeights(getPartFeatures,getAttributeFeatures));
  int n = GetFeatures(tmp, locs, has_attribute, getPartFeatures, getAttributeFeatures);
  for(int i = 0; i < n; i++) {
    //fprintf(stderr, " %f", tmp[i]); assert(tmp[i] > -10 && tmp[i] < 10);
    f[i] = tmp[i];
  }
  free(tmp);
  return n;
}

void ImageProcess::SetCustomWeights(float *w, bool setPartFeatures, bool setAttributeFeatures) {
  int i, j;
  float *curr = w;
  custom_weights = w;
  if(setPartFeatures) {
    for(i = 0; i < classes->NumParts(); i++) {
      partInstances[i]->SetCustomStaticWeights(curr);
      if(w) curr += classes->GetPart(i)->NumStaticWeights();
    }

    int *pose_offsets, *spatial_offsets;
    int numOffsets = classes->GetWeightOffsets(&pose_offsets, &spatial_offsets);
    for(i = 0; i < classes->NumParts(); i++) 
      for(j = 0; j < classes->GetPart(i)->NumPoses(); j++) 
	partInstances[i]->GetPose(j)->SetCustomWeights(curr, pose_offsets, spatial_offsets);
    if(w) curr += numOffsets;
  }
  if(setAttributeFeatures) {
    for(i = 0; i < classes->NumAttributes(); i++) {
      attributeInstances[i]->SetCustomWeights(curr);
      if(w) curr += classes->GetAttribute(i)->NumWeights();
    }
  }
}

void ImageProcess::DetectParts(bool secondPass) {
  /*// Temporary debug code
  IplImage *siftImg = feat->GetSiftImage();
  SiftParams *params = feat->GetParams();
  int numX = siftImg->width/params->cellWidth;
  int numY = siftImg->height/params->cellWidth;
  float *descriptor = (float*)malloc(sizeof(float)*numX*numY*params->numBins);
  int num = ComputeHOGDescriptor(siftImg, descriptor, 0, 0, numX, numY, params);
  IplImage *tmp = VisualizeHOGDescriptor(descriptor, numX, numY, params->cellWidth, params->numBins, params->combineOpposite);
  cvSaveImage("data/img.png", tmp);
  cvReleaseImage(&tmp);
  free(descriptor);
  */

  if(!part_scores) {
    part_scores = (IplImage****)malloc(sizeof(IplImage***)*classes->NumParts());
    memset(part_scores, 0, sizeof(IplImage***)*classes->NumParts());
  }
  if(secondPass) {
    for(int i = classes->NumParts()-1; i >= 0; i--) 
      part_scores[i] = partInstances[i]->Detect(true);
  } else {
    for(int i = 0; i < classes->NumParts(); i++) 
      part_scores[i] = partInstances[i]->Detect(classes->GetPart(i)->GetParent() == NULL);
  }
}

void ImageProcess::DetectClickParts() {
  if(!part_click_scores) {
    part_click_scores = (IplImage****)malloc(sizeof(IplImage***)*classes->NumClickParts());
    memset(part_click_scores, 0, sizeof(IplImage***)*classes->NumClickParts());
  }
  for(int i = 0; i < classes->NumClickParts(); i++) 
    part_click_scores[i] = partClickInstances[i]->Detect(true);
}

void ImageProcess::Clear(bool clearBuffers, bool clearFeatures, bool clearFeaturesFull, bool clearScores) {
  if(clearBuffers) {
    for(int i = 0; i < classes->NumParts(); i++) 
      partInstances[i]->Clear(clearScores);
    for(int i = 0; i < classes->NumAttributes(); i++) 
      attributeInstances[i]->Clear();
    for(int i = 0; i < classes->NumClickParts(); i++) 
      partClickInstances[i]->Clear(clearScores);
    //for(int i = 0; i < classes->NumClasses(); i++) 
    //  classInstances[i]->Clear();
    if(clearScores && part_scores) { free(part_scores); part_scores = NULL; }
    if(clearScores && part_click_scores) { free(part_click_scores); part_click_scores = NULL; }
  }

  if(clearFeatures) 
    feat->Clear(clearFeaturesFull);
}

float ImageProcess::Detect() {
  if(g_debug > 0) fprintf(stderr, "Detect %s...\n", feat->Name());

  DetectParts(false);


  // DetectParts() computes the maximum likelihood position of each part, as well as a dense computation of the 
  // maximum likelihood position of each part conditioned on any given location of the root part.  When the computeBidirectional
  // flag is set, we also densely compute the maximum likelihood position of each part conditioned on any possible location of
  // any single part.  This is done by making a 2nd pass through the tree, propagating information from parent parts into the
  // child parts
  if(computeBidirectional) {
    if(g_debug > 1) fprintf(stderr, "Second Pass Detect %s...\n", feat->Name());
    DetectParts(true);
  }

  if(computeClickParts) {
    assert(computeBidirectional);
    if(g_debug > 1) fprintf(stderr, "Click Parts Detect %s...\n", feat->Name());
    DetectClickParts();
  }

  
  float score = 0;
  for(int i = 0; i < classes->NumParts(); i++) {
    if(classes->GetPart(i)->GetParent() == NULL) {
      float s = partInstances[i]->GetPartLocation()->GetScore();
      score += s;
    }
  }
 
  return score;
}



void ImageProcess::Draw(IplImage *img, PartLocation *locs2, CvScalar color, bool labelParts, bool mixColors, 
			bool labelPoints, bool labelRects, bool showAnchors, int selectedPart, bool showTree) {
  PartLocation *locs = locs2 ? locs2 : ExtractPartLocations();
  float zoom = img->width / (float)feat->GetImage()->width;
 
  //CvScalar colors[7] = { CV_RGB(255,255,255), CV_RGB(0,0,255), CV_RGB(255,255,0), CV_RGB(255,0,255), CV_RGB(0,255,255), CV_RGB(0,255,0), CV_RGB(255,0,0) };
  
  CvScalar colors1[15] =  { CV_RGB(255,0,0), CV_RGB(128,0,0), CV_RGB(0,255,0), CV_RGB(0,128,0), CV_RGB(255,191,74), CV_RGB(0,0,128),  CV_RGB(255,255,0), CV_RGB(98,98,0), CV_RGB(0,255,255), CV_RGB(0,98,98), CV_RGB(255,0,255), CV_RGB(98,0,98),  CV_RGB(255,255,255), CV_RGB(0,0,0),  CV_RGB(68,32,15) }; 
  CvScalar colors2[15] = { CV_RGB(0,0,0), CV_RGB(255,255,255), CV_RGB(0,0,0), CV_RGB(255,255,255), CV_RGB(0,0,0), CV_RGB(255,255,255), CV_RGB(0,0,0), CV_RGB(255,255,255), CV_RGB(0,0,0), CV_RGB(255,255,255), CV_RGB(0,0,0), CV_RGB(255,255,255), CV_RGB(0,0,0), CV_RGB(255,255,255), CV_RGB(255,255,255) };

  int n = ceil(LOG2(classes->NumParts())/3)+1, ind1 = 0, ind2 = 1;
  for(int i = 0; i < classes->NumParts(); i++) {
    if(mixColors) {
      color = CV_RGB((i%n)*99/(n-1), ((i/n)%n)*99/(n-1), (int)((1.0f-((float)i)/n/n/n)*99));
    }    
    CvScalar c_inner = i==selectedPart ? CV_RGB(0,255,0) : ((showAnchors && partInstances[i]->IsAnchor()) ? CV_RGB(255,0,0) : colors1[ind1]);
    CvScalar c_outer = i==selectedPart ? CV_RGB(0,255,0) : ((showAnchors && partInstances[i]->IsAnchor()) ? CV_RGB(255,0,0) : 
							    (classes->NumParts() > 15 ? colors1[ind2] : colors2[ind1]));
    CvScalar c_text = c_outer;
    if(classes->NumParts() > 15) {
      do {
	ind1++;
	if(ind1 >= 15) { ind1 = 0; ind2++; }
      } while(classes->NumParts() > 15 && ind1 == ind2);
    } else
      ind1++;

    if(showAnchors && partInstances[i]->IsAnchor()) {
      c_inner = c_outer = CV_RGB(255,0,0);
      c_text = CV_RGB(0,0,0);
    }
    

    ObjectPart *part = classes->GetPart(i);
    ObjectPart *parent = part->GetParent();
    int pose, par_pose=-1;
    float x, y, par_x=0, par_y=0;
    locs[i].GetDetectionLocation(NULL, NULL, NULL, NULL, &pose);
    locs[i].GetImageLocation(&x, &y);
    if(parent) {
      locs[parent->Id()].GetDetectionLocation(NULL, NULL, NULL, NULL, &par_pose);
      locs[parent->Id()].GetImageLocation(&par_x, &par_y);
    }
    if(showTree && parent && !part->GetPose(pose)->IsNotVisible() && !parent->GetPose(par_pose)->IsNotVisible() ) {
      cvLine(img, cvPoint(x*zoom,y*zoom), cvPoint(par_x*zoom,par_y*zoom), color, 2);
    }
    partInstances[i]->Draw(img, &locs[i], c_inner, c_outer, c_text, labelParts ? partInstances[i]->Model()->Name() : NULL, labelPoints, labelRects, zoom);
  }
  if(selectedPart >= 0) 
    partInstances[selectedPart]->Draw(img, &locs[selectedPart], CV_RGB(0,255,0), CV_RGB(0,255,0), CV_RGB(0,0,0), 
				      partInstances[selectedPart]->Model()->Name(), labelPoints, labelRects, zoom);

  if(locs != locs2) delete [] locs;
}

void ImageProcess::DrawClicks(IplImage *img, CvScalar color, bool labelParts, bool mixColors, bool labelPoints, bool labelRects) {
  int n = ceil(LOG2(classes->NumClickParts())/3)+1;
  for(int i = 0; i < classes->NumClickParts(); i++) {
    if(partClickInstances[i]->IsObserved()) {
      if(mixColors) {
	color = CV_RGB((i%n)*99/(n-1), ((i/n)%n)*99/(n-1), (int)((1.0f-((float)i)/n/n/n)*99));
      }
      PartLocation l(partClickInstances[i]->GetBestLoc());
      partClickInstances[i]->Draw(img, &l, color, color, color, labelParts ? partClickInstances[i]->Model()->Name() : NULL, labelPoints, labelRects);
    }
  }
}

PartLocationSampleSet *ImageProcess::DrawRandomPartLocationSet(int numSamples, ObjectPartInstance *root_part, bool useNMS) {
  int sz = sizeof(PartLocationSampleSet)+(numSamples)*(sizeof(PartLocation**)) + numSamples*sizeof(int);
  PartLocationSampleSet *samples = (PartLocationSampleSet*)malloc(sz);
  memset(samples, 0, sz);
  
  // Create a set of num_samples random assignments to all part locations, sampled roughly according to the 
  // probability of some configuration of part location 
  samples->num_samples = numSamples;
  samples->samples = (PartLocation**)(samples+1);
  samples->root_samples = new PartLocation[numSamples];
  PartLocation *l;
  ObjectPartInstance *part;
  for(int i = 0; i < numSamples; i++) {
    samples->samples[i] = PartLocation::NewPartLocations(classes, feat->GetImage()->width, feat->GetImage()->height, feat, root_part ? root_part->Model()->IsClick() : false);

    // If root_part is set, pick a random location of root_part with probability proportional to the value of its 
    // probability map at that location. If root part is not set, pick a part uniformly at random, then pick
    // a location randomly according to its probability map
    if(root_part) 
      part = root_part;
    else 
      part = partInstances[rand()%classes->NumParts()];
    l = &samples->root_samples[i];
    l->Init(classes, feat->GetImage()->width, feat->GetImage()->height, feat);
    l->SetPart(classes->GetPart(part->Id())); 
    for(int t = 0; t < 50; t++) {
      // HACK: Only select poses that are visible
      if(useNMS)
	l->Copy(part->DrawRandomPartLocationNMS());
      else
	l->Copy(part->DrawRandomPartLocation());

      int pose;
      l->GetDetectionLocation(NULL, NULL, NULL, NULL, &pose);
      if(!partInstances[part->Id()]->GetPose(pose)->IsNotVisible())
	break;
    }
    part->ExtractPartLocations(samples->samples[i], l, NULL);
  }
  return samples;
}

// Compute an array of approximate probabilities for each class, by sampling part configurations and
// evaluating the class probabilities at that location
void ImageProcess::ComputeClassProbabilitiesFromSamples(double *classProbs, PartLocationSampleSet *locs) {
  int i, j;
  double *classProbsCurr = (double*)malloc(sizeof(double)*classes->NumClasses());
  for(j = 0; j < classes->NumClasses(); j++)
    classProbs[j] = 0;
  for(i = 0; i < locs->num_samples; i++) {
    ComputeClassProbabilitiesAtLocation(classProbsCurr, locs->samples[i]);
    for(j = 0; j < classes->NumClasses(); j++)
      classProbs[j] += classProbsCurr[j];
  }
  NormalizeProbabilities(classProbs, classProbs, classes->NumClasses()); 
  free(classProbsCurr);
}

void ImageProcess::ComputeClassProbabilities(double *classProbs) {
  PartLocationSampleSet *locs = DrawRandomPartLocationSet(100);
  ComputeClassProbabilitiesFromSamples(classProbs, locs);
  for(int i = 0; i < locs->num_samples; i++)
    delete [] locs->samples[i];
  delete [] locs->root_samples;
  free(locs);
}

#include "class.h"
void ImageProcess::ComputeClassLogLikelihoodsAtLocation(double *classLikelihoods, PartLocation *locs, bool useGamma) {
  // Compute class likelihoods as the sum over part likelihoods
  int i, j;
  for(j = 0; j < classes->NumClasses(); j++)
    classLikelihoods[j] = 0;
  for(i = 0; i < classes->NumParts(); i++) 
    partInstances[i]->UpdateClassLogLikelihoodAtLocation(classLikelihoods, locs+i, useGamma&&classes->ScaleAttributesIndependently());

  if(useGamma) {
    float gamma = classes->GetClassGamma();
    double m = -1000000; for(j = 0; j < classes->NumClasses(); j++) m = my_max(m, gamma*classLikelihoods[j]); 
    double sum = 0; for(j = 0; j < classes->NumClasses(); j++) sum += exp(gamma*classLikelihoods[j]-m);
    double delta = -m - log(sum);  // delta is a per example constant that normalizes the class probabilities
    for(j = 0; j < classes->NumClasses(); j++) classLikelihoods[j] = gamma*classLikelihoods[j] + delta;
  }
}

// Compute an array of probabilities for each class, evaluated at a particular assignment to all part locations,
// stored in 'locs'
void ImageProcess::ComputeClassProbabilitiesAtLocation(double *classProbs, PartLocation *locs) {
  int j;

  ComputeClassLogLikelihoodsAtLocation(classProbs, locs);
    
  // Convert class likelihoods to probabilities
  for(j = 0; j < classes->NumClasses(); j++)
    classProbs[j] = exp(classProbs[j]);
  NormalizeProbabilities(classProbs, classProbs, classes->NumClasses()); 
}



void ImageProcess::SaveProbabilityMaps(const char *fname, const char *dir, char *html, bool saveClicks, bool mergeResults, bool generateHeatMaps, bool mergePoses, bool keepBigImage) {
  int i, j;
  ObjectPartInstance **insts = saveClicks ? partClickInstances : partInstances;
  int numParts = saveClicks ? classes->NumParts() : classes->NumClickParts();

  if(html) strcpy(html, "");
  for(i = 0; i < numParts; i++) 
    insts[i]->ComputeCumulativeSums(true);

  IplImage *maps[1000][100];
  for(i = 0; i < numParts; i++) {
    ObjectPartInstance *part = insts[i];
    part->ComputeCumulativeSums();
    PartLocation loc(GetRootPart()->GetBestLoc());
    double maxResponse = 0;//exp(gamma*my_max(loc.score,mi)+delta);
    char pname[1000];
    
    strcpy(pname, insts[i]->Model()->Name());
    StringReplaceChar(pname, ' ', '_');

    for(j = 0; j < insts[i]->Model()->NumPoses(); j++) {
      maps[i][j] = insts[i]->GetPose(j)->SaveProbabilityMaps(fname, dir, html ? html+strlen(html) : NULL, mergeResults);
      if(mergeResults && maps[i][j]) {
	double m, ma; cvMinMaxLoc(maps[i][j], &m, &ma);
	if(ma > maxResponse) maxResponse = ma;
	if(mergePoses && j) {
	  if(maps[i][0]) {
	    cvMax(maps[i][j], maps[i][0], maps[i][0]);
	    cvReleaseImage(&maps[i][j]);
	  } else {
	    maps[i][0] = maps[i][j];
	    maps[i][j] = NULL;
	  }
	}
      } else
	assert(!maps[i][j]);
    }
    
    if(mergeResults) {
      for(j = 0; j < insts[i]->Model()->NumPoses(); j++) {
	if(maps[i][j]) {
	  char str[4000];
	  IplImage *subImg2 = cvCreateImage(cvSize(maps[i][j]->width,maps[i][j]->height), IPL_DEPTH_8U, 1);
	  cvCvtScale(maps[i][j], subImg2, 255/maxResponse, 0);

	  if(mergePoses) {
	    if(!fname) sprintf(str, "test/%s_%s.png", feat->Name(), insts[i]->Model()->Name());
	    else sprintf(str, "%s/%s_%s.png", dir, fname, insts[i]->Model()->Name());
	  } else {
	    if(!fname) sprintf(str, "test/%s_%s.png", feat->Name(), insts[i]->Model()->GetPose(j)->Name());
	    else sprintf(str, "%s/%s_%s.png", dir, fname, insts[i]->Model()->GetPose(j)->Name());
	  }

	  if(!generateHeatMaps) {
	    cvSaveImage(str, subImg2);
	    if(html) { 
	      sprintf(html+strlen(html), "<td><center><img src=\"%s\" width=300><h3>%s%s</h3></center></td>", 
		      str + (dir ? strlen(dir)+1 : 0), insts[i]->Model()->GetPose(j)->IsClick() ? "click " : "", 
		      insts[i]->Model()->GetPose(j)->Name());
	    }
	  } else {
	    IplImage *img = feat->GetImage();
	    IplImage *vis = NULL;
	    IplImage *alpha = NULL;
	    IplImage *alpha_small = cvCreateImage(cvSize(subImg2->width,subImg2->height),IPL_DEPTH_8U,3);
	    IplImage *gray = cvCloneImage(img);
	    IplImage *gray2 = cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_8U,1);
	    cvZero(alpha_small);
	    cvSetChannel(alpha_small, subImg2, 2, 0, 1);
	    cvCvtColor(img,gray2,CV_RGB2GRAY);
	    cvSetChannel(gray, gray2, 0, 0); cvSetChannel(gray, gray2, 1, 0); cvSetChannel(gray, gray2, 2, 0);
	    if(keepBigImage) {
	      alpha = cvCloneImage(gray);
	      cvResize(alpha_small, alpha);
	      vis = cvCloneImage(gray);
	      cvAdd(vis, alpha, vis);
	    } else {
	      vis = cvCloneImage(alpha_small);
	      cvResize(gray, vis);
	      cvAdd(vis, alpha_small, vis);
	    }
	    if(mergePoses) {
	      if(!fname) sprintf(str, "test/%s_%s_heat.png", feat->Name(), pname);
	      else sprintf(str, "%s/%s_%s_heat.png", dir, fname, pname);
	      if(html) { 
		sprintf(html+strlen(html), "<td><center><img src=\"%s\" width=300><h3>%s%s</h3></center></td>", 
			str + (dir ? strlen(dir)+1 : 0), insts[i]->Model()->IsClick() ? "click " : "", 
			pname);
	      }
	    } else {
	      if(!fname) sprintf(str, "test/%s_%s_heat.png", feat->Name(), insts[i]->Model()->GetPose(j)->Name());
	      else sprintf(str, "%s/%s_%s_heat.png", dir, fname, insts[i]->Model()->GetPose(j)->Name());
	      if(html) { 
		sprintf(html+strlen(html), "<td><center><img src=\"%s\" width=300><h3>%s%s</h3></center></td>", 
			str + (dir ? strlen(dir)+1 : 0), insts[i]->Model()->GetPose(j)->IsClick() ? "click " : "", 
			insts[i]->Model()->GetPose(j)->Name());
	      }
	    }
	    cvSaveImage(str, vis);
	    cvReleaseImage(&vis); cvReleaseImage(&alpha); cvReleaseImage(&alpha_small); cvReleaseImage(&gray);  cvReleaseImage(&gray2); 
	  }


	  cvReleaseImage(&maps[i][j]);
	  cvReleaseImage(&subImg2);
	}
      }
    }
  }
}

/*
void ImageProcess::SaveClickProbabilityMaps(const char *fname, const char *dir, char *html) {
  int i, j;
  for(i = 0; i < classes->NumClickParts(); i++) 
    partClickInstances[i]->ComputeCumulativeSums();
  for(i = 0; i < classes->NumClickParts(); i++) 
    for(j = 0; j < partClickInstances[i]->Model()->NumPoses(); j++) 
      partClickInstances[i]->GetPose(j)->SaveProbabilityMaps(fname, dir, html ? html+strlen(html) : NULL);
}
*/

void ImageProcess::VisualizeSpatialModel(PartLocation *locs, const char *fname_prefix, char *html) {
  // Visualize spatial model
  char fname[1000], base[1000];
  ExtractFilename(fname_prefix, base);
  //IplImage *img2 = cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_32F,1);
  //cvSet(img2, cvScalar(-100000000));
  IplImage *img = Image();
  IplImage *img4 = cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_8U,3);
  cvZero(img4);
  ObjectPartInstance *part = partInstances[classes->NumParts()-1];
  int pose;
  locs[part->Id()].GetDetectionLocation(NULL, NULL, NULL, NULL, &pose);
  part->GetPose(pose)->VisualizeSpatialModel(img4, locs);
  //mi=-1; cvMaxS(img2, mi, img2);
  //cvConvertScale(img2, img2, part->Model()->GetGamma());
  //cvExp(img2, img2);
  /*IplImage *img3 = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_8U, 1);
  double mii, ma;  cvMinMaxLoc(img2, &mii, &ma); 
  cvCvtScale(img2, img3, 255/ma);
  IplImage *img4 = cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_8U,3);
  cvMerge(img3, img3, img3, NULL, img4); */

  Draw(img4, locs, CV_RGB(0,0,200), false, false, true, false, false, -1, false);
  sprintf(fname, "%s_spatial.png", fname_prefix);
  cvSaveImage(fname, img4);
  sprintf(html, "<table><center><tr><td><img src=\"%s_spatial.png\"></td></tr><tr><td><h2>Spatial Model</h2></td></tr></center></table>", base);
  //cvReleaseImage(&img2); cvReleaseImage(&img3); 
  cvReleaseImage(&img4);
}

void ImageProcess::VisualizeFeatures(PartLocation *locs, const char *fname_prefix) {
  char fname[1000], base[1000], *html = (char*)malloc(100000);
  IplImage *img = Image();

  sprintf(fname, "%s.html", fname_prefix);
  FILE *fout = fopen(fname, "w");
  assert(fout);
  ExtractFilename(fname_prefix, base);
  
  sprintf(fname, "%s_original.png", fname_prefix);
  IplImage *img2 = cvCloneImage(img);
  Draw(img2, locs);
  cvSaveImage(fname, img2);
  cvReleaseImage(&img2);
  fprintf(fout, "<html><body><table><center><tr><td><img src=\"%s_original.png\"></td></tr><tr><td><h1>Original Image</h1></td></tr></center></table>", base);
  
  if(locs) VisualizeSpatialModel(locs, fname_prefix, html);
  fprintf(fout, "%s\n", html);

  fprintf(fout, "%s\n", feat->Visualize(classes, locs, fname_prefix, html));
  fprintf(fout, "</body></html>");
  fclose(fout);

  free(html);
}

void ImageProcess::VisualizePartModel(PartLocation *locs, const char *fname_prefix) {
  char fname[1000], base[1000], *html=(char*)malloc(100000);

  sprintf(fname, "%s.html", fname_prefix);
  FILE *fout = fopen(fname, "w");
  assert(fout);
  ExtractFilename(fname_prefix, base);

  VisualizeSpatialModel(locs, fname_prefix, html) ;
  fprintf(fout, "<html><body>%s\n", html);

  // Visualize appearance features
  fprintf(fout, "<h1>Visualization of Learned Part Model Weights</h1><i>Spatial model not depicted</i>%s\n", feat->Visualize(classes, locs, fname_prefix, html, true));
  fprintf(fout, "</body></html>");
  fclose(fout);
  free(html);
}

void ImageProcess::VisualizeAttributeModels(const char *fname_prefix) {
  char fname[1000], *html=(char*)malloc(100000);

  sprintf(fname, "%s.html", fname_prefix);
  FILE *fout = fopen(fname, "w");
  assert(fout);
  fprintf(fout, "<html><body>");

  for(int i = 0; i < classes->NumAttributes(); i++) {
    sprintf(fname, "%s_%s", fname_prefix, classes->GetAttribute(i)->Name());
    fprintf(fout, "<h1>Detector for Attribute \"%s\"</h1>%s<br><br><br>\n", classes->GetAttribute(i)->Name(), feat->Visualize(classes, NULL, fname, html, true, attributeInstances[i]));
  }
  fprintf(fout, "</body></html>");
  fclose(fout);
  free(html);
}

void ImageProcess::SanityCheckDynamicProgramming(PartLocation *gt_locs) {
  for(int i = 0; i < classes->NumParts(); i++) 
    partInstances[i]->SanityCheckDynamicProgramming(gt_locs);
}


float *g_f = NULL, *g_w, g_loss = 0; 
void ImageProcess::Debug(float *w, float *f, PartLocation *locs,  bool debug_scores, bool print_weights,
			 bool getPartFeatures, bool getAttributeFeatures, float *f_gt) {
  int i;
  float *currF = f, *currW = w, *currF_gt = f_gt;

  g_f = f; g_w = w;
  g_loss = 0;

  fprintf(stderr, "\nDEBUG: ");  

  if(getPartFeatures) {
    if(!locs) {
      fprintf(stderr, "not visible\n");  
      //return;
    } 
    for(i = 0; i < classes->NumParts(); i++) {
      int m = partInstances[i]->DebugStatic(currW, currF, locs, debug_scores, print_weights, currF_gt);
      currF += m; currW += m;
      if(f_gt) currF_gt += m;
    }
    int *poseOffsets, *spatialOffsets;
    int n = classes->GetWeightOffsets(&poseOffsets, &spatialOffsets);
    for(i = 0; i < classes->NumParts(); i++) {
      if(locs) partInstances[i]->Debug(currW, currF, poseOffsets, spatialOffsets, locs, debug_scores, print_weights, currF_gt);
    }
    currF += n; currW += n;
    if(f_gt) currF_gt += n;
  }
  if(getAttributeFeatures) {
    for(i = 0; i < classes->NumAttributes(); i++) {
      int m = attributeInstances[i]->Debug(currW, currF, print_weights, currF_gt);
      currF += m;  currW += m;  
      if(f_gt) currF_gt += m;
    }
  }
  fprintf(stderr, "\n");  

  assert(currF-f == classes->NumWeights(getPartFeatures, getAttributeFeatures));
}


char *ImageProcess::PrintPartLocations(PartLocation *locs, char *str) {
  strcpy(str, "");
  int x, y, scale, rot, pose;
  for(int i = 0; locs && i < classes->NumParts(); i++) {
    locs[i].GetDetectionLocation(&x, &y, &scale, &rot, &pose);
    sprintf(str+strlen(str), " (%s: loc=(%d,%d), scale=%d, rot=%d)",
            classes->GetPart(i)->GetPose(pose)->Name(), x, y, scale, rot);
  }
  return str;
}


void ImageProcess::RestrictLocationsToAgreeWithAtLeastOneUser(MultipleUserResponses *users, int radius) {
  for(int i = 0; i < classes->NumParts(); i++) {
     for(int j = 0; j < classes->GetPart(i)->NumPoses(); j++) {
       partInstances[i]->GetPose(j)->RestrictLocationsToAgreeWithAtLeastOneUser(users, radius);
     }
  }
}
