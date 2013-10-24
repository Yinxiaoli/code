/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#include "pose.h"
#include "part.h"
#include "attribute.h"
#include "imageProcess.h"
#include "classes.h"
#include "spatialModel.h"
#include "dataset.h"


static int yinxiao_count = 0;

ObjectPose::ObjectPose(const char *n, bool isC) {
  name = n ? StringCopy(n) : NULL;
  id = -1;
  classes = NULL;

  isClick = isC;
  isFlipped = false;

  segmentation = NULL;
  segmentation_name = NULL;

  flippedName = NULL;
  flipped = NULL;
  appearanceModel = NULL;
  visualization_image = NULL;
  isNotVisible = name && (strstr(name, "not_visible") || strstr(name, "not visible"));
}

ObjectPose::~ObjectPose() {
  if(name)
    StringFree(name);

  if(appearanceModel && !IsFlipped())
    delete appearanceModel;

  if(flippedName)
    StringFree(flippedName);

  if(visualization_image) free(visualization_image);

  if(segmentation_name) free(segmentation_name);
  if(segmentation) cvReleaseImage(&segmentation);
}

Json::Value ObjectPose::Save() {
  Json::Value root;

  if(name) root["name"] = name;
  root["id"] = id;
  root["isClick"] = isClick;
  if(flipped) root["flipped"] = flipped->Name();
  if(isFlipped) root["isFlipped"] = isFlipped;
  if(visualization_image) root["visualization"] = visualization_image;
  if(segmentation_name) root["segmentation"] = segmentation_name;
  if(appearanceModel && !IsFlipped()) root["appearanceModel"] = appearanceModel->Save();
  
  return root;
}
bool ObjectPose::Load(const Json::Value &root) {
  name = StringCopy(root.get("name", "").asString().c_str());
  id = root.get("id", -1).asInt();
  isClick = root.get("isClick", false).asBool();
  if(root.isMember("flipped"))
    flippedName = StringCopy(root.get("flipped", "").asString().c_str());
  isFlipped = root.get("isFlipped", false).asBool();
  visualization_image = root.isMember("visualization") ? StringCopy(root["visualization"].asString().c_str()) : NULL;
  segmentation_name = root.isMember("segmentation") ? StringCopy(root["segmentation"].asString().c_str()) : NULL;
  if(root.isMember("appearanceModel")) {
    appearanceModel = new Attribute(NULL);
    appearanceModel->SetClasses(classes);
    if(!appearanceModel->Load(root["appearanceModel"])) return false;
  }
  isNotVisible = strstr(name, "not_visible") || strstr(name, "not visible");
  return true;
}

bool ObjectPose::ResolveLinks(Classes *c) {
  if(appearanceModel) 
    appearanceModel->SetClasses(c);
  classes = c;
  if(flippedName) 
    flipped = isClick ? classes->FindClickPose(flippedName) : classes->FindPose(flippedName);
  if(IsFlipped()) {
    assert(!appearanceModel);
    appearanceModel = GetFlipped()->Appearance();
  }
  return true;
}

int ObjectPose::GetWeights(float *w) { 
  if(IsFlipped()) 
    return 0;
  return appearanceModel ? appearanceModel->GetWeights(w) : 0;
}
void ObjectPose::SetWeights(float *w) { 
  if(appearanceModel && !IsFlipped()) 
    appearanceModel->SetWeights(w);  
}
int ObjectPose::NumWeights() { return (appearanceModel && !IsFlipped() ? appearanceModel->NumWeights() : 0); }

float ObjectPose::MaxFeatureSumSqr() { return appearanceModel ? appearanceModel->MaxFeatureSumSqr() : 0; }


ObjectPose *ObjectPose::FlippedCopy() {
  char str[1000]; 
  sprintf(str, "%s flipped", name);
  ObjectPose *f = new ObjectPose();
  
  f->SetName(str);
  f->id = -1;
  f->flipped = this;
  f->isFlipped = true;
  f->isClick = isClick;
  f->visualization_image = visualization_image ? StringCopy(visualization_image) : 0;
  this->flipped = f;
  f->ResolveLinks(classes);
  return f;
}


ObjectPoseInstance::ObjectPoseInstance(ObjectPose *pose, ObjectPartInstance *part, ImageProcess *p) { 
  this->pose = pose; 
  this->part = part;
  responses = NULL; 
  Attribute *app = pose->Appearance();
  appearance = app && app->NumWeights() ? new AttributeInstance(app, p, pose->IsFlipped()) : NULL;
  process = p;
  childPartScores = NULL;
  childPartBestPoseIndices = NULL;

  maxResponse = maxResponseInds = NULL;
  losses = NULL;
  useLoss = false;
  unaryExtra = NULL;
  minScale = maxScale = 0;

  loss_buff = NULL;
  max_loss = 0;
  dontDeleteChildPartPoseTransitions = NULL;
}

void ObjectPoseInstance::ResolveLinks() {
  // Create pose part transitions
  int **n;
  int numParts = part->Model()->NumParts();
  int po = part->Model()->PoseInd(pose->IsFlipped() ? pose->GetFlipped() : pose);
  ObjectPartPoseTransition ****p = part->Model()->GetPosePartTransitions(&n);

  childPartPoseTransitions = (ObjectPartPoseTransitionInstance***)malloc(sizeof(ObjectPartPoseTransitionInstance**)*(numParts+1));
  dontDeleteChildPartPoseTransitions = (bool**)malloc(sizeof(bool*)*(numParts+1));
  memset(childPartPoseTransitions, 0, sizeof(ObjectPartPoseTransitionInstance**)*(numParts+1));
  memset(dontDeleteChildPartPoseTransitions, 0, sizeof(bool*)*(numParts+1));
  numChildPartPoseTransitions = (int*)malloc(sizeof(int)*(numParts+1));
  memset(numChildPartPoseTransitions, 0, sizeof(int)*(numParts+1));
  for(int i = 0; n && i < (numParts + (pose->IsClick() ? 1 : 0)); i++) {
    int ii = pose->IsFlipped() && part->GetPart(i)->Model()->GetFlipped() ? part->Model()->PartInd(part->GetPart(i)->Model()->GetFlipped()) : i;
    numChildPartPoseTransitions[ii] = n[po][i];
    childPartPoseTransitions[ii] = (ObjectPartPoseTransitionInstance**)malloc(sizeof(ObjectPartPoseTransitionInstance*)*n[po][i]);
    dontDeleteChildPartPoseTransitions[ii] = (bool*)malloc(sizeof(bool)*(n[po][i]+1));
    memset(dontDeleteChildPartPoseTransitions[ii], 0, sizeof(bool)*(n[po][i]+1));
    for(int j = 0; j < n[po][i]; j++)
      childPartPoseTransitions[ii][j] = new ObjectPartPoseTransitionInstance(p[po][i][j], process, pose->IsFlipped());
  }
}

void ObjectPoseInstance::ResolveParentLinks() {
  int **num;
  int numParts = part->Model()->NumParts();
  int po = part->Model()->PoseInd(pose->IsFlipped() ? pose->GetFlipped() : pose);
  ObjectPartPoseTransition ****p = part->Model()->GetPosePartTransitions(&num);
  for(int i = 0; num && i < numParts; i++) {
    int ii = pose->IsFlipped() && part->GetPart(i)->Model()->GetFlipped() ? part->Model()->PartInd(part->GetPart(i)->Model()->GetFlipped()) : i;
    for(int j = 0; j < num[po][i]; j++) {
      ObjectPartPoseTransitionInstance *c = NULL;
      for(int l = 0; l < childPartPoseTransitions[ii][j]->NumChildPoses(); l++) {
        ObjectPoseInstance *childPose = childPartPoseTransitions[ii][j]->GetChildPose(l);
        int n = childPose->part->Model()->NumParts();
        childPose->childPartPoseTransitions[n] = (ObjectPartPoseTransitionInstance**)realloc(childPose->childPartPoseTransitions[n], sizeof(ObjectPartPoseTransitionInstance*)*(childPose->numChildPartPoseTransitions[n]+1));
        childPose->dontDeleteChildPartPoseTransitions[n] = (bool*)realloc(childPose->dontDeleteChildPartPoseTransitions[n], sizeof(bool)*(childPose->numChildPartPoseTransitions[n]+1));
	if(l == 0) c = new ObjectPartPoseTransitionInstance(p[po][i][j], process, pose->IsFlipped(), true);
	childPose->dontDeleteChildPartPoseTransitions[n][childPose->numChildPartPoseTransitions[n]] = l > 0;
        childPose->childPartPoseTransitions[n][childPose->numChildPartPoseTransitions[n]++] = c;
      }
    }
  }
}


void ObjectPoseInstance::ExtractPartLocations(PartLocation *locs, PartLocation *l, ObjectPartInstance *par) {
  int x, y, scale, rot;
  l->GetDetectionLocation(&x, &y, &scale, &rot);
 
  
  for(int i = 0; i <= part->NumParts(); i++) {
    if(!part->GetPart(i) || part->GetPart(i) == par)
      continue;

    IplImage *c;
    int bestPose;

    if((part->GetPart(i)->IsObserved() || !numChildPartPoseTransitions[i]) && !IsNotVisible()) {
      locs[part->GetPart(i)->Id()].Copy(part->GetPart(i)->GetBestLoc());
    } else {
      if(IsNotVisible()) {
	int pose = -1;
	for(int j = 0; j < part->GetPart(i)->Model()->NumPoses(); j++) 
	  if(part->GetPart(i)->GetPose(j)->IsNotVisible()) 
	    pose = j;
	assert(pose != -1);
	locs[part->GetPart(i)->Id()].SetDetectionLocation(LATENT, LATENT, LATENT, LATENT, pose, LATENT, LATENT);
      } else {
	c = childPartBestPoseIndices[i][scale][rot];
	bestPose = ((int*)(c->imageData + y*c->widthStep))[x];
	assert(bestPose >= 0 && bestPose < part->GetPart(i)->Model()->NumPoses());
	locs[part->GetPart(i)->Id()].Copy(childPartPoseTransitions[i][bestPose]->GetChildPartLocation(l));
      }
    }

     part->GetPart(i)->ExtractPartLocations(locs, &locs[part->GetPart(i)->Id()], part);
  }
}

void ObjectPoseInstance::ClearLoss() { 
  process->Features()->ReleaseResponses(&losses); 
} 

void ObjectPoseInstance::Clear(bool clearAppearance, bool clearBuffers, bool clearResponses, bool clearLoss) { 
  //fprintf(stderr, "Clear pose %s %x\n", pose->Name(), process);
  if(clearResponses) {
    if(responses) 
      process->Features()->ReleaseResponses(&responses); 
    responses = NULL;
  }

  if(clearBuffers) {
    if(maxResponse) {
      cvReleaseImage(&maxResponse);     maxResponse = NULL;
      cvReleaseImage(&maxResponseInds); maxResponseInds = NULL;
    }

    if(childPartBestPoseIndices) {
      FeatureOptions *feat = process->Features();
      for(int i = 0; i <= part->NumParts(); i++) 
	for(int j = 0; j < feat->NumScales(); j++) 
	  for(int k = 0; k < feat->NumOrientations(); k++) 
	    if(childPartBestPoseIndices[i][j][k])
	      cvReleaseImage(&childPartBestPoseIndices[i][j][k]);
      free(childPartBestPoseIndices); childPartBestPoseIndices = NULL;
    }
    
    if(childPartScores) {
      FeatureOptions *feat = process->Features();
      for(int i = 0; i <= part->NumParts(); i++) 
	for(int j = 0; j < feat->NumScales(); j++) 
	  for(int k = 0; k < feat->NumOrientations(); k++) 
	    if(childPartScores[i][j][k])
	      cvReleaseImage(&childPartScores[i][j][k]);
      free(childPartScores); childPartScores = NULL;
    }

    for(int i = 0; i <= part->NumParts(); i++)
      for(int j = 0; numChildPartPoseTransitions && j < numChildPartPoseTransitions[i]; j++)
        if(!dontDeleteChildPartPoseTransitions[i][j])
	  childPartPoseTransitions[i][j]->Clear();

    if(losses && clearLoss)
      process->Features()->ReleaseResponses(&losses); 
  }

  if(appearance && clearAppearance)
    appearance->Clear();

  //for(int j = 0; j < pose->NumPoses(); j++) 
  //poses[j]->Clear();
}

ObjectPoseInstance::~ObjectPoseInstance() {
  Clear();

  if(appearance)
    delete appearance;

  if(childPartPoseTransitions) {
    for(int i = 0; i <= part->NumParts(); i++) {
      for(int j = 0; j < numChildPartPoseTransitions[i]; j++) 
	if(!dontDeleteChildPartPoseTransitions[i][j])
	  delete childPartPoseTransitions[i][j];
      if(childPartPoseTransitions[i]) free(childPartPoseTransitions[i]);
      if(dontDeleteChildPartPoseTransitions[i]) free(dontDeleteChildPartPoseTransitions[i]);
    }
    free(childPartPoseTransitions);
    free(dontDeleteChildPartPoseTransitions);
  }
  if(numChildPartPoseTransitions) free(numChildPartPoseTransitions);

  if(loss_buff) cvReleaseImage(&loss_buff);

  FreeUnaryExtra();
}

void ObjectPoseInstance::FreeUnaryExtra() {
  if(unaryExtra) {
    if(responses) {
      for(int scale = 0; scale < process->Features()->NumScales(); scale++)
	for(int rot = 0; rot < process->Features()->NumOrientations(); rot++)
	  if(responses[scale][rot] && unaryExtra[scale][rot])
	    cvSub(responses[scale][rot], unaryExtra[scale][rot], responses[scale][rot]);
    }
    process->Features()->ReleaseResponses(&unaryExtra); 
  }
  unaryExtra = NULL;
}



void ObjectPoseInstance::AllocateCacheTables() {
  FeatureParams *feat = process->Features()->GetParams();
  int memSize = feat->numScales*(sizeof(IplImage**) + sizeof(IplImage*)*feat->numOrientations);
  responses = (IplImage***)malloc(memSize); memset(responses, 0, memSize);
  
  for(int scale = 0; scale < feat->numScales; scale++) {
    responses[scale] = ((IplImage**)(responses+feat->numScales))+scale*feat->numOrientations;
  }

  int memSize2 = (part->NumParts()+1)*(sizeof(IplImage***) + (feat->numScales*((sizeof(IplImage**)+(feat->numOrientations)*sizeof(IplImage*)))));
  childPartScores = (IplImage****)malloc(memSize2); memset(childPartScores, 0, memSize2);
  childPartBestPoseIndices = (IplImage****)malloc(memSize2); memset(childPartBestPoseIndices, 0, memSize2);
  char *sPtr = ((char*)childPartScores)+(part->NumParts()+1)*sizeof(IplImage***);
  char *iPtr = ((char*)childPartBestPoseIndices)+(part->NumParts()+1)*sizeof(IplImage***);
  for(int j = 0; j <= part->NumParts(); j++) {
    childPartScores[j] = (IplImage***)sPtr;  sPtr += sizeof(IplImage**)*feat->numScales;
    childPartBestPoseIndices[j] = (IplImage***)iPtr;  iPtr += sizeof(IplImage**)*feat->numScales;
    for(int scale = 0; scale < feat->numScales; scale++) {
      childPartScores[j][scale] = (IplImage**)sPtr;  sPtr += sizeof(IplImage*)*feat->numOrientations;
      childPartBestPoseIndices[j][scale] = (IplImage**)iPtr;  iPtr += sizeof(IplImage*)*feat->numOrientations;
    }
  }
}

/**
 * Use img to update a per pixel maximum score in currMax.  If a particular pixel is updated, set 
 * the corresponding pixel in currMaxInds to newInd
 */
void MaxWithIndex(IplImage *currMax, IplImage *img, IplImage *currMaxInds, int newInd) {
  assert(currMax->width == img->width && currMaxInds->width == img->width &&
	 currMax->height == img->height && currMaxInds->height == img->height);
  char *ptrCurrMax2 = currMax->imageData, *ptrCurrMaxInds2 = currMaxInds->imageData, *ptrImg2 = img->imageData;
  float *ptrCurrMax, *ptrImg;
  int i, j;
  for(i = 0; i < img->height; i++, ptrCurrMax2+=currMax->widthStep, 
	ptrCurrMaxInds2+=currMaxInds->widthStep, ptrImg2+=img->widthStep) {
    for(j = 0, ptrCurrMax=(float*)ptrCurrMax2, 
	  ptrImg=(float*)ptrImg2; j < img->width; j++) {
      if(ptrImg[j] > ptrCurrMax[j]) { 
        ptrCurrMax[j] = ptrImg[j];
        if(currMaxInds->depth == IPL_DEPTH_16S) ((int16_t*)ptrCurrMaxInds2)[j] = newInd;
        else ((int32_t*)ptrCurrMaxInds2)[j] = newInd;
      }
    }
  }
}


/*
 * Compute the score for this object pose for each scale/rotation/pixel location in the image
 * using dynamic programming.  A pose's response can incorporate an appearance model
 * using a conventional sliding window classifier as well as the scores of all child parts
 */
IplImage ***ObjectPoseInstance::Detect(ObjectPartInstance *parentPart) {
  FeatureOptions *fo = process->Features();
  FeatureParams *feat = fo->GetParams();

  // We can either parallelize running inference over multiple child parts/poses at the same time,
  // or parallelize different scales/orientations for a given child part/pose
  //int nthreads = process->NumThreads();

  if(g_debug > 1) fprintf(stderr, "  Detecting pose instance %s %s...\n", fo->Name(), pose->Name());

  // Initialize the unary potential for this part, which is the appearance score, plus 
  // any additional terms (like the loss term, or click location terms, or attribute answer terms)
  if(!responses) {
    AllocateCacheTables();
    IplImage ***a = appearance ? appearance->GetResponses(pose) : NULL;
    for(int scale = 0; scale < feat->numScales; scale++) {
      for(int rot = 0; rot < feat->numOrientations; rot++) {
	assert(!responses[scale][rot]);
	if(a) responses[scale][rot] = cvCloneImage(a[scale][rot]);
	if(useLoss && max_loss) {
	  if(!losses) ComputeLoss(&ground_truth_loc, max_loss);
	  if(responses[scale][rot]) cvScaleAdd(losses[scale][rot], cvScalar(1.0), responses[scale][rot], responses[scale][rot]);
	  else {
	    responses[scale][rot] = cvCloneImage(losses[scale][rot]);
	  }
	}
	if(unaryExtra) {
	  if(responses[scale][rot]) cvAdd(responses[scale][rot], unaryExtra[scale][rot], responses[scale][rot]);
	  else responses[scale][rot] = cvCloneImage(unaryExtra[scale][rot]);
	}
      }
    }
  }

  // Build the list of child part/pose pairs that we need to optimize over
  int l_parts[10000], l_poses[10000], l_parts3[10000], l_poses3[10000], l_scales[10000], l_rotations[10000], l_parts2[10000], l_scales2[10000], l_rotations2[10000];
  int num = 0, num2 = 0, num3 = 0;
  for(int k = 0; k <= part->NumParts(); k++) {
    ObjectPartInstance *childPart = part->GetPart(k);
    if(childPart == parentPart || !childPart || !numChildPartPoseTransitions[k] || childPartScores[k][0][0])
      continue;
    for(int l = 0; l < numChildPartPoseTransitions[k]; l++) {
      childPartPoseTransitions[k][l]->InitDetect(l_parts, l_poses, l_scales, l_rotations, k, l, &num);
      l_parts3[num3] = k; 
      l_poses3[num3++] = l; 
    }
    for(int rot = 0; rot < feat->numOrientations; rot++) {
      for(int scale = 0; scale < feat->numScales; scale++) {
	l_scales2[num2] = scale; 
	l_rotations2[num2] = rot; 
	l_parts2[num2++] = k; 
      }
    }
  }
  int nthreads = num > 20 ? fo->NumThreads() : 1;
  
  if(num) {
    // Compute the transition scores for every position of this part, which considers
    // every possible position of each child part
#ifdef USE_OPENMP
    #pragma omp parallel for num_threads(nthreads)
#endif
    for(int i = 0; i < num; i++) 
      childPartPoseTransitions[l_parts[i]][l_poses[i]]->Detect(this, l_scales[i], l_rotations[i]);

    /*
    if(process->GetScaleOrientationMethod() == SO_SCALE_ORIENTATION_ARBITRARY_COST)
#ifdef USE_OPENMP
    #pragma omp parallel for num_threads(nthreads)
#endif
    for(int j = 0; j < num3; j++) 
      childPartPoseTransitions[l_parts3[j]][l_poses3[j]]->FinishDetect();
    */
  }

  
  if(num2) {
#ifdef USE_OPENMP
    #pragma omp parallel for num_threads(nthreads)
#endif
    for(int l = 0; l < num2; l++) {
      int scale = l_scales2[l], rot = l_rotations2[l], j = l_parts2[l];
      assert(!childPartScores[j][scale][rot]);
	  
      childPartScores[j][scale][rot] = cvCloneImage(childPartPoseTransitions[j][0]->scores[scale][rot]);
      childPartBestPoseIndices[j][scale][rot] = cvCreateImage(cvSize(childPartScores[j][scale][rot]->width,
								     childPartScores[j][scale][rot]->height), IPL_DEPTH_32S, 1);
      cvZero(childPartBestPoseIndices[j][scale][rot]);
      for(int k = 1; k < numChildPartPoseTransitions[j]; k++) {
	MaxWithIndex(childPartScores[j][scale][rot], childPartPoseTransitions[j][k]->scores[scale][rot], 
		     childPartBestPoseIndices[j][scale][rot], k);
      }
	 
	  
      // The score of this pose is the unary term plus the sum over all child parts
      if(process->GetInferenceMethod() == IM_MAXIMUM_LIKELIHOOD) {
	if(!responses[scale][rot]) {
	  responses[scale][rot] = cvCreateImage(cvSize(childPartScores[j][scale][rot]->width, childPartScores[j][scale][rot]->height), IPL_DEPTH_32F, 1);
	  cvZero(responses[scale][rot]);
	}
	cvAdd(responses[scale][rot], childPartScores[j][scale][rot], responses[scale][rot]);

      } else {
	if(!responses[scale][rot]) {
	  responses[scale][rot] = cvCreateImage(cvSize(childPartScores[j][scale][rot]->width, childPartScores[j][scale][rot]->height), IPL_DEPTH_32F, 1);
	  cvSet(responses[scale][rot], cvScalar(1));
	} 
	cvMul(responses[scale][rot], childPartScores[j][scale][rot], responses[scale][rot]);
      }
    }
  } else if(responses && IsInvalid()) {
    for(int scale = 0; scale < feat->numScales; scale++) {
      for(int rot = 0; rot < feat->numOrientations; rot++) {
	if(responses[scale][rot])
	  cvSet(responses[scale][rot], cvScalar(-1000000));
      }
    }
  }

  return responses;
}

IplImage *ObjectPoseInstance::SaveProbabilityMaps(const char *fname, const char *dir, char *html, bool mergeResults) {
  FeatureOptions *fo = process->Features();
  FeatureParams *feat = fo->GetParams();
  int scale, rot, ind = 0;

  //if(IsNotVisible()) return NULL;

  if(g_debug > 0)
    fprintf(stderr, "    Saving responses %s...\n", pose->Name());
  

  part->ComputeCumulativeSums();
  float maxScore = part->GetMaxScore();
  float gamma = part->Model()->GetGamma(), delta = part->GetDelta(), mi = part->GetLowerBound();
  minScale = 0;
  maxScale = exp(gamma*my_max(maxScore,mi)+delta);

  IplImage *subImg2, *merged = NULL;

  for(scale = 0; scale < feat->numScales; scale++) {
    for(rot = 0; rot < feat->numOrientations; rot++, ind++) {
      IplImage *subImg = GetResponse(scale, rot);
      if(!subImg) {
	return NULL;
      }
      float *dstPtr;
      IplImage *dst = fo->ConvertDetectionImageCoordinates(subImg, scale, rot, 0, 0, -10000000);//cvCloneImage(subImg);
      char *dstPtr2 = dst->imageData;
      int i, j;

      if(!pose->IsClick() || !part->IsObserved()) {
	for(i = 0; i < dst->height; i++, dstPtr2 += dst->widthStep) 
	  for(j = 0, dstPtr=(float*)dstPtr2; j < dst->width; j++) 
	    dstPtr[j] = exp(gamma*my_max(dstPtr[j],mi)+delta);
      } else {
	cvZero(dst);
        PartLocation *l = part->GetPartLocation();
	int x, y, sca, ro;
	l->GetDetectionLocation(&x, &y, &sca, &ro);
	if(scale == sca && rot == ro) 
	  ((float*)(dst->imageData+y*dst->widthStep))[x] = 1;
      }
      
      if(!mergeResults) {
	if((!minScale && !maxScale)) {
	  subImg2 = MinMaxNormImage(dst, &minScale, &maxScale);
	} else {
	  subImg2 = cvCreateImage(cvSize(dst->width,dst->height), IPL_DEPTH_8U, 1);
	  cvCvtScale(dst, subImg2, 255/(maxScale-minScale), -minScale*255/(maxScale-minScale));
	}
	cvReleaseImage(&dst);


	char str[4000];
	if(!fname) sprintf(str, "test/%s_%s_%d_%d.png", fo->Name(), pose->Name(), scale, rot);
	else sprintf(str, "%s/%s_%s_%d_%d.png", dir, fname, pose->Name(), scale, rot);
	
	char scale_str[1000];
	if(feat->numScales>1 || feat->numOrientations>1) sprintf(scale_str, " scale=%f, rot=%f", fo->Scale(scale), fo->Rotation(rot));
	else strcpy(scale_str, "");
	if(html) { 
	  sprintf(html+strlen(html), "<td><center><img src=\"%s\" width=300><h3>%s%s%s</h3></center></td>", str + (dir ? strlen(dir)+1 : 0), pose->IsClick() ? "click " : "", pose->Name(), scale_str);
	}
	cvReleaseImage(&subImg2);
      } else {
	if(!merged) merged = dst;
	else {
	  cvAdd(merged, dst, merged);
	  cvReleaseImage(&dst);
	}
      }
    }
  }


  if(merged) {

    if((!minScale && !maxScale)) {
      subImg2 = MinMaxNormImage(merged, &minScale, &maxScale);
    } else {
      subImg2 = cvCreateImage(cvSize(merged->width, merged->height), IPL_DEPTH_8U, 1);
      cvCvtScale(merged, subImg2, 255/(maxScale-minScale), -minScale*255/(maxScale-minScale));
    }

    cvReleaseImage(&subImg2);
  }

  return merged;
}


IplImage *ObjectPoseInstance::GetResponse(int scale, int rot, ObjectPartInstance *parentPart, IplImage **tmp) { 
  assert(responses || unaryExtra);
  IplImage ***res = responses;
  int parentPartInd = parentPart ? part->PartInd(parentPart) : -1;

  
  if(parentPartInd >= 0 && childPartScores && childPartScores[parentPartInd][scale][rot]) {
    // If responses is computed by summing out the location of every child part, we can compute a new response that undos summing 
    // out the child part 'subtractPoseInd' (by removing its score component from responses).  This is used to propagate information 
    // from the parent part back into the child part
    assert(res && res[scale][rot]);
    *tmp = cvCloneImage(res[scale][rot]);
    if(process->GetInferenceMethod() == IM_MAXIMUM_LIKELIHOOD) 
      cvSub(*tmp, childPartScores[parentPartInd][scale][rot], *tmp);
    else
      cvDiv(*tmp, childPartScores[parentPartInd][scale][rot], *tmp);

    /*
      CvPoint min_loc, max_loc;
      double min_val, max_val;
      cvMinMaxLoc(res[scale][rot], &min_val, &max_val, &min_loc, &max_loc);
      cvMinMaxLoc(*tmp, &min_val, &max_val, &min_loc, &max_loc);
      cvMinMaxLoc(childPartScores[parentPartInd][scale][rot], &min_val, &max_val, &min_loc, &max_loc);
    */

    return *tmp;
  }

  if((!res || !res[scale][rot]) && unaryExtra)
    return unaryExtra[scale][rot];

  return res[scale][rot]; 
}

PartLocation ObjectPoseInstance::GetBestLoc() {
  FeatureOptions *fo = process->Features();
  FeatureParams *feat = fo->GetParams();
  PartLocation loc(part, process->Image()->width, process->Image()->height);
  double best = -INFINITY, mi, ma;
  CvPoint pt;
  for(int scale = 0; scale < feat->numScales; scale++) {
    for(int rot = 0; rot < feat->numOrientations; rot++) {
      cvMinMaxLoc(responses[scale][rot], &mi, &ma, &pt);
      if(ma > best) {
	best = ma;
	loc.SetDetectionLocation(pt.x, pt.y, scale, rot, part->PoseInd(this), LATENT, LATENT);
      }
    }
  }
  return loc;
}


// The maximum response among any scale/orientation
IplImage *ObjectPoseInstance::GetMaxResponse(ObjectPartInstance *parentPart, IplImage **tmp) { 
  int parentPartInd = parentPart ? part->PartInd(parentPart) : -1;
  FeatureOptions *fo = process->Features();
  FeatureParams *feat = fo->GetParams();
  int scale, rot;

  assert(responses);
  if(!maxResponse) 
    ::GetMaxResponse(responses, feat->numScales, feat->numOrientations, &best_x, &best_y, &best_scale, &best_rot, 
		     &maxResponse, &maxResponseInds);

  for(scale = 0; scale < feat->numScales; scale++) {
    for(rot = 0; rot < feat->numOrientations; rot++) {
      if(parentPartInd >= 0 && childPartScores && childPartScores[parentPartInd][scale][rot]) {
	// If responses is computed by summing out the location of every child part, we can compute a new response that undos summing 
	// out the child part 'subtractPoseInd' (by removing its score component from responses).  This is used to propagate information 
	// from the parent part back into the child part
	assert(maxResponse);
	*tmp = cvCloneImage(maxResponse);
	if(process->GetInferenceMethod() == IM_MAXIMUM_LIKELIHOOD) 
	  cvSub(*tmp, childPartScores[parentPartInd][scale][rot], *tmp);
	else
	  cvDiv(*tmp, childPartScores[parentPartInd][scale][rot], *tmp);
	return *tmp;
      }
    }
  }
  return maxResponse;
}


// If the unary score of the parentPart is changed, we can remove the component of our response score
// that is attributed to partPart, such that that component can be re-computed in the future
void ObjectPoseInstance::InvalidateChildScores(ObjectPartInstance *parentPart) { 
  if(childPartScores) {
    int scale, rot;
    FeatureOptions *fo = process->Features();
    FeatureParams *feat = fo->GetParams();
    int parentPartInd = parentPart ? part->PartInd(parentPart) : -1;
    assert(parentPartInd >= 0);
    for(scale = 0; scale < feat->numScales; scale++) {
      for(rot = 0; rot < feat->numOrientations; rot++) {
	if(childPartScores[parentPartInd][scale][rot]) {
	  cvSub(responses[scale][rot], childPartScores[parentPartInd][scale][rot], responses[scale][rot]);
	  cvReleaseImage(&childPartScores[parentPartInd][scale][rot]);
	  childPartScores[parentPartInd][scale][rot] = NULL;
	  cvReleaseImage(&childPartBestPoseIndices[parentPartInd][scale][rot]);
	  childPartBestPoseIndices[parentPartInd][scale][rot] = NULL;
	}
      }
    }
    if(parentPartInd >= 0)
      for(int j = 0; numChildPartPoseTransitions && j < numChildPartPoseTransitions[parentPartInd]; j++)
	childPartPoseTransitions[parentPartInd][j]->Clear();
  }
}

void ObjectPoseInstance::AddToUnaryExtra(IplImage ***add, bool useMax, float scalar) { 
  FeatureOptions *fo = process->Features();
  FeatureParams *feat = fo->GetParams();
  if(!unaryExtra) {
    unaryExtra = (IplImage***)malloc(sizeof(IplImage**)*feat->numScales + sizeof(IplImage*)*feat->numScales*feat->numOrientations);
    for(int scale = 0; scale < feat->numScales; scale++) {
      unaryExtra[scale] = ((IplImage**)(unaryExtra+feat->numScales))+scale*feat->numOrientations;
      for(int rot = 0; rot < feat->numOrientations; rot++) {
        unaryExtra[scale][rot] = cvCloneImage(add[scale][rot]); 
        if(scalar) cvAddS(unaryExtra[scale][rot], cvScalar(scalar), unaryExtra[scale][rot]);
        if(responses && responses[scale][rot]) {
          cvAdd(responses[scale][rot], unaryExtra[scale][rot], responses[scale][rot]);
  
	  //double mi, ma;
	  //cvMinMaxLoc(responses[scale][rot], &mi, &ma);
	  //fprintf(stderr, "add to unary extra %s %f %f\n", pose->Name(), (float)mi, (float)ma);
        }
      }
    }
  } else {
    for(int scale = 0; scale < feat->numScales; scale++) {
      for(int rot = 0; rot < feat->numOrientations; rot++) {
        if(useMax) {
          if(responses) 
            cvSub(responses[scale][rot], unaryExtra[scale][rot], responses[scale][rot]);  
          IplImage *tmp = scalar ? cvCloneImage(add[scale][rot]) : add[scale][rot];
          if(scalar) cvAddS(tmp, cvScalar(scalar), tmp);
          cvMax(unaryExtra[scale][rot], tmp, unaryExtra[scale][rot]);
          if(scalar) cvReleaseImage(&tmp);
          if(responses) 
            cvAdd(responses[scale][rot], unaryExtra[scale][rot], responses[scale][rot]);  
        } else {
          cvAdd(unaryExtra[scale][rot], add[scale][rot], unaryExtra[scale][rot]);   
          if(scalar) cvAddS(unaryExtra[scale][rot], cvScalar(scalar), unaryExtra[scale][rot]);
          if(responses && responses[scale][rot]) {
            cvAdd(responses[scale][rot], add[scale][rot], responses[scale][rot]);  
            if(scalar) cvAddS(responses[scale][rot], cvScalar(scalar), responses[scale][rot]);
          }
        }
      }
    }
  }
}

void ObjectPoseInstance::Draw(IplImage *img, PartLocation *l, CvScalar color, CvScalar color2, CvScalar color3, const char *str, const char *key, bool labelPoint, bool labelRect, float zoom) {
  float x, y, scale, rot;
  float width, height;
  l->GetImageLocation(&x, &y, &scale, &rot, NULL, &width, &height);

  if(labelRect)
    cvRotatedRect(img, cvPoint((int)(x*zoom),(int)(y*zoom)), (int)(width*zoom), (int)(height*zoom), -rot, color);
  if(labelPoint) {
    cvCircle(img, cvPoint((int)(x*zoom),(int)(y*zoom)), CIRCLE_RADIUS, color2, -1);
    cvCircle(img, cvPoint((int)(x*zoom),(int)(y*zoom)), CIRCLE_RADIUS2, color, -1);
  }
  if(str) {
    CvFont font;
    CvSize sz;
    int baseline;
    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, .5, .5, 0, 1, CV_AA);
    cvGetTextSize(str, &font, &sz, &baseline);
    cvPutText(img, str, cvPoint((int)((x*zoom-sz.width/2)),(int)((y*zoom+sz.height+CIRCLE_RADIUS))), &font, CV_RGB(0,0,0));
  }
  if(key) {
    CvFont font;
    CvSize sz;
    int baseline;
    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, .3, .3, 0, 1, CV_AA);
    cvGetTextSize(key, &font, &sz, &baseline);
    cvPutText(img, key, cvPoint((int)((x*zoom-sz.width/2)),(int)((y*zoom+sz.height/2))), &font, color3);
  }
}

void ObjectPoseInstance::GetScaleRot(int x, int y, int *scale, int *rot) {
  int *ptr2 = ((int*)(maxResponseInds->imageData+y*maxResponseInds->widthStep))+x*2;
  *scale = ptr2[0]; *rot = ptr2[1]; 
}


void ObjectPoseInstance::SetLoss(PartLocation *gt_loc, float max_loss) {
  if(gt_loc) this->ground_truth_loc.Copy(*gt_loc);
  this->max_loss = max_loss;
}


void ObjectPoseInstance::ComputeLoss(PartLocation *gt_loc, float max_loss, bool useMultiObject) {
  bool penalizePose = false;  // No current methods penalize pose
  bool penalizeScale = true;
  bool penalizeRotation = true;

  int gt_orig_x, gt_orig_y, gt_orig_scale=0, gt_orig_rot=0, gt_pose, gt_scale=0, gt_rot=0;
  float gt_x, gt_y;
  gt_loc->GetDetectionLocation(&gt_orig_x, &gt_orig_y, &gt_orig_scale, &gt_orig_rot, &gt_pose);
  process->GetClasses()->ConvertDetectionCoordinates(gt_orig_x, gt_orig_y, gt_orig_scale, gt_orig_rot, gt_scale, gt_rot, 
						     process->Image()->width, process->Image()->height, &gt_x, &gt_y);


  bool gtNotVisible = ((gt_pose >= 0 && gt_pose < part->Model()->NumPoses()) ? part->GetPose(gt_pose)->IsNotVisible() : true);
  bool badVisible = IsNotVisible() != gtNotVisible;
  float poseScore = penalizePose ? ((gt_pose >= 0 && gt_pose < part->Model()->NumPoses()) ? (gt_pose == part->PoseInd(this) ? 1 : .5f) : 0) : 1;
  if(badVisible) poseScore = 0;
  int ww, hh; process->Features()->GetDetectionImageSize(&ww, &hh, gt_scale, gt_rot);
  int y, x, scale, rot;
  FeatureOptions *feat = process->Features();
  bool bothNotVisible = IsNotVisible() && !badVisible;
  PartDetectionLossType lossMethod = process->GetClasses()->GetDetectionLossMethod();
  double gammaClick = process->GetClasses()->GetClickGamma();

  IplImage *inter = loss_buff;
  if(!inter) {
    inter = cvCreateImage(cvSize(ww, hh), IPL_DEPTH_32F, 1);
    unsigned char *interPtr2 = (unsigned char*)inter->imageData;
    float *interPtr;
    int gx = (int)(gt_x+.5f), gy = (int)(gt_y+.5f);
    loss_buff = inter;

    if(lossMethod == LOSS_BOUNDING_BOX_AREA_UNION_OVER_INTERSECTION && part->GetParent() != NULL) { 
      // If the loss is to be measured over just the object bounding box and this part is just a subpart, then no loss is used
      cvSet(inter, cvScalar(1)); 
      penalizeScale = false;
      penalizeRotation = false;
    } else {
      if(badVisible) 
	cvZero(inter);
      else {
	if(IsNotVisible()) {
	  cvSet(inter, cvScalar(1));
	} else {
	  if(lossMethod == LOSS_PART_AVERAGE_AREA_UNION_OVER_INTERSECTION || lossMethod == LOSS_BOUNDING_BOX_AREA_UNION_OVER_INTERSECTION ||
	     lossMethod == LOSS_DETECTION) {
	    // VOC-style loss: Assume there is a bounding box of the size of the detection window centered at
	    // the predicted location, and another one at the ground truth location.  For every candidate
	    // predicted pixel location, densely compute the area of intersection divided by the area of union
		//yinxiao_count++;
		//printf("the current number is %4d \n", yinxiao_count);
	    Attribute *attr = pose->Appearance();
	    FeatureWindow *templ = attr->Feature(0);
	    float w = (float)templ->w, h = (float)templ->h;
	    float A2 = w*h*2, a, int_x, int_y;
	    for(y = 0; y < inter->height; y++, interPtr2+=inter->widthStep) {
	      int_y = (h-my_min(my_abs(y-gy),h));
	      for(x = 0, interPtr=(float*)interPtr2; x < inter->width; x++) {
		int_x = (w-my_min(my_abs(x-gx),w));
		a = int_x*int_y;
		interPtr[x] = a / (A2-a);
	      }
	    }
	  } else {
	    // Loss measured from user statistics.  Assume the standard deviation of user click responses as compared to
	    // the ground truth locations was computed using LearnUserClickProbabilities().  For every candidate
	    // predicted pixel location, densely compute the number of user click standard deviations
	    penalizeScale = false;
	    penalizeRotation = false;
	    int partID = part->Id();
	    int *num, dx;
	    ObjectPartPoseTransitionInstance ***t = process->GetClickPartInst(partID)->GetVisiblePose()->GetPosePartTransitions(&num);
            float wx, wy, wxx, wyy;
	    t[0][0]->Model()->GetWeights(&wx, &wxx, &wy, &wyy);
	    FeatureOptions *fo = process->Features();
	    FeatureParams *feat = fo->GetParams();
	    int g = feat->spatialGranularity;
	    float offn = t[0][0]->Model()->offset_norm*g;
	    wxx *= SQR(offn)/2/gammaClick; wyy *= SQR(offn)/2/gammaClick; 
	    for(int dy = -gy; dy < inter->height-gy; dy++, interPtr2+=inter->widthStep) 
	      for(dx = -gx, x=0, interPtr=(float*)interPtr2; x < inter->width; x++, dx++) 
		interPtr[x] = sqrt((SQR(dx)*wxx + SQR(dy)*wyy));
	    
	    if(lossMethod == LOSS_USER_STANDARD_DEVIATIONS) {
	      // Convert standard deviations to a similarity between 0 and 1 (it will be converted back to a loss later on)
	      cvThreshold(inter, inter, MAX_DEVIATIONS, MAX_DEVIATIONS, CV_THRESH_TRUNC);
	      cvConvertScale(inter, inter, -1.0f/MAX_DEVIATIONS, 1.0);
	    } else if(lossMethod == LOSS_NUM_INCORRECT) {
	      // If the number of deviations is outside of NUM_DEVIATIONS_BEFORE_INCORRECT, set the intersection to 0, otherwise set it to 1
	      cvThreshold(inter, inter, NUM_DEVIATIONS_BEFORE_INCORRECT, 1, CV_THRESH_BINARY_INV);
	    }
	  }
	}
      }
    } 
  }

  IplImage ***losses_new = (IplImage***)malloc(feat->NumScales()*(sizeof(IplImage***)+sizeof(IplImage**)*feat->NumOrientations()));
  memset(losses_new, 0, feat->NumScales()*(sizeof(IplImage***)+sizeof(IplImage**)*feat->NumOrientations()));
  for(scale = 0; scale < feat->NumScales(); scale++) 
    losses_new[scale] = ((IplImage**)(losses_new+feat->NumScales()))+feat->NumOrientations()*scale;

  for(scale = 0; scale < feat->NumScales(); scale++) {
    //losses_new[scale] = (IplImage**)malloc(sizeof(IplImage**)*feat->NumOrientations());
    float s = feat->Scale(scale);
    float scaleScore = penalizeScale ? my_min(s/feat->Scale(gt_scale), feat->Scale(gt_scale)/s) : 1;
    for(rot = 0; rot < feat->NumOrientations(); rot++) {
      float rotScore = penalizeRotation ? my_max(0, cos(feat->Rotation(rot)-feat->Rotation(gt_rot))) : 1;
      // Set loss to 1 - percent_intersection*poseScore*scaleScore*rotScore;
      losses_new[scale][rot] = process->Features()->ConvertDetectionImageCoordinates(inter, gt_scale, gt_rot, scale, rot, max_loss);
      if(bothNotVisible)
	cvZero(losses_new[scale][rot]);
      else
	cvConvertScale(losses_new[scale][rot], losses_new[scale][rot], -poseScore*scaleScore*rotScore*max_loss, max_loss);
    }
  }
  if(!bothNotVisible && !gtNotVisible && gt_orig_scale >= 0 && gt_orig_scale < feat->NumScales() && gt_orig_rot >= 0 && gt_orig_rot < feat->NumOrientations())
    ((float*)(losses_new[gt_orig_scale][gt_orig_rot]->imageData+gt_orig_y*losses_new[gt_orig_scale][gt_orig_rot]->widthStep))[gt_orig_x] = max_loss-poseScore*max_loss;
  
  if(losses) {
    if(useMultiObject) {
      for(scale = 0; scale < feat->NumScales(); scale++) 
	for(rot = 0; rot < feat->NumOrientations(); rot++) 
	  cvMin(losses_new[scale][rot], losses[scale][rot], losses_new[scale][rot]);
    }
    process->Features()->ReleaseResponses(&losses); 
  } 
  losses = losses_new;
}

float ObjectPoseInstance::GetLoss(PartLocation *pred_loc) {
  int gt_x, gt_y, gt_scale, gt_rot, gt_pose;
  bool gt_visible = false;
  if(ground_truth_loc.IsValid()) {
    ground_truth_loc.GetDetectionLocation(&gt_x, &gt_y, &gt_scale, &gt_rot, &gt_pose);
    gt_visible = !part->GetPose(gt_pose)->IsNotVisible();
  }
  int pred_x, pred_y, pred_scale, pred_rot, pred_pose;
  pred_loc->GetDetectionLocation(&pred_x, &pred_y, &pred_scale, &pred_rot, &pred_pose);

  
  if(IsNotVisible()) 
    return !gt_visible ? 0 : max_loss;
  else if(!gt_visible)
    return max_loss;
  

  if(!losses) ComputeLoss(&ground_truth_loc, max_loss);
  IplImage *l = losses[pred_scale][pred_rot];
  int y = my_max(0,my_min(l->height-1,pred_y));
  int x = my_max(0,my_min(l->width-1,pred_x));

  return ((float*)(l->imageData+y*l->widthStep))[x];
}

int ObjectPoseInstance::GetFeatures(float *f, PartLocation *locs, PartLocation *loc) { 
  if(!loc) loc = locs + part->Id(); 
  
  int x, y, scale, rot, pose;
  loc->GetDetectionLocation(&x, &y, &scale, &rot, &pose);

  assert(part->GetPose(pose) == this);
  return appearance ? appearance->GetFeatures(f, loc) : 0; 
}

float ObjectPoseInstance::ScoreAt(PartLocation *l) {
  int x, y, scale, rot;
  l->GetDetectionLocation(&x, &y, &scale, &rot);
  return ((float*)(responses[scale][rot]->imageData+responses[scale][rot]->widthStep*y))[x];
}

bool ObjectPoseInstance::IsInvalid() { 
  bool valid = process->GetClasses()->NumParts() <= 1;
  for(int i = 0; i <= part->NumParts(); i++)
    if(numChildPartPoseTransitions[i])
      valid = true;
  return !valid;
}


extern float g_score2, *g_w2;
float g_spatial_scores[1000];
void ObjectPoseInstance::UpdateTransitionFeatures(float *f, PartLocation *locs, PartLocation *loc, int *offsets) {
  if(!loc) loc = locs + part->Id(); 
  int numP = (part->Model()->NumParts()+(pose->IsClick()?1:0));
  float f_tmp[1000];
  for(int j = 0; j < numP; j++) {
    for(int k = 0; k < numChildPartPoseTransitions[j]; k++) {
      int n = childPartPoseTransitions[j][k]->GetFeatures(f_tmp, loc, locs+part->GetPart(j)->Model()->Id());
      float *fptr = f+offsets[childPartPoseTransitions[j][k]->Model()->Id()];
      //int num = 0;
      for(int i = 0; i < n; i++) { fptr[i] += f_tmp[i]; /*if(g_w2) g_score2 += g_w2[offsets[childPartPoseTransitions[j][k]->Model()->Id()]+i]*f_tmp[i]; num += f_tmp[i] != 0; */}
      //if(num) fprintf(stderr, " [*spatial %d %d:%d]", childPartPoseTransitions[j][k]->Model()->Id(), offsets[childPartPoseTransitions[j][k]->Model()->Id()], offsets[childPartPoseTransitions[j][k]->Model()->Id()]+n);
      //g_spatial_scores[childPartPoseTransitions[j][k]->Model()->Id()] = g_score2;
    }
  }
}

void ObjectPoseInstance::SetPartLocationsAtIdealOffset(PartLocation *locs) {
  float x, y, scale, rot;
  float offset_norm, offset_x, offset_y, offset_scale, offset_rotation, wt;
  locs[part->Model()->Id()].GetImageLocation(&x, &y, &scale, &rot);
  for(int j = 0; j < part->Model()->NumParts(); j++) {
    float best = -1000000;
    int pid=-1, pid2 = -1;
    for(int k = 0; k < numChildPartPoseTransitions[j]; k++) {
      for(int l = 0; l < childPartPoseTransitions[j][k]->NumChildPoses(); l++) {
        childPartPoseTransitions[j][k]->GetWeights(NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &wt, l);
        if(wt > best) { best = wt; pid = k; pid2 = l; }
      }
    }
    if(pid >= 0) {
      childPartPoseTransitions[j][pid]->GetOffsets(&offset_norm, &offset_x, &offset_y, &offset_scale, &offset_rotation);
      locs[part->Model()->GetPart(j)->Id()].SetImageLocation(x+offset_x*scale, y+offset_y*scale, scale+offset_scale, rot+offset_rotation, childPartPoseTransitions[j][pid]->GetChildPose(pid2)->Model()->Name());
      part->GetPart(j)->GetPose(pid)->SetPartLocationsAtIdealOffset(locs);
    }
  }
}

void ObjectPoseInstance::SetCustomWeights(float *w, int *poseOffsets, int *spatialOffsets) {
  if(appearance) appearance->SetCustomWeights(w+poseOffsets[pose->Id()]);
  for(int j = 0; j <= part->Model()->NumParts(); j++) 
    for(int k = 0; k < numChildPartPoseTransitions[j]; k++) 
      if(childPartPoseTransitions[j][k])
  childPartPoseTransitions[j][k]->SetCustomWeights(w+spatialOffsets[childPartPoseTransitions[j][k]->Model()->Id()]);
}

float ObjectPoseInstance::VisualizeSpatialModel(IplImage *img, PartLocation *locs) {
  double mi; 
  double min_val = 10000000;

  float par_x, par_y, par_scale, par_rot;
  locs[part->Id()].GetImageLocation(&par_x, &par_y, &par_scale, &par_rot);

  for(int j = 0; j < part->Model()->NumParts(); j++) {
    int id = part->GetPart(j)->Id();
    for(int i = 0; i < numChildPartPoseTransitions[j]; i++) {
      ObjectPartPoseTransitionInstance *t = childPartPoseTransitions[j][i];
      for(int l = 0; l < t->NumChildPoses(); l++) {
        int pos;
        locs[id].GetDetectionLocation(NULL,NULL,NULL,NULL, &pos);
        if(pos == part->GetPart(j)->PoseInd(t->GetChildPose(l))) {
          if(!t->GetChildPose(l)->IsNotVisible() && !t->GetParentPose()->IsNotVisible()) {
            float x, y, child_x, child_y;
            CvPoint2D32f offset = t->Offset();
            locs[id].GetImageLocation(&child_x, &child_y);
            x = par_x + par_scale*(cos(par_rot)*offset.x - sin(par_rot)*offset.y);
            y = par_y + par_scale*(sin(par_rot)*offset.x + cos(par_rot)*offset.y);
            cvLine(img, cvPoint(par_x, par_y), cvPoint(x, y), CV_RGB(255,0,0), 3);
            cvLine(img, cvPoint(x, y), cvPoint(child_x, child_y), CV_RGB(0,255,0), 1);
            cvCircle(img, cvPoint(child_x,child_y), 4, CV_RGB(0,255,0));
          }

          mi = t->GetChildPose(l)->VisualizeSpatialModel(img, locs);
          if(mi < min_val) min_val = mi;
        }
      }
    }
  }
  return (float)min_val;
}

void ObjectPoseInstance::SetClickPoint(PartLocation *l, bool useMultiObject) {
  assert(pose->IsClick());

  int i;
  bool isValid[10000];
  for(i = 0; i < part->GetParent()->Model()->NumPoses(); i++) 
    isValid[i] = false;

  int l_pose;
  l->GetDetectionLocation(NULL, NULL, NULL, NULL, &l_pose);

  for(i = 0; i < numChildPartPoseTransitions[0]; i++) {
    ObjectPartPoseTransitionInstance *t = childPartPoseTransitions[0][i];
    t->BuildSpatialScores(l);
	float wt;
    for(int c = 0; c < t->NumChildPoses(); c++) {
      t->GetWeights(NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &wt, c);
      isValid[part->GetParent()->PoseInd(t->GetChildPose(c))] = true;
  
      // Add the probability into the unary potential of the part 
      t->GetChildPose(c)->AddToUnaryExtra(t->GetSpatialScores(), useMultiObject, -wt*POSE_TRANSITION_FEATURE);
    }
  }

  // If the pose is unspecified, allow any pose except the not-visible pose
  if(l_pose < 0) {
    for(i = 0; i < part->GetParent()->Model()->NumPoses(); i++) 
      isValid[i] = !part->GetParent()->GetPose(i)->IsNotVisible();
  }

  // Eliminate poses that are impossible (e.g. have 0 probability given click location l)
  // by adding a large negative unary score to each such pose
  for(i = 0; i < part->GetParent()->Model()->NumPoses(); i++) {
    if(!isValid[i]) {
      FeatureOptions *feat = process->Features();
      ObjectPoseInstance *p = part->GetParent()->GetPose(i);
      int j, k, w, h;
      int sz = feat->NumScales()*(sizeof(IplImage**) + feat->NumOrientations()*sizeof(IplImage*));
      IplImage ***invalid = (IplImage***)malloc(sz); memset(invalid, 0, sz);
      IplImage **curr = (IplImage**)(invalid+feat->NumScales());
      for(j = 0; j < feat->NumScales(); j++) {
        invalid[j] = curr;  curr += feat->NumOrientations();
        for(k = 0; k < feat->NumOrientations(); k++) {
          feat->GetDetectionImageSize(&w, &h, j, k);
          invalid[j][k] = cvCreateImage(cvSize(w,h), IPL_DEPTH_32F, 1);
          cvSet(invalid[j][k], cvScalar(-1000000));
        }
      }
      p->AddToUnaryExtra(invalid, useMultiObject);
      feat->ReleaseResponses(&invalid);
    }
  }
}


void ObjectPoseInstance::SetLocation(PartLocation *l, bool useMultiObject) {
  FreeUnaryExtra();

  // Set all pixel locations to -\infty 
  int scale, rot, pose;
  float i_x, i_y;
  l->GetDetectionLocation(NULL, NULL, &scale, &rot, &pose);
  l->GetImageLocation(&i_x, &i_y);

  FeatureOptions *feat = process->Features();
  int j, k, w, h;
  int sz = feat->NumScales()*(sizeof(IplImage**) + feat->NumOrientations()*sizeof(IplImage*));
  IplImage ***invalid = (IplImage***)malloc(sz); memset(invalid, 0, sz);
  IplImage **curr = (IplImage**)(invalid+feat->NumScales());
  int x, y;
  int ind = part->PoseInd(this);
  bool isNotVisible = !IS_LATENT(pose) && this->pose->IsNotVisible() && part->GetPose(pose)->Model()->IsNotVisible();
  for(j = 0; j < feat->NumScales(); j++) {
    invalid[j] = curr;  curr += feat->NumOrientations();
    for(k = 0; k < feat->NumOrientations(); k++) {
      feat->ImageLocationToDetectionLocation(i_x, i_y, j, k, &x, &y); 
      feat->GetDetectionImageSize(&w, &h, j, k);
      invalid[j][k] = cvCreateImage(cvSize(w,h), IPL_DEPTH_32F, 1);
      cvSet(invalid[j][k], cvScalar(isNotVisible ? 0 : -1000));
      if((IS_LATENT(scale) || scale == j) && (IS_LATENT(rot) || rot == k) && (IS_LATENT(pose) || pose == ind) && !this->pose->IsNotVisible())
	((float*)(invalid[j][k]->imageData + invalid[j][k]->widthStep*y))[x] = 0;
    }
  }
  AddToUnaryExtra(invalid, useMultiObject);
  feat->ReleaseResponses(&invalid);
}

void ObjectPoseInstance::RestrictLocationsToAgreeWithAtLeastOneUser(MultipleUserResponses *users, int radius) {
  FreeUnaryExtra();

  int partId = part->Model()->Id();
  FeatureOptions *feat = process->Features();
  int j, k, w, h;
  int sz = feat->NumScales()*(sizeof(IplImage**) + feat->NumOrientations()*sizeof(IplImage*));
  IplImage ***invalid = (IplImage***)malloc(sz); memset(invalid, 0, sz);
  IplImage **curr = (IplImage**)(invalid+feat->NumScales());
  int x, y;
  for(j = 0; j < feat->NumScales(); j++) {
    invalid[j] = curr;  curr += feat->NumOrientations();
    for(k = 0; k < feat->NumOrientations(); k++) {
      feat->GetDetectionImageSize(&w, &h, j, k);
      invalid[j][k] = cvCreateImage(cvSize(w,h), IPL_DEPTH_32F, 1);
      cvSet(invalid[j][k], cvScalar(-1000000));
    }
  }
  for(int i = 0; i < users->NumUsers(); i++) {
    PartLocation *locs = users->GetPartClickLocations(i);
    int scale, rot, pos;
    float i_x, i_y;
    locs[partId].GetDetectionLocation(NULL, NULL, &scale, &rot, &pos);
    locs[partId].GetImageLocation(&i_x, &i_y);
    if((pos >= 0 && process->GetClasses()->GetClickPart(partId)->GetPose(pos)->IsNotVisible() == pose->IsNotVisible()) || 
       (pos < 0 && !pose->IsNotVisible())) {
      for(j = 0; j < feat->NumScales(); j++) {
        for(k = 0; k < feat->NumOrientations(); k++) {
          if((scale >= 0 && scale != j) || (rot >= 0 && rot != k)) 
            continue;
          if(pose->IsNotVisible())
            cvZero(invalid[j][k]);
          else {
            feat->ImageLocationToDetectionLocation(i_x, i_y, j, k, &x, &y); 
            feat->GetDetectionImageSize(&w, &h, j, k);
			cvCircle(invalid[j][k], cvPoint(x,y), ceil(radius/feat->Scale(j)/feat->SpatialGranularity()), CV_RGB(0,0,0), -1);
            ((float*)(invalid[j][k]->imageData + invalid[j][k]->widthStep*y))[x] = 0;
          }
        }
      }
    }
  }

  AddToUnaryExtra(invalid, false);
  feat->ReleaseResponses(&invalid);
}


extern double g_score;
extern float g_loss;
int ObjectPoseInstance::DebugAppearance(float *w, float *f, PartLocation *loc, bool debug_scores, bool print_weights, float *f_gt) { 
  int n = appearance ? appearance->Debug(w, f, print_weights) : 0;
  float loss;

  if(debug_scores) {
    float score_wf = 0, score_gt = 0;
    if(appearance && appearance->GetCustomWeights()) 
      w = appearance->GetCustomWeights();

    int i;
    for(i = 0; i < n; i++) 
      score_wf += w[i]*f[i];
    if(f_gt)
      for(i = 0; i < n; i++) 
        score_gt += w[i]*f_gt[i];


    int x, y, scale, rot;
    loc->GetDetectionLocation(&x, &y, &scale, &rot);
    IplImage ***resps = appearance ? appearance->GetResponses(pose) : NULL;
    IplImage *resp = resps ? resps[scale][rot] : NULL;
    float score_resp = resp ? ((float*)(resp->imageData+resp->widthStep*y))[x] : 0;    
    if(losses) 
      loss = GetLoss(loc);

    g_score += score_wf + loss;
    g_loss += loss;
    fprintf(stderr, " (%s w*f=%f r=%f r_gt=%f loss=%f)", pose->Name(), score_wf, 
	    score_resp, score_gt, loss);
    //assert(my_abs(score_wf-score_resp) < .001);
  }

  return n;
}

void ObjectPoseInstance::DebugSpatial(float *w, float *f, int *offsets, PartLocation *locs, bool debug_scores, bool print_weights, float *f_gt) { 
  if(locs) {
    int numP = (part->Model()->NumParts()+(pose->IsClick()?1:0));
    for(int j = 0; j < numP; j++) {
      for(int k = 0; k < numChildPartPoseTransitions[j]; k++) {
        int n = offsets[childPartPoseTransitions[j][k]->Model()->Id()];
        childPartPoseTransitions[j][k]->Debug(this, w+n, f+n, 
					      locs+childPartPoseTransitions[j][k]->GetParentPart()->Model()->Id(), 
					      locs+childPartPoseTransitions[j][k]->GetChildPart()->Model()->Id(),
					      debug_scores, print_weights, f_gt ? f_gt+n : NULL);
      }
    }
  }
}

void ObjectPoseInstance::SanityCheckDynamicProgramming(PartLocation *gt_locs) {
  if(pose->IsNotVisible()) return;

  int par_pose, child_pose;
  gt_locs[part->Model()->Id()].GetDetectionLocation(NULL, NULL, NULL, NULL, &par_pose);
  if(par_pose == part->Model()->PoseInd(pose)) {
    int numP = (part->Model()->NumParts()+(pose->IsClick()?1:0));
    for(int j = 0; j < numP; j++) {
      gt_locs[part->Model()->GetPart(j)->Id()].GetDetectionLocation(NULL, NULL, NULL, NULL, &child_pose);
      for(int k = 0; k < numChildPartPoseTransitions[j]; k++) {
	if(childPartPoseTransitions[j][k]->ChildPoseInd(part->GetPart(j)->GetPose(child_pose)) >= 0 && !childPartPoseTransitions[j][k]->GetChildPose(0)->IsNotVisible()) 
	  childPartPoseTransitions[j][k]->SanityCheckLocation(&gt_locs[part->Model()->Id()]);
      }
    }
  }
}
