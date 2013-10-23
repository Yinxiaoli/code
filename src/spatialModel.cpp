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

#define PART_WIDTH 50
#define NO_TRANSLATION_BIAS 1

#define ENFORCE_NON_NEGATIVE_WEIGHTS

//#define USE_MAX_CHANGE

ObjectPartPoseTransition::ObjectPartPoseTransition() {
  parentPart = childPart = NULL;
  parentPose = NULL;
  childPoses = NULL;
  numChildPoses = 0;
  numTrainExamples = NULL;
  parentPartName = childPartName = parentPoseName = NULL;
  childPoseNames = NULL;
  isClick = false;
  W = NULL;
  Reset();
}

int ObjectPartPoseTransition::AddChildPose(ObjectPose *pose) {
  childPoses = (ObjectPose**)realloc(childPoses, sizeof(ObjectPose*)*(numChildPoses+1));
  childPoseNames = (char**)realloc(childPoseNames, sizeof(char*)*(numChildPoses+1));
  numTrainExamples = (int*)realloc(numTrainExamples, sizeof(int)*(numChildPoses+1));
  W = (float*)realloc(W, sizeof(float)*(8+numChildPoses+1));
  childPoses[numChildPoses] = pose;
  childPoseNames[numChildPoses] = StringCopy(pose->Name());
  numTrainExamples[numChildPoses] = 0;
  W[8+numChildPoses] = 0;
  return numChildPoses++;
}

void ObjectPartPoseTransition::Reset() {
  numTrainExamples = numChildPoses ? (int*)realloc(numTrainExamples, (sizeof(int)*numChildPoses)) : NULL;
  W = (float*)realloc(W, sizeof(float)*(8+numChildPoses+1));
  for(int i = 0; i < numChildPoses; i++) numTrainExamples[i] = 0;
  offset_norm = 0;
  offset.x = offset.y = 0;
  max_change_x = max_change_y = max_change_scale = max_change_rot = 10000;
  offset_orientation_count = NULL;
  offset_orientation = offset_scale = 0;
  ZeroWeights();
}

void ObjectPartPoseTransition::ZeroWeights() {
  for(int i = 0; i < 8+numChildPoses; i++) W[i] = 0;
}

ObjectPartPoseTransition::~ObjectPartPoseTransition() {
  if(parentPartName) free(parentPartName);
  if(childPartName) free(childPartName);
  if(parentPoseName) free(parentPoseName);
  if(childPoseNames) {
    for(int i = 0; i < numChildPoses; i++)
      free(childPoseNames[i]);
    free(childPoseNames);
  }
  if(childPoses) free(childPoses);
  if(numTrainExamples) free(numTrainExamples);
  if(W) free(W);
}

Json::Value ObjectPartPoseTransition::Save() {
  Json::Value root;

  root["parentPart"] = parentPart->Name();
  root["childPart"] = childPart->Name();
  root["parentPose"] = parentPose->Name();
  if(childPoses) {
    Json::Value a;
    for(int i = 0; i < numChildPoses; i++)
      a[i] = childPoses[i]->Name();
    root["childPoses"] = a;
  }
  Json::Value n;
  for(int i = 0; i < numChildPoses; i++)
    n[i] = numTrainExamples[i];
  root["numTrainExamples"] = n;
  root["maxChangeX"] = max_change_x;
  root["maxChangeY"] = max_change_y;
  root["maxChangeScale"] = max_change_scale;
  root["maxChangeRot"] = max_change_rot;
  root["isClick"] = isClick;

  Json::Value o;
  o["norm"] = offset_norm;
  o["x"] = offset.x;
  o["y"] = offset.y;
  o["rotation"] = offset_orientation;
  o["scale"] = offset_scale;
  root["offsets"] = o;

  Json::Value wo;
  wo["dx"] = W[0];
  wo["dy"] = W[1];
  wo["dr"] = W[2];
  wo["ds"] = W[3];
  wo["dxx"] = W[4];
  wo["dyy"] = W[5];
  wo["drr"] = W[6];
  wo["dss"] = W[7];
  Json::Value a;
  for(int i = 0; i < numChildPoses; i++)
    a[i] = W[8+i];
  wo["const"] = a;
  root["weights"] = wo;

  return root;
}

bool ObjectPartPoseTransition::Load(const Json::Value &root) {
  parentPartName = StringCopy(root.get("parentPart", "").asString().c_str());
  childPartName = StringCopy(root.get("childPart", "").asString().c_str());
  parentPoseName = StringCopy(root.get("parentPose", "").asString().c_str());
  if(root.isMember("childPose")) {
    numChildPoses = 1;
    childPoseNames = (char**)malloc(sizeof(char*)*numChildPoses);
    childPoseNames[0] = StringCopy(root.get("childPose", "").asString().c_str());
  } else if(root.isMember("childPoses")) {
    numChildPoses = root["childPoses"].size();
    childPoseNames = (char**)malloc(sizeof(char*)*numChildPoses);
    for(int i = 0; i < numChildPoses; i++)
      childPoseNames[i] = StringCopy(root["childPoses"][i].asString().c_str());
  }
  if(root["numTrainExamples"].isArray()) {
    assert(numChildPoses == root["numTrainExamples"].size());
    numTrainExamples = (int*)malloc(sizeof(int)*numChildPoses);
    for(int i = 0; i < numChildPoses; i++)
      numTrainExamples[i] = root["numTrainExamples"][i].asInt();
  } else {
    assert(numChildPoses == 1);
    numTrainExamples = (int*)malloc(sizeof(int)*numChildPoses);
    numTrainExamples[0] = root.get("numTrainExamples", 0).asInt();
  }
  max_change_x = root.get("maxChangeX", 10000).asInt();
  max_change_y = root.get("maxChangeY", 10000).asInt();
  max_change_scale = root.get("maxChangeScale", 10000).asInt();
  max_change_rot = root.get("maxChangeRot", 10000).asInt();
  isClick = root.get("isClick", false).asBool();
  if(!(strlen(parentPartName) > 0 && strlen(childPartName) > 0 && strlen(parentPoseName) >= 0)) {
    fprintf(stderr, "Error reading object part pose transition\n");
    return false;
  }

  Json::Value o = root["offsets"];
  offset_norm = o.get("norm", 0).asFloat();
  offset.x = o.get("x", 0).asFloat();
  offset.y = o.get("y", 0).asFloat();
  offset_scale = o.get("scale", 0).asFloat();
  offset_orientation = o.get("rotation", 0).asFloat();

  Json::Value wo = root["weights"];
  W = (float*)realloc(W, sizeof(float)*(8+numChildPoses));
  W[0] = wo.get("dx", 0).asFloat();
  W[1] = wo.get("dy", 0).asFloat();
  W[2] = wo.get("dr", 0).asFloat();
  W[3] = wo.get("ds", 0).asFloat();
  W[4] = wo.get("dxx", 0).asFloat();
  W[5] = wo.get("dyy", 0).asFloat();
  W[6] = wo.get("drr", 0).asFloat();
  W[7] = wo.get("dss", 0).asFloat();
  if(wo["const"].isArray()) {
    assert(wo["const"].size() == numChildPoses);
    for(int i = 0; i < numChildPoses; i++)
      W[8+i] = wo["const"][i].asFloat();
  } else
    W[8] = wo.get("const", 0).asFloat();
  
  if(!offset_norm) {
    offset_norm = sqrt(SQR(offset.x)+SQR(offset.y))*(NORMALIZE_TEMPLATES ? 21 : 1);
    offset_norm = (offset_norm&&!isClick) ? 1.0f/offset_norm : 1;
  }

  return true;
}


bool ObjectPartPoseTransition::ResolveLinks(Classes *classes) {
  if(isClick) {
    parentPart = classes->FindClickPart(parentPartName);
    parentPose = classes->FindClickPose(parentPoseName);
  } else {
    parentPart = classes->FindPart(parentPartName);
    parentPose = classes->FindPose(parentPoseName);
  }
  childPart = classes->FindPart(childPartName);
  childPoses = (ObjectPose**)realloc(childPoses, sizeof(ObjectPose*)*numChildPoses);
  for(int i = 0; i < numChildPoses; i++)
    childPoses[i] = childPoseNames[i] ? classes->FindPose(childPoseNames[i]) : NULL;

  return true;
}

int ObjectPartPoseTransition::GetWeights(float *w) {
  if(childPoses[0]->IsNotVisible() || parentPose->IsNotVisible()) { 
    w[0] = W[8]; 
    return 1; 
  } else { 
    memcpy(w, W, sizeof(float)*(8+numChildPoses));
    return 8+numChildPoses; 
  } 
}


#define LEARN_POSE_TRANSITIONS true
#define REGULARIZE_POSE_TRANSITIONS false
#define LEARN_SPATIAL_SCORES true
#define REGULARIZE_SPATIAL_SCORES true

//#define LEARN_SPATIAL_SCORES false 


int ObjectPartPoseTransition::GetWeightConstraints(int *wc, bool *learn_weights, bool *regularize) {
  if(GetChildPose(0)->IsNotVisible() || parentPose->IsNotVisible()) { 
    learn_weights[0] = LEARN_POSE_TRANSITIONS;
    regularize[0] = REGULARIZE_POSE_TRANSITIONS;
    return 1; 
  } else {
    wc[4] = wc[5] = wc[6] = wc[7] = 1;
    learn_weights[4] = learn_weights[5] = learn_weights[6] = learn_weights[7] = LEARN_SPATIAL_SCORES;
    regularize[4] = regularize[5] = regularize[6] = regularize[7] = REGULARIZE_SPATIAL_SCORES;
    for(int i = 0; i < numChildPoses; i++) {
      learn_weights[8+i] = LEARN_POSE_TRANSITIONS;
      regularize[8+i] = REGULARIZE_POSE_TRANSITIONS;
	}
    return 8+numChildPoses; 
  } 
}

void ObjectPartPoseTransition::SetWeights(float *w) {
  if(childPoses[0]->IsNotVisible() || parentPose->IsNotVisible()) 
    W[8] = w[0]; 
  else {
    //assert(w[4] >= 0 && w[5] >= 0 && w[6] >= 0 && w[7] >= 0);
#ifdef ENFORCE_NON_NEGATIVE_WEIGHTS
    if(w[4] < 0.0000001) w[4] = 0.0000001;
    if(w[5] < 0.0000001) w[5] = 0.0000001;
    if(w[6] < 0.0000001) w[6] = 0.0000001;
    if(w[7] < 0.0000001) w[7] = 0.0000001;
#endif
    memcpy(W, w, sizeof(float)*(8+numChildPoses)); 
  }
}

int ObjectPartPoseTransition::NumWeights() { 
  return ((numChildPoses && childPoses[0]->IsNotVisible()) || (parentPose && parentPose->IsNotVisible())) ? 1 : (8+numChildPoses); 
}

float ObjectPartPoseTransition::MaxFeatureSumSqr() {
  if(((numChildPoses && childPoses[0]->IsNotVisible()) ||(parentPose && parentPose->IsNotVisible())))
    return SQR(POSE_TRANSITION_FEATURE);
  else {
    Classes *classes = parentPart->GetClasses();
    float g = (float)classes->SpatialGranularity();
    int num_o = classes->GetFeatureParams()->numOrientations;
    int num_s = classes->GetFeatureParams()->numScales;
    float n = SQR(POSE_TRANSITION_FEATURE)*numChildPoses + SQR(max_change_x*g*offset_norm) + SQR(max_change_y*g*offset_norm) + 
      SQR(SQR(max_change_x*g*offset_norm)) + SQR(SQR(max_change_y*g*offset_norm));
    bool noSO = classes->GetScaleOrientationMethod() == SO_PARENT_CHILD_SAME_SCALE_ORIENTATION || 
      classes->GetScaleOrientationMethod() == SO_SCALE_ORIENTATION_NO_COST;
    if(!noSO) 
      n += SQR(num_o/2) + SQR(SQR(num_o/2)) + SQR(num_s) + SQR(SQR(num_s));
    return n;
  }
}

int ObjectPartPoseTransition::ChildPoseInd(ObjectPose *pose) {
  for(int i = 0; i < numChildPoses; i++)
    if(childPoses[i] == pose)
      return i;
  return -1;
}

int ObjectPartPoseTransitionInstance::ChildPoseInd(ObjectPoseInstance *pose) {
  ObjectPose *m = pose->Model();
  if(!isReversed) {
    for(int i = 0; i < NumChildPoses(); i++)
      if(GetChildPose(i) == pose)
	return i;
  } else {
    if(!isFlipped) 
      return model->ChildPoseInd(pose->Model());
    else 
      return model->ChildPoseInd(m->GetFlipped() ? m->GetFlipped() : m);
  }
  return -1;
}

ObjectPartInstance *ObjectPartPoseTransitionInstance::GetParentPart() {
  ObjectPart *parentPartM = isReversed ? model->GetChildPart() : model->GetParentPart();
  if(isFlipped) parentPartM = parentPartM->GetFlipped() ? parentPartM->GetFlipped() : parentPartM;
  return parentPartM->IsClick() ? process->GetClickPartInst(parentPartM->Id()) : process->GetPartInst(parentPartM->Id());
}
  
ObjectPartInstance *ObjectPartPoseTransitionInstance::GetChildPart() {
  ObjectPart *childPartM  = isReversed ? model->GetParentPart() : model->GetChildPart();
  if(isFlipped) childPartM = childPartM->GetFlipped() ? childPartM->GetFlipped() : childPartM;
  return childPartM->IsClick() ? process->GetClickPartInst(childPartM->Id()) : process->GetPartInst(childPartM->Id());
}
  
ObjectPoseInstance *ObjectPartPoseTransitionInstance::GetParentPose() {
  ObjectPart *parentPartM = isReversed ? model->GetChildPart() : model->GetParentPart();
  ObjectPose *parentPoseM = isReversed ? model->GetChildPose(0) : model->GetParentPose();
  if(isFlipped) {
    if(parentPoseM->GetFlipped()) {
      parentPoseM = parentPoseM->GetFlipped();
    } else {
      parentPoseM = parentPartM->GetFlipped() ? parentPartM->GetFlipped()->GetPose(parentPartM->PoseInd(parentPoseM)) : parentPoseM;
    }
    parentPartM = parentPartM->GetFlipped() ? parentPartM->GetFlipped() : parentPartM;
  }
  return GetParentPart()->GetPose(parentPartM->PoseInd(parentPoseM));
}
  
ObjectPoseInstance *ObjectPartPoseTransitionInstance::GetChildPose(int i) {
  ObjectPart *childPartM = isReversed ? model->GetParentPart() : model->GetChildPart();
  ObjectPose *childPoseM = isReversed ? model->GetParentPose() : model->GetChildPose(i);
  if(isFlipped) {
    if(childPoseM->GetFlipped()) {
      childPoseM = childPoseM->GetFlipped();
    } else {
      childPoseM = childPartM->GetFlipped() ? childPartM->GetFlipped()->GetPose(childPartM->PoseInd(childPoseM)) : childPoseM;
    }
    childPartM = childPartM->GetFlipped() ? childPartM->GetFlipped() : childPartM;
  }
  return GetChildPart()->GetPose(childPartM->PoseInd(childPoseM));
}

int ObjectPartPoseTransitionInstance::NumChildPoses() { 
  return isReversed ? 1 : model->NumChildPoses(); 
}

ObjectPartPoseTransitionInstance::ObjectPartPoseTransitionInstance(ObjectPartPoseTransition *m, ImageProcess *p, bool isFlip, bool isReverse) {
  model = m;
  process = p;
  isFlipped = isFlip;
  isReversed = isReverse;
  custom_weights_buff = (float*)malloc(sizeof(float)*(8+m->NumChildPoses()));
  
  childPoseMaxInds = NULL;
  scores = scores_no_wt = NULL;
  scores_scale_rot = NULL;
  best_offsets = NULL;
  spatialScores = NULL;
  custom_weights = NULL;
  offsets = NULL;
}

ObjectPartPoseTransitionInstance::~ObjectPartPoseTransitionInstance() {
  Clear();
  if(custom_weights_buff)
    free(custom_weights_buff);
}

int ObjectPartPoseTransitionInstance::SetCustomWeights(float *w) { 
  custom_weights = w; 
  if(w) {
    if(model->NumWeights() == 1)
      custom_weights_buff[8] = w[0]; 
    else {
#ifdef ENFORCE_NON_NEGATIVE_WEIGHTS
      if(w[4] < 0.0000001) w[4] = 0.0000001;
      if(w[5] < 0.0000001) w[5] = 0.0000001;
      if(w[6] < 0.0000001) w[6] = 0.0000001;
      if(w[7] < 0.0000001) w[7] = 0.0000001;
#endif
      memcpy(custom_weights_buff, w, sizeof(float)*model->NumWeights());
    }
  } 

  return model->NumWeights();
}


void ObjectPartPoseTransitionInstance::Clear() {
  FeatureParams *feat = process->Features()->GetParams();
  if(scores) {
    for(int scale = 0; scale < feat->numScales; scale++) {
      for(int rot = 0; rot < feat->numOrientations; rot++) {
        cvReleaseImage(&best_offsets[scale][rot]);
        if(scores[scale][rot] != scores_no_wt[scale][rot]) 
          cvReleaseImage(&scores_no_wt[scale][rot]);
        cvReleaseImage(&scores[scale][rot]);
      }
    }
    if(scores_no_wt && scores != scores_no_wt) free(scores_no_wt);
    free(scores);
    free(best_offsets);
    scores = NULL;
    scores_no_wt = NULL;
    best_offsets = NULL;
  }

  if(scores_scale_rot)
    process->Features()->ReleaseResponses(&scores_scale_rot);
  scores_scale_rot = NULL;
  if(offsets)
    process->Features()->ReleaseResponses(&offsets);
  offsets = NULL;
        
  if(childPoseMaxInds) {
    for(int scale = 0; scale < feat->numScales; scale++) 
      for(int rot = 0; rot < feat->numOrientations; rot++) 
	cvReleaseImage(&childPoseMaxInds[scale][rot]);
    free(childPoseMaxInds);
  }
  childPoseMaxInds = NULL;

  FreeSpatialScores();
}

void ObjectPartPoseTransitionInstance::AllocateCacheTables() {
  FeatureParams *feat = process->Features()->GetParams();
  int memSize = feat->numScales*(sizeof(IplImage**) + sizeof(IplImage*)*feat->numOrientations);
  bool addWtLater = isReversed && model->NumChildPoses() > 1;

assert(!scores && !scores_no_wt);

  if(!best_offsets) { best_offsets = (IplImage***)malloc(memSize); memset(best_offsets, 0, memSize); }
  if(!scores) { scores = (IplImage***)malloc(memSize); memset(scores, 0, memSize); }
  if(!scores_no_wt) { scores_no_wt = addWtLater ? (IplImage***)malloc(memSize) : scores;  memset(scores_no_wt, 0, memSize); }
  if(!childPoseMaxInds) { childPoseMaxInds = (IplImage***)malloc(memSize); memset(childPoseMaxInds, 0, memSize); }
  char *bPtr = ((char*)best_offsets)+feat->numScales*sizeof(IplImage**);
  char *cPtr = ((char*)scores)+feat->numScales*sizeof(IplImage**);
  char *cPtr2= ((char*)scores_no_wt)+feat->numScales*sizeof(IplImage**);
  char *cPtr3= ((char*)childPoseMaxInds)+feat->numScales*sizeof(IplImage**);

  for(int scale = 0; scale < feat->numScales; scale++) {
    best_offsets[scale] = (IplImage**)bPtr;  bPtr += sizeof(IplImage*)*feat->numOrientations;
    scores[scale] = (IplImage**)cPtr;  cPtr += sizeof(IplImage*)*feat->numOrientations;
    scores_no_wt[scale] = (IplImage**)cPtr2;  cPtr2 += sizeof(IplImage*)*feat->numOrientations;
    childPoseMaxInds[scale] = (IplImage**)cPtr3;  cPtr3 += sizeof(IplImage*)*feat->numOrientations;
  }
}


void ObjectPartPoseTransitionInstance::ConvertDetectionImageCoordinatesUpdateMax(IplImage *srcImg, IplImage *dstImg, IplImage *srcOffset, IplImage *dstOffset, 
										 int srcScale, int srcRot, int dstScale, int dstRot) {
  int w, h;
  float mat[6];
  FeatureOptions *fo = process->Features();
  RotationInfo rSrc = fo->GetRotationInfo(srcRot, srcScale);
  RotationInfo rDst = fo->GetRotationInfo(dstRot, dstScale);
  MultiplyAffineMatrices(rSrc.invMat, rDst.mat, mat);
  fo->GetDetectionImageSize(&w, &h, dstScale, dstRot);

  assert(srcImg->nChannels == 1 && dstImg->nChannels == 1 && srcOffset->nChannels == 5 && dstOffset->nChannels == 5);
  int oc = srcOffset->nChannels;

  // Assume every pixel in the destination image has already been filled by a candidate pixel from the source image.
  // Now loop through pixels in the source image, find the corresponding pixel in the destination image, and update it
  // if the score of the source pixel is greater.  This procedure is designed to ensure 1) every pixel in the
  // destination image is mapped to a pixel in the source image, and 2) when multiple pixels in the source image
  // map to the same pixel in the destination image, take the one with maximum score
  float dx = mat[0], dy = mat[3], x, y;
  int i, j, ix, iy;
  float *sPtr, *ptr;
  int *iPtr, *oPtr;
  int pad = 0;
  char *ptr2=srcImg->imageData+srcImg->widthStep*pad, 
    *iPtr2=srcOffset->imageData+srcOffset->widthStep*pad;
  for(i = pad; i < srcImg->height-pad; i++, ptr2+=srcImg->widthStep, iPtr2 += srcOffset->widthStep) {
    x = mat[2] + i*mat[1] + dx*pad + .5f;
    y = mat[5] + i*mat[4] + dy*pad + .5f;
    for(j = pad, ptr = ((float*)ptr2), iPtr = ((int*)iPtr2)+oc*pad; j < srcImg->width-pad; j++, iPtr += oc) {
      ix = (int)x; iy = (int)y;
      if(ix < 0) ix = 0;
      if(ix >= w) ix = w-1;
      if(iy < 0) iy = 0;
      if(iy >= h) iy = h-1;
      sPtr = ((float*)(dstImg->imageData+iy*dstImg->widthStep));
      if(ptr[j] > sPtr[ix]) {
        // The source image has higher score than the current pixel in the destination image
        sPtr[ix] = ptr[j];
        oPtr = ((int*)(dstOffset->imageData+iy*dstOffset->widthStep))+(ix*oc);
        oPtr[0] = iPtr[0];  oPtr[1] = iPtr[1];  oPtr[2] = iPtr[2];  oPtr[3] = iPtr[3];  oPtr[4] = iPtr[4]; 
      }
      x += dx;  y += dy;  
    }
  }
}



void ObjectPartPoseTransitionInstance::InitDetect(int *l_parts, int *l_poses, int *l_scales, int *l_rotations, int part_ind, int pose_ind, int *num) {
  //if(scores) return;
  bool addWtLater = (isReversed && model->NumChildPoses() > 1);
  assert(!scores_scale_rot);

  if(!scores)
    AllocateCacheTables();
  
  FeatureOptions *fo = process->Features();
  FeatureParams *feat = fo->GetParams();
  for(int j = 0; j < feat->numOrientations; j++) {
    for(int i = 0; i < feat->numScales; i++) {
      l_scales[*num] = i; 
      l_rotations[*num] = j;
      l_parts[*num] = part_ind;
      l_poses[*num] = pose_ind;
      *num = *num + 1;
    }
  }

  // Infer the best scale/orientation of the child part for every pixel/scale/orientation of the parent part
  if(process->GetScaleOrientationMethod() == SO_SCALE_ORIENTATION_ARBITRARY_COST && 
     !GetChildPose(0)->IsNotVisible() && !GetParentPose()->IsNotVisible()) {

    // Convert all detection score images into the coordinate system for scale=0, orientation=0, such that
    // pixels from all scales/orientations are aligned
    float offset_norm, offset_x, offset_y, offset_scale, offset_rotation, wx, wxx, wy, wyy, ws, wss, wr, wrr, wt = 0;
    GetOffsets(&offset_norm, &offset_x, &offset_y, &offset_scale, &offset_rotation); 
    int i, j, k, l, k5, k3, s, o;
    int *optr, *ptr;
    int w, h;   fo->GetDetectionImageSize(&w, &h, 0, 0);
    FeatureOptions *fo = process->Features();
    FeatureParams *feat = fo->GetParams();
    IplImage ***scale_offsets, ***orientation_offsets;
    IplImage ***scores_tmp = (IplImage***)malloc(feat->numScales*(sizeof(IplImage**)+feat->numOrientations*sizeof(IplImage*)));
    IplImage ***offsets_tmp = (IplImage***)malloc(feat->numScales*(sizeof(IplImage**)+feat->numOrientations*sizeof(IplImage*)));
    scores_scale_rot = (IplImage***)malloc(feat->numScales*(sizeof(IplImage**)+feat->numOrientations*sizeof(IplImage*)));
    offsets = (IplImage***)malloc(feat->numScales*(sizeof(IplImage**)+feat->numOrientations*sizeof(IplImage*)));
    for(j = 0; j < feat->numScales; j++) {
      scores_tmp[j] = ((IplImage**)(scores_tmp+feat->numScales)) + j*feat->numOrientations;
      scores_scale_rot[j] = ((IplImage**)(scores_scale_rot+feat->numScales)) + j*feat->numOrientations;
      offsets[j] = ((IplImage**)(offsets+feat->numScales)) + j*feat->numOrientations;
      offsets_tmp[j] = ((IplImage**)(offsets_tmp+feat->numScales)) + j*feat->numOrientations;
      for(i = 0; i < feat->numOrientations; i++) {
        IplImage *tmpImg = NULL;
        IplImage *subImg = NULL, *inds = NULL;
        offsets_tmp[j][i] = cvCreateImage(cvSize(w, h), IPL_DEPTH_32S, 3);
        for(l = 0; l < NumChildPoses(); l++) {
          GetWeights(&wx, &wxx, &wy, &wyy, &ws, &wss, &wr, &wrr, !addWtLater ? &wt : NULL, l);
          IplImage *curr = cvCloneImage(GetChildPose(l)->GetResponse(j, i, GetParentPart(), &tmpImg));
          if(wt) cvAddS(curr, cvScalar(-POSE_TRANSITION_FEATURE*wt), curr);
          if(!l) 
            subImg = curr;
          else {
            if(!inds) inds = cvCreateImage(cvSize(w, h), IPL_DEPTH_32S, 1);
            MaxWithIndex(subImg, curr, inds, l); 
            cvReleaseImage(&curr);
          }
          if(tmpImg) cvReleaseImage(&tmpImg);
        }
        cvZero(offsets_tmp[j][i]);
        if(inds) {
          cvSetChannel(offsets_tmp[j][i], inds, 2);
          cvReleaseImage(&inds);
        } 
                
	//double minVal, maxVal; CvPoint minLoc, maxLoc; cvMinMaxLoc(subImg, &minVal, &maxVal, &minLoc, &maxLoc); assert(minVal > -1e+14 && maxVal < 10000000);
        scores_tmp[j][i] = fo->ConvertDetectionImageCoordinates(subImg, j, i, 0, 0, -10000000000000.0f, offsets_tmp[j][i]);
	//cvMinMaxLoc(scores_tmp[j][i], &minVal, &maxVal, &minLoc, &maxLoc); assert(minVal > -1e+14 && maxVal < 10000000);
        if(subImg) cvReleaseImage(&subImg);
        //cvReleaseImage(&scores[j][i]);
      }
    }

    //if(wrr < 0) wrr = 0;
    //if(wss < 0) wss = 0;
    //W[6]=W[2]=0;
    //W[7]=W[3]=0;

    // For each pixel/scale/orientation of the parent part, compute the scale/orientation of the child part that maximizes the
    // scale and orientation component of the spatial cost between parent and child, obtaining new detection score maps 
#ifdef USE_MAX_CHANGE
    float ms = model->max_change_scale, mr = model->max_change_rot;
#else
    float ms = 10000, mr = 10000;
#endif
    IplImage ***scores_new = DistanceTransformScaleOrientation(scores_tmp,  &scale_offsets, &orientation_offsets, feat->numScales, 
							       feat->numOrientations, my_round(offset_scale), my_round(offset_rotation), 
							       -wss, -wrr, -ws, -wr, ms, mr);

    //fprintf(stderr, "%s %s\n", parentPose->Model()->Name(), childPose->Model()->Name());

    // Convert new detection score images back into the coordinate system for their respective scale/orientation
    for(j = 0; j < feat->numScales; j++) {
      for(i = 0; i < feat->numOrientations; i++) {
	//double minVal, maxVal; CvPoint minLoc, maxLoc; cvMinMaxLoc(scores_new[j][i], &minVal, &maxVal, &minLoc, &maxLoc); assert(minVal > -1e+14 && maxVal < 10000000);
        scores_scale_rot[j][i] = fo->ConvertDetectionImageCoordinates(scores_new[j][i], 0, 0, j, i, -10000000000000.0f);
	//cvMinMaxLoc(scores_scale_rot[j][i], &minVal, &maxVal, &minLoc, &maxLoc); assert(minVal > -1e+14 && maxVal < 10000000);

        // For each pixel location in the parent, offsets[scale][rot] stores the pixel location, scale, and orientation of the
        // child part that maximizes the scale/orientation part of the spatial transition cost
        IplImage *offset = cvCreateImage(cvSize(scores_new[j][i]->width,scores_new[j][i]->height), IPL_DEPTH_32S, 5);
        assert(offset->widthStep*offset->height == (int)(offset->width*offset->height*5*sizeof(int)));
        assert(offsets_tmp[j][i]->widthStep*offsets_tmp[j][i]->height == (int)(offsets_tmp[j][i]->width*offsets_tmp[j][i]->height*3*sizeof(int)));
        assert(orientation_offsets[j][i]->widthStep*orientation_offsets[j][i]->height == (int)(orientation_offsets[j][i]->width*orientation_offsets[j][i]->height*sizeof(int)));
        assert(scale_offsets[j][i]->widthStep*scale_offsets[j][i]->height == (int)(scale_offsets[j][i]->width*scale_offsets[j][i]->height*sizeof(int)));
        assert(scores_new[j][i]->width == scale_offsets[j][i]->width && scores_new[j][i]->width == orientation_offsets[j][i]->width);
        ptr = (int*)offset->imageData;
        for(k = 0, k5 = 0, k3 = 0; k < offset->width*offset->height; k++, k5+=5, k3 += 3) {
          s = ((int*)scale_offsets[j][i]->imageData)[k];
          o = ((int*)orientation_offsets[j][i]->imageData)[k];
          optr = (int*)offsets_tmp[s][o]->imageData;
          ptr[k5] = optr[k3]; ptr[k5+1] = optr[k3+1]; ptr[k5+2] = s; ptr[k5+3] = o; ptr[k5+4] = optr[k3+2]; 
        }
        offsets[j][i] = fo->ConvertDetectionImageCoordinates(offset, 0, 0, j, i, -1);

        // Avoid potentially skipping relevant pixel locations in the child when the parent is lower resolution
        ConvertDetectionImageCoordinatesUpdateMax(scores_new[j][i], scores_scale_rot[j][i], offset, offsets[j][i], 0, 0, j, i);

#ifdef EXTRA_DEBUG
	double minVal, maxVal, minVal2, maxVal2; 
	CvPoint minLoc, maxLoc,  minLoc2, maxLoc2; 
	cvMinMaxLoc(scores_scale_rot[j][i], &minVal, &maxVal, &minLoc, &maxLoc); 
	cvMinMaxLoc(scores_new[j][i], &minVal2, &maxVal2, &minLoc2, &maxLoc2);
	assert(maxVal == maxVal2);
#endif

        cvReleaseImage(&offset);
      }
    }

    fo->ReleaseResponses(&scores_tmp);
    fo->ReleaseResponses(&scores_new);
    fo->ReleaseResponses(&scale_offsets);
    fo->ReleaseResponses(&orientation_offsets);
    fo->ReleaseResponses(&offsets_tmp);
  }

  if(g_debug > 2) fprintf(stderr, "    distance transform %s...\n", GetChildPose(0)->Model()->Name());
}

void ObjectPartPoseTransitionInstance::SanityCheckLocation(PartLocation *par_loc) {
  int par_s, par_o, par_x, par_y, par_p;
  float offset_norm, offset_x, offset_y, offset_scale, offset_rotation,
    wx, wxx, wy, wyy, ws, wss, wr, wrr, wt = 0;
  FeatureOptions *fo = process->Features();
  FeatureParams *feat = fo->GetParams();
  int g = feat->spatialGranularity;
  float ff = 1.0f/g;
  int dx, dy;
  float sp, ss, sr, sy, bestS= -INFINITY, score;
  int best[5];

  par_loc->GetDetectionLocation(&par_x, &par_y, &par_s, &par_o, &par_p);
  GetOffsets(&offset_norm, &offset_x, &offset_y, &offset_scale, &offset_rotation); 

  for(int i = 0; i < model->NumChildPoses(); i++) {
    GetWeights(&wx, &wxx, &wy, &wyy, &ws, &wss, &wr, &wrr, &wt, i);
    sp = -POSE_TRANSITION_FEATURE*wt;
    for(int s = 0; s < feat->numScales; s++) {
      if(process->GetScaleOrientationMethod() == SO_PARENT_CHILD_SAME_SCALE_ORIENTATION && s != par_s) 
	continue;
      int ds = par_s+my_round(offset_scale)-s;
      ss = sp + ws*ds - wss*SQR(ds);
      for(int o = 0; o < feat->numOrientations; o++) {
	if(process->GetScaleOrientationMethod() == SO_PARENT_CHILD_SAME_SCALE_ORIENTATION && o != par_o) 
	  continue;
	int dr = par_o+my_round(offset_rotation)-o;
	if(dr < -feat->numOrientations/2) dr += feat->numOrientations;
	if(dr > feat->numOrientations/2) dr -= feat->numOrientations;
	sr = ss + wr*dr - wrr*SQR(dr);
	IplImage *img = GetChildPose(i)->GetResponse(s, o);
	char *ptr = img->imageData;
	for(int y = 0; y < img->height; y++, ptr+=img->widthStep) {
	  dy = -(y-par_y - (int)(offset_y*ff)) * g * offset_norm;
	  sy = sr + wy*dy - wyy*SQR(dy);
	  for(int x = 0; x < img->width; x++) {
	    dx = -(x-par_x - (int)(offset_x*ff)) * g * offset_norm;
	    score = ((float*)ptr)[x] + sy + wx*dx - wxx*SQR(dx);
	    if(score > bestS) { 
	      bestS = score;
	      best[0] = i; best[1] = s; best[2] = o; best[3] = y; best[4] = x;
	    }
	  }
	}
      }
    }
  }

  if(process->GetScaleOrientationMethod() == SO_SCALE_ORIENTATION_ARBITRARY_COST) {
    int16_t *ptr2 = ((int16_t*)(best_offsets[par_s][par_o]->imageData+par_y*best_offsets[par_s][par_o]->widthStep))+par_x*2;
    int x = ptr2[0], y=ptr2[1];
    int *ptr, poseInd;
    if(offsets) {
      ptr = ((int*)(offsets[par_s][par_o]->imageData+y*offsets[par_s][par_o]->widthStep))+x*5;
      poseInd = GetChildPart()->PoseInd(GetChildPose(ptr[4]));
    }
    float s = ((float*)(scores[par_s][par_o]->imageData+par_y*scores[par_s][par_o]->widthStep))[par_x];
    assert(my_abs(s - bestS) < .0001);
  }
}

/*
 * Compute the score for this object part/pose transition for each scale/rotation/pixel location in the image
 * using dynamic programming.  
 */
IplImage *ObjectPartPoseTransitionInstance::Detect(ObjectPoseInstance *parPose, int scale, int rot) {
  float wt_later = 0;
  bool addWtLater = (isReversed && model->NumChildPoses() > 1);
  if(isReversed && model->NumChildPoses() > 1) {
    GetWeights(NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &wt_later, ChildPoseInd(parPose));
  }
  if(scores_no_wt[scale][rot]) {
    AddUnaryWeights(scale, rot, -POSE_TRANSITION_FEATURE*wt_later);
    return scores[scale][rot];
  }
  FeatureOptions *fo = process->Features();
  FeatureParams *feat = fo->GetParams();
  int g = feat->spatialGranularity;
  float offset_norm, offset_x, offset_y, offset_scale, offset_rotation, wx, wxx, wy, wyy, ws, wss, wr, wrr, wt = 0;
  GetOffsets(&offset_norm, &offset_x, &offset_y, &offset_scale, &offset_rotation); 
  float ss = scale < fo->ScaleOffset() ? 1 / fo->Scale(scale) : 1;
  float ff = 1.0f/g;
  float offn = model->offset_norm*g;//*ss*g;
    
  int w, h;
  process->Features()->GetDetectionImageSize(&w, &h, scale, rot);
  IplImage *subImg, *tmpImg = NULL;
  if(GetChildPose(0)->IsNotVisible() || GetParentPose()->IsNotVisible()) {
    // Special case corresponding to part not being visible.  The score at every pixel is just the pose
    // transition cost
    for(int i = 0; i < NumChildPoses(); i++) {
      IplImage *r = GetChildPose(i)->GetResponse(scale,rot,GetParentPart(),&tmpImg), *curr;
      GetWeights(&wx, &wxx, &wy, &wyy, &ws, &wss, &wr, &wrr, wt_later ? NULL : &wt, i);
      if(r) {
        curr = cvCloneImage(r);
        if(wt) cvAddS(curr, cvScalar(-POSE_TRANSITION_FEATURE*wt), curr);
      } else {
        curr = cvCreateImage(cvSize(w, h), IPL_DEPTH_32F, 1);
        cvSet(curr, cvScalar(-POSE_TRANSITION_FEATURE*wt));
      }
      if(!i) {
        scores_no_wt[scale][rot] = curr;
        if(childPoseMaxInds[scale][rot]) cvReleaseImage(&childPoseMaxInds[scale][rot]);
          childPoseMaxInds[scale][rot] = NULL;
      } else {
        if(!childPoseMaxInds[scale][rot]) {
          childPoseMaxInds[scale][rot] = cvCreateImage(cvSize(w,h), IPL_DEPTH_16S, 1);
          cvZero(childPoseMaxInds[scale][rot]);
        }
        MaxWithIndex(scores_no_wt[scale][rot], curr, childPoseMaxInds[scale][rot], i);
        cvReleaseImage(&curr);
      }
      if(tmpImg) cvReleaseImage(&tmpImg);
    }
    AddUnaryWeights(scale, rot, -POSE_TRANSITION_FEATURE*wt_later);
    return scores[scale][rot];
  }
  
  bool freeSubImg = true;
  for(int i = 0; i < NumChildPoses(); i++) {
    GetWeights(&wx, &wxx, &wy, &wyy, &ws, &wss, &wr, &wrr, wt_later ? NULL : &wt, i);
    IplImage *curr;
    if(process->GetScaleOrientationMethod() == SO_PARENT_CHILD_SAME_SCALE_ORIENTATION) {
      // parent and child parts must have the same scale and orientation
      curr = GetChildPose(i)->GetResponse(scale, rot, GetParentPart(), &tmpImg);
    } else if(process->GetScaleOrientationMethod() == SO_SCALE_ORIENTATION_NO_COST) {
      // child part can freely change in scale and orientation from the parent part with zero cost
      curr = GetChildPose(i)->GetMaxResponse(GetParentPart(), &tmpImg);
    } else {
      assert(scores_scale_rot && scores_scale_rot[scale][rot]);
      subImg = scores_scale_rot[scale][rot];
      freeSubImg = false;
      break;
    }

    if(!i) {
      subImg = cvCloneImage(curr);
      if(wt) cvAddS(subImg, cvScalar(-POSE_TRANSITION_FEATURE*wt), subImg);
      if(childPoseMaxInds[scale][rot]) cvReleaseImage(&childPoseMaxInds[scale][rot]);
      childPoseMaxInds[scale][rot] = NULL;
    } else {
      if(!childPoseMaxInds[scale][rot]) {
        childPoseMaxInds[scale][rot] = cvCreateImage(cvSize(w,h), IPL_DEPTH_16S, 1);
        cvZero(childPoseMaxInds[scale][rot]);
      }
      IplImage *tmp = cvCloneImage(curr);
      if(wt) cvAddS(tmp, cvScalar(-POSE_TRANSITION_FEATURE*wt), tmp);
      MaxWithIndex(subImg, tmp, childPoseMaxInds[scale][rot], i);
      cvReleaseImage(&tmp);
    } 
    if(tmpImg) cvReleaseImage(&tmpImg);
  } 

  if(GetParentPose()->IsNotVisible()) {
    cvAdd(scores_no_wt[scale][rot], subImg, scores_no_wt[scale][rot]);
    if(tmpImg) cvReleaseImage(&tmpImg);
    AddUnaryWeights(scale, rot, -POSE_TRANSITION_FEATURE*wt_later);
    return scores[scale][rot];
  }

  if(process->GetInferenceMethod() == IM_MAXIMUM_LIKELIHOOD) {
    // For every possible position of the parent part, find the position of the child part that maximizes the 
    // score (which is a sum of the log unary score map for the child and log pairwise spatial offset score)
    // Assumes subImg is a log probability
    
    if(g_debug > 3) fprintf(stderr, "    distance transform %s %d %d...\n", GetChildPose(0)->Model()->Name(), scale, rot);
#if NO_TRANSLATION_BIAS
    wx = wy = wr = ws = 0;
#endif
    //if(W[4] == 0) W[4] = .000000001f;
    //if(W[5] == 0) W[5] = .000000001f;
    //if(W[6] == 0) W[6] = .000000001f;
    //if(W[7] == 0) W[7] = .000000001f;

#ifdef USE_MAX_CHANGE
    float mx = model->max_change_x, my = model->max_change_y;
#else
    float mx = 10000, my = 10000;
#endif
    assert(!best_offsets[scale][rot]);
    //wxx = my_max(.000001,wxx);
    //wyy = my_max(.000001,wyy);
    DistanceTransform(subImg, -wxx*SQR(offn), -wyy*SQR(offn),  -wx*offn, -wy*offn, 
		      (int)(offset_x*ff), (int)(offset_y*ff), &best_offsets[scale][rot], 
		      &scores_no_wt[scale][rot], ceil(mx/*ss*//g), ceil(my/*ss*//g));
  } else {
    // For every possible position of the parent part, integrate out the location of the child part by summing
    // over every possible location of the child part and multiplying its unary score times the pairwise spatial
    // offset score
    // Assumes subImg is a regular probability
    if(g_debug > 2) fprintf(stderr, "    convolution %s %d %d...\n", GetChildPose(0)->Model()->Name(), scale, rot);
	  
    assert(0); // TODO: compute kernels, convert unary and pairwise log probabilities to regular probabilities
    //cvFilter2D(subImgShift, score, spatialScores[scale][rot][j]);
  }

  if(tmpImg) cvReleaseImage(&tmpImg);
  if(freeSubImg) cvReleaseImage(&subImg);
  
  if(addWtLater) AddUnaryWeights(scale, rot, -POSE_TRANSITION_FEATURE*wt_later);


  return scores[scale][rot];
}

void ObjectPartPoseTransitionInstance::AddUnaryWeights(int scale, int rot, float wt) {
  if(!wt) {
    if(scores[scale][rot] && scores[scale][rot] != scores_no_wt[scale][rot])
      cvReleaseImage(&scores[scale][rot]);
    scores[scale][rot] = scores_no_wt[scale][rot];
  } else {
    assert(scores != scores_no_wt);
    if(!scores[scale][rot] || scores[scale][rot] == scores_no_wt[scale][rot]) 
      scores[scale][rot] = cvCreateImage(cvSize(scores_no_wt[scale][rot]->width,scores_no_wt[scale][rot]->height), scores_no_wt[scale][rot]->depth, scores_no_wt[scale][rot]->nChannels);
    cvAddS(scores_no_wt[scale][rot], cvScalar(wt), scores[scale][rot]);
  }
}

PartLocation ObjectPartPoseTransitionInstance::GetChildPartLocation(PartLocation *l) {
  PartLocation t(*l); 
  t.SetPart(GetChildPart()->Model());
  t.SetResponseTime(0);

  float ff = 1.0f/process->Features()->SpatialGranularity();
  int par_x, par_y, par_scale, par_rot, par_pose;
  float offset_norm, offset_x, offset_y, offset_scale, offset_rotation;
  GetOffsets(&offset_norm, &offset_x, &offset_y, &offset_scale, &offset_rotation); 
  l->GetDetectionLocation(&par_x, &par_y, &par_scale, &par_rot, &par_pose);

  if(!GetChildPose(0)->IsNotVisible() && !GetParentPose()->IsNotVisible()) {
    assert(par_y >= 0 && par_y <= best_offsets[par_scale][par_rot]->height);
    assert(par_x >= 0 && par_x <= best_offsets[par_scale][par_rot]->width);
    int16_t *ptr = ((int16_t*)(best_offsets[par_scale][par_rot]->imageData+par_y*best_offsets[par_scale][par_rot]->widthStep))+par_x*2;
    float score = ((float*)(scores[par_scale][par_rot]->imageData+par_y*scores[par_scale][par_rot]->widthStep))[par_x];
    int x = ptr[0], y=ptr[1];
    assert(x >= 0 && y >= 0);
    int poseInd = !childPoseMaxInds[par_scale][par_rot] ? 0 : ((int16_t*)(childPoseMaxInds[par_scale][par_rot]->imageData+childPoseMaxInds[par_scale][par_rot]->widthStep*y))[x];
    t.SetDetectionLocation(x, y, par_scale, par_rot, GetChildPart()->PoseInd(GetChildPose(poseInd)), 
			   (x-par_x - (int)(offset_x*ff)), (y-par_y - (int)(offset_y*ff)));
    t.SetScore(score);
    if(offsets) {
      int *ptr = ((int*)(offsets[par_scale][par_rot]->imageData+y*offsets[par_scale][par_rot]->widthStep))+x*5;
      poseInd = GetChildPart()->PoseInd(GetChildPose(ptr[4]));
      t.SetDetectionLocation(ptr[0], ptr[1], ptr[2], ptr[3], GetChildPart()->PoseInd(GetChildPose(ptr[4])), 
			     (x-par_x - (int)(offset_x*ff)), (y-par_y - (int)(offset_y*ff)));
    }

    //t.centerPt.x = p.x/s+ptr[0]; t.centerPt.y = p.y/s+ptr[1]; 
    if(process->GetScaleOrientationMethod() == SO_SCALE_ORIENTATION_NO_COST) {
      GetChildPose(poseInd)->GetScaleRot(par_x, par_y, &par_scale, &par_rot);
      t.SetDetectionLocation(x, y, par_scale, par_rot, GetChildPart()->PoseInd(GetChildPose(poseInd)), 
			     (x-par_x - (int)(offset_x*ff)), (y-par_y - (int)(offset_y*ff)));
    }
  } else {
    FeatureOptions *fo = process->Features();
    FeatureParams *feat = fo->GetParams();
    int r = par_rot+my_round(offset_rotation);
    if(r >= feat->numOrientations) r -= feat->numOrientations;
    if(r < 0) r += feat->numOrientations;
    t.SetDetectionLocation(par_x+(int)(offset_x*ff), par_y+(int)(offset_y*ff), par_scale+offset_scale, r, 
			   GetChildPart()->PoseInd(GetChildPose(0)), par_x+(int)(offset_x*ff), par_y+(int)(offset_y*ff));
  }

  return t;
}

void ObjectPartPoseTransitionInstance::FreeSpatialScores() {
  if(spatialScores) {
    FeatureOptions *feat = process->Features();
    for(int j = 0; j < feat->NumScales(); j++) 
      for(int k = 0; k < feat->NumOrientations(); k++) 
	  cvReleaseImage(&spatialScores[j][k]);
    free(spatialScores);
    spatialScores = NULL;
  }
}

IplImage ***ObjectPartPoseTransitionInstance::BuildSpatialScores(PartLocation *loc) {
  int j, k;

  FreeSpatialScores();
  int loc_x, loc_y, scale, rotation, loc_pose;
  loc->GetDetectionLocation(&loc_x, &loc_y, &scale, &rotation, &loc_pose);

  scale = scale >= 0 ? scale : 0;
  rotation = rotation >= 0 ? rotation : 0;
  FeatureOptions *feat = process->Features();
  int numScales = feat->NumScales();
  int numOrientations = feat->NumOrientations();
  float ss = scale < feat->ScaleOffset() ? 1 / feat->Scale(scale) : 1;
  float g = (float)feat->SpatialGranularity();
  float ff = 1.0f/*ss*//g;
  float offset_norm, offset_x, offset_y, offset_scale, offset_rotation, wx, wxx, wy, wyy, ws, wss, wr, wrr;
  GetOffsets(&offset_norm, &offset_x, &offset_y, &offset_scale, &offset_rotation); 
  GetWeights(&wx, &wxx, &wy, &wyy, &ws, &wss, &wr, &wrr, NULL);

  // allocation spatial scores
  int sz = numScales*(sizeof(IplImage**) + numOrientations*sizeof(IplImage*));
  spatialScores = (IplImage***)malloc(sz);
  memset(spatialScores, 0, sz);
  unsigned char *curr = (unsigned char*)(spatialScores+numScales);
  for(j = 0; j < numScales; j++) {
    spatialScores[j] = (IplImage**)curr;
    curr += sizeof(IplImage*)*numOrientations;
    for(k = 0; k < numOrientations; k++) {
      spatialScores[j][k] = (IplImage*)curr;
    }
  }

  float mu_x = loc_x + offset_x*ff;
  float mu_y = loc_y + offset_y*ff;
  unsigned char *ptr;
  float wS=0, wO=0, wY, *ptr2;
  int x, y;
  float offn, offn_s;
  int w = PART_WIDTH, h = PART_WIDTH;
  float s, mm[6];
  CvMat mat = cvMat(2, 3, CV_32FC1, mm);
  int num_o = feat->NumOrientations();
  int num_o2 = num_o/2;
  float dor, ds;
  bool visible = !GetParentPose()->IsNotVisible() && !GetChildPose(0)->IsNotVisible();

  // Compute spatial scores for the ground truth scale/orientation 
  feat->GetDetectionImageSize(&w, &h, scale, rotation);
  //offn = model->offset_norm*g/feat->Scale(scale);
  offn = offset_norm*g/*ss*/;
  offn_s = SQR(offn);
  feat->GetDetectionImageSize(&w, &h, scale, rotation);
  spatialScores[scale][rotation] = cvCreateImage(cvSize(w,h), IPL_DEPTH_32F, 1);
  ptr = (unsigned char*)spatialScores[scale][rotation]->imageData;
  if(!visible)
    cvZero(spatialScores[scale][rotation]);
  else {
    for(y = 0; y < h; y++, ptr += spatialScores[scale][rotation]->widthStep) {
      wY = -offn*wy*(y-mu_y);
      wY -= offn_s*wyy*(y-mu_y)*(y-mu_y);
      for(x = 0, ptr2=(float*)ptr; x < w; x++) {
        ptr2[x] = wY - offn*wx*(x-mu_x) - offn_s*wxx*(x-mu_x)*(x-mu_x);
      }
    }
  }


  for(j = 0; j < numScales; j++) {
    s = feat->Scale(j);
    offn = offset_norm*g;
    offn_s = SQR(offn);
    if(scale >= 0) {
      ds = (j - scale - my_round(offset_scale));
      wS = -ws*ds - wss*ds*ds;
    }
    for(k = 0; k < numOrientations; k++) {
      feat->GetDetectionImageSize(&w, &h, j, k);
      if(rotation >= 0) {
        dor = (k-(rotation+my_round(offset_rotation)+num_o)%num_o);
        while(dor > num_o2) dor -= num_o;
        while(dor < -num_o2) dor += num_o;
        wO = wS - wr*dor - wrr*dor*dor;
      }
      if(j != scale || k != rotation) {
        spatialScores[j][k] = feat->ConvertDetectionImageCoordinates(spatialScores[scale][rotation], 
								     scale, rotation, j, k, -10000000000000.0f);
        if(visible)
          cvConvertScale(spatialScores[j][k], spatialScores[j][k], 1, wO);
      }
    }
  }

  //cvSaveImage("tmp.png", MinMaxNormImage(spatialScores[0][0]));


  return spatialScores;
}


int ObjectPartPoseTransitionInstance::GetFeatures(float *f, PartLocation *parent_loc, PartLocation *child_loc, float *w) { 
  // If the part's predicted pose is something different than this one, zero out all features
  int i;
  float offset_norm, offset_x, offset_y, offset_scale, offset_rotation;
  GetOffsets(&offset_norm, &offset_x, &offset_y, &offset_scale, &offset_rotation); 

  int par_x, par_y, par_scale, par_rot, par_pose, par_dx, par_dy;
  int child_x, child_y, child_scale, child_rot, child_pose, child_dx, child_dy;
  parent_loc->GetDetectionLocation(&par_x, &par_y, &par_scale, &par_rot, &par_pose, NULL, NULL, &par_dx, &par_dy);
  child_loc->GetDetectionLocation(&child_x, &child_y, &child_scale, &child_rot, &child_pose, NULL, NULL, &child_dx, &child_dy);

  if(w) model->GetWeights(w);

  for(i = 0; i < model->NumWeights(); i++) 
    f[i] = 0;

  if(IS_LATENT(par_pose) || IS_LATENT(child_pose) || GetParentPart()->GetPose(par_pose) != GetParentPose() ||
     ChildPoseInd(GetChildPart()->GetPose(child_pose)) < 0) {
    return model->NumWeights();
  }

  if(!GetChildPose(0)->IsNotVisible() && !GetParentPose()->IsNotVisible()) {
    FeatureOptions *feat = process->Features();
    int num_o = feat->NumOrientations();
    int childPoseInd = ChildPoseInd(GetChildPart()->GetPose(child_pose));
    assert(childPoseInd >= 0);
    float g = (float)process->Features()->SpatialGranularity();
    FeatureOptions *fo = process->Features();
    float ss = par_scale < fo->ScaleOffset() ? 1 / fo->Scale(par_scale) : 1;
    float ff = 1.0f/*ss*//g;
    int num_o2 = num_o/2, dor;
    bool noSO = process->GetScaleOrientationMethod() == SO_PARENT_CHILD_SAME_SCALE_ORIENTATION || 
      process->GetScaleOrientationMethod() == SO_SCALE_ORIENTATION_NO_COST;
    
    // Set the feature values for costs related the relative offset between the parent and child
    if(IS_LATENT(child_dx)) {
      float x, y;  
      fo->ConvertDetectionCoordinates(child_x, child_y, child_scale, child_rot, par_scale, par_rot, &x, &y);
      f[0] = -(my_round(x)-par_x - (int)(offset_x*ff)) * g/*ss*/ * offset_norm;
      f[1] = -(my_round(y)-par_y - (int)(offset_y*ff)) * g/*ss*/ * offset_norm;
    } else {
      // Avoid rounding error quirks: use the same offsets used in distance transform computation
      f[0] = -child_dx*g/*ss*/*offset_norm;
      f[1] = -child_dy*g/*ss*/*offset_norm;
	  //float x, y;  
      //fo->ConvertDetectionCoordinates(child_x, child_y, child_scale, child_rot, par_scale, par_rot, &x, &y);
	  //assert(f[0] == -(my_round(x)-par_x - (int)(offset_x*ff)) * g/*ss*/ * offset_norm);
    }
    assert(!isReversed);
    if(isFlipped)
      f[0] = -f[0];
    f[4] = -SQR(f[0]);
    f[5] = -SQR(f[1]);
    if(noSO) f[2] = f[3] = f[6] = f[7] = 0;
    else {
      dor = child_rot - (par_rot + my_round(offset_rotation)+num_o)%num_o;
      while(dor > num_o2) dor -= num_o;
      while(dor < -num_o2) dor += num_o;
      f[2] = -(dor);
      f[3] = -(child_scale - par_scale - my_round(offset_scale));
      f[6] = -SQR(f[2]);
      f[7] = -SQR(f[3]);
    }
#if NO_TRANSLATION_BIAS
    f[0]=f[1]=f[2]=f[3]=0;
#endif
    f[8+childPoseInd] = -POSE_TRANSITION_FEATURE;
  } else
    f[0] = -POSE_TRANSITION_FEATURE;

  return model->NumWeights();
}

double g_score = 0;
float g_score2 = 0, *g_w2;
int ObjectPartPoseTransitionInstance::Debug(ObjectPoseInstance *parPose, float *w, float *f, PartLocation *parent_loc, PartLocation *child_loc, 
					    bool debug_scores, bool print_weights, float *f_gt) {
  int par_x, par_y, par_scale, par_rot, par_pose, par_dx, par_dy;
  int child_x, child_y, child_scale, child_rot, child_pose, child_dx, child_dy;
  parent_loc->GetDetectionLocation(&par_x, &par_y, &par_scale, &par_rot, &par_pose, NULL, NULL, &par_dx, &par_dy);
  child_loc->GetDetectionLocation(&child_x, &child_y, &child_scale, &child_rot, &child_pose, NULL, NULL, &child_dx, &child_dy);

  if(IS_LATENT(par_pose) || IS_LATENT(child_pose) || GetParentPart()->GetPose(par_pose) != GetParentPose() ||
     ChildPoseInd(GetChildPart()->GetPose(child_pose)) < 0) 
    return model->NumWeights();

  
  //float ff[100]; GetFeatures(ff, parent_loc, child_loc);
  int childPoseInd = ChildPoseInd(GetChildPart()->GetPose(child_pose));
  fprintf(stderr, " (%s[(%d,%d),%d,%d]->%s[(%d,%d),%d,%d]", GetParentPose()->Model()->Name(),par_x,par_y,par_scale,par_rot, 
	  GetChildPose(childPoseInd)->Model()->Name(), child_x,child_y,child_scale,child_rot);
  if(print_weights || 1) {
    for(int i = 0; i < model->NumWeights(); i++) {
      if(f_gt) {
        fprintf(stderr, " %.7f:%f:%f", w[i], f[i], f_gt[i]);
      } else
        fprintf(stderr, " %.7f:%f", w[i], f[i]);
      //assert(f[i] == ff[i]);
    }
  }
  

  if(debug_scores && !GetChildPose(childPoseInd)->IsNotVisible() && !GetParentPose()->IsNotVisible()) {
    int i;
    float score_wf = 0, score_gt = 0;
    if(custom_weights) w = custom_weights;

    for(i = 0; i < model->NumWeights(); i++) 
      score_wf += w[i]*f[i];

    if(f_gt)
      for(i = 0; i < model->NumWeights(); i++) 
        score_gt += w[i]*f_gt[i];
    
    g_score += score_wf;

    IplImage *resp = GetChildPose(childPoseInd)->GetResponse(child_scale, child_rot);
    IplImage *score = scores_no_wt[par_scale][par_rot];
    float score_app_spat = ((float*)(score->imageData+score->widthStep*par_y))[par_x];
    if(isReversed && model->NumChildPoses() > 1) {    
      float wt_later = 0;
      GetWeights(NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &wt_later, ChildPoseInd(parPose));
      score_app_spat -= POSE_TRANSITION_FEATURE*wt_later;
    } 
    float score_app = !resp ? 0 : ((float*)(resp->imageData+resp->widthStep*child_y))[child_x];
    float score_spat = score_app_spat - score_app;
    if(f_gt) fprintf(stderr, " w*f=%f r=%f r_gt=%f)",  score_wf,  score_spat,  score_gt);
    else fprintf(stderr, " w*f=%f r=%f)",  score_wf,  score_spat);

    if(my_abs(score_wf-score_spat)>=.0001) {
      float ff[100]; 
      GetChildPartLocation(parent_loc); 
      GetFeatures(ff, parent_loc, child_loc);
      int x = 1;
    }

    if(f_gt) assert(my_abs(score_wf-score_spat)<.0001);
    else assert((score_wf-score_spat)<.01);
  }
  return model->NumWeights();
}


