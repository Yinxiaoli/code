/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#include "part.h"
#include "pose.h"
#include "imageProcess.h"
#include "classes.h"
#include "class.h"
#include "attribute.h"
#include "spatialModel.h"
#include <assert.h>


ObjectPart::ObjectPart(const char *n, bool isC, const char *abbrev) {
  name = n ? StringCopy(n) : NULL;
  id = -1;
  parent = NULL;
  parent_name = NULL;
  isClick = isC;
  flipped = NULL;
  flipped_name = NULL;
  color = 0;

  poses = NULL;
  pose_names = NULL;
  numPoses = 0;

  parts = NULL;
  part_names = NULL;
  numParts = 0;

  gamma = 1;

  attributes = NULL;
  numAttributes = 0;

  childPartPoseTransitions = NULL;
  numChildPartPoseTransitions = NULL;
 
  visualization_image = NULL;

  abbreviation = abbrev ? StringCopy(abbrev) : NULL;

  memset(&staticFeatures, 0, sizeof(staticFeatures));
}

ObjectPart::~ObjectPart() {
  if(name) StringFree(name);

  if(poses)
    free(poses);
  if(pose_names) {
    for(int i = 0; i < numPoses; i++)
      free(pose_names[i]);
    free(pose_names);
  }
  if(parts)
    free(parts);

  if(part_names) {
    for(int i = 0; i < numParts; i++)
      free(part_names[i]);
    free(part_names);
  }

  if(parent_name)
    free(parent_name);

  if(flipped_name)
    free(flipped_name);

  if(attributes)
    free(attributes);

  if(childPartPoseTransitions) {
    for(int i = 0; i < numPoses; i++) {
      for(int j = 0; j <= numParts; j++) { 
        for(int k = 0; k < numChildPartPoseTransitions[i][j]; k++) 
          if(childPartPoseTransitions[i][j][k])
            delete childPartPoseTransitions[i][j][k];
        free(childPartPoseTransitions[i][j]);
      }
      free(childPartPoseTransitions[i]);
      free(numChildPartPoseTransitions[i]);
    }
    free(childPartPoseTransitions);
    free(numChildPartPoseTransitions);
  }

  if(visualization_image) free(visualization_image);

  if(abbreviation) free(abbreviation);

  if(staticFeatures.weights) free(staticFeatures.weights);
}


Json::Value ObjectPart::Save() {
  Json::Value root;

  root["name"] = name;
  root["id"] = id;
  if(parent) root["parent"] = parent->Name();
  root["gamma"] = gamma;
  root["isClick"] = isClick;
  if(abbreviation) root["abbreviation"] = abbreviation;
  if(visualization_image) root["visualization"] = visualization_image;
  if(flipped) root["flipped"] = flipped->Name();
  if(color) root["color"] = color;

  int i;
  Json::Value po;
  for(i = 0; i < NumPoses(); i++)
    po[i] = GetPose(i)->Name();
  root["poses"] = po;

  if(NumParts()) {
    Json::Value pa;
    for(i = 0; i < NumParts(); i++)
      pa[i] = GetPart(i)->Name();
    root["parts"] = pa;
  }

  if(NumStaticWeights()) 
    root["staticFeatures"] = SaveStaticFeatures();

  return root;
}

bool ObjectPart::Load(const Json::Value &root) {
  name = StringCopy(root.get("name", "").asString().c_str());
  id = root.get("id", -1).asInt();
  parent_name = root.isMember("parent") ? StringCopy(root.get("parent", "").asString().c_str()) : NULL;
  flipped_name = root.isMember("flipped") ? StringCopy(root.get("flipped", "").asString().c_str()) : NULL;
  gamma = root.get("gamma", 1).asFloat();
  isClick = root.get("isClick", false).asBool();
  color = root.get("color", 0).asInt();
  visualization_image = root.isMember("visualization") ? StringCopy(root["visualization"].asString().c_str()) : NULL;
  abbreviation = root.isMember("abbreviation") ? StringCopy(root["abbreviation"].asString().c_str()) : NULL;

  if(root.isMember("poses") && root["poses"].isArray()) {
    for(int i = 0; i < (int)root["poses"].size(); i++) {
      poses = (ObjectPose**)realloc(poses, sizeof(ObjectPose*)*(numPoses+1));
      poses[numPoses] = NULL;
      pose_names = (char**)realloc(pose_names, sizeof(char*)*(numPoses+1));
      pose_names[numPoses++] = StringCopy(root["poses"][i].asString().c_str());
    }
  }

  if(root.isMember("parts") && root["parts"].isArray()) {
    for(int i = 0; i < (int)root["parts"].size(); i++) {
       parts = (ObjectPart**)realloc(parts, sizeof(ObjectPart*)*(numParts+2));
      parts[numParts] = parts[numParts+1] = NULL;
      part_names = (char**)realloc(part_names, sizeof(char*)*(numParts+1));
      part_names[numParts++] = StringCopy(root["parts"][i].asString().c_str());
    }
  }

  if(staticFeatures.weights) free(staticFeatures.weights);
  memset(&staticFeatures, 0, sizeof(staticFeatures));
  if(root.isMember("staticFeatures")) 
    if(!LoadStaticFeatures(root["staticFeatures"]))
      return false;

  return true;
}


bool ObjectPart::ResolveLinks(Classes *c) {
  int i;
  classes = c;

  for(i = 0; i < numPoses; i++) {
    if(!(poses[i] = (isClick ? classes->FindClickPose(pose_names[i]) : classes->FindPose(pose_names[i])))) {
      fprintf(stderr, "Part %d(%s) couldn't resolve pose %s\n", id, name, pose_names[i]);
      return false;
    }
  }

  for(i = 0; i < numParts; i++) {
    assert(!isClick);
    if(!(parts[i] = classes->FindPart(part_names[i]))) {
      fprintf(stderr, "Part %d(%s) couldn't resolve part %s\n", id, name, part_names[i]);
      return false;
    }
    parts[i]->SetParent(this);
  }

  if(parent_name) {
    if(!(parent = classes->FindPart(parent_name))) {
      fprintf(stderr, "Part %d(%s) couldn't resolve parent %s\n", id, name, parent_name);
      return false;
    }
    parts = (ObjectPart**)realloc(parts, sizeof(ObjectPart*)*(numParts+1));
    parts[numParts] = parent;
  }

  if(flipped_name) {
    if(!(flipped = classes->FindPart(flipped_name))) {
      fprintf(stderr, "Part %d(%s) couldn't resolve flipped %s\n", id, name, flipped_name);
      return false;
    }
  }

  if(!staticFeatures.weights && NumStaticWeights())
    staticFeatures.weights = (float*)malloc(sizeof(float)*NumStaticWeights());

  return true;
}

void ObjectPart::AddSpatialTransition(ObjectPartPoseTransition *t) {
  int pa = PartInd(t->childPart);
  int po = PoseInd(t->parentPose);
  assert(!t->parentPose->IsFlipped());

  if(!childPartPoseTransitions) {
    childPartPoseTransitions = (ObjectPartPoseTransition****)malloc(sizeof(ObjectPartPoseTransition***)*numPoses);
    numChildPartPoseTransitions = (int**)malloc(sizeof(int*)*numPoses);
    for(int i = 0; i < numPoses; i++) {
      childPartPoseTransitions[i] = (ObjectPartPoseTransition***)malloc(sizeof(ObjectPartPoseTransition**)*(numParts+1));
      memset(childPartPoseTransitions[i], 0, sizeof(ObjectPartPoseTransition**)*(numParts+1));
      numChildPartPoseTransitions[i] = (int*)malloc(sizeof(int)*(numParts+1));
      memset(numChildPartPoseTransitions[i], 0, sizeof(int)*(numParts+1));
    }
  }

  childPartPoseTransitions[po][pa] = (ObjectPartPoseTransition**)realloc(childPartPoseTransitions[po][pa],
                        sizeof(ObjectPartPoseTransition*)*(numChildPartPoseTransitions[po][pa]+1));
  childPartPoseTransitions[po][pa][numChildPartPoseTransitions[po][pa]] = t;
  numChildPartPoseTransitions[po][pa]++;
}

void ObjectPart::AddSpatialTransitions(PartLocation *locs, PartLocation *parent_loc,  bool computeSigma, bool combinePoses) {
  if(!parent_loc) parent_loc = locs+id;

  assert(parent_loc);
  int parent_x, parent_y, parent_scale, parent_rot, parent_pose;
  int w, h;
  parent_loc->GetImageSize(&w, &h);
  Classes *classes = parent_loc->GetClasses();
  parent_loc->GetDetectionLocation(&parent_x, &parent_y, &parent_scale, &parent_rot, &parent_pose);
  //if(poses[parent_pose]->IsNotVisible() && !IsClick())
  //return;

  int num_o = classes->GetFeatureParams()->numOrientations;
  int num_o2 = num_o/2;
  for(int i = 0; i <= numParts; i++) {
    if(!parts || !parts[i]) continue;
	
    int child_x, child_y, child_scale, child_rot, child_pose;
    locs[parts[i]->id].GetDetectionLocation(&child_x, &child_y, &child_scale, &child_rot, &child_pose);	
   		
    ObjectPartPoseTransition *r = NULL;
    int childPoseInd = -1;
    for(int j = 0; childPartPoseTransitions && j < numChildPartPoseTransitions[parent_pose][i]; j++) {
      int ind = childPartPoseTransitions[parent_pose][i][j]->ChildPoseInd(parts[i]->poses[child_pose]);
      if(ind != -1 || (combinePoses && !parts[i]->poses[child_pose]->IsNotVisible() && 
                       !childPartPoseTransitions[parent_pose][i][j]->GetChildPose(0)->IsNotVisible())) {
        r = childPartPoseTransitions[parent_pose][i][j];
        childPoseInd = ind;
        if(ind == -1) {
          childPoseInd = childPartPoseTransitions[parent_pose][i][j]->AddChildPose(parts[i]->poses[child_pose]);
	}
        break;
      }
    }
    if(!r) {
      assert(!computeSigma);
      r = new ObjectPartPoseTransition();
      r->parentPart = this; r->parentPose = poses[parent_pose]; r->childPart = parts[i]; r->isClick = isClick;
	  childPoseInd = r->AddChildPose(parts[i]->poses[child_pose]); 
      r->offset_orientation_count = (int*)malloc(sizeof(int)*num_o);
      r->max_change_x = r->max_change_y = r->max_change_scale = r->max_change_rot = 0;
      memset(r->offset_orientation_count, 0, sizeof(int)*num_o);
      classes->AddSpatialTransition(r);
      AddSpatialTransition(r);
    }
    float s = parent_scale < classes->GetFeatureParams()->scaleOffset ? 1/classes->Scale(parent_scale) : 1;
    float g = classes->SpatialGranularity();
    float x, y;  
    classes->ConvertDetectionCoordinates(child_x, child_y, child_scale, child_rot, parent_scale, parent_rot, w, h, &x, &y);
    float dx = (my_round(x) - parent_x)*g; //*s;
    float dy = (my_round(y) - parent_y)*g; //*s;
    float ds = (child_scale - parent_scale);

    if(computeSigma) {
      int dor = child_rot-(parent_rot+my_round(r->offset_orientation)+num_o)%num_o;
      while(dor > num_o2) dor -= num_o;
      while(dor < -num_o2) dor += num_o;
      assert(r->numTrainExamples[childPoseInd]);
      if(!r->GetChildPose(0)->IsNotVisible() && !r->parentPose->IsNotVisible()) {
        r->W[4] += SQR(dx-r->offset.x);  
        r->W[5] += SQR(dy-r->offset.y);
        r->W[6] += SQR(dor);
        r->W[7] += SQR(ds-r->offset_scale);
      }
      r->max_change_x = my_max(my_abs(dx-r->offset.x), r->max_change_x);
      r->max_change_y = my_max(my_abs(dy-r->offset.y), r->max_change_y);
      r->max_change_scale = my_max(ceil(my_abs(ds-r->offset_scale)), r->max_change_scale);
      r->max_change_rot = my_max((my_abs(dor)), r->max_change_rot);
    } else {
      // Update the sum squared angle difference for every possible choice of mean angle offset
      for(int k = -num_o2; k < num_o-num_o2; k++) {
        int dor = child_rot-(parent_rot+k+num_o)%num_o;
        while(dor > num_o2) dor -= num_o;
        while(dor < -num_o2) dor += num_o;
        r->offset_orientation_count[k+num_o2] += SQR(dor);
      }

      if(!r->GetChildPose(childPoseInd)->IsNotVisible() && !r->parentPose->IsNotVisible()) {
        r->offset.x += dx;
        r->offset.y += dy;
        r->offset_scale += ds;
      }
      r->numTrainExamples[childPoseInd]++;
    }
  }
}

void ObjectPart::NormalizePartPoseTransitions(int minExamples, bool computeSigma) {
  int i, j, k, l;
  int num_o = classes->GetFeatureParams()->numOrientations;
  int num_o2 = num_o/2;
  for(i = 0; i < numPoses; i++) {
    for(j = 0; j <= numParts; j++) {
      int numGood = 0;
      int numPose = 0;
      for(k = 0; childPartPoseTransitions && k < numChildPartPoseTransitions[i][j]; k++)
        for(l = 0; l < childPartPoseTransitions[i][j][k]->NumChildPoses(); l++)
          numPose += childPartPoseTransitions[i][j][k]->numTrainExamples[l];
      for(k = 0; childPartPoseTransitions && k < numChildPartPoseTransitions[i][j]; k++) {
        ObjectPartPoseTransition *r = childPartPoseTransitions[i][j][k];
        int numTrainExamples = 0;
        for(l = 0; l < childPartPoseTransitions[i][j][k]->NumChildPoses(); l++)
          numTrainExamples += r->numTrainExamples[l];
        if(computeSigma) {
          if(r->NumWeights() > 1) {
            assert(r->NumWeights()==8+r->NumChildPoses());
          
            if(numTrainExamples) { 
              r->W[4] /= numTrainExamples; 
              r->W[5] /= numTrainExamples; 
              r->W[6] /= numTrainExamples; 
              r->W[7] /= numTrainExamples; 
            }
            r->offset_norm = sqrt(r->W[4] + r->W[5]); 
            r->offset_norm = (r->offset_norm&&!isClick) ? 1.0f/r->offset_norm/(NORMALIZE_TEMPLATES ? 21 : 1) : 1;
            float offn = SQR(r->offset_norm);

            //fprintf(stderr, "%s->%s: %f(%f) %f(%f)\n", r->childPose->Name(), r->parentPose->Name(), r->offset.x, sqrt(r->W[4]/r->numTrainExamples), r->offset.y, sqrt(r->W[5]/r->numTrainExamples));

            r->W[4] = numTrainExamples ? my_min(10,1.0f / (offn*r->W[4]*2)) : 0;   
            r->W[5] = numTrainExamples ? my_min(10,1.0f / (offn*r->W[5]*2)) : 0;
            r->W[6] = numTrainExamples ? my_min(10,1.0f / (r->W[6]*2)) : 0;   
            r->W[7] = numTrainExamples ? my_min(10,1.0f / (r->W[7]*2)) : 0;
          } else {
            assert(r->NumWeights()==1);
          }
          for(l = 0; l < childPartPoseTransitions[i][j][k]->NumChildPoses(); l++)
            r->W[8+l] = r->numTrainExamples[l] ? -log(r->numTrainExamples[l]/(float)numPose)/POSE_TRANSITION_FEATURE : 0;
        } else if(numTrainExamples >= minExamples) {
          r->offset.x /= numTrainExamples;
          r->offset.y /= numTrainExamples;
          r->offset_scale /= numTrainExamples;
          r->offset_orientation /= numTrainExamples;

          //fprintf(stderr, "%s->%s: %f %f\n", r->childPose->Name(), r->parentPose->Name(), r->offset.x, r->offset.y);

          // Select the mean angle offset by finding the offset that minimizes the sum squared difference
          int best = 1<<30;
          for(int l = 0; l < num_o; l++) {
            if(r->offset_orientation_count[l] < best) {
              best = r->offset_orientation_count[l];
              r->offset_orientation = l-num_o2;
            }
          }

          childPartPoseTransitions[i][j][numGood++] = r;
        } else
          delete r;
      }
      if(numChildPartPoseTransitions && !computeSigma) 
        numChildPartPoseTransitions[i][j] = numGood;
    }
  }
}


void ObjectPart::AddPose(ObjectPose *pose) {
  poses = (ObjectPose**)realloc(poses, sizeof(ObjectPose*)*(numPoses+1));
  poses[numPoses++] = pose;
}

void ObjectPart::AddPart(ObjectPart *part) {
  part->parent = this;
  parts = (ObjectPart**)realloc(parts, sizeof(ObjectPart*)*(numParts+2));
  parts[numParts++] = part;
  parts[numParts] = parent;
  part->parts = (ObjectPart**)realloc(part->parts, sizeof(ObjectPart*)*(part->numParts+1));
  part->parts[part->numParts] = this;
}

int ObjectPart::NumStaticWeights() { 
  FeatureParams *params = classes->GetFeatureParams();
  int n = (staticFeatures.useBiasTerm ? 2 : 0) + 
    (staticFeatures.useScalePrior ? params->numScales : 0) +
    (staticFeatures.useOrientationPrior ? params->numOrientations : 0);
  for(int i = 1; i < staticFeatures.numSpatialLocationLevels; i++) 
    n += (1<<(2*i));
  return n;
}




int ObjectPart::GetStaticWeightConstraints(int *w, bool *learn_weights, bool *regularize) {
  int num = NumStaticWeights();
  for(int i = 0; i < num; i++) {
    learn_weights[i] = regularize[i] = true;
    w[i] = 1;
  }
  return num;
}

float ObjectPart::MaxFeatureSumSqr() {
  float n = 0;
  int num = numParts + (isClick ? 1 : 0);
  for(int i = 0; i < numPoses; i++)
    for(int j = 0; j < num; j++)
      for(int k = 0; k < numChildPartPoseTransitions[i][j]; k++)
        n += childPartPoseTransitions[i][j][k]->MaxFeatureSumSqr();
  
  return n;
}


void ObjectPart::SetStaticWeights(float *w) {
  int numW = NumStaticWeights();
  if(numW) 
    memcpy(staticFeatures.weights, w, numW*sizeof(float));
}




ObjectPartInstance::ObjectPartInstance(ObjectPart *m, ImageProcess *p) {
  model = m;
  parts = (ObjectPartInstance**)malloc(sizeof(ObjectPartInstance*) * (model->NumParts()+1));
  memset(parts, 0, sizeof(ObjectPartInstance*) * (model->NumParts()+1));

  process = p;

  theta.SetPart(model);

  delta = 0;
  lower_bound = 0;
  maxScore = 0;
  cumulativeSums = NULL;
  cumulativeSumMaps = NULL;
  cumulativeSumTotal = 0;
  
  cumulativeSumsNMS_keypoints = NULL;
  cumulativeSumsNMS_numKeypoints = 0;
  cumulativeSumsNMS_sum = 0;

  isObserved = isAnchor = false;

  responses = NULL;
  pose_responses = NULL;

  clickPart = NULL;
  customStaticWeights = NULL;

  poses = (ObjectPoseInstance**)malloc(sizeof(ObjectPoseInstance*)*model->NumPoses());
  for(int i = 0; i < model->NumPoses(); i++)
    poses[i] = new ObjectPoseInstance(model->GetPose(i), this, process);
}

ObjectPartInstance::~ObjectPartInstance() {
  FreeResponses();

  if(parts) 
    free(parts);

  if(poses) {
    for(int i = 0; i < model->NumPoses(); i++)
      delete poses[i];
    free(poses);
  }
}

ObjectPoseInstance *ObjectPartInstance::FindPose(int id) { 
  for(int i = 0; i < model->NumPoses(); i++) { 
    if(poses[i]->Model()->Id() == id) 
      return poses[i]; 
  } 
  return NULL; 
}


ObjectPartInstance *ObjectPartInstance::GetParent() { 
  return model->GetParent() ? process->GetPartInst(model->GetParent()->Id()) : NULL; 
}

void ObjectPartInstance::UseLoss(bool u) {
  for(int i = 0; i < model->NumPoses(); i++)
    poses[i]->UseLoss(u);
}

void ObjectPartInstance::ResolveLinks() {
  int i;
  
  for(i = 0; i <= model->NumParts(); i++) {
    assert(i==model->NumParts() || !model->IsClick());
    parts[i] = model->GetPart(i) ? process->GetPartInst(model->GetPart(i)->Id()) : NULL;
  }

  for(i = 0; i < model->NumPoses(); i++)
    poses[i]->ResolveLinks();
}

void ObjectPartInstance::ResolveParentLinks() {
  for(int i = 0; i < model->NumPoses(); i++)
    poses[i]->ResolveParentLinks();
}

void ObjectPartInstance::FreeResponses() {
  int i, j;

  FeatureParams *features = process ? process->Features()->GetParams() : NULL;
  if(responses) {
    for(i = 0; i < features->numScales; i++) {
      for(j = 0; j < features->numOrientations; j++) 
	cvReleaseImage(&responses[i][j]);
    }
    free(responses);
    responses = NULL;
  }

  if(pose_responses) {
    /*for(k = 0; k < model->NumPoses(); k++) 
      for(i = 0; i < features->numScales; i++) 
	for(j = 0; j < features->numOrientations; j++) 
	  cvReleaseImage(&pose_responses[k][i][j]);
    */
    free(pose_responses);
    pose_responses = NULL;
  }

  ClearCumulativeSums();
}

void ObjectPartInstance::ClearCumulativeSums() {
  if(cumulativeSums) {
    int i, j, k;
    FeatureParams *features = process ? process->Features()->GetParams() : NULL;
    for(k = 0; k < model->NumPoses(); k++) {
      for(i = 0; i < features->numScales; i++) { 
	for(j = 0; j < features->numOrientations; j++) {
	  cvReleaseImage(&cumulativeSumMaps[k][i][j]);
	}
	free(cumulativeSumMaps[k][i]); free(cumulativeSums[k][i]);
      }
      free(cumulativeSumMaps[k]); free(cumulativeSums[k]);
    }
    free(cumulativeSumMaps); free(cumulativeSums);
    cumulativeSums = NULL;
    cumulativeSumMaps = NULL;
    cumulativeSumTotal = 0;
  }
  
  if(cumulativeSumsNMS_keypoints) {
    for(int i = 0; i < cumulativeSumsNMS_numKeypoints; i++) delete cumulativeSumsNMS_keypoints[i];
    free(cumulativeSumsNMS_keypoints);
  }
  cumulativeSumsNMS_keypoints = NULL;
  cumulativeSumsNMS_numKeypoints = 0;
  cumulativeSumsNMS_sum = 0;
}

void ObjectPartInstance::Clear(bool clearResponses, bool clearPose) {
  if(clearResponses)
    FreeResponses();

  //fprintf(stderr, "Clear %s\n", model->Name());
  ClearCumulativeSums();
  
  if(clearPose)
    for(int i = 0; i < model->NumPoses(); i++)
      poses[i]->Clear(true, true, clearResponses);
}


// Greedy non-maximal suppression that is applicable toward picking a set of detected locations that are
// high scoring and fairly independent from one another (e.g. have different location, scale, or pose)
PartLocation **ObjectPartInstance::GreedyNonMaximalSuppression(int num_samples, double min_score, int *num, 
							       double suppressWidth, double suppressScale, 
							       double suppressOrientation, bool suppressAllPoses,
							       bool getDiversePoses) {
  FeatureOptions *feat = process->Features();
  int image_width = feat->GetImage()->width, image_height = feat->GetImage()->height;
  int numScales = feat->NumScales(), numOrientations = feat->NumOrientations();
  CvPoint ptMin, ptMax;
  int ***best_x = Create3DArray<int>(model->NumPoses(), numScales, numOrientations);
  int ***best_y = Create3DArray<int>(model->NumPoses(), numScales, numOrientations);
  double ***best_scores = Create3DArray<double>(model->NumPoses(), numScales, numOrientations);
  int p, s, o;
  double mi, ma;
  Classes *classes = model->GetClasses();
  int numSuppressScales = my_round(LOG_B(suppressScale, feat->GetParams()->hogParams.subsamplePower));
  int numSuppressOris = my_round(numOrientations*suppressOrientation/M_PI);
  PartLocation **samples = new PartLocation*[num_samples];
  int id = model->Id();
  int *num_used_pose = new int[model->NumPoses()];
  *num = 0;

  for(p = 0; p < model->NumPoses(); p++) {
    num_used_pose[p] = 0;
    for(s = 0; s < numScales; s++) 
      for(o = 0; o < numOrientations; o++) {
	if(pose_responses[p][s][o]) {
	  cvMinMaxLoc(pose_responses[p][s][o], &mi, &best_scores[p][s][o], &ptMin, &ptMax);
	  best_x[p][s][o] = ptMax.x;
	  best_y[p][s][o] = ptMax.y;
	} else
	  best_scores[p][s][o] = -INFINITY;
      }
  }

  for(int i = 0; i < num_samples; i++) {
    // Find the best scoring part location
    CvPoint best_loc;
    int best_pose=-1, best_scale, best_ori;
    double best_score = min_score;
    int min_used = num_samples+1;
    for(p = 0; p < model->NumPoses(); p++) {
      for(s = 0; s < numScales; s++) {
	for(o = 0; o < numOrientations; o++) {
	  if(pose_responses[p][s][o] && best_scores[p][s][o] > best_score && (!getDiversePoses || num_used_pose[p] <= min_used)) {
	    best_score = best_scores[p][s][o];
	    best_pose = p;
	    best_scale = s;
	    best_ori = o;
	    best_loc = cvPoint(best_x[p][s][o],best_y[p][s][o]);
	    min_used = num_used_pose[p];
	  }
	}
      }
    }
    if(best_score <= min_score)
      break;
    assert(best_pose != -1);

    num_used_pose[best_pose]++;

    cvMinMaxLoc(pose_responses[best_pose][best_scale][best_ori], &mi, &best_scores[best_pose][best_scale][best_ori], &ptMin, &ptMax); 
    best_x[best_pose][best_scale][best_ori] = ptMax.x;
    best_y[best_pose][best_scale][best_ori] = ptMax.y;
    samples[i] = PartLocation::NewPartLocations(classes, image_width, image_height, feat, false);
    samples[i][id].SetDetectionLocation(best_loc.x, best_loc.y, best_scale, best_ori, best_pose, LATENT, LATENT);
    samples[i][id].SetScore(best_score);
    poses[best_pose]->ExtractPartLocations(samples[i], &samples[i][id], GetParent());

    // Suppress all "overlapping" sets of part locations
    int width = poses[best_pose]->Model()->Appearance() ? poses[best_pose]->Model()->Appearance()->Width() : 10;
    for(p = 0; p < model->NumPoses(); p++) {
      if(suppressAllPoses || p == best_pose) {
	for(s = my_max(0,best_scale-numSuppressScales); s < my_min(numScales,best_scale+numSuppressScales+1); s++) {
	  for(o = my_max(0,best_ori-numSuppressOris); o < my_min(numOrientations,best_ori+numSuppressOris+1); o++) {
	    float xx, yy;
	    float w = suppressWidth*width*feat->Scale(s)/feat->Scale(best_scale);
	    IplImage *r = pose_responses[p][s][o];
	    if(!r) continue;
	    feat->ConvertDetectionCoordinates(best_loc.x, best_loc.y, best_scale, best_ori, s, o, &xx, &yy);
	    int xStart = (int)my_max(0,xx-w),
	      xEnd = (int)my_min(r->width,xx+width*feat->Scale(s)/feat->Scale(best_scale)+.99),
	      yStart = (int)my_max(0,yy-width*feat->Scale(s)/feat->Scale(best_scale)),
	      yEnd = (int)my_min(r->height,yy+width*feat->Scale(s)/feat->Scale(best_scale)+.99);
	    for(int y = yStart; y < yEnd; y++) {
	      float *ptr = ((float*)(r->imageData+y*r->widthStep));
	      for(int x = xStart; x < xEnd; x++) 
		ptr[x] = -INFINITY;
	    }
	  }
	}
      }
    }
    (*num)++;
  }
 
    

  free(best_x);
  free(best_y);
  free(best_scores);
  delete [] num_used_pose;
  return samples;
}

// Greedy non-maximal suppression that is applicable toward picking a set of detected locations that are
// high scoring and don't have overlapping bounding boxes
PartLocation **ObjectPartInstance::GreedyNonMaximalSuppressionByBoundingBox(int num_samples, double min_score, int *num) {
  FeatureOptions *feat = process->Features();
  Classes *classes = model->GetClasses();
  int num_boxes;
  int wi, he;
  process->Features()->GetDetectionImageSize(&wi, &he, 0, 0);
  int image_width = feat->GetImage()->width, image_height = feat->GetImage()->height;
  PartLocation **samples = new PartLocation*[num_samples];
  CvRectScore *boxes = GetBoundingBoxes(&num_boxes, true, true, false);
  int num_boxes_nms = NonMaximalSuppression(boxes, num_boxes, DEFAULT_OVERLAP, wi, he);
  int i;
  int id = model->Id();
  *num = 0;
  for(i = 0; i < num_boxes_nms && i < num_samples; i++) {
    //if(y_gt && !y_gt_locs) boxes[i].score += maxLoss;
    if(i == 0 || boxes[i].score - min_score > 0) {
      samples[i] = (PartLocation*)boxes[i].data;
      if(!samples[i]) {
	samples[i] = PartLocation::NewPartLocations(classes, image_width, image_height, feat, false);
	samples[i][id].SetDetectionLocation(boxes[i].det_x, boxes[i].det_y, boxes[i].scale_ind, boxes[i].rot_ind, LATENT, LATENT, LATENT);
	ExtractPartLocations(samples[i], &samples[i][id]);
	boxes[i].data = NULL;
      }
      samples[i][id].SetScore(boxes[i].score);
      *num = *num+1;
    }
  }
  for(i = 0; i < num_boxes; i++)
    if(boxes[i].data)
      delete [] (PartLocation*)boxes[i].data;
  free(boxes);
  return samples;
}

CvRectScore *ObjectPartInstance::GetBoundingBoxes(int *num, bool includeThisPart, bool storePartLocations, bool computeBoundingFromParts) {
  assert(responses);

  int s, o;
  FeatureOptions *feat = process->Features();
  CvRectScore *rects = NULL;
  int im_w = feat->GetImage()->width, im_h = feat->GetImage()->height;
  PartLocation *locs_static = PartLocation::NewPartLocations(process->GetClasses(), im_w, im_h, feat, false);
  PartLocation loc(this, im_w, im_h);
  int i = 0, x, y, xx, yy, scale, rotation;
  int id = Id();
  float ix, iy, width, height;
  PartLocation *locs = locs_static;
  int numScales = feat->NumScales(), numOrientations = feat->NumOrientations();

  for(s = 0; s < numScales; s++) {
    for(o = 0; o < numOrientations; o++) {
      int w, h; 
      feat->GetDetectionImageSize(&w, &h, s, o);
      rects = (CvRectScore*)realloc(rects,sizeof(CvRectScore)*(i+w*h));
      for(y = 0; y < h; y++) {
	for(x = 0; x < w; x++, i++) {
	  rects[i].det_x = x;
	  rects[i].det_y = y;
	  rects[i].scale_ind = s;
	  rects[i].rot_ind = o;
	  if(computeBoundingFromParts) {
	    if(storePartLocations) 
	      rects[i].data = locs = PartLocation::NewPartLocations(process->GetClasses(), im_w, im_h, feat, false);
	    locs[id].SetDetectionLocation(x, y, s, o, LATENT, LATENT, LATENT);
	    ExtractPartLocations(locs, &locs[id], NULL);
	    locs[id].GetDetectionLocation(&xx, &yy, &scale, &rotation);
	    GetBoundingBox(&rects[i].rect, locs, includeThisPart);
	  } else {
	    loc.SetDetectionLocation(x, y, s, o, LATENT, LATENT, LATENT);
	    PredictLatentValues(&loc);
	    loc.GetImageLocation(&ix, &iy, NULL, NULL, NULL, &width, &height);
	    loc.GetDetectionLocation(&xx, &yy, &scale, &rotation);
	    rects[i].rect.x = my_round(ix-width/2);
	    rects[i].rect.y = my_round(iy-height/2);
	    rects[i].rect.width = my_round(width);
	    rects[i].rect.height = my_round(height);
	    rects[i].data = NULL;
	  }
	  IplImage *r = responses[scale][rotation];
	  rects[i].score = ((float*)(r->imageData+r->widthStep*yy))[xx];
	  locs[id].SetScore(rects[i].score);
	}
      }
    }
  }
  delete [] locs_static;

  *num = i;
  return rects;
}
 
void ObjectPartInstance::GetBoundingBox(CvRect *rect, PartLocation *locs, bool includeThisPart, float s) {
  rect->x = rect->y = 100000000;
  rect->width = rect->height = -100000000;

  // Find the bounding box around all part bounding boxes
  for(int i = 0; i < process->GetClasses()->NumParts(); i++) {
    if(includeThisPart || model->Id() != i) {
      float matf[6], x[4], y[4];
      CvMat mat = cvMat(2, 3, CV_32FC1, matf);
      float lx, ly, lscale, lrot, width, height;
      locs[i].GetImageLocation(&lx, &ly, &lscale, &lrot, NULL, &width, &height);
      cv2DRotationMatrix(cvPoint2D32f(0,0), -lrot*180/M_PI, 1, &mat);
      AffineTransformPoint(matf, (float)-width*s/2, (float)-height*s/2, &x[0], &y[0]);
      AffineTransformPoint(matf, (float)width*s/2, (float)-height*s/2, &x[1], &y[1]);
      AffineTransformPoint(matf, (float)width*s/2, (float)height*s/2, &x[2], &y[2]);
      AffineTransformPoint(matf, (float)-width*s/2, (float)height*s/2, &x[3], &y[3]);
      for(int j = 0; j < 4; j++) {
	if(lx+x[j] < rect->x) rect->x = lx+x[j];
	if(ly+y[j] < rect->y) rect->y = ly+y[j];
	if(lx+x[j] > rect->width) rect->width = lx+x[j];
	if(ly+y[j] > rect->height) rect->height = ly+y[j];
      }
    }
  }

  IplImage *img = process->Image();
  rect->x = my_max(0, rect->x);
  rect->y = my_max(0, rect->y);
  rect->width = my_min(rect->width, img->width);
  rect->height = my_min(rect->height, img->height);
  rect->width -= rect->x;
  rect->height -= rect->y;
}

void ObjectPartInstance::ZeroScoreMap(PartLocation loc, float s) {
  int w, h;
  FeatureOptions *fo = process->Features();
  FeatureParams *params = fo->GetParams();
  int loc_x, loc_y, loc_scale, loc_rot;
  float loc_width, loc_height;
  loc.GetDetectionLocation(&loc_x, &loc_y, &loc_scale, &loc_rot, NULL, &loc_width, &loc_height);
  float fs = fo->Scale(loc_scale);
  float ss = fs*fo->CellWidth() / s;

  fo->GetDetectionImageSize(&w, &h, loc_scale, loc_rot);
    
  IplImage *zeroed = cvCreateImage(cvSize(w,h), IPL_DEPTH_32F, 0);
  cvZero(zeroed);
  for(int y = my_max(0,floor(loc_y-loc_height/ss)); y < my_min(h,ceil(loc_y+loc_height/ss)); y++) {
    float *ptr = ((float*)(zeroed->imageData+y*zeroed->widthStep));
    for(int x = my_max(0,floor(loc_x-loc_width/ss)); x < my_min(w,ceil(loc_x+loc_width/ss)); x++) 
      ptr[x] = -10000000;
  }

  theta.SetScore(-10000000);
  for(int i = 0; i < model->NumPoses(); i++) {
    if(pose_responses && pose_responses[i]) {
      // Update the best scoring detection for any pose or pixel location in this image
      CvPoint min_loc, max_loc;
      double min_val, max_val;
      for(int scale = 0; scale < params->numScales; scale++) {
	for(int rot = 0; rot < params->numOrientations; rot++) {
	  if(!pose_responses[i][scale][rot]) continue;

	  // Warp the zeroed out score map in the original scale/orientation to this one
	  IplImage *curr = fo->ConvertDetectionImageCoordinates(zeroed, loc_scale, loc_rot, scale, rot, 0); 
	  cvAdd(pose_responses[i][scale][rot], curr, pose_responses[i][scale][rot]);
	  if(responses && responses[scale][rot]) cvMax(pose_responses[i][scale][rot], responses[scale][rot], responses[scale][rot]);
	  cvReleaseImage(&curr);
	
	  // Find the new best score
	  cvMinMaxLoc(pose_responses[i][scale][rot], &min_val, &max_val, &min_loc, &max_loc);
	  if(max_val > theta.GetScore()) {
	    theta.SetDetectionLocation(loc_x, loc_y, loc_scale, loc_rot, i, LATENT, LATENT);
	    theta.SetScore((float)max_val);
	    theta.SetResponseTime(0);
	  }
	}
      }
    }
  }
  cvReleaseImage(&zeroed);

  if(responses && model->NumStaticWeights()) {
    theta.SetScore(-1000000000000.0);
    CvPoint min_loc, max_loc;
    double min_val, max_val;
    for(int scale = 0; scale < params->numScales; scale++) {
      for(int rot = 0; rot < params->numOrientations; rot++) { 
	if(responses[scale][rot]) {
	  AddStaticFeaturesToDetectionMap(responses[scale][rot], scale, rot);
	  cvMinMaxLoc(responses[scale][rot], &min_val, &max_val, &min_loc, &max_loc);
	  if(max_val > theta.GetScore()) {
	    this->theta.SetScore((float)max_val);
	    float best = -100000000;
	    int pose = LATENT;
	    for(int i = 0; i < model->NumPoses(); i++) {
	      float f = ((float*)(pose_responses[i][scale][rot]->imageData+responses[scale][rot]->widthStep*max_loc.y))[max_loc.x];
	      if(f > best) { pose = i; best = f; }
	    }
            this->theta.SetDetectionLocation(max_loc.x, max_loc.y, scale, rot, pose, LATENT, LATENT);
	  }
	}
      }
    }
  }
}

float ObjectPartInstance::ScoreAt(PartLocation *l) {
  int loc_x, loc_y, loc_scale, loc_rot, loc_pose;
  l->GetDetectionLocation(&loc_x, &loc_y, &loc_scale, &loc_rot, &loc_pose);

  if(IS_LATENT(loc_pose))
    return ((float*)(responses[loc_scale][loc_rot]->imageData+responses[loc_scale][loc_rot]->widthStep*loc_y))[loc_x];
  else
    return ((float*)(pose_responses[loc_pose][loc_scale][loc_rot]->imageData+responses[loc_scale][loc_rot]->widthStep*loc_y))[loc_x];
}

IplImage ***ObjectPartInstance::Detect(bool isRoot) {
  if(IsObserved()) return NULL;  // an observed click point

  FeatureOptions *feat = process->Features();
  FeatureParams *params = feat->GetParams();
  if(g_debug > 1) fprintf(stderr, "  Detecting part instance %s %s...\n", feat->Name(), model->Name());

  if(responses)
    return responses;
  theta.Clear();
  
  theta.Init(process->GetClasses(), process->Image()->width,process->Image()->height, feat);
  theta.SetScore(-1000000000000.0);
  theta.SetPart(model);

  int numPoses = model->NumPoses();
  //ObjectPose **poseModels = model->GetPoses(&numPoses);

  if(!responses && (isRoot || model->NumStaticWeights())) {
    int memSize = params->numScales*(sizeof(IplImage**) + sizeof(IplImage*)*params->numOrientations);
    responses = (IplImage***)malloc(memSize);
    memset(responses, 0, memSize);
  }
  if(!pose_responses)
    pose_responses = (IplImage****)malloc(sizeof(IplImage***)*numPoses);

  for(int i = 0; i < numPoses; i++) {
    //if(poses[i]->IsNotVisible()) 
      //continue;

    pose_responses[i] = poses[i]->Detect(isRoot ? NULL : GetParent());

    if(responses && pose_responses[i]) {
      // Update the best scoring detection for any pose or pixel location in this image
      CvPoint min_loc, max_loc;
      double min_val, max_val;
      for(int scale = 0; scale < params->numScales; scale++) {
	if(!i) responses[scale] = ((IplImage**)(responses+params->numScales))+scale*params->numOrientations;
	for(int rot = 0; rot < params->numOrientations; rot++) {
	  if(!pose_responses[i][scale][rot]) continue;

	  if(!responses[scale][rot]) {
	    responses[scale][rot] = cvCreateImage(cvSize(pose_responses[i][scale][rot]->width, 
							 pose_responses[i][scale][rot]->height),
						  IPL_DEPTH_32F, 1);
	    cvSet(responses[scale][rot], cvScalar(-10000000000000.0));
	  }
	
	  // Maintain the best scoring pixel location, scale, orientation, and pose
	  cvMax(pose_responses[i][scale][rot], responses[scale][rot], responses[scale][rot]);
	  cvMinMaxLoc(pose_responses[i][scale][rot], &min_val, &max_val, &min_loc, &max_loc);
	  if(max_val > theta.GetScore()) {
	    this->theta.SetScore((float)max_val);
            this->theta.SetDetectionLocation(max_loc.x, max_loc.y, scale, rot, i, LATENT, LATENT);
	  }
	}
      }
    }
  }
      
  if(responses && model->NumStaticWeights()) {
    theta.SetScore(-1000000000000.0);
    CvPoint min_loc, max_loc;
    double min_val, max_val;
    for(int scale = 0; scale < params->numScales; scale++) {
      for(int rot = 0; rot < params->numOrientations; rot++) { 
	if(responses[scale][rot]) {
	  AddStaticFeaturesToDetectionMap(responses[scale][rot], scale, rot);
	  cvMinMaxLoc(responses[scale][rot], &min_val, &max_val, &min_loc, &max_loc);
	  if(max_val > theta.GetScore()) {
	    this->theta.SetScore((float)max_val);
	    float best = -100000000;
	    int pose = LATENT;
	    for(int i = 0; i < model->NumPoses(); i++) {
	      float f = ((float*)(pose_responses[i][scale][rot]->imageData+responses[scale][rot]->widthStep*max_loc.y))[max_loc.x];
	      if(f > best) { pose = i; best = f; }
	    }
            this->theta.SetDetectionLocation(max_loc.x, max_loc.y, scale, rot, pose, LATENT, LATENT);
	  }
	}
      }
    }
  }

  return responses;
}

void ObjectPartInstance::Draw(IplImage *img, PartLocation *l, CvScalar color, CvScalar color2, CvScalar color3, const char *str, bool labelPoint, bool labelRect, float zoom) {
  if(!l) l = &theta;
  int pose;
  l->GetDetectionLocation(NULL, NULL, NULL, NULL, &pose);
  if(pose >= 0 && !poses[pose]->IsNotVisible())
    poses[pose]->Draw(img, l, color, color2, color3, str, model->GetAbbreviation(), labelPoint, labelRect, zoom);
}

void ObjectPartInstance::PredictLatentValues(PartLocation *l, bool useUnaryExtra) {	
  Classes *classes = process->GetClasses();
  int loc_x, loc_y, loc_scale, loc_rot, loc_pose;
  float loc_i_x, loc_i_y;
  l->GetDetectionLocation(&loc_x, &loc_y, &loc_scale, &loc_rot, &loc_pose);
  l->GetImageLocation(&loc_i_x, &loc_i_y);
  
  if(IS_LATENT(loc_pose) || IS_LATENT(loc_scale) || IS_LATENT(loc_rot)) {
    // If the pose, scale, or orientation are left as latent variables, 
    // find the pose/scale/orientation with maximum score
    float best = -10000000000000000000.0f, f;
    FeatureOptions *feat = process->Features();
    int numScales = feat->NumScales(), numOrientations = feat->NumOrientations();
    int x, y;
    for(int i = 0; i < model->NumPoses(); i++) {
      if(((!IS_LATENT(loc_pose) || poses[i]->Model()->IsNotVisible()) && i != loc_pose)) continue;
      if(!poses[i]->NumChildPartPoseTransitions()[0] && (model->GetParent() || model->NumParts())) 
        continue;  // pose never occurs in training set?

      IplImage ***unaryExtra = poses[i]->GetUnaryExtra();
      for(int scale = 0; scale < numScales; scale++) {
        if(!IS_LATENT(loc_scale) && scale != loc_scale) continue;
        for(int rotation = 0; rotation < numOrientations; rotation++) {
          if(!IS_LATENT(loc_rot) && rotation != loc_rot) continue;
          feat->ImageLocationToDetectionLocation(loc_i_x, loc_i_y, scale, rotation, &x, &y);
          if(!pose_responses[i][scale][rotation]) {
            assert(poses[i]->IsNotVisible());
            if(best < -10000000) {
              l->SetImageLocation(loc_i_x, loc_i_y, classes->Scale(scale), classes->Rotation(rotation), model->GetPose(i)->Name());
            }
            //f = par->childPartPoseTransitions[i][j][k]
          } else {
            f = ((float*)(pose_responses[i][scale][rotation]->imageData+pose_responses[i][scale][rotation]->widthStep*y))[x];
            if(useUnaryExtra && unaryExtra) f -= ((float*)(unaryExtra[scale][rotation]->imageData+unaryExtra[scale][rotation]->widthStep*y))[x];
            f += GetStaticScore(loc_x, loc_y, scale, rotation);
            if(f > best) { 
              best=f;
              l->SetImageLocation(loc_i_x, loc_i_y, classes->Scale(scale), classes->Rotation(rotation), model->GetPose(i)->Name());
	      l->SetScore(best);
            }
          }
        }
      }
    }
  }
}

void ObjectPartInstance::ExtractPartLocations(PartLocation *locs, PartLocation *l, ObjectPartInstance *par) {
  int id = Id();
  if(!l) l = &theta;
  locs[id].Copy(*l);
  locs[id].SetPart(model);
  PredictLatentValues(&locs[id], !par);

  int loc_pose;
  locs[id].GetDetectionLocation(NULL, NULL, NULL, NULL, &loc_pose);
  poses[loc_pose]->ExtractPartLocations(locs, &locs[id], par);
}

float ObjectPartInstance::GetStaticScore(int x, int y, int scale, int rot) {
  int num_static = model->NumStaticWeights();
  float f = 0;
  if(!num_static) return 0;
  else {
    float *static_features = new float[num_static];
    float *static_weights = NULL;
    static_weights = customStaticWeights ? customStaticWeights : model->GetStaticPartFeatures().weights;  
    PartLocation tmp(this, process->Image()->width, process->Image()->height);
    tmp.SetDetectionLocation(x, y, scale, rot, LATENT, LATENT, LATENT);
    GetStaticFeatures(static_features, &tmp);
    for(int i = 0; i < num_static; i++) 
      f += static_weights[i]*static_features[i];
    delete [] static_features;
  }
  return f;
}

IplImage *ObjectPartInstance::GetResponse(int s, int r) {
  if(!responses)
    responses = Detect(false);
  return responses[s][r];
}


void ObjectPartInstance::SetLoss(PartLocation *gt_loc, float l) {
  if(gt_loc) ground_truth_loc.Copy(*gt_loc);
  for(int i = 0; i < model->NumPoses(); i++) {
    poses[i]->SetLoss(gt_loc, l);
  }
}


float ObjectPartInstance::GetLoss(PartLocation *pred_loc) {
  int pose;
  pred_loc->GetDetectionLocation(NULL, NULL, NULL, NULL, &pose);
  if(pose >= 0) return poses[pose]->GetLoss(pred_loc);
  else return pose >= 0 ? 1.0f : 0.0f;
}


// Update unnormalized probabilities using a class-attribute naive bayes class probability estimate 
// \prod_{a_i,part(a_i)=p} p(a_i|c,theta_p) at a given part location theta_p
// TODO: we should have a standard class probability 
void ObjectPartInstance::UpdateClassLogLikelihoodAtLocation(double *classLikelihoods, PartLocation *theta_p, bool useGamma) {
  int i, j, theta_pose;
  theta_p->GetDetectionLocation(NULL, NULL, NULL, NULL, &theta_pose);
  Classes *classes = process->GetClasses();
  if(theta_pose < 0 || !GetPose(theta_pose)->IsNotVisible()) {
    float *features = NULL;
    int n;
    if(classes->AllAttributesHaveSaveFeatures()) {
      features = (float*)malloc(sizeof(float)*classes->NumWindowFeatures());
      n = GetLocalizedFeatures(features, theta_p);
      assert(n == classes->NumWindowFeatures());
    }

    for(j = 0; j < model->NumAttributes(); j++) {
      if(model->GetAttribute(j)->NumWeights() && model->GetAttribute(j)->Weights()) {
	int a_id = model->GetAttribute(j)->Id();
	AttributeInstance *a_i = process->GetAttributeInst(a_id);
	assert(a_i->Model()->Part() == model);
	double s = a_i->GetLogLikelihoodAtLocation(theta_p, 1, false, features);
	double gamma = model->GetAttribute(j)->GetGamma();
	double m = my_max(gamma*s, -gamma*s);
	double sum = exp(gamma*s-m) + exp(-gamma*s-m);
	double delta = -m - log(sum);
	for(i = 0; i < classes->NumClasses(); i++) {
	  double attributeWeight = classes->GetClass(i)->GetAttributeWeight(a_id);
	  classLikelihoods[i] += useGamma ? (gamma*attributeWeight*s+delta) : (attributeWeight*s);
	  assert(!isnan(classLikelihoods[i]));
	}
      }
    }

    double gamma = classes->GetClassGamma();
    for(i = 0; i < classes->NumClasses(); i++) {
      float *weights = classes->GetClass(i)->GetWeights();
      if(weights) {
	weights += n*model->Id();
	assert(features);
	float sum = 0;
	for(int k = 0; k < n; k++)
	  sum += weights[k]*features[k];
	classLikelihoods[i] += useGamma ? (gamma*sum+delta) : sum;
	assert(!isnan(classLikelihoods[i]));
      }
    }

    if(features)
      free(features);
  }
}

int ObjectPartInstance::GetLocalizedFeatures(float *f, PartLocation *loc) {
  float *fptr = f;

  Classes *classes = process->GetClasses();
  FeatureOptions *feat = process->Features();
  FeatureWindow *featureWindows = classes->FeatureWindows();
  int numWindows = classes->NumFeatureWindows();
  
  for(int k = 0; k < numWindows; k++) {
    SlidingWindowFeature *fe = feat->Feature(featureWindows[k].name);
    if(loc->IsVisible()) {
      int n = fe->GetFeaturesAtLocation(fptr, featureWindows[k].w, featureWindows[k].h, featureWindows[k].scale, loc, false);
      assert(n == fe->NumFeatures(featureWindows[k].w, featureWindows[k].h));
      fptr += n;
    } else {
      int n = fe->NumFeatures(featureWindows[k].w, featureWindows[k].h);
      for(int j = 0; j < n; j++)
	fptr[j] = 0;
      fptr += n;
    }
  }
  return fptr-f;
}

float ObjectPartInstance::ComputeCumulativeSums(bool recompute) {
  FeatureOptions *feat = process->Features();
  FeatureParams *params = feat->GetParams();
  if(!cumulativeSums) recompute = false;


  float gamma = model->GetGamma();
  double m, mi;
  int i;
  if(!cumulativeSums || recompute) {
    maxScore = -10000000000000.0f;
    float minScore = 10000000000000.0f;
    for(i = 0; i < model->NumPoses(); i++) {
      for(int scale = 0; pose_responses[i] && scale < params->numScales; scale++) {
        for(int rot = 0; rot < params->numOrientations; rot++) {
	  if(pose_responses[i][scale][rot]) {
	    cvMinMaxLoc(pose_responses[i][scale][rot], &mi, &m); 
	    if(m > maxScore) maxScore = (float)m;
	    if(mi < minScore) minScore = (float)mi;
	  }
        }
      }
    }
    //fprintf(stderr, "%s: max %f, min %f\n", process->ImageName(), (float)maxScore, (float)minScore);

    lower_bound = maxScore-4;//maxScore-process->GetClasses()->NumParts()*4;
    if(!recompute) {
      cumulativeSumMaps = (IplImage****)malloc(sizeof(IplImage***)*model->NumPoses());
      cumulativeSums = (float***)malloc(sizeof(float**)*model->NumPoses());
    }
    cumulativeSumTotal = 0;
    float sum;
    for(i = 0; i < model->NumPoses(); i++) {
      if(!recompute) {
        cumulativeSumMaps[i] = (IplImage***)malloc(sizeof(IplImage**)*params->numScales);
        cumulativeSums[i] = (float**)malloc(sizeof(float*)*params->numScales);
      }
      for(int scale = 0; pose_responses[i] && scale < params->numScales; scale++) {
        if(!recompute) {
          cumulativeSumMaps[i][scale] = (IplImage**)malloc(sizeof(IplImage*)*params->numOrientations);
          cumulativeSums[i][scale] = (float*)malloc(sizeof(float)*params->numOrientations);
          memset(cumulativeSumMaps[i][scale], 0, sizeof(IplImage*)*params->numOrientations);
        }
        for(int rot = 0; rot < params->numOrientations; rot++) {
	  if(pose_responses[i][scale][rot] && !GetPose(i)->IsInvalid()) {
	    cumulativeSumMaps[i][scale][rot] = cvCumulativeSum(pose_responses[i][scale][rot], &sum, 
							       cumulativeSumMaps[i][scale][rot], gamma, -gamma*maxScore, lower_bound);
	    cumulativeSums[i][scale][rot] = cumulativeSumTotal = cumulativeSumTotal + sum;
	    /*if(!scale && !rot) {
	      double min_val, max_val; CvPoint min_loc, max_loc;
	      cvMinMaxLoc(pose_responses[i][scale][rot], &min_val, &max_val, &min_loc, &max_loc);
	      //fprintf(stderr, "Use %s %f %f %f\n", GetPose(i)->Model()->Name(), (float)sum, (float)min_val, (float)max_val);
	    }
	    */
	  } else {
	    //if(!scale && !rot) fprintf(stderr, "Skip %s\n", GetPose(i)->Model()->Name());
	    cumulativeSums[i][scale][rot] = cumulativeSumTotal;
	  }
        }
      }
      //fprintf(stderr, "Cumulative Sum %s %f\n", GetPose(i)->Model()->Name(), cumulativeSumTotal-lastPose);
    }
    if(cumulativeSumTotal == 0)
      delta = -10000000;
    else
      delta = -gamma*maxScore - log(cumulativeSumTotal);
    //fprintf(stderr, "%s %f %f %f\n", model->Name(), gamma, delta, ma);
    //assert(delta >= -10000000000000.0f && delta < 10000000000000.0f);
  }
  return cumulativeSumTotal;
}

float ObjectPartInstance::ComputeCumulativeSumsNMS(bool recompute) {
  if(!cumulativeSumsNMS_keypoints) 
    recompute = false;

  if(!cumulativeSumsNMS_keypoints || recompute) {
    FeatureOptions *feat = process->Features();
    FeatureParams *params = feat->GetParams();
    float maxScore = -10000000000000.0f;
    float gamma = model->GetGamma();
    double m, mi;
    int i;

    for(i = 0; i < model->NumPoses(); i++) {
      for(int scale = 0; pose_responses[i] && scale < params->numScales; scale++) {
        for(int rot = 0; rot < params->numOrientations; rot++) {
	  if(pose_responses[i][scale][rot]) {
	    cvMinMaxLoc(pose_responses[i][scale][rot], &mi, &m); 
	    if(m > maxScore) maxScore = (float)m;
	  }
        }
      }
    }
    float minScore = maxScore-4;//process->GetClasses()->NumParts()*4;

    cumulativeSumsNMS_numKeypoints = 0;
    cumulativeSumsNMS_sum = 0;
    int numAlloc = 0;
    for(i = 0; i < model->NumPoses(); i++) {
      for(int scale = 0; pose_responses[i] && scale < params->numScales; scale++) {
        for(int rot = 0; rot < params->numOrientations; rot++) {
	  if(pose_responses[i][scale][rot] && !GetPose(i)->IsInvalid()) {
	    for(int y = 0; y < pose_responses[i][scale][rot]->height; y++) {
	      float *ptr = (float*)(pose_responses[i][scale][rot]->imageData + y*pose_responses[i][scale][rot]->widthStep);
	      float *ptr_n = (float*)(pose_responses[i][scale][rot]->imageData + (y+1)*pose_responses[i][scale][rot]->widthStep);
	      float *ptr_p = (float*)(pose_responses[i][scale][rot]->imageData + (y-1)*pose_responses[i][scale][rot]->widthStep);
	      for(int x = 0; x < pose_responses[i][scale][rot]->width; x++) {
		float dx1 = x > 0 ? ptr[x] - ptr[x-1] : 1;
		float dx2 = x < pose_responses[i][scale][rot]->width-1 ? ptr[x] - ptr[x+1] : 1;
		float dy1 = y > 0 ? ptr[x] - ptr_p[x] : 1;
		float dy2 = y < pose_responses[i][scale][rot]->height-1 ? ptr[x] - ptr_n[x] : 1;
		if((dx1 > 0 && dx2 > 0 && dy1 > 0 && dy2 > 0) || 
		   (cumulativeSumsNMS_numKeypoints == 0 && dx1 >= 0 && dx2 >= 0 && dy1 >= 0 && dy2 >= 0 && 
		    x == pose_responses[i][scale][rot]->width/2 && y == pose_responses[i][scale][rot]->height/2)) {
		  float s = ((float*)(pose_responses[i][scale][rot]->imageData + y*pose_responses[i][scale][rot]->widthStep))[x];
		  cumulativeSumsNMS_sum += exp(gamma*my_max(s,minScore)-gamma*maxScore);
		  if(cumulativeSumsNMS_numKeypoints >= numAlloc) {
		    numAlloc += 128;
		    cumulativeSumsNMS_keypoints = (PartLocation**)realloc(cumulativeSumsNMS_keypoints, sizeof(PartLocation*)*(numAlloc));
		  }
		  cumulativeSumsNMS_keypoints[cumulativeSumsNMS_numKeypoints] = new PartLocation(this, process->Image()->width, process->Image()->height);
		  cumulativeSumsNMS_keypoints[cumulativeSumsNMS_numKeypoints]->SetDetectionLocation(x, y, scale, rot, i, LATENT, LATENT);
		  cumulativeSumsNMS_keypoints[cumulativeSumsNMS_numKeypoints++]->SetScore(cumulativeSumsNMS_sum);
		}
	      }
	    }
	  }
	}
      }
    }
  }
  assert(cumulativeSumsNMS_keypoints);
  return cumulativeSumsNMS_sum;
}


float ObjectPartInstance::GetLogLikelihoodAtLocation(PartLocation *loc) {
  int x, y, scale, rot, pose;
  loc->GetDetectionLocation(&x, &y, &scale, &rot, &pose);
  IplImage *rimg = GetPose(pose)->GetResponse(scale,rot);
  return rimg ? (((float*)(rimg->imageData + rimg->widthStep*y))[x]) : 0; 
}


// Draw a  part location randomly according to a non-max-suppressed version of this part's probability maps
PartLocation ObjectPartInstance::DrawRandomPartLocationNMS(int numTries) {
  ComputeCumulativeSumsNMS();

  float r = RAND_FLOAT*cumulativeSumsNMS_sum;
  int i;
  for(i = 0; i < cumulativeSumsNMS_numKeypoints-1; i++) {
    if(cumulativeSumsNMS_keypoints[i]->GetScore() >= r)
      break;
  }
  return *cumulativeSumsNMS_keypoints[i];
}

// Draw a  part location randomly according to this part's probability maps
PartLocation ObjectPartInstance::DrawRandomPartLocation(int numTries) {
  PartLocation l(this, process->Image()->width, process->Image()->height);
  ComputeCumulativeSums();
  float target = RAND_FLOAT*cumulativeSumTotal;
  FeatureOptions *feat = process->Features();
  FeatureParams *params = feat->GetParams();

  float last = 0;
  for(int i = 0; i < model->NumPoses(); i++) {
    for(int scale = 0; scale < params->numScales; scale++) {
      for(int rot = 0; rot < params->numOrientations; rot++) {
	if(cumulativeSums[i][scale][rot] >= target) {
	  CvPoint pt;
	  if(!cvFindCumulativeSumLocation(cumulativeSumMaps[i][scale][rot], target-last, &pt)) {
	    assert(numTries>0);
	    fprintf(stderr, "ERROR: DrawRandomPartLocation() failed.  Trying again %d...\n", numTries);
	    return DrawRandomPartLocation(numTries-1);
	  }
	  l.SetDetectionLocation(pt.x, pt.y, scale, rot, i, LATENT, LATENT);
	  if(pose_responses) l.SetScore(cvGet2D(pose_responses[i][scale][rot],pt.x,pt.y).val[0]);
	  return l;
	}
	last = cumulativeSums[i][scale][rot];
      }
    }
  }
  assert(0);
  return l;
}

void ObjectPartInstance::DrawRandomPartLocationsWithoutDetector(PartLocation *locs, ObjectPartInstance *parent, int pad) {
  FeatureOptions *feat = process->Features();
  if(!parent) {
    int pose_ind = rand() % model->NumPoses();
    while(model->GetPose(pose_ind)->IsNotVisible())
      pose_ind = rand() % model->NumPoses();
    int scale_ind = rand() % feat->NumScales();
    int rot_ind = rand() % my_max(feat->NumOrientations(),1);
    float scale = feat->Scale(scale_ind), rot = feat->Rotation(rot_ind);
    char *pose = model->GetPose(pose_ind)->Name();
    while((int)(feat->GetImage()->width-2*pad*scale) <= 0 || (int)(feat->GetImage()->height-2*pad*scale) <= 0) {
      scale_ind = rand() % feat->NumScales();
      scale = feat->Scale(scale_ind), rot = feat->Rotation(rot_ind);
    }
    float x = pad*scale + rand()%(int)(feat->GetImage()->width-2*pad*scale);
    float y = pad*scale + rand()%(int)(feat->GetImage()->height-2*pad*scale);
    locs[model->Id()].SetImageLocation(x, y, scale, rot, pose);
  } 

  int *numChildPoses;
  int x_det, y_det, scale_ind, rot_ind, pose_ind;
  float offset_norm, off_x, off_y, off_scale, off_rot, offn;
  int g = feat->SpatialGranularity();
  float ff = 1.0f/g;
  locs[model->Id()].GetDetectionLocation(&x_det, &y_det, &scale_ind, &rot_ind, &pose_ind);
  ObjectPartPoseTransitionInstance ***trans = poses[pose_ind]->GetPosePartTransitions(&numChildPoses);
  
  for(int i = 0; i < model->NumParts(); i++) {
    ObjectPartInstance *child = GetPart(i);
    if(child == parent) continue;

    float wx, wxx, wy, wyy, ws, wss, wr, wrr, wt;
    int j, k, ind = 0;
    float sumP = 0, p[1000];
    for(j = 0; j < numChildPoses[i]; j++) {
      for(k = 0; k < trans[i][j]->NumChildPoses(); k++, ind++) {
        trans[i][j]->Model()->GetWeights(&wx, &wxx, &wy, &wyy, &ws, &wss, &wr, &wrr, &wt, k);
        p[ind] = trans[i][j]->Model()->NumTrainExamples(k);
        sumP += p[ind];
      }
    }
    float r = RAND_FLOAT*sumP;
    sumP = 0;
    for(j = 0, ind = 0; j < numChildPoses[i]; j++) {
      for(k = 0; k < trans[i][j]->NumChildPoses(); k++, ind++) {
        sumP += p[ind];
        if(sumP >= r) {
          trans[i][j]->Model()->GetWeights(&wx, &wxx, &wy, &wyy, &ws, &wss, &wr, &wrr, &wt, k);
          trans[i][j]->Model()->GetOffsets(&offset_norm, &off_x, &off_y, &off_scale, &off_rot);
          offn = offset_norm*g;
          break;
        }
      }
    }
    int x_child = my_round(x_det + off_x*ff + RAND_GAUSSIAN/(2*wxx*SQR(offn)));
    int y_child = my_round(y_det + off_y*ff + RAND_GAUSSIAN/(2*wyy*SQR(offn)));
    int scale_child = my_round(scale_ind + off_scale + RAND_GAUSSIAN/(2*wss));
    int rot_child = my_round(rot_ind + off_rot + RAND_GAUSSIAN/(2*wrr));
    rot_child = my_max(0, my_min(feat->NumOrientations(), rot_child));
    scale_child = my_max(0, my_min(feat->NumScales(), scale_child));
    locs[child->Model()->Id()].SetDetectionLocation(x_child, y_child, scale_child, rot_child, child->Model()->PoseInd(trans[i][j]->GetChildPose(k)->Model()), LATENT, LATENT);

    child->DrawRandomPartLocationsWithoutDetector(locs, this, pad);
  }
}


void ObjectPartInstance::SetAttributeAnswer(int attribute_ind, struct _AttributeAnswer *a) {
  // TODO: update the unary probability map to favor locations that agree with the attribute answer
  //inst->PropagateUnaryMapChanges();
}

void ObjectPartInstance::SetClickPoint(PartLocation *l, bool useMultiObject) {
  theta.Copy(*l);					
  isObserved = true;

  // Build a log probability cloud map centered around the specified click point
  assert(model->IsClick() && model->NumParts() == 0);
  int loc_pose;
  l->GetDetectionLocation(NULL, NULL, NULL, NULL, &loc_pose);
  if(loc_pose >= 0)
    poses[loc_pose]->SetClickPoint(l, useMultiObject);
  else {
    for(int i = 0; i < model->NumPoses(); i++)
      poses[i]->SetClickPoint(l, useMultiObject);
  }
  ClearCumulativeSums();
  if(GetParent()) GetParent()->ClearCumulativeSums();
}


void ObjectPartInstance::SetLocation(PartLocation *l, bool detect, bool useMultiObject) {
  // Build a log probability cloud map centered around the specified click point
  for(int i = 0; i < model->NumPoses(); i++)
    poses[i]->SetLocation(l, useMultiObject);
  if(detect) {
    Clear(true, false);
    isObserved = false;
    Detect(true);				
  }


  theta.Copy(*l);	
  isAnchor = true;
}

// After someone has called pose->AddToUnaryExtra(), we can propagate the changes to 
// every other node in the tree using a traversal of the tree starting at
// the altered node.  The result is that we can precompute the maximum likelihood part
// configuration when fixing any single part or for any possible click point location
void ObjectPartInstance::PropagateUnaryMapChanges(ObjectPartInstance *pre) {
  // If this part is the one whose unary map changed, then the parameter 'pre' should
  // be NULL.  Otherwise, it is the preceding node in a traversal of the tree
  
  //fprintf(stderr, "Before Propagate %s %f\n", model->Name(), (float)theta.score);
  
  int i;
  
  for(i = 0; i < model->NumPoses(); i++) {
    if(pre)  {
      // First remove the component of this node's combined unary-pairwise response that came 
      // from the preceding node
      poses[i]->InvalidateChildScores(pre);  

      // Now recompute the component of this node's combined unary-pairwise response from
      // the preceding node's updated unary map
      poses[i]->Detect(NULL);
    }
  }
  if(!IsObserved()) {
    Clear(true, false);
    Detect(true);
    //fprintf(stderr, "After Propagate %s %f\n", model->Name(), (float)theta.score);
  } 
  

  // Continue to propagate information to other parts in the tree
  for(i = 0; i < model->NumParts(); i++)
    if(parts[i] != pre && (!parts[i]->IsObserved())) 
      parts[i]->PropagateUnaryMapChanges(this);
  if(GetParent() && GetParent() != pre && (!GetParent()->IsObserved()))
    GetParent()->PropagateUnaryMapChanges(this);

  // Densely compute the maximum likelihood set of part configurations for every possible
  // place the user can click
  if(clickPart && !clickPart->IsObserved() && process->ComputeClickParts()) {
    clickPart->PropagateUnaryMapChanges(this);
  }
}

void ObjectPartInstance::AddStaticFeaturesToDetectionMap(IplImage *response, int scale, int rot) {
  FeatureParams *params = process->GetClasses()->GetFeatureParams();
  StaticPartFeatures staticFeatures = model->GetStaticPartFeatures();
  float *wptr = customStaticWeights ? customStaticWeights : staticFeatures.weights;
  if(staticFeatures.useBiasTerm || staticFeatures.useScalePrior || staticFeatures.useOrientationPrior) {
    float a = 0;
    if(staticFeatures.useBiasTerm) {
      a += wptr[1]*BIAS_FEATURE;
      wptr += 2;
    }
    if(staticFeatures.useScalePrior) {
      a += wptr[scale]*BIAS_FEATURE/(float)params->numScales;
      wptr += params->numScales;
    }
    if(staticFeatures.useOrientationPrior) {
      a += wptr[rot]*BIAS_FEATURE/(float)params->numOrientations;
      wptr += params->numOrientations;
    }
    cvAddS(response, cvScalar(a), response);
  }
  if(staticFeatures.numSpatialLocationLevels > 1) {
    RotationInfo r = process->Features()->GetRotationInfo(rot, scale);
    int ix, iy;
    for(int i = 1, num = 2; i < staticFeatures.numSpatialLocationLevels; i++, num*=2) {
      float weight = BIAS_FEATURE/(float)SQR(num);
      float w = process->Image()->width / (float)num;
      float h = process->Image()->height / (float)num;
      float dx = r.invMat[0]/w,  dy = r.invMat[3]/h;
      for(int yy = 0; yy < response->height; yy++) {
	float x = (r.invMat[1]*yy + r.invMat[2])/w;
	float y = (r.invMat[4]*yy + r.invMat[5])/h;
	float *ptr = (float*)(response->imageData+response->widthStep*yy);
	for(int xx = 0; xx < response->width; xx++, x += dx, y += dy) {
	  if((int)(x+.01) != (int)(x-.01) || (int)(y+.01) != (int)(y-.01)) {
	    float x2, y2; AffineTransformPoint(r.invMat, xx, yy, &x2, &y2); x2/=w; y2/=h;  x=x2; y=y2;
	  }
	  ix = (int)x;
	  iy = (int)y;
	  if(iy < 0) iy = 0;
	  if(ix < 0) ix = 0;
	  if(iy >= num) iy = num-1;
	  if(ix >= num) ix = num-1;
	  ptr[xx] += wptr[iy*num+ix]*weight;
	}
      }
      wptr += SQR(num);
    }
  }
} 

int ObjectPartInstance::GetStaticFeatures(float *f_orig, PartLocation *loc) {
  float *f = f_orig;
  StaticPartFeatures staticFeatures = model->GetStaticPartFeatures();
  int num = model->NumStaticWeights();
  if(!num) return 0;
  for(int i = 0; i < num; i++) 
    f[i] = 0;
  if(!loc) {
    if(staticFeatures.useBiasTerm) *f = BIAS_FEATURE; 
    return num;
  }
  float x, y;
  int x1, y1;
  int scaleInd, rotInd;
  FeatureParams *params = process->GetClasses()->GetFeatureParams();
  loc->GetDetectionLocation(&x1, &y1, &scaleInd, &rotInd);
  RotationInfo r = process->Features()->GetRotationInfo(rotInd, scaleInd);
  AffineTransformPoint(r.invMat, x1, y1, &x, &y);
  if(x < 0) x = 0;
  if(y < 0) y = 0;
  if(x >= process->Image()->width) x = process->Image()->width-1;
  if(y >= process->Image()->height) y = process->Image()->height-1;

  if(staticFeatures.useBiasTerm) { 
    *f = 0;
    f++;
    *f = BIAS_FEATURE; 
    f++; 
  }
  if(staticFeatures.useScalePrior) { 
    f[scaleInd] = BIAS_FEATURE/(float)params->numScales; 
    f += params->numScales; 
  }
  if(staticFeatures.useOrientationPrior) { 
    f[rotInd] = BIAS_FEATURE/(float)params->numOrientations; 
    f += params->numOrientations; 
  }
  for(int i = 1, num = 2; i < staticFeatures.numSpatialLocationLevels; i++, num*=2) {
    float w = process->Image()->width / (float)num;
    float h = process->Image()->height / (float)num;
    f[((int)(y/h))*num+(int)(x/w)] = BIAS_FEATURE/(float)SQR(num); 
    f += SQR(num);
  }

  return f - f_orig;
}

extern float g_score2, *g_w2;
float g_pose_scores[1000];
void ObjectPartInstance::UpdatePoseFeatures(float *f, PartLocation *locs, int *poseOffsets, PartLocation *loc) {
  int pid = model->Id();
  int i;
  locs[pid].GetDetectionLocation(NULL, NULL, NULL, NULL, &i);
  int id = poses[i]->Model()->Id();
  Attribute *app = poses[i]->Model()->Appearance();
  if(app && app->NumWeights()) {
    float *ff = (float*)malloc(sizeof(float)*poses[i]->Model()->Appearance()->NumWeights());
    int n = poses[i]->GetFeatures(ff, locs, loc);
    //fprintf(stderr, " [*pose %d %d:%d %s]", id, poseOffsets[id], poseOffsets[id]+n, poses[i]->Model()->Name());
    float *fptr = f+poseOffsets[id];
    for(int j = 0; j < n; j++) {
      fptr[j] += ff[j];
      //if(g_w2) g_score2 += g_w2[poseOffsets[id] + j]*ff[j];
    }
    //g_pose_scores[id] = g_score2;
    free(ff);
  }
}

void ObjectPartInstance::UpdateSpatialFeatures(float *f, PartLocation *locs, int *spatialOffsets, PartLocation *loc) {
  int pid = model->Id();
  int i;
  locs[pid].GetDetectionLocation(NULL, NULL, NULL, NULL, &i);
  poses[i]->UpdateTransitionFeatures(f, locs, loc, spatialOffsets);
}


void ObjectPartInstance::SetCustomStaticWeights(float *w) {
  if(model->NumStaticWeights()) 
    customStaticWeights = w;
}

int ObjectPartInstance::DebugStatic(float *w, float *f, PartLocation *locs, bool debug_scores, bool print_weights, float *f_gt) { 
  int num = model->NumStaticWeights();
  if(!num) return 0;

  if(customStaticWeights) w = customStaticWeights;
  float score = 0, score_gt = 0;
  fprintf(stderr, "%s static:", model->Name());
  for(int i = 0; i < num; i++) {
    score += w[i]*f[i];
    if(f_gt) score_gt += w[i]*f_gt[i];
    if((print_weights||1) && (f[i] || (f_gt && f_gt[i]))) {
      if(f_gt) {
        fprintf(stderr, " %d:%.7f:%f:%f", i, w[i], f[i], f_gt[i]);
      } else
        fprintf(stderr, " %d:%.7f:%f", i, w[i], f[i]);
    }
  }
  if(debug_scores) {
    if(responses && locs) {
      int x, y, scale, rot, pose;
      locs[model->Id()].GetDetectionLocation(&x, &y, &scale, &rot, &pose);
      float pose_score = ((float*)(pose_responses[pose][scale][rot]->imageData+pose_responses[pose][scale][rot]->widthStep*y))[x];
      float part_score = ((float*)(responses[scale][rot]->imageData+responses[scale][rot]->widthStep*y))[x];
      assert(my_abs(score - (part_score-pose_score)) < .01);
      fprintf(stderr, " score=%f:%f score_gt=%f", score, part_score-pose_score, score_gt);
    } else
      fprintf(stderr, " score=%f score_gt=%f", score, score_gt);
  }
  fprintf(stderr, "\n");

  return num;
}


void ObjectPartInstance::Debug(float *w, float *f, int *poseOffsets, int *spatialOffsets, PartLocation *locs, bool debug_scores, bool print_weights, float *f_gt) { 
  int pid = model->Id();
  int i;
  locs[pid].GetDetectionLocation(NULL, NULL, NULL, NULL, &i);
  int id = poses[i]->Model()->Id();
  poses[i]->DebugSpatial(w, f, spatialOffsets, locs, debug_scores, print_weights, f_gt);
  //fprintf(stderr, " [pose %d %s]", poseOffsets[id], poses[i]->Model()->Name());
  poses[i]->DebugAppearance(w+poseOffsets[id], f+poseOffsets[id], locs+pid, debug_scores, print_weights, f_gt ? (f_gt+poseOffsets[id]) : NULL);
}

void ObjectPartInstance::SanityCheckDynamicProgramming(PartLocation *gt_locs) {
  for(int i = 0; i < model->NumPoses(); i++) 
    poses[i]->SanityCheckDynamicProgramming(gt_locs);
}


float ObjectPartInstance::GetUserLoss(PartLocation *pred_loc, float maxDev) {
  int *num;
  int gt_pose_ind, pred_pose_ind;
  float gt_x, gt_y, pred_x, pred_y;
  ground_truth_loc.GetDetectionLocation(NULL, NULL, NULL, NULL, &gt_pose_ind);
  ground_truth_loc.GetImageLocation(&gt_x, &gt_y);
  pred_loc->GetDetectionLocation(NULL, NULL, NULL, NULL, &pred_pose_ind);
  pred_loc->GetImageLocation(&pred_x, &pred_y);
  ObjectPartPoseTransitionInstance ***t = process->GetClickPartInst(Id())->GetVisiblePose()->GetPosePartTransitions(&num);
  ObjectPoseInstance *gt_pose = gt_pose_ind >= 0 ? process->GetPartInst(Id())->GetPose(gt_pose_ind) : NULL;
  ObjectPoseInstance *pose = pred_pose_ind >= 0 ? process->GetPartInst(Id())->GetPose(pred_pose_ind) : NULL;
  float wx, wxx, wy, wyy;
  t[0][0]->Model()->GetWeights(&wx, &wxx, &wy, &wyy);
  if(gt_pose->IsNotVisible() != pose->IsNotVisible())
    return maxDev;
  else if(gt_pose->IsNotVisible())
    return 0;
  else {
    float d = sqrt((SQR(pred_x - gt_x) * wxx + SQR(pred_y - gt_y) * wyy)/2/process->GetClasses()->GetClickGamma());
    return my_min(maxDev, d);
  }
}

ObjectPoseInstance *ObjectPartInstance::GetVisiblePose() { 
  for(int i = 0; i < model->NumPoses(); i++) 
    if(!poses[i]->Model()->IsNotVisible()) 
      return poses[i]; 
  return NULL;
}

ObjectPoseInstance *ObjectPartInstance::GetNotVisiblePose() { 
  for(int i = 0; i < model->NumPoses(); i++) 
    if(poses[i]->Model()->IsNotVisible()) 
      return poses[i]; 
  return NULL;
}

PartLocation::PartLocation() {
  Init(NULL, -1, -1, NULL);
}

void PartLocation::Init(Classes *c, int w, int h, FeatureOptions *f) {
  poseID = partID = LATENT;
  partName = poseName = NULL;
  x_img = y_img = scale_img = rotation_img = width = height = LATENT;
  flipHorizontal = flipVertical = false;
  x_det = y_det = scale_det = rotation_det = dx = dy = LATENT;
  responseTimeSec = 0;
  score = 0;
  hasDetectionCoords = false;
  hasImageCoords = false;
  classes = c;
  image_width = w;
  image_height = h;
  isClick = false;
  visible = true;
  feat = f;
}

PartLocation::PartLocation(const PartLocation &rhs) {
  Init(NULL, -1, -1, NULL);
  this->Copy(rhs);
}

void PartLocation::Copy(const PartLocation &rhs) {
  if(this->partName && this->partName != rhs.partName) {
    free(this->partName);
    this->partName = NULL;
  }
  if(this->poseName && this->poseName != rhs.poseName) {
    free(this->poseName);
    this->poseName = NULL;
  }
  this->classes = rhs.classes;
  this->image_width = rhs.image_width;
  this->image_height = rhs.image_height;
  this->poseID = rhs.poseID;  this->partID = rhs.partID;
  if(rhs.partName != this->partName)
    this->partName = rhs.partName ? StringCopy(rhs.partName) : NULL;   
  if(rhs.poseName != this->poseName )
    this->poseName = rhs.poseName ? StringCopy(rhs.poseName) : NULL;
  this->x_img = rhs.x_img;   this->y_img = rhs.y_img;
  this->scale_img = rhs.scale_img;   this->rotation_img = rhs.rotation_img;
  this->width = rhs.width;   this->height = rhs.height;
  this->flipHorizontal = rhs.flipHorizontal;   
  this->x_det = rhs.x_det;   this->y_det = rhs.y_det;  
  this->scale_det = rhs.scale_det;  this->rotation_det = rhs.rotation_det;
  this->dx = rhs.dx;  this->dy = rhs.dy;
  this->responseTimeSec = rhs.responseTimeSec;  this->score = rhs.score;
  this->hasDetectionCoords = rhs.hasDetectionCoords;  this->hasImageCoords = rhs.hasImageCoords;
  this->isClick = rhs.isClick;
  this->visible = rhs.visible;
  this->feat = rhs.feat;
}

PartLocation::PartLocation(ObjectPartInstance *part, int image_width, int image_height) {
  Init(part->Model()->GetClasses(), image_width, image_height, part->Process()->Features());
  SetPart(part->Model());
}

int PartLocation::GetPartID() {
  if(!hasDetectionCoords) ComputeDetectionLocations();
  return partID;
}

void PartLocation::SetPart(ObjectPart *part) {
  if(partName) free(partName);
  partName = StringCopy(part->Name());
  partID = part->Id();
  isClick = part->IsClick();
}


PartLocation::~PartLocation() { 
	Clear();
}

void PartLocation::Clear() { 
  if(partName) free(partName);
  if(poseName) free(poseName);
  partName = poseName = NULL;
  Init(NULL, -1, -1, NULL);
}

bool PartLocation::load(const Json::Value &r) {
  hasImageCoords = true;
  x_img = r.get("x", LATENT).asFloat();
  y_img = r.get("y", LATENT).asFloat();
  scale_img = r.get("scale", LATENT).asFloat();
  rotation_img = r.get("rotation", LATENT).asFloat();
  flipHorizontal = r.get("flipHorizontal", false).asBool();
  flipVertical = r.get("flipVertical", false).asBool();
  responseTimeSec = r.get("responseTime", 0).asFloat();

  hasDetectionCoords = false;
  score = 0;
  x_det = y_det = dx = dy = LATENT;
  width = height = LATENT;
  partID = poseID = LATENT;
  visible = r.get("visible", true).asBool();

  if(poseName) free(poseName);
  poseName = NULL;

  char tmp[1000];
  if(r.isMember("pose")) {
    strcpy(tmp, r["pose"].asString().c_str());
    poseName = StringCopy(tmp);
  }
  if(r.isMember("part")) {
    if(partName) free(partName);
    partName = NULL;
    strcpy(tmp, r["part"].asString().c_str());
    partName = StringCopy(tmp);
  }

  if(image_width >= 0 && image_height >= 0) 
    ComputeDetectionLocations();

  return true;
}

Json::Value PartLocation::save() {
  assert(hasImageCoords);

  Json::Value r;
  if(!IS_LATENT(x_img)) r["x"] = x_img;
  if(!IS_LATENT(y_img)) r["y"] = y_img;
  if(!IS_LATENT(scale_img)) r["scale"] = scale_img;
  if(!IS_LATENT(rotation_img)) r["rotation"] = rotation_img;

  if(partName) r["part"] = partName;
  if(poseName) r["pose"] = poseName;
  if(responseTimeSec) r["responseTime"] = responseTimeSec;
  if(flipVertical) r["flipVertical"] = flipVertical;
  if(flipHorizontal) r["flipHorizontal"] = flipHorizontal;
  r["visible"] = visible;
  r["score"] = score;

  return r;
}


bool ObjectPart::LoadStaticFeatures(const Json::Value &root) {
  staticFeatures.useBiasTerm = root.get("useBiasTerm", false).asBool();
  staticFeatures.useScalePrior = root.get("useScalePrior", false).asBool();
  staticFeatures.useOrientationPrior = root.get("useOrientationPrior", false).asBool();
  staticFeatures.numSpatialLocationLevels = root.get("numSpatialLocationLevels", 0).asInt();

  if(root.isMember("weights") && root["weights"].isArray()) {
    staticFeatures.weights = (float*)malloc(sizeof(float)*root["weights"].size());
    for(int i = 0; i < (int)root["weights"].size(); i++)
      staticFeatures.weights[i] = root["weights"][i].asFloat();
  }
  return true;
}

Json::Value ObjectPart::SaveStaticFeatures() {
  Json::Value root;
  root["useBiasTerm"] = staticFeatures.useBiasTerm;
  root["useScalePrior"] = staticFeatures.useScalePrior;
  root["useOrientationPrior"] = staticFeatures.useOrientationPrior;
  root["numSpatialLocationLevels"] = staticFeatures.numSpatialLocationLevels;

  int num = NumStaticWeights();
  if(num) {
    Json::Value w;
    for(int i = 0; i < num; i++)
      w[i] = staticFeatures.weights[i];
    root["weights"] = w;
  }
  return root;
}


bool PartLocation::ComputeDetectionLocations() {
  // Fill in entries of the PartLocation struct that are dependent on the image size
  assert(hasImageCoords && classes && image_width > 0 && image_height > 0);
  scale_det = IS_LATENT(scale_img) ? LATENT : classes->ScaleInd(scale_img);
  rotation_det = IS_LATENT(rotation_img) ? LATENT : classes->RotationInd(rotation_img);
  if(feat)
    feat->ImageLocationToDetectionLocation(x_img, y_img, IS_LATENT(scale_det) ? 0 : scale_det, IS_LATENT(rotation_det) ? 0 : rotation_det, &x_det, &y_det);
  else
    classes->ImageLocationToDetectionLocation(x_img, y_img, IS_LATENT(scale_det) ? 0 : scale_det, IS_LATENT(rotation_det) ? 0 : rotation_det, image_width, image_height, &x_det, &y_det);
  hasDetectionCoords = true;

  poseID = LATENT;

  if(!partName) return false;
  ObjectPart *part = isClick ? classes->FindClickPart(partName) : classes->FindPart(partName);
  if(!part) { fprintf(stderr, "Couldn't find part %s\n", partName); return false; }
  partID = part->Id();

  if(poseName) {
    if(!partName) return false;
    ObjectPose *pose = isClick ? classes->FindClickPose(poseName) : classes->FindPose(poseName);
    if(!pose) { 
      fprintf(stderr, "Couldn't find pose %s\n", poseName); return false; 
    }
    if((poseID=part->PoseInd(pose)) < 0) { 
      fprintf(stderr, "Couldn't find part pose %s\n", poseName); return false; 
    }
    visible = !part->GetPose(poseID)->IsNotVisible();
  } else
    visible = true;

  Attribute *appearance = IS_LATENT(poseID) ? NULL : part->GetPose(poseID)->Appearance();
  float fs = (IS_LATENT(scale_img) ? 1 : scale_img)*classes->SpatialGranularity();
  width = appearance ? appearance->Width()*fs : 0;
  height = appearance ? appearance->Height()*fs : 0;

  return true;
}

bool PartLocation::ComputeImageLocations() {
  assert(hasDetectionCoords && classes && image_width > 0 && image_height > 0);
  scale_img = IS_LATENT(scale_det) ? LATENT : classes->Scale(scale_det);
  rotation_img = IS_LATENT(rotation_det) ? LATENT : classes->Rotation(rotation_det);
  
  if(feat)
    feat->DetectionLocationToImageLocation(x_det, y_det, IS_LATENT(scale_det) ? 0 : scale_det, IS_LATENT(rotation_det) ? 0 : rotation_det, &x_img, &y_img);
  else
    classes->DetectionLocationToImageLocation(x_det, y_det, IS_LATENT(scale_det) ? 0 : scale_det, IS_LATENT(rotation_det) ? 0 : rotation_det, image_width, image_height, &x_img, &y_img);
  
  if(IS_LATENT(partID)) 
    return false;
  ObjectPart *part = isClick ? classes->GetClickPart(partID) : classes->GetPart(partID);
  if(partName) free(partName);
  partName = StringCopy(part->Name());

  if(!IS_LATENT(poseID)) { 
    if(poseName) free(poseName);
    poseName = StringCopy(part->GetPose(poseID)->Name());
    visible = !part->GetPose(poseID)->IsNotVisible();
  } else
    visible = true;

  Attribute *appearance = IS_LATENT(poseID) ? NULL : part->GetPose(poseID)->Appearance();
  float fs = (IS_LATENT(scale_img) ? 1 : scale_img)*classes->SpatialGranularity();
  width = appearance ? appearance->Width()*fs : 0;
  height = appearance ? appearance->Height()*fs : 0;
  hasImageCoords = true;

  return true;
}

bool PartLocation::IsLatent() { 
  if(!hasDetectionCoords) ComputeDetectionLocations();
  return IS_LATENT(x_det) && IS_LATENT(y_det) && IS_LATENT(scale_det) && IS_LATENT(rotation_det) && IS_LATENT(poseID); 
}

void PartLocation::GetImageLocation(float *x, float *y, float *scale, float *rot, 
				    const char **pose, float *width, float *height) {
  if(!hasImageCoords) ComputeImageLocations();
  if(x) *x = x_img; 
  if(y) *y = y_img; 
  if(scale) *scale = scale_img; 
  if(rot) *rot = rotation_img; 
  if(pose) *pose = poseName; 
  if(width) *width = this->width; 
  if(height) *height = this->height; 
}

void PartLocation::SetImageLocation(float x, float y, float scale, float rot, const char *pose) {
  x_img = x; 
  y_img = y; 
  scale_img = scale; 
  rotation_img = rot; 
  if(pose != poseName) {
    if(poseName) free(poseName);
    poseName = StringCopy(pose); 
  }
  hasImageCoords = true;
  if(x != LATENT && y != LATENT) 
    ComputeDetectionLocations();
  else {
    ObjectPose *pose = poseName ? (isClick ? classes->FindClickPose(poseName) : classes->FindPose(poseName)) : NULL;
    visible = pose ? !pose->IsNotVisible() : true;
  } 
}

void PartLocation::GetDetectionLocation(int *x, int *y, int *scale, int *rot, 
					int *pose, float *width, float *height, int *dx, int *dy) {
  if(!hasDetectionCoords) ComputeDetectionLocations();
  if(x) *x = x_det; 
  if(y) *y = y_det; 
  if(scale) *scale = scale_det; 
  if(rot) *rot = rotation_det; 
  if(pose) *pose = poseID; 
  if(width) *width = this->width; 
  if(height) *height = this->height; 
  if(dx) *dx = this->dx;
  if(dy) *dy = this->dy;
}

void PartLocation::SetDetectionLocation(int x, int y, int scale, int rot, int pose, int dx, int dy) {
  x_det = x; 
  y_det = y; 
  scale_det = scale; 
  rotation_det = rot; 
  poseID = pose;
  this->dx = dx;
  this->dy = dy;
  hasDetectionCoords = true;
  ComputeImageLocations();
}

PartLocation *PartLocation::NewPartLocations(Classes *classes, int image_width, int image_height, FeatureOptions *feat, bool isClick) {
  PartLocation *retval = new PartLocation[classes->NumParts()];
  for(int i = 0; i < classes->NumParts(); i++) {
    retval[i].Init(classes, image_width, image_height, feat);
    retval[i].SetPart(isClick ? classes->GetClickPart(i) : classes->GetPart(i));
  }
  return retval;
}

PartLocation *PartLocation::CopyPartLocations(PartLocation *locs) {
  int w, h;
  locs[0].GetImageSize(&w, &h);
  PartLocation *n = NewPartLocations(locs[0].GetClasses(), w, h, locs[0].feat, locs[0].isClick);
  for(int i = 0; i < locs[0].GetClasses()->NumParts(); i++) 
    n[i].Copy(locs[i]);
  return n;
}

PartLocation *PartLocation::FlippedPartLocations(PartLocation *locs) {
  PartLocation *retval = NewPartLocations(locs[0].classes, locs[0].image_width, locs[0].image_height, locs[0].feat, locs[0].isClick);
  Classes *classes = locs[0].GetClasses();
  for(int i = 0; i < classes->NumParts(); i++) {
    float x, y, scale, rot;
    const char *poseName;
    locs[i].GetImageLocation(&x, &y, &scale, &rot, &poseName);
    int ind = i;
    ObjectPose *pose = classes->FindPose(poseName);
    if(classes->GetPart(i)->GetFlipped()) {
      ind = classes->GetPart(i)->GetFlipped()->Id();
      if(!pose->GetFlipped())
	pose = classes->GetPart(i)->GetFlipped()->GetPose(classes->GetPart(i)->PoseInd(pose));
    }
    if(pose && pose->GetFlipped()) 
      pose = pose->GetFlipped();
    retval[ind].SetImageLocation(locs[0].image_width-1-x, y, scale, -rot, pose->Name());
  }
  return retval;
}


