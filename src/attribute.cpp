/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#include "attribute.h"
#include "imageProcess.h"
#include "part.h"
#include "pose.h"
#include "classes.h"

Attribute::Attribute(const char *n) {
  name = n ? StringCopy(n) : NULL;
  weights = NULL;
  features = NULL;
  numFeatureTypes = 0;
  id = -1;
  part = NULL;
  part_name = NULL;
  binaryQuestion = NULL;
  gamma = 1;
  classes = NULL;
  property_name = value_name = visualization_image = NULL;
}

Attribute::~Attribute() {
  if(part_name) free(part_name);
  if(weights) free(weights);
  if(features) {
    for(int i = 0; i < numFeatureTypes; i++) 
      if(features[i].name) 
	free(features[i].name);
    free(features);
  }
  if(name) StringFree(name);
  if(property_name) StringFree(property_name);
  if(value_name) StringFree(value_name);
  if(visualization_image) StringFree(visualization_image);
};

int Attribute::Width(int i) { return i<numFeatureTypes ? features[i].w : 0; }
int Attribute::Height(int i) { return i<numFeatureTypes ? features[i].h : 0; }


Json::Value Attribute::Save() {
  Json::Value root;

  if(name) {
    root["name"] = name;
    root["gamma"] = gamma;
    root["id"] = id;
  }
  if(part) root["part"] = part->Name();
  if(property_name) root["property"] = property_name;
  if(value_name) root["value"] = value_name;
  if(visualization_image) root["visualization"] = visualization_image;

  float *wptr = weights;
  Json::Value feats;
  for(int i = 0; i < numFeatureTypes; i++) {
    Json::Value fe, w;
    fe["window"] = SaveFeatureWindow(&features[i]);
    if(weights) {
      FILE *fout = classes->GetBinaryWeightsFile();
      if(!fout) {
	// Save weights in human-readable form
	for(int j = 0; j < features[i].dim; j++) 
	  w[j] = wptr[j];
	fe["weights"] = w;
      } else {
	// Save weights in a separate binary file
	long seek = ftell(fout);
	fe["weights_seek_u"] = (unsigned int)((seek & 0xFFFFFFFF00000000L)>>32);
	fe["weights_seek_l"] = (unsigned int)(seek & 0x00000000FFFFFFFFL);
	int n = fwrite(wptr, sizeof(float), features[i].dim, fout);
        assert(n == features[i].dim);
      }
    }
    feats[i] = fe;
    wptr += features[i].dim;
  }
  root["features"] = feats;

  return root;
}

bool Attribute::Load(const Json::Value &root) {
  name = StringCopy(root.get("name", "").asString().c_str());
  id = root.get("id", -1).asInt();
  part_name = root.isMember("part") ? StringCopy(root.get("part", "").asString().c_str()) : NULL;
  gamma = root.get("gamma", 1).asFloat();
  property_name = root.isMember("property") ? StringCopy(root["property"].asString().c_str()) : NULL;
  value_name = root.isMember("value") ? StringCopy(root["value"].asString().c_str()) : NULL;
  visualization_image = root.isMember("visualization") ? StringCopy(root["visualization"].asString().c_str()) : NULL;
  
  //weights = NULL;
  if(root.isMember("features") && root["features"].isArray()) {
    for(int i = 0; i < (int)root["features"].size(); i++) {
      FeatureWindow f;
      Json::Value fe = root["features"][i];
      if(!fe.isObject() || !fe.isMember("window") || !LoadFeatureWindow(&f, fe["window"])) {
	fprintf(stderr, "Error parsing attribute feature\n");
	return false;
      }
      if(weights || fe.isMember("weights") || fe.isMember("weights_seek_u")) {
	weights = (float*)realloc(weights, (NumWeights()+f.dim)*sizeof(float));
	float *wptr = weights+NumWeights();
	for(int j = 0; j < f.dim; j++) 
	  wptr[j] = 0;
      }
      if(fe.isMember("weights") && fe["weights"].isArray()) {
	float *wptr = weights+NumWeights();
	Json::Value w = fe["weights"];
	if((int)w.size() != f.dim) { 
	  fprintf(stderr, "Error loading feature, weight array size is incorrect\n"); 
	  return false; 
	}
	for(int j = 0; j < f.dim; j++) 
	  wptr[j] = w[j].asFloat();
      } else if(fe.isMember("weights_seek_u")) {
	float *wptr = weights+NumWeights();
	FILE *fin = classes->GetBinaryWeightsFile();
	assert(fin);
	fseek(fin, (long)(((unsigned long)(fe["weights_seek_u"].asUInt())<<32) | ((unsigned long)fe["weights_seek_l"].asUInt())),  SEEK_SET);
	if(fread(wptr, sizeof(float), f.dim, fin) != f.dim) {
	  fprintf(stderr, "Error loading feature weights from binary file\n"); 
	  return false; 
	}
      }
      features = (FeatureWindow*)realloc(features, (numFeatureTypes+1)*sizeof(FeatureWindow));
      features[numFeatureTypes++] = f;
    }
  }
  return true;
}

bool Attribute::ResolveLinks(Classes *c) {
  classes = c;
  if(part_name) {
    if(!(part = classes->FindPart(part_name))) {
      fprintf(stderr, "Part %d(%s) couldn't resolve part %s\n", id, name, part_name);
      return false;
    }
    part->AddAttribute(this);
  }
  return true;
}


int Attribute::GetWeights(double *w) { 
  if(!weights) 
    for(int i = 0; i < NumWeights(); i++)
      w[i] = 0;
  else
    for(int i = 0; i < NumWeights(); i++)
      w[i] = weights[i];
  return NumWeights();
}

int Attribute::GetWeights(float *w) { 
  if(!weights) 
    for(int i = 0; i < NumWeights(); i++)
      w[i] = 0;
  else
    for(int i = 0; i < NumWeights(); i++)
      w[i] = weights[i];
  return NumWeights();
}

float Attribute::MaxFeatureSumSqr() { 
  float n = 0;
  FeatureOptions *fo = new FeatureOptions(NULL, classes->GetFeatureParams(), NULL, classes);

  for(int i = 0; i < numFeatureTypes; i++) {
    FeatureWindow *fw = &features[i];
    SlidingWindowFeature *fe = fo->Feature(fw->name);
    n += fe->MaxFeatureSumSqr(fw->w, fw->h);
  }
  delete fo;

  return n;
}

void Attribute::SetWeights(double *w) { 
  if(!weights) weights = (float*)malloc(NumWeights()*sizeof(float));
  for(int i = 0; i < NumWeights(); i++) {
    weights[i] = (float)w[i];
  }
}

void Attribute::SetWeights(float *w) { 
  if(!weights) weights = (float*)malloc(NumWeights()*sizeof(float));
  for(int i = 0; i < NumWeights(); i++) {
    weights[i] = (float)w[i];
  }
}

int Attribute::GetWeights(float *w, const char *feature_type) {
  float *ptr = weights;
  for(int i = 0; i < numFeatureTypes; i++) {
    FeatureWindow *fw = &features[i];
    assert(fw);
    if(!strcmp(feature_type, fw->name)) {
      if(ptr)
	memcpy(w, ptr, sizeof(float)*fw->dim);
      else
	for(int i = 0; i < fw->dim; i++) 
	  w[i] = 0;
      return fw->dim;
    }
    ptr += fw->dim;
  }
  return 0;
}

IplImage ***AttributeInstance::Detect(ObjectPose *pose) {
  FeatureOptions *fo = process->Features();
  fo->SetNumThreads(process->NumThreads());
  return fo->SlidingWindowDetect(custom_weights ? custom_weights : model->Weights(), model->Features(), model->NumFeatureTypes(), isFlipped, pose);
};

IplImage ***AttributeInstance::GetResponses(ObjectPose *pose) { 
  if(!responses)
    responses = Detect(pose);
  return responses; 
}

void AttributeInstance::Clear() { 
  if(responses) {
    process->Features()->ReleaseResponses(&responses); 
    responses = NULL;
  }
}

float AttributeInstance::GetMaxResponse(int *best_x, int *best_y, int *best_scale, int *best_rot, 
					IplImage **reponseMap, IplImage **bestIndMap) { 
  IplImage ***resps = GetResponses();
  assert(resps);
  FeatureOptions *feat = process->Features();
  return ::GetMaxResponse(responses, feat->NumScales(), feat->NumOrientations(), best_x, best_y, best_scale, best_rot, 
			  reponseMap, bestIndMap);
}


int AttributeInstance::GetFeatures(float *f, PartLocation *loc, float *has_attribute, const char *feat_name, FeatureWindow *features, int numFeatures) {
  int l_x=-1, l_y=-1, l_scale=-1, l_rot=-1, l_pose=-1;
  int numWeights = 0; 
  if(!features) {
    features = model->Features();
    numFeatures = model->NumFeatureTypes();
  }

  for(int i = 0; i < numFeatures; i++) 
    numWeights += features[i].dim;
  if(loc)
    loc->GetDetectionLocation(&l_x, &l_y, &l_scale, &l_rot, &l_pose);

  if(model->Part() && model->Part()->GetPose(l_pose)->IsNotVisible()) {
    // Part is not visible
    for(int i = 0; i < numWeights; i++)
      f[i] = 0;
    return numWeights;
  }

  FeatureOptions *fo = process->Features();
  float *ptr = f;
  fo->SetNumThreads(process->NumThreads());
  for(int i = 0; i < numFeatures; i++) {
    FeatureWindow *fw = &features[i];
    SlidingWindowFeature *fe = fo->Feature(fw->name);
    float xx, yy;
    assert(fe && fw);
    if(!feat_name || !strcmp(fw->name, feat_name)) {
      int n;
      if(l_pose >= 0 && fw->poseInd >= 0 && l_pose != fw->poseInd) {
	// If this feature is associated with a particular pose other than the pose in 'loc', then
	// zero out the features
	n = fw->dim;
	for(int j = 0; j < n; j++) 
	  ptr[j] = 0;
      } else if(loc) {
	// Add in any x,y,scale, or orientation offsets defining where to extract features from
	PartLocation l(*loc);
        int x = l_x+fw->dx, y = l_y+fw->dy, scale = l_scale+fw->scale;
        int rot = (l_rot+fw->orientation+fo->NumOrientations())%fo->NumOrientations();
	if(scale >= fo->ScaleOffset() && scale < fo->NumScales()) {
	  fo->ConvertDetectionCoordinates(x, y, l_scale, l_rot, scale, rot, &xx, &yy);
	  l.SetDetectionLocation((int)(xx+.5f), (int)(yy+.5f), scale, rot, l_pose, LATENT, LATENT);
	}
	n = fe->GetFeaturesAtLocation(ptr, fw->w, fw->h, fw->scale, &l, isFlipped);
      } else {
	// Extract features from the whole image
      }
      if(has_attribute) {
        float h = has_attribute[model->Id()];
        for(int j = 0; h && j < n; j++) 
	  ptr[j] *= h;
      }
      ptr += n;
    }
  }
  return ptr-f;
}

float GetMaxResponse(IplImage ***responses, int numScales, int numOrientations, 
		     int *best_x, int *best_y, int *best_scale, int *best_rot, 
		     IplImage **responseMap, IplImage **bestIndMap) { 
  // For each pixel, combine all scales and orientations into a single max response.
  // Store the indices of the scale and orientation with the max response into a 2-channel image 'best'
  int scale, rot, x, y;
  IplImage *bestResponse = NULL, *best = NULL;
  for(scale = 0; scale < numScales; scale++) {
    assert(responses[scale][0]);
    IplImage *rS = cvCloneImage(responses[scale][0]);
    IplImage *bestS = cvCreateImage(cvSize(responses[scale][0]->width, responses[scale][0]->height), IPL_DEPTH_32S, 1);
    cvZero(bestS);
    for(rot = 1; rot < numOrientations; rot++) {
      IplImage *img = responses[scale][rot];
      unsigned char *imgPtr2 = (unsigned char*)img->imageData, *rSPtr2 = (unsigned char*)rS->imageData, 
	*bestSPtr2=(unsigned char*)bestS->imageData;
      float *imgPtr, *rSPtr;
      int *bestSPtr;
      for(y = 0; y < img->height; y++, imgPtr2+=img->widthStep, rSPtr2+=rS->widthStep, bestSPtr2+=bestS->widthStep) {
	for(x = 0, imgPtr=(float*)imgPtr2, rSPtr=(float*)rSPtr2, bestSPtr=(int*)bestSPtr2; x < img->width; x++) {
	  if(imgPtr[x] > rSPtr[x]) { rSPtr[x]=imgPtr[x]; bestSPtr[x] = rot; }
	}
      }
    }
    if(!bestResponse) {
      best = cvCreateImage(cvSize(responses[scale][0]->width, responses[scale][0]->height), IPL_DEPTH_32S, 2);
      cvZero(best);
      cvSetChannel(best, bestS, 1);
      bestResponse = rS;
      cvReleaseImage(&bestS);
    } else {
      IplImage *rS2 = cvCreateImage(cvSize(responses[0][0]->width, responses[0][0]->height), IPL_DEPTH_32F, 1);
      IplImage *bestS2 = cvCreateImage(cvSize(responses[0][0]->width, responses[0][0]->height), IPL_DEPTH_32S, 2);
      IplImage *bestS_tmp = cvCreateImage(cvSize(responses[0][0]->width, responses[0][0]->height), IPL_DEPTH_32S, 1);
      cvResize(rS, rS2);
      cvResize(bestS, bestS_tmp, CV_INTER_NN);
      cvSetChannel(bestS_tmp, bestS2, 1);
      cvReleaseImage(&bestS_tmp);

      unsigned char *imgPtr2 = (unsigned char*)rS2->imageData, *rSPtr2 = (unsigned char*)bestResponse->imageData, 
	*bestPtr2=(unsigned char*)best->imageData, *bestSPtr2=(unsigned char*)bestS2->imageData;
      float *imgPtr, *rSPtr;
      int *bestPtr, *bestSPtr;
      for(y = 0; y < rS2->height; y++, imgPtr2+=rS2->widthStep, rSPtr2+=bestResponse->widthStep, bestPtr2+=best->widthStep, 
	    bestSPtr2+=bestS2->widthStep) {
	for(x = 0, imgPtr=(float*)imgPtr2, rSPtr=(float*)rSPtr2, bestSPtr=((int*)bestSPtr2), bestPtr=((int*)bestPtr2);
	    x < rS2->width; x++) {
	  if(imgPtr[x] > rSPtr[x]) { rSPtr[x]=imgPtr[x]; bestPtr[(x<<1)] = scale; bestPtr[(x<<1)+1] = bestSPtr[x]; }
	}
      }

      cvReleaseImage(&rS); cvReleaseImage(&rS2); 
      cvReleaseImage(&bestS); cvReleaseImage(&bestS2); 
    }
  }

  assert(best && bestResponse);

  double min_val, max_val; CvPoint min_loc, max_loc;
  cvMinMaxLoc(bestResponse, &min_val, &max_val, &min_loc, &max_loc);
  if(best_x) *best_x = max_loc.x;
  if(best_y) *best_y = max_loc.y;
  if(best_scale) *best_scale = ((int*)(bestResponse->imageData+max_loc.y*bestResponse->widthStep))[max_loc.x*2];
  if(best_rot) *best_rot = ((int*)(bestResponse->imageData+max_loc.y*bestResponse->widthStep))[max_loc.x*2+1];

  if(responseMap) *responseMap = bestResponse;
  else cvReleaseImage(&bestResponse);

  if(bestIndMap) *bestIndMap = best;
  else cvReleaseImage(&best);

  return (float)max_val; 
}

AttributeInstance::AttributeInstance(Attribute *a, ImageProcess *p, bool isFlip) {
  process = p;

  model = a;
  
  responses = NULL;
  custom_weights = NULL;
  isFlipped = isFlip;
}

AttributeInstance::~AttributeInstance() {
  if(responses) 
    process->Features()->ReleaseResponses(&responses);
}

int AttributeInstance::Width() { return model->Feature(0)->w*process->Features()->CellWidth(); }
int AttributeInstance::Height() { return model->Feature(0)->h*process->Features()->CellWidth(); }


double AttributeInstance::GetLogLikelihoodAtLocation(PartLocation *loc, float attributeWeight, bool useGamma, float *features) {
  float *w = custom_weights;
  int nw = model->NumWeights();
  int n = nw;
  float *f = features;
  if(!f) {
    f = (float*)malloc(sizeof(float)*nw);
    n = GetFeatures(f, loc);
  }
  float *ww = (float*)malloc(sizeof(float)*nw);
  if(!w) {
    int n2 = model->GetWeights(ww);
    assert(n2 == n);
    w = ww;
  }

  float resp = 0;
  for(int i = 0; i < n; i++)
    resp += w[i]*f[i];
 
  if(f != features) 
    free(f);
  free(ww);
  assert(!isnan(resp));

  if(useGamma) {
    float gamma = model->GetGamma();
    float m = my_max(gamma*resp, -gamma*resp);
    float sum = exp(gamma*resp-m) + exp(-gamma*resp-m);
    float delta = -m - log(sum);  // delta is a per example constant that normalizes the attribute probabilities
    
    //fprintf(stderr, "Attribute %s %f %f, gamma=%f, delta=%f, ll=%f\n", model->Name(), exp(gamma*resp+delta), exp(-gamma*resp+delta), gamma, delta, (hasAttribute ? gamma*resp : -gamma*resp) + delta); 
    
    return attributeWeight*gamma*resp + delta;
  } else
    return attributeWeight*resp;
}

ObjectPartInstance *AttributeInstance::Part() { 
  return model->Part() ? process->GetPartInst(model->Part()->Id()) : NULL; 
}



int AttributeInstance::Debug(float *w, float *f, bool print_weights, float *f_gt) {
  if(print_weights) {
    for(int i = 0; i < model->NumWeights(); i++) {
      if(f[i] || w[i] || (f_gt && f_gt[i])) {
        if(f_gt) fprintf(stderr, " (%d:%s:%.7f:%f:%f)", (int)((f+i)-g_f), model->Name(), w[i], f[i], f_gt[i]);
	else fprintf(stderr, " (%d:%s:%.7f:%f)", (int)((f+i)-g_f), model->Name(), w[i], f[i]);
      }
    }
  }
  return model->NumWeights();
}
