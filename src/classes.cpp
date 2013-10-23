/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#include "classes.h"
#include "attribute.h"
#include "pose.h"
#include "part.h"
#include "class.h"
#include "question.h"
#include "dataset.h"
#include "spatialModel.h"
#include "histogram.h"
#include "fisher.h"
#include "hog.h"

Classes::Classes(ScaleOrientationMethod s) {
  attributes = NULL;
  numAttributes = 0;

  allAttributesHaveSaveFeatures = true;

  parts = NULL;
  numParts = 0;

  poses = NULL;
  numPoses = 0;

  classes = NULL;
  numClasses = 0;

  binaryWeightFile = NULL;

  clickParts = NULL;
  numClickParts = 0;
  clickPoses = NULL;
  numClickPoses = 0;
  supportsMultiObjectDetection = false;

  spatialTransitions = NULL;
  numSpatialTransitions = 0;

  certainties = NULL;
  numCertainties = 0;
  certaintyWeights = NULL;

  questions = NULL;
  numQuestions = 0;
  imageScale = 1;

  attributeMeans = NULL;
  attributeCovariances = NULL;
  classAttributeMeans = NULL;
  classAttributeCovariances = NULL;

  scaleAttributesIndependently = true;
  gamma_class = gamma_click = 1;

  codebooks = NULL;
  numCodebooks = 0;

  fisherCodebooks = NULL;
  numFisherCodebooks = 0;

  scaleOrientationMethod = s;

  features = NULL;
  numFeatureTypes = 0;

  featureWindows = NULL;
  numFeatureWindows = 0;

  params.scaleOffset = 0;
  params.numScales = 100;
  params.numOrientations = 1;
  params.spatialGranularity = 8;
  InitHOGParams(&params.hogParams, params.numScales);
  detectionLoss = LOSS_PART_AVERAGE_AREA_UNION_OVER_INTERSECTION;

  poseOffsets = spatialOffsets = NULL;
  numOffsets = 0;
}

Classes::~Classes() {
  int i;

  if(poseOffsets) delete [] poseOffsets;
  if(spatialOffsets) delete [] spatialOffsets;

  if(features)
    free(features);

  if(featureWindows) {
    for(int i = 0; i < numFeatureWindows; i++) 
      if(featureWindows[i].name) 
	free(featureWindows[i].name);
    free(featureWindows);
  }

  if(attributes) {
    for(i = 0; i < numAttributes; i++)
      delete attributes[i];
    free(attributes);
  }

  if(parts) {
    for(i = 0; i < numParts; i++)
      delete parts[i];
    free(parts);
  }

  if(poses) {
    for(i = 0; i < numPoses; i++)
      delete poses[i];
    free(poses);
  }

  if(clickParts) {
   for(i = 0; i < numClickParts; i++)
      delete clickParts[i];
    free(clickParts);
  }

  if(clickPoses) {
    for(i = 0; i < numClickPoses; i++)
      delete clickPoses[i];
    free(clickPoses);
  }

  if(classes) {
    for(i = 0; i < numClasses; i++)
      delete classes[i];
    free(classes);
  }

  if(certainties) {
    for(i = 0; i < numCertainties; i++)
      free(certainties[i]);
    free(certainties);
  }
  if(certaintyWeights) 
    free(certaintyWeights);

  if(questions) {
    for(i = 0; i < numQuestions; i++)
      delete questions[i];
    free(questions);
  }

  if(spatialTransitions) {
    free(spatialTransitions);
  }

  RemoveCodebooks();
}

void Classes::RemoveCodebooks() {
  if(codebooks) {
    for(int i = 0; i < numCodebooks; i++)
      delete codebooks[i];
    free(codebooks);
  }
  if(fisherCodebooks) {
    for(int i = 0; i < numFisherCodebooks; i++)
      delete fisherCodebooks[i];
    free(fisherCodebooks);
  }
  codebooks = NULL;
  numCodebooks = 0;
  fisherCodebooks = NULL;
  numFisherCodebooks = 0;
}

void Classes::ResolveLinks() {
  int i;
  for(i = 0; i < numParts; i++) 
    parts[i]->ResolveLinks(this);
  for(i = 0; i < numPoses; i++) 
    poses[i]->ResolveLinks(this);
  for(i = 0; i < numAttributes; i++) 
    attributes[i]->ResolveLinks(this);
  for(i = 0; i < numClickParts; i++) 
    clickParts[i]->ResolveLinks(this);
  for(i = 0; i < numClickPoses; i++) 
    clickPoses[i]->ResolveLinks(this);
  for(i = 0; i < numClasses; i++) 
    classes[i]->ResolveLinks(this);
  for(i = 0; i < numQuestions; i++) 
    questions[i]->ResolveLinks(this);
  for(i = 0; i < numSpatialTransitions; i++) {
    spatialTransitions[i]->ResolveLinks(this);
    if(!spatialTransitions[i]->IsClick())
      parts[spatialTransitions[i]->GetParentPart()->Id()]->AddSpatialTransition(spatialTransitions[i]);
    else
      clickParts[spatialTransitions[i]->GetParentPart()->Id()]->AddSpatialTransition(spatialTransitions[i]);
  }
}

// Sort the parts so that the child most parts occur first in the array
void Classes::TopologicallySortParts(int *inds) {
  int i;

  currPart = 0;
  newPartsArray = (ObjectPart**)malloc(sizeof(ObjectPart*)*numParts);
  for(i = 0; i < numParts; i++) 
    parts[i]->SetClassInd(-1);
  for(i = 0; i < numParts; i++) 
    TopologicallySortParts(parts[i]);
  ObjectPart **oldParts = parts;
  parts = newPartsArray;
  free(oldParts);
  newPartsArray = NULL;
  for(i = 0; i < numParts; i++) {
    if(inds) inds[i] = parts[i]->Id();
    parts[i]->SetId(i);
  }
}

void Classes::TopologicallySortParts(ObjectPart *part) {
  int i;

  if(part->GetClassInd() != -1)
    return;
  for(i = 0; i < part->NumParts(); i++) 
    TopologicallySortParts(part->GetPart(i));
  int ind = part->GetClassInd();
  assert(ind == -1);
  newPartsArray[currPart] = part;
  part->SetClassInd(currPart++);
}


void Classes::OnModelChanged() {
  if(poseOffsets) delete [] poseOffsets;
  if(spatialOffsets) delete [] spatialOffsets;
  poseOffsets = spatialOffsets = NULL;
  numOffsets = 0;
}

void Classes::AddPart(ObjectPart *part) {
  if(part->Id() < 0) part->SetId(numParts);
  assert(part->Id() == numParts);
  parts = (ObjectPart**)realloc(parts, sizeof(ObjectPart*)*(numParts+1));
  parts[numParts++] = part;
  OnModelChanged();
  part->classes = this;
}

void Classes::AddPose(ObjectPose *pose) {
  if(pose->Id() < 0) pose->SetId(numPoses);
  assert(pose->Id() == numPoses);
  poses = (ObjectPose**)realloc(poses, sizeof(ObjectPose*)*(numPoses+1));
  poses[numPoses++] = pose;
  pose->classes = this;
  if(pose->appearanceModel) 
    pose->appearanceModel->SetClasses(this);
  OnModelChanged();
}

void Classes::AddClickPart(ObjectPart *part) {
  if(part->Id() < 0) part->SetId(numClickParts);
  assert(part->Id() == numClickParts);
  clickParts = (ObjectPart**)realloc(clickParts, sizeof(ObjectPart*)*(numClickParts+1));
  clickParts[numClickParts++] = part;
  part->classes = this;
}

void Classes::AddClickPose(ObjectPose *pose) {
  if(pose->Id() < 0) pose->SetId(numClickPoses);
  assert(pose->Id() == numClickPoses);
  clickPoses = (ObjectPose**)realloc(clickPoses, sizeof(ObjectPose*)*(numClickPoses+1));
  clickPoses[numClickPoses++] = pose;
  pose->classes = this;
  OnModelChanged();
}

void Classes::AddCertainty(const char *c) {
  certainties = (char**)realloc(certainties, sizeof(char*)*(numCertainties+1));
  certainties[numCertainties++] = StringCopy(c);
  OnModelChanged();
}

void Classes::AddQuestion(Question *q) {
  if(q->Id() < 0) q->SetId(numQuestions);
  assert(q->Id() == numQuestions);
  questions = (Question**)realloc(questions, sizeof(Question*)*(numQuestions+1));
  questions[numQuestions++] = q;
  OnModelChanged();
}

void Classes::AddSpatialTransition(ObjectPartPoseTransition *t) {
  t->SetId(numSpatialTransitions);
  spatialTransitions = (ObjectPartPoseTransition**)realloc(spatialTransitions, sizeof(ObjectPartPoseTransition*)*(numSpatialTransitions+1));
  spatialTransitions[numSpatialTransitions++] = t;
}

ObjectPart *Classes::CreateClickPart(ObjectPart *part) {
  ObjectPart *clickPart = new ObjectPart(part->Name(), true, part->GetAbbreviation());
  clickPart->SetParent(part);
  clickPart->SetId(part->Id());
  clickPart->classes = this;
  clickParts = (ObjectPart**)realloc(clickParts, sizeof(ObjectPart*)*(numClickParts+1));
  clickParts[numClickParts++] = clickPart;

  for(int i = 0; i < part->NumPoses(); i++) {
    ObjectPose *clickPose = CreateClickPose(part->GetPose(i));
    clickPart->AddPose(clickPose);
  }
  OnModelChanged();
  
  return clickPart;
}

ObjectPose *Classes::CreateClickPose(ObjectPose *pose) {
  ObjectPose *clickPose = new ObjectPose(pose->Name(), true);
  clickPose->SetId(numClickPoses);
  clickPose->classes = this;
  clickPoses = (ObjectPose**)realloc(clickPoses, sizeof(ObjectPose*)*(numClickPoses+1));
  clickPoses[numClickPoses++] = clickPose;
  OnModelChanged();
  return clickPose;
}

void Classes::AddAttribute(Attribute *attribute) {
  if(numAttributes && allAttributesHaveSaveFeatures) {
    if(attribute->NumFeatureTypes() != attributes[0]->NumFeatureTypes())
      allAttributesHaveSaveFeatures = false;
    else {
      for(int i = 0; i < attribute->NumFeatureTypes(); i++) {
	if(!IsSameFeatureWindow(*attribute->Feature(i), *attributes[0]->Feature(i)))
	  allAttributesHaveSaveFeatures = false;
      }
    }
  }

  attribute->SetId(numAttributes);
  attributes = (Attribute**)realloc(attributes, sizeof(Attribute*)*(numAttributes+1));
  attributes[numAttributes++] = attribute;
  attribute->classes = this;
  OnModelChanged();
}

void Classes::AddClass(ObjectClass *cl) {
  cl->SetId(numClasses);
  classes = (ObjectClass**)realloc(classes, sizeof(ObjectClass*)*(numClasses+1));
  classes[numClasses++] = cl;
  cl->classes = this;
  OnModelChanged();
}

int *Classes::Subset(int *subset, int num) {
  ObjectClass **classes_new = (ObjectClass**)malloc(sizeof(ObjectClass*)*(num));
  int *new_id = new int[numClasses];
  for(int i = 0; i < numClasses; i++)
    new_id[i] = -1;
  for(int i = 0; i < num; i++) {
    new_id[subset[i]] = i;
    classes_new[i] = classes[subset[i]];
    classes_new[i]->SetId(i);
  }
  for(int i = 0; i < numClasses; i++)
    if(new_id[i] < 0) delete classes[i];
  free(classes);
  classes = classes_new;  
  numClasses = num;
  return new_id;
}


void Classes::CreateClickParts() {
  for(int i = 0; i < numParts; i++)
    CreateClickPart(parts[i]);
}

PartDetectionLossType Classes::DetectionLossMethodFromString(const char *l) {
  PartDetectionLossType lossMethod = LOSS_PART_AVERAGE_AREA_UNION_OVER_INTERSECTION;
  if(!strcmp(l, "intersection_parts")) 
    lossMethod = LOSS_PART_AVERAGE_AREA_UNION_OVER_INTERSECTION;
  else if(!strcmp(l, "intersection_object")) 
    lossMethod = LOSS_BOUNDING_BOX_AREA_UNION_OVER_INTERSECTION;
  else if(!strcmp(l, "num_click_std_dev")) 
    lossMethod = LOSS_USER_STANDARD_DEVIATIONS;
  else if(!strcmp(l, "num_parts_incorrect")) 
    lossMethod = LOSS_NUM_INCORRECT;
  else if(!strcmp(l, "detection")) 
    lossMethod = LOSS_DETECTION;
  else {
    fprintf(stderr, "Unknown loss type %s\n", l);
    assert(0);
  }
  return lossMethod;
}
void Classes::DetectionLossMethodToString(PartDetectionLossType m, char *l) {
  if(m == LOSS_PART_AVERAGE_AREA_UNION_OVER_INTERSECTION) strcpy(l, "intersection_parts");
  else if(m == LOSS_BOUNDING_BOX_AREA_UNION_OVER_INTERSECTION) strcpy(l, "intersection_object");
  else if(m == LOSS_USER_STANDARD_DEVIATIONS) strcpy(l, "num_click_std_dev");
  else if(m == LOSS_NUM_INCORRECT) strcpy(l, "num_parts_incorrect");
  else if(m == LOSS_DETECTION) strcpy(l, "detection");
}

Json::Value Classes::Save() {
  Json::Value root;
  int i;
  
  Json::Value hog = SiftParamsToJSON(&params.hogParams);
  root["version"] = API_VERSION;
  root["hogParams"] = hog;
  root["numScales"] = params.numScales;
  root["numOrientations"] = params.numOrientations ;
  root["spatialGranularity"] = params.spatialGranularity;
  root["scaleOffset"] = params.scaleOffset;
  root["gammaClass"] = gamma_class;
  root["gammaClick"] = gamma_click;
  root["scaleAttributesIndependently"] = scaleAttributesIndependently;
  root["scaleOrientationMethod"] = scaleOrientationMethod;
  char detectionLossStr[1000];  DetectionLossMethodToString(detectionLoss, detectionLossStr);
  root["detectionLoss"] = detectionLossStr;
  root["supportsMultiObjectDetection"] = supportsMultiObjectDetection;
  root["imageScale"] = imageScale;

  Json::Value cb(Json::arrayValue);
  for(i = 0; i < numCodebooks; i++)
    cb[i] = codebooks[i]->FileName();
  root["codebooks"] = cb;

  Json::Value fcb(Json::arrayValue);
  for(i = 0; i < numFisherCodebooks; i++)
    fcb[i] = fisherCodebooks[i]->FileName();
  root["fisherCodebooks"] = fcb;

  Json::Value cert(Json::arrayValue);
  for(i = 0; i < numCertainties; i++) {
    Json::Value w;
    w["name"] = certainties[i];
    if(certaintyWeights) w["weight"] = certaintyWeights[i];
    cert[i] = w;
  }
  root["certainties"] = cert;

  Json::Value pa(Json::arrayValue);
  for(i = 0; i < numParts; i++)
    pa[i] = parts[i]->Save();
  root["parts"] = pa;

  Json::Value po(Json::arrayValue);
  for(i = 0; i < numPoses; i++)
    po[i] = poses[i]->Save();
  root["poses"] = po;

  Json::Value cpa(Json::arrayValue);
  for(i = 0; i < numClickParts; i++)
    cpa[i] = clickParts[i]->Save();
  root["clickParts"] = cpa;

  Json::Value cpo(Json::arrayValue);
  for(i = 0; i < numClickPoses; i++)
    cpo[i] = clickPoses[i]->Save();
  root["clickPoses"] = cpo;

  Json::Value attr(Json::arrayValue);
  for(i = 0; i < numAttributes; i++)
    attr[i] = attributes[i]->Save();
  root["attributes"] = attr;

  Json::Value cl(Json::arrayValue);
  for(i = 0; i < numClasses; i++)
    cl[i] = classes[i]->Save();
  root["classes"] = cl;

  Json::Value qu(Json::arrayValue);
  for(i = 0; i < numQuestions; i++)
    qu[i] = questions[i]->Save();
  root["questions"] = qu;

  int num = 0;
  Json::Value tr;
  for(i = 0; i < numSpatialTransitions; i++)
    if(spatialTransitions[i]->IsClick() || spatialTransitions[i]->GetParentPart()->GetParent() != spatialTransitions[i]->GetChildPart())
      tr[num++] = spatialTransitions[i]->Save();
  root["spatialTransitions"] = tr;

  Json::Value fe;
  for(i = 0; i < (int)root["features"].size(); i++) 
    fe[i] = SiftParamsToJSON(&features[i]);
  root["features"] = fe;

  Json::Value feats;
  for(int i = 0; i < numFeatureWindows; i++) 
    feats[i] = SaveFeatureWindow(&featureWindows[i]);
  root["featureWindows"] = feats;

  return root;
}

void Classes::AddDefaultFeatures() {
  numFeatureTypes = 6;
  features = (SiftParams*)realloc(features, sizeof(SiftParams)*(numFeatureTypes));
  features[0] = features[1] = features[2] = features[3] = params.hogParams;
  features[1].numBins = features[2].numBins = 3;
  features[3].numBins = 1;
  features[1].cellWidth = features[2].cellWidth = features[3].cellWidth = 1;
  strcpy(features[1].name, "RGB");   strcpy(features[1].type, "RGB");
  strcpy(features[2].name, "CIE");   strcpy(features[2].type, "CIE");
  strcpy(features[3].name, "MASK");   strcpy(features[3].type, "MASK");
  InitSiftParams(&features[4]);  
  InitSiftParams(&features[5]);  strcpy(features[5].name, "SIFT4"); features[5].cellWidth = 4;  features[5].smoothWidth = 0; 
}

bool Classes::Load(const Json::Value &root) {
  params.numScales = root.get("numScales", params.hogParams.maxScales).asInt();
  params.numOrientations = root.get("numOrientations", params.hogParams.maxOrientations).asInt();
  params.spatialGranularity = root.get("spatialGranularity", params.hogParams.cellWidth).asInt();
  params.scaleOffset = root.get("scaleOffset", 0).asInt();

  if(root.isMember("hogParams"))
    params.hogParams = SiftParamsFromJSON(root["hogParams"]);

  if(root.isMember("features") && root["features"].isArray() && root["features"].size()) {
    numFeatureTypes = root["features"].size();
    features = (SiftParams*)realloc(features, sizeof(SiftParams)*(numFeatureTypes));
    for(int i = 0; i < (int)root["features"].size(); i++) {
      features[i] = SiftParamsFromJSON(root["features"][i]);
      if(i == 0 && !root.isMember("hogParams"))
	params.hogParams = features[i];
    }
  } else {
    AddDefaultFeatures();
  }

  if(root.isMember("featureWindows") && root["featureWindows"].isArray()) {
    for(int i = 0; i < (int)root["featureWindows"].size(); i++) {
      featureWindows = (FeatureWindow*)realloc(featureWindows, (numFeatureWindows+1)*sizeof(FeatureWindow));
      if(!LoadFeatureWindow(&featureWindows[numFeatureWindows], root["featureWindows"][i])) {
	fprintf(stderr, "Error parsing feature\n");
	return false;
      }
      numFeatureWindows++;
    }
  }

  gamma_class = root.get("gammaClass", 1).asFloat();
  gamma_click = root.get("gammaClick", 1).asFloat();
  imageScale = root.get("imageScale", 1).asFloat();
  scaleAttributesIndependently = root.get("scaleAttributesIndependently", true).asBool();
  scaleOrientationMethod = (ScaleOrientationMethod)root.get("scaleOrientationMethod", SO_PARENT_CHILD_SAME_SCALE_ORIENTATION).asInt();
  char dl[1000]; strcpy(dl, root.get("detectionLoss", "intersection_parts").asString().c_str());
  detectionLoss = DetectionLossMethodFromString(dl);
  supportsMultiObjectDetection = root.get("supportsMultiObjectDetection", false).asBool();

  if(root.isMember("codebooks") && root["codebooks"].isArray()) {
    char fname[1000];
    for(int i = 0; i < (int)root["codebooks"].size(); i++) {
      strcpy(fname, root["codebooks"][i].asString().c_str());
      FeatureDictionary *d = new FeatureDictionary();
      bool b = d->Load(fname); assert(b);
      AddCodebook(d);
    }
  }

  if(root.isMember("fisherCodebooks") && root["fisherCodebooks"].isArray()) {
    char fname[1000];
    for(int i = 0; i < (int)root["fisherCodebooks"].size(); i++) {
      strcpy(fname, root["fisherCodebooks"][i].asString().c_str());
      FisherFeatureDictionary *d = new FisherFeatureDictionary();
      bool b = d->Load(fname); assert(b);
      AddFisherCodebook(d);
    }
  }

  if(root.isMember("certainties") && root["certainties"].isArray()) {
    char certainty[1000];
    for(int i = 0; i < (int)root["certainties"].size(); i++) {
      if(!root["certainties"][i].isObject() || !root["certainties"][i].isMember("name")) {
	fprintf(stderr, "Error parsing certainty\n");
	return false;
      }
      strcpy(certainty, root["certainties"][i]["name"].asString().c_str());
      AddCertainty(certainty);
      if(root["certainties"][i].isMember("weight")) {
	certaintyWeights = (float*)realloc(certaintyWeights, sizeof(float)*numCertainties);
	certaintyWeights[numCertainties-1] = root["certainties"][i].get("weight",0).asFloat();
      }
    }
  }

  if(root.isMember("parts") && root["parts"].isArray()) {
    for(int i = 0; i < (int)root["parts"].size(); i++) {
      ObjectPart *part = new ObjectPart();
      if(!part->Load(root["parts"][i])) { delete part; return false; }
      AddPart(part);
    }
  }

  if(root.isMember("poses") && root["poses"].isArray()) {
    for(int i = 0; i < (int)root["poses"].size(); i++) {
      ObjectPose *pose = new ObjectPose();
      pose->SetClasses(this);
      if(!pose->Load(root["poses"][i])) { delete pose; return false; }
      AddPose(pose);
    }
  }

  if(root.isMember("clickParts") && root["clickParts"].isArray()) {
    for(int i = 0; i < (int)root["clickParts"].size(); i++) {
      ObjectPart *part = new ObjectPart();
      if(!part->Load(root["clickParts"][i])) { delete part; return false; }
      AddClickPart(part);
    }
  }

  if(root.isMember("clickPoses") && root["clickPoses"].isArray()) {
    for(int i = 0; i < (int)root["clickPoses"].size(); i++) {
      ObjectPose *pose = new ObjectPose();
      if(!pose->Load(root["clickPoses"][i])) { delete pose; return false; }
      AddClickPose(pose);
    }
  }


  if(root.isMember("attributes") && root["attributes"].isArray()) {
    for(int i = 0; i < (int)root["attributes"].size(); i++) {
      Attribute *attribute = new Attribute();
      attribute->SetClasses(this);
      if(!attribute->Load(root["attributes"][i])) { delete attribute; return false; }
      AddAttribute(attribute);
    }
  }

  if(root.isMember("classes") && root["classes"].isArray()) {
    for(int i = 0; i < (int)root["classes"].size(); i++) {
      ObjectClass *c = new ObjectClass(this);
      c->SetClasses(this);
      if(!c->Load(root["classes"][i])) { delete c; return false; }
      AddClass(c);
    }
  }

  if(root.isMember("spatialTransitions") && root["spatialTransitions"].isArray()) {
    for(int i = 0; i < (int)root["spatialTransitions"].size(); i++) {
      ObjectPartPoseTransition *c = new ObjectPartPoseTransition;
      if(!c->Load(root["spatialTransitions"][i])) { delete c; return false; }
      AddSpatialTransition(c);
    }
  }

  if(root.isMember("questions") && root["questions"].isArray()) {
    for(int i = 0; i < (int)root["questions"].size(); i++) {
      Question *q = Question::New(root["questions"][i], this);
      if(!q) { delete q; return false; }
      AddQuestion(q);
    }
  }

  ResolveLinks();
  TopologicallySortParts();

  return true;
}
bool Classes::Save(const char *fname) {
  if(!fname) fname = this->fname;

  char bname[1000];
  bool saveWeightsInBinaryFormat = NumWeights(true, true) > 100000 || attributeMeans;
  if(saveWeightsInBinaryFormat) {
    sprintf(bname, "%s.weights", fname);
    binaryWeightFile = fopen(bname, "wb");
  }

  Json::StyledWriter writer;
  FILE *fout = fopen(fname,"w");
  if(!fout) { fprintf(stderr, "Failed to open %s for writing\n", fname); return false; }
  Json::Value v = Save();
  if(saveWeightsInBinaryFormat)
    v["binaryWeightsFile"] = bname;
  fprintf(fout, "%s",  writer.write(v).c_str()); 
  fclose(fout);

  if(saveWeightsInBinaryFormat) {
    fclose(binaryWeightFile);
    binaryWeightFile = NULL;
  }
  return true;
}

bool Classes::Load(const char *fname) {
  strcpy(this->fname, fname);
  char *str = ReadStringFile(fname);
  if(!str) { fprintf(stderr, "Failed to read classes from %s\n", fname); return false; }
  Json::Reader reader;
  Json::Value root;
  bool retval = true;

  if(!reader.parse(str, root)) {
    fprintf(stderr, "Failed to parse classes from %s\n", fname); 
    retval = false;
  } else {
    if(root.isMember("binaryWeightsFile")) {
      binaryWeightFile = fopen(root["binaryWeightsFile"].asString().c_str(), "rb");
      if(!binaryWeightFile) {
	fprintf(stderr, "Failed to load class weights from %s\n", fname); 
	retval = false;
      }
    }

    if(!Load(root)) {
      fprintf(stderr, "Failed to load classes from %s\n", fname); 
      retval = false;
    }
  }

  if(binaryWeightFile) {
    fclose(binaryWeightFile);
    binaryWeightFile = NULL;
  }

  free(str);
  return retval;
}

ObjectClass *Classes::FindClass(const char *p) {
  for(int i = 0; i < numClasses; i++)
    if(!strcmp(classes[i]->Name(), p)) 
      return classes[i];
  return NULL;
}

ObjectPose *Classes::FindPose(const char *p) {
  for(int i = 0; i < numPoses; i++) 
    if(!strcmp(poses[i]->Name(), p))
      return poses[i];
  return NULL;
}

ObjectPart *Classes::FindPart(const char *p) {
  for(int i = 0; i < numParts; i++) 
    if(!strcmp(parts[i]->Name(), p))
      return parts[i];
  return NULL;
}

ObjectPose *Classes::FindClickPose(const char *p) {
  for(int i = 0; i < numClickPoses; i++) 
    if(!strcmp(clickPoses[i]->Name(), p))
      return clickPoses[i];
  return NULL;
}

ObjectPart *Classes::FindClickPart(const char *p) {
  for(int i = 0; i < numClickParts; i++) 
    if(!strcmp(clickParts[i]->Name(), p))
      return clickParts[i];
  return NULL;
}

Attribute *Classes::FindAttribute(const char *a) {
  for(int i = 0; i < numAttributes; i++) 
    if(!strcmp(attributes[i]->Name(), a))
      return attributes[i];
  return NULL;
}

int Classes::FindCertainty(const char *a) {
  for(int i = 0; i < numCertainties; i++) 
    if(!strcmp(certainties[i], a))
      return i;
  return -1;
}

int Classes::GetWeights(double *w, bool getPartFeatures, bool getAttributeFeatures) {
  int n = NumWeights(getPartFeatures, getAttributeFeatures);
  float *ww = (float*)malloc(sizeof(float)*(n+1));
  GetWeights(ww, getPartFeatures, getAttributeFeatures);
  for(int i = 0; i < n; i++)
    w[i] = (double)ww[i];
  free(ww);
  return n;
}

int Classes::GetWeightContraints(int *wc, bool *learn_weights, bool *regularize, bool getPartFeatures, 
				 bool getAttributeFeatures) {
  int *curr = wc, i;
  int m = NumWeights(getPartFeatures, getAttributeFeatures);
  for(i = 0; i < m; i++) {
    wc[i] = 0;
    learn_weights[i] = regularize[i] = true;
  }


  bool *currL = learn_weights, *currR = regularize;
  if(getPartFeatures) {
    for(i = 0; i < NumParts(); i++) {
      if(!GetPart(i)->IsClick()) {
        int n = GetPart(i)->GetStaticWeightConstraints(curr, currL, currR);
        curr += n; currL += n;  currR += n; 
	  }
    }
    for(i = 0; i < NumSpatialTransitions(); i++) {
      if(!GetSpatialTransition(i)->IsClick()) {
        int n = GetSpatialTransition(i)->GetWeightConstraints(curr, currL, currR);
        curr += n; currL += n;  currR += n; 
	  }
    }
    for(i = 0; i < NumPoses(); i++) {
      if(!GetPose(i)->IsClick()) {
        int n = GetPose(i)->NumWeights();
        curr += n; currL += n;  currR += n; 
	  }
    }
  }
  
  if(getAttributeFeatures) {
    for(i = 0; i < NumAttributes(); i++) {
      int n = GetAttribute(i)->NumWeights();
      curr += n; currL += n;  currR += n; 
    }
  }

  return (int)(curr-wc);
}

int Classes::GetWeights(float *w, bool getPartFeatures, bool getAttributeFeatures) {
  int i;
  float *curr = w;
  if(getPartFeatures) {
    for(i = 0; i < NumParts(); i++) 
      if(!GetPart(i)->IsClick())
        curr += GetPart(i)->GetStaticWeights(curr);
    for(i = 0; i < NumSpatialTransitions(); i++) 
      if(!GetSpatialTransition(i)->IsClick())
        curr += GetSpatialTransition(i)->GetWeights(curr);
    for(i = 0; i < NumPoses(); i++) 
      if(!GetPose(i)->IsClick())
        curr += GetPose(i)->GetWeights(curr);
  }

  if(getAttributeFeatures) {
    for(i = 0; i < NumAttributes(); i++) 
      curr += GetAttribute(i)->GetWeights(curr);
  }

  return (int)(curr-w);
}

int Classes::NumWeights(bool getPartFeatures, bool getAttributeFeatures) {
  int n = 0, i;
  if(getPartFeatures) {
    for(i = 0; i < NumParts(); i++) 
      if(!GetPart(i)->IsClick())
        n += GetPart(i)->NumStaticWeights();
    for(i = 0; i < NumSpatialTransitions(); i++)
      if(!GetSpatialTransition(i)->IsClick())
        n += GetSpatialTransition(i)->NumWeights();
    for(i = 0; i < NumPoses(); i++) 
      if(!GetPose(i)->IsClick())
        n += GetPose(i)->NumWeights();
  }

  if(getAttributeFeatures) {
    for(i = 0; i < NumAttributes(); i++) 
      n += GetAttribute(i)->NumWeights();
  }

  return n;
}

float Classes::MaxFeatureSumSqr(bool getPartFeatures, bool getAttributeFeatures) {
  float n = 0;
  int i;
  if(getPartFeatures) {
    for(i = 0; i < NumParts(); i++) 
      if(!GetPart(i)->IsClick())
        n += GetPart(i)->MaxFeatureSumSqr();
    for(i = 0; i < NumPoses(); i++) 
      if(!GetPose(i)->IsClick())
        n += GetPose(i)->MaxFeatureSumSqr();
  }

  if(getAttributeFeatures) {
    for(i = 0; i < NumAttributes(); i++) 
      n += GetAttribute(i)->MaxFeatureSumSqr();
  }

  return n;
}

SiftParams *Classes::Feature(const char *n) {
  for(int i = 0; i < numFeatureTypes; i++)
    if(!strcmp(n, features[i].name))
      return &features[i];
  return NULL;
}

void Classes::SetWeights(double *w, bool setPartFeatures, bool setAttributeFeatures) {
  int n = NumWeights(setPartFeatures, setAttributeFeatures);
  float *ww = (float*)malloc(sizeof(float)*(n+1));
  for(int i = 0; i < n; i++)
    ww[i] = (float)w[i];
  SetWeights(ww, setPartFeatures, setAttributeFeatures);
  free(ww);
}

void Classes::SetWeights(float *w, bool setPartFeatures, bool setAttributeFeatures) {
  int i;
  float *curr = w;
  if(setPartFeatures) {
    for(i = 0; i < NumParts(); i++) {
      if(!GetPart(i)->IsClick()) {
        GetPart(i)->SetStaticWeights(curr);
        curr += GetPart(i)->NumStaticWeights();
      }
    }
    for(i = 0; i < NumSpatialTransitions(); i++) {
      if(!GetSpatialTransition(i)->IsClick()) {
        GetSpatialTransition(i)->SetWeights(curr);
        curr += GetSpatialTransition(i)->NumWeights();
      }
    }
    for(i = 0; i < NumPoses(); i++) {
      if(!GetPose(i)->IsClick()) {
        GetPose(i)->SetWeights(curr);
        curr += GetPose(i)->NumWeights();
      }
    }
  }
  if(setAttributeFeatures) {
    for(i = 0; i < NumAttributes(); i++) {
      GetAttribute(i)->SetWeights(curr);
      curr += GetAttribute(i)->NumWeights();
    }
  }
}

int Classes::GetWeightOffsets(int **pose_offsets, int **spatial_offsets) {
  if(!poseOffsets) {
    int i;
    poseOffsets = new int[10000];
    spatialOffsets = new int[10000];
    numOffsets = 0;
    for(i = 0; i < NumSpatialTransitions(); i++) {
      spatialOffsets[i] = numOffsets;
      numOffsets += GetSpatialTransition(i)->IsClick() ? 0 : GetSpatialTransition(i)->NumWeights();
    }
    for(i = 0; i < NumPoses(); i++) {
      poseOffsets[i] = numOffsets;
      numOffsets += GetPose(i)->IsClick() ? 0 : GetPose(i)->NumWeights();
    }
    for(i = 0; i < NumPoses(); i++) {
      if(poses[i]->IsFlipped())
        poseOffsets[i] = poseOffsets[poses[i]->GetFlipped()->Id()];
    }
  }
  *pose_offsets = poseOffsets;
  *spatial_offsets = spatialOffsets;
  
  return numOffsets;
}


bool Classes::HasPartPoseTransitions() {
  for(int p = 0; p < numParts; p++) {
    int **numChild = NULL;
    ObjectPartPoseTransition ****c = parts[p]->GetPosePartTransitions(&numChild);
    for(int i = 0; i < numPoses; i++) 
      for(int j = 0; j <= numParts; j++)  
        for(int k = 0; c && numChild && c[i] && c[i][j] && k < numChild[i][j]; k++) 
	  if(c[i][j][k])
	    return true;
  }
  return false;
}
bool Classes::HasClickPartPoseTransitions() {
  for(int p = 0; p < numClickParts; p++) {
    int **numChild = NULL;
    ObjectPartPoseTransition ****c = clickParts[p]->GetPosePartTransitions(&numChild);
    for(int i = 0; i < numPoses; i++) 
      for(int j = 0; j <= numParts; j++)  
        for(int k = 0; c && numChild && c[i] && c[i][j] && k < numChild[i][j]; k++) 
	  if(c[i][j][k])
	    return true;
  }
  return false;
}

void Classes::AddSpatialTransitions(Dataset *d, int minExamples, bool computeClickParts, bool computeParts, bool combinePoses) {
  int i, j;
  // Make sure all spatial weights are zeroed out initially
  for(j = 0; j < numSpatialTransitions; j++) 
    if((computeParts && !spatialTransitions[j]->IsClick()) || (computeClickParts && spatialTransitions[j]->IsClick())) 
      spatialTransitions[j]->Reset();

  for(i = 0; i < d->NumExamples(); i++) {
    for(int o = 0; o < d->GetExampleLabel(i)->NumObjects(); o++) {
      PartLocation *locs = d->GetExampleLabel(i)->GetObject(o)->GetPartLocations();
      if(!locs) continue;
      PartLocation *locs_flipped = PartLocation::FlippedPartLocations(locs);
      if(computeParts) {
        for(j = 0; j < numParts; j++) {
	  int parent_pose;
	  locs[j].GetDetectionLocation(NULL, NULL, NULL, NULL, &parent_pose);
	  if(GetPart(j)->GetPose(parent_pose)->IsFlipped()) {
	    if(parts[j]->GetFlipped()) parts[j]->GetFlipped()->AddSpatialTransitions(locs_flipped, NULL, false, combinePoses);
	    else parts[j]->AddSpatialTransitions(locs_flipped, NULL, false, combinePoses);
	  } else
	    parts[j]->AddSpatialTransitions(locs, NULL, false, combinePoses);
	}
      }

      if(computeClickParts) {
        for(int user = 0; user < d->GetExampleLabel(i)->GetObject(o)->NumUsers(); user++) {
          PartLocation *click_locs = d->GetExampleLabel(i)->GetObject(o)->GetPartClickLocations(user);
          if(click_locs) {
            for(j = 0; j < numClickParts; j++) {
              clickParts[j]->AddSpatialTransitions(locs, &click_locs[j]);
	    }
          }
        }
      }

      if(locs_flipped) 
	delete [] locs_flipped;
    }
  }
  if(computeParts) {
    for(j = 0; j < numParts; j++) 
      parts[j]->NormalizePartPoseTransitions(minExamples);
  }
  if(computeClickParts) {
    for(j = 0; j < numClickParts; j++) 
      clickParts[j]->NormalizePartPoseTransitions(minExamples);
  }
}

void Classes::ComputePartPoseTransitionWeights(Dataset *d, bool computeClickParts, bool computeParts) {
  int i, j;
  int n = NumWeights(true, false);

  // Make sure all spatial weights are zeroed out initially
  for(j = 0; j < numSpatialTransitions; j++) 
    if((computeParts && !spatialTransitions[j]->IsClick()) || (computeClickParts && spatialTransitions[j]->IsClick())) 
      spatialTransitions[j]->ZeroWeights();

  if(computeParts) {
   
    // Compute the standard deviation of parent/child part offsets 
    for(i = 0; i < d->NumExamples(); i++) {
      for(int o = 0; o < d->GetExampleLabel(i)->NumObjects(); o++) {
        PartLocation *locs = d->GetExampleLabel(i)->GetObject(o)->GetPartLocations();
	if(!locs) continue;
	PartLocation *locs_flipped = PartLocation::FlippedPartLocations(locs);
        if(!locs) continue;
        for(j = 0; j < numParts; j++) {
	  int parent_pose;
	  locs[j].GetDetectionLocation(NULL, NULL, NULL, NULL, &parent_pose);
	  if(GetPart(j)->GetPose(parent_pose)->IsFlipped()) {
	    if(parts[j]->GetFlipped()) parts[j]->GetFlipped()->AddSpatialTransitions(locs_flipped, NULL, true);
	    else parts[j]->AddSpatialTransitions(locs_flipped, NULL, true);
	  } else
	    parts[j]->AddSpatialTransitions(locs, NULL, true);
	}
      }
    }
    for(j = 0; j < numParts; j++)
      parts[j]->NormalizePartPoseTransitions(0, true);
  }

  if(computeClickParts) {
    // Compute the standard deviation of parent/child part offsets
    for(i = 0; i < d->NumExamples(); i++) {
      for(int o = 0; o < d->GetExampleLabel(i)->NumObjects(); o++) {
        PartLocation *locs = d->GetExampleLabel(i)->GetObject(o)->GetPartLocations();
        if(!locs) continue;
        for(int user = 0; user < d->GetExampleLabel(i)->GetObject(o)->NumUsers(); user++) {
          PartLocation *click_locs = d->GetExampleLabel(i)->GetObject(o)->GetPartClickLocations(user);
          if(click_locs) {
            for(j = 0; j < numClickParts; j++)
              clickParts[j]->AddSpatialTransitions(locs, &click_locs[j], true);
          }
        }
      }
    } 
    for(j = 0; j < numClickParts; j++)
      clickParts[j]->NormalizePartPoseTransitions(0, true);
  }
}

void Classes::ImageLocationToDetectionLocation(float x, float y, int scale, int rot, int width, int height, int *xx, int *yy) {
  float xxx, yyy;
  int w, h;
 
  RotationInfo r = GetRotationInfo(rot, scale, width, height);
  AffineTransformPoint(r.mat, x, y, &xxx, &yyy);
  *xx = (int)(xxx+.5f); *yy = (int)(yyy+.5f);

  w = (int)(r.maxX - r.minX);
  h = (int)(r.maxY - r.minY);

  if(*xx >= w) *xx = w-1;
  if(*yy >= h) *yy = h-1;
  if(*xx < 0) *xx = 0;
  if(*yy < 0) *yy = 0;
}

void Classes::DetectionLocationToImageLocation(int x, int y, int scale, int rot, int width, int height, float *xx, float *yy) {
  RotationInfo r = GetRotationInfo(rot, scale, width, height);
  AffineTransformPoint(r.invMat, x, y, xx, yy);
}

void Classes::ConvertDetectionCoordinates(float x, float y, int srcScale, int srcRot, int dstScale, int dstRot, int width, int height, float *xx, float *yy) {
  RotationInfo rSrc = GetRotationInfo(srcRot, srcScale, width, height);
  RotationInfo rDst = GetRotationInfo(dstRot, dstScale, width, height);
  float mat[6];
  MultiplyAffineMatrices(rSrc.invMat, rDst.mat, mat);
  AffineTransformPoint(mat, x, y, xx, yy);  
}

PartLocation *Classes::LoadPartLocations(const Json::Value &root, int image_width, int image_height, bool isClick) {
  if(!root.isArray()) { fprintf(stderr, "Part location struct is not an array\n"); return false; }
  char tmp[1000];
  PartLocation *locs = PartLocation::NewPartLocations(this, image_width, image_height, NULL, isClick);
  for(int i = 0; i  < (int)root.size(); i++) {
    strcpy(tmp, root[i].get("part","").asString().c_str());
    ObjectPart *part = isClick ? FindClickPart(tmp) : FindPart(tmp);
    if(part && !locs[part->Id()].load(root[i])) { 
      delete [] locs; 
      fprintf(stderr, "Couldn't load part location %s\n", tmp); 
      return NULL; 
    } 
  }
  return locs;
}

Json::Value Classes::SavePartLocations(PartLocation *locs) {
  Json::Value root;
  int num = 0;
  for(int i = 0; i < NumParts(); i++) {
    //if(!locs[i].IsLatent()) 
      root[num++] = locs[i].save();
  }
  return root;
}

AttributeAnswer *Classes::LoadAttributeResponses(const Json::Value &root) {
  if(!root.isArray() || (int)root.size() != NumAttributes()) { 
    fprintf(stderr, "attribute_responses is not an array of the appropriate size\n"); return NULL; 
  }
  char tmp[10000];
  AttributeAnswer *r = (AttributeAnswer*)malloc(sizeof(AttributeAnswer)*NumAttributes());
  for(int i = 0; i  < (int)root.size(); i++) {
    strcpy(tmp, root[i].asString().c_str());
    if(sscanf(tmp, "%d,%d,%f", &r[i].answer, &r[i].certainty, &r[i].responseTimeSec) != 3) {
      fprintf(stderr, "attribute response has bad format: '%s'\n", tmp); 
      free(r);
      return NULL;
    }
  }
  return r;
}

Json::Value Classes::SaveAttributeResponses(AttributeAnswer *r) {
  Json::Value root;
  char tmp[1000];
  for(int i = 0; i < NumAttributes(); i++) {
    sprintf(tmp, "%d,%d,%f", r[i].answer, r[i].certainty, r[i].responseTimeSec);
    root[i] = tmp;
  }
  return root;
}

FeatureDictionary *Classes::FindCodebook(const char *baseName) { 
  for(int i = 0; i < numCodebooks; i++)
    if(!strcmp(codebooks[i]->BaseFeatureName(), baseName))
      return codebooks[i];
  return NULL;
}

FisherFeatureDictionary *Classes::FindFisherCodebook(const char *baseName) { 
  for(int i = 0; i < numFisherCodebooks; i++)
    if(!strcmp(fisherCodebooks[i]->BaseFeatureName(), baseName))
      return fisherCodebooks[i];
  return NULL;
}
