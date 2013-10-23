#include "online_interactive_parts_server.h"
#include "pose.h"
#include "class.h"
#include "question.h"
#include "interactiveLabel.h"
#include "dataset.h"
#include "distance.h"



class InteractiveParts : public PartModelStructuredSVM {
public:
  StructuredData *NewStructuredData() { return new InteractivePartsData; }
};


PartModelStructuredLearnerRpc::PartModelStructuredLearnerRpc() : StructuredLearnerRpc(new InteractiveParts) {
  Init();
}

void PartModelStructuredLearnerRpc::Init() {
  classesJSON = partsJSON = posesJSON = attributesJSON = certaintiesJSON = questionsJSON = NULL;
  classes = NULL;
  nonVisibleCost = 15;
  useMirroredPoses = false;
  dataset = NULL;
  all_group_points = NULL;
};

PartModelStructuredLearnerRpc::~PartModelStructuredLearnerRpc() {
  if(classes) delete classes;
}


// Finalize the location of a part with index part_id at a specified location
bool PartModelStructuredLearnerRpc::LabelPart(const Json::Value& root, Json::Value& response) {
  int sess_ind;
  if((sess_ind=FindSession(root, response)) < 0) {
    JSON_ERROR("session_id parameter is invalid", -1);
  }


  // Parse the part location sent from the client
  InteractivePartsData *x = (InteractivePartsData*)sessions[sess_ind].example->x;
  ImageProcess *process = x->GetProcess(((PartModelStructuredSVM*)learner)->GetClasses());
  PartLocation loc;
  loc.Init(process->GetClasses(), x->Width(), x->Height(), process->Features()); 
  int partID;
  if(loc.load(root) && (partID=loc.GetPartID()) >= 0) {
    x->GetSession()->FinalizePartLocation(&loc);
    
    // Return a string encoding of the solution
    PartLocation *locs = x->GetSession()->GetPartLocations();
    PartLocalizedStructuredLabel *m_ybar = (PartLocalizedStructuredLabel*)sessions[sess_ind].example->y;
    PartLocation *locs_copy = PartLocation::NewPartLocations(process->GetClasses(), process->Image()->width, process->Image()->height, process->Features(), false);
    for(int i = 0; i < process->GetClasses()->NumParts(); i++)
      locs_copy[i] = locs[i];
    m_ybar->SetPartLocations(locs_copy);
    response["y"] = m_ybar->save(learner);
  } else {
    JSON_ERROR("Parameters part and x,y must be specified", sess_ind);
  }

  UnlockSession(sess_ind);

  return true;
}

// Quickly compute the max likelihood location of all parts if the user were to
// move the location of a part with index part_id to the specified location
bool PartModelStructuredLearnerRpc::PreviewPartLocations(const Json::Value& root, Json::Value& response) {
  int sess_ind;
  if((sess_ind=FindSession(root, response)) < 0) {
    JSON_ERROR("session_id parameter is invalid", -1);
  }

  InteractivePartsData *x = (InteractivePartsData*)sessions[sess_ind].example->x;
  ImageProcess *process = x->GetProcess(((PartModelStructuredSVM*)learner)->GetClasses());
  PartLocation loc;
  int partID;
  loc.Init(process->GetClasses(), x->Width(), x->Height(), process->Features()); 

  // Parse the part location sent from the client
  if(loc.load(root) && (partID=loc.GetPartID()) >= 0) {
    PartLocation *locs = PartLocation::NewPartLocations(process->GetClasses(), process->Image()->width, process->Image()->height, process->Features(), false);
    locs[partID] = loc;
    process->GetPartInst(partID)->ExtractPartLocations(locs, &loc, NULL);

    // Return a string encoding of the solution
    PartLocalizedStructuredLabel *m_ybar = (PartLocalizedStructuredLabel*)sessions[sess_ind].example->y;
    m_ybar->SetPartLocations(locs);
    response["y"] = m_ybar->save(learner);
  } else {
    JSON_ERROR("Part location invalid", sess_ind);
  }

  UnlockSession(sess_ind);

  return true;
}


bool PartModelStructuredLearnerRpc::Preprocess(const Json::Value& root, Json::Value& response) {
  if(!StructuredLearnerRpc::InitializeSession(root, response))
    return false;

  char errStr[1000];
  int sess_ind = -1;
  if((sess_ind=FindSession(root, response)) < 0) {
    JSON_ERROR("session_id parameter is invalid", -1);
  }
  
  char debugDir[1000];
  bool debug = root.get("debug", false).asBool();
  bool debugProbabilityMaps = root.get("debug_probability_maps", true).asBool();

  InteractivePartsData *x = (InteractivePartsData*)sessions[sess_ind].example->x;
  IplImage *img = cvLoadImage(x->GetImageName());
  if(!img) {
    sprintf(errStr, "Couldn't open image %s", x->GetImageName());
    JSON_ERROR(errStr, sess_ind);
  }
  ImageProcess *process = x->GetProcess(((PartModelStructuredSVM*)learner)->GetClasses());
  SparseVector *w = learner->GetCurrentWeights(true);
  float *ww = w->get_non_sparse<float>(learner->GetSizePsi());
  process->SetCustomWeights(ww, true, false);
  
  x->SetSize(img->width, img->height);
  process->Features()->SetImage(img);
  process->SetBidirectional(true);
  process->SetMultithreaded(false);
  sprintf(debugDir, "%s/%s", session_dir, sessions[sess_ind].id);
  InteractiveLabelingSession *session = new InteractiveLabelingSession(process, NULL, true, true, false, false, true, 1,
								       debug ? debugDir : false, debugProbabilityMaps);
  x->SetSession(session);
  double score = session->Preprocess();
  PartLocation *locs = session->GetPartLocations();
  PartLocalizedStructuredLabel *m_ybar = (PartLocalizedStructuredLabel*)sessions[sess_ind].example->y;
  PartLocation *locs_copy = PartLocation::NewPartLocations(process->GetClasses(), process->Image()->width, process->Image()->height, process->Features(), false);
  for(int i = 0; i < process->GetClasses()->NumParts(); i++)
    locs_copy[i] = locs[i];
  m_ybar->SetPartLocations(locs_copy);
  response["y"] = m_ybar->save(learner);
  if(!isnan(score)) response["score"] = score;
  
  free(ww);
  delete w;

  UnlockSession(sess_ind);

  return true;
}


void PartModelStructuredLearnerRpc::parse_command_line_arguments(int argc, const char **argv){
  srand(time(NULL));
  StructuredLearnerRpc::parse_command_line_arguments(argc, argv);

  PartDetectionLossType lossMethod = LOSS_BOUNDING_BOX_AREA_UNION_OVER_INTERSECTION;
  bool customLoss = false;


  // Add additional parameter
  int i = 1;
  this->classes = NULL;
  while(i < argc) {
    if(!strcmp(argv[i], "-l")) {
      // Specify which loss function to use for training a part model
      lossMethod = Classes::DetectionLossMethodFromString(argv[i+1]);
      customLoss = true;
      i += 2;
    } else if(!strcmp(argv[i], "-c")) {
      // The class definition file
      classes = new Classes();
      bool b = classes->Load(argv[i+1]);  assert(b);
      ((PartModelStructuredSVM*)learner)->SetClasses(classes);
      VFLOAT *w = new VFLOAT[classes->NumWeights(true,false)];
      classes->GetWeights(w, true, false); 
      ((PartModelStructuredSVM*)learner)->SetWeights(w);
      delete [] w;
      i += 2;
    } else if(!strcmp(argv[i], "-R")) {
      MultiSampleMethod m = SAMPLE_ALL_POSES;
      if(!strcmp(argv[i+1], "pose")) m = SAMPLE_ALL_POSES;
      else if(!strcmp(argv[i+1], "bounding_box")) m = SAMPLE_BY_BOUNDING_BOX;
      else if(!strcmp(argv[i+1], "random")) m = SAMPLE_RANDOMLY;
      else if(!strcmp(argv[i+1], "multi_update")) m = SAMPLE_UPDATE_EACH_POSE_IMMEDIATELY;
      ((PartModelStructuredSVM*)learner)->SetSamplingMethod(m);
      i += 2;
    } else if(!strcmp(argv[i], "-D")) {
      dataset = new Dataset(classes);
      bool b = dataset->Load(argv[i+1]);  assert(b);
      BuildDatasetPoseCache();
      i += 2;
    } else {
      i++;
    } /*else {
      Classes *classes = new Classes();
      bool b = classes->Load(argv[i]);  assert(b);
      ((PartModelStructuredSVM*)learner)->SetClasses(classes);
      i++;
    } */
  }

  Classes *classes2 = ((PartModelStructuredSVM*)learner)->GetClasses();
  StructuredSVMTrainParams *params = learner->GetTrainParams();
  if(classes2) {
    if(customLoss)
      classes2->SetDetectionLossMethod(lossMethod);
    if(!params->lambda)
      learner->SetC(.1*classes->NumPoses());
    if(!params->eps)
      params->eps = .05;
  }
}

// Add methods specific to realtime interactive part labeling, and document their parameters/return values 
void PartModelStructuredLearnerRpc::AddMethods() {
  StructuredLearnerRpc::AddMethods();

  if(server) {
    Json::Value preprocess_parameters, preprocess_returns;
    preprocess_parameters["session_id"] = "Session id returned by new_session().";
    preprocess_parameters["x"] = "String encoding of an example in the format of PartLocalizedStructuredData::read() (the image file name).";
    preprocess_returns["y"] = "A string encoding of the predicted label y in the format of PartLocalizedStructuredLabel::read()";
    preprocess_returns["score"] = "The score of the predicted label y";
    preprocess_returns["session_id"] = "A string encoding of the session id.  The client should pass this as a parameter to all future accesses to x";
    server->RegisterMethod(new JsonRpcMethod<PartModelStructuredLearnerRpc>(this, &PartModelStructuredLearnerRpc::Preprocess, "initialize_interactive_parts", "Preprocess an image, predicting the most likely part configuration and initializing the session for interactive part labeling.  The prediction is the same behavior as classify_example.", preprocess_parameters, preprocess_returns));
    
    Json::Value label_part_parameters, label_part_returns;
    label_part_parameters["session_id"] = "Session id returned by new_session().";
    label_part_parameters["part"] = "The name of the part being labeled";
    label_part_parameters["x"] = "The x pixel location of the center of the part";
    label_part_parameters["y"] = "The y pixel location of the center of the part";
    label_part_parameters["scale"] = "Optional scale of the part";
    label_part_parameters["rotation"] = "Optional rotation of the part";
    label_part_parameters["pose"] = "Optional pose name of the part";
    label_part_returns["y"] = "A string encoding of the predicted label y (the highest scoring label consistent with all labeled part locations) in the format of PartLocalizedStructuredLabel::read()";
    label_part_returns["score"] = "The score of the predicted label y";
    label_part_returns["session_id"] = "A string encoding of the session id.  The client should pass this as a parameter to all future accesses to x";
    server->RegisterMethod(new JsonRpcMethod<PartModelStructuredLearnerRpc>(this, &PartModelStructuredLearnerRpc::LabelPart, "label_part", "Label the location of one part (should be called each time a user finalizes a part for interactive labeling)", label_part_parameters, label_part_returns));

    server->RegisterMethod(new JsonRpcMethod<PartModelStructuredLearnerRpc>(this, &PartModelStructuredLearnerRpc::PreviewPartLocations, "preview_part_locations", "Preview the highest scoring part configuration if the user moves one new part that is consistent with all previous labeled parts", label_part_parameters, label_part_returns));

    Json::Value get_definitions_parameters, get_definitions_returns;
    get_definitions_parameters["classes"] = "If true, get the list of all classes";
    get_definitions_parameters["questions"] = "If true, get the list of all questions";
    get_definitions_parameters["parts"] = "If true, get the list of all parts";
    get_definitions_parameters["poses"] = "If true, get the list of all poses";
    get_definitions_parameters["certainties"] = "If true, get the list of all certainties";
    get_definitions_parameters["attributes"] = "If true, get the list of all attributes";
    get_definitions_returns["classes"] = "An array of classes structs";
    get_definitions_returns["questions"] = "An array of questions structs";
    get_definitions_returns["parts"] = "An array of part structs";
    get_definitions_returns["poses"] = "An array of poses structs";
    get_definitions_returns["certainties"] = "An array of certainties structs";
    get_definitions_returns["attributes"] = "An array of attributes structs";
    server->RegisterMethod(new JsonRpcMethod<PartModelStructuredLearnerRpc>(this, &PartModelStructuredLearnerRpc::GetDefinitions, "get_definitions", "Get a list of the definition of all classes, questions, parts, poses, certainties, or attributes", get_definitions_parameters, get_definitions_returns));
  }
}




bool PartModelStructuredLearnerRpc::GetDefinitions(const Json::Value& root, Json::Value& response) {
  if(root.get("classes", false).asBool())
    GetClasses(root, response);
  if(root.get("questions", false).asBool())
    GetQuestions(root, response);
  if(root.get("parts", false).asBool())
    GetParts(root, response);
  if(root.get("poses", false).asBool())
    GetPoses(root, response);
  if(root.get("attributes", false).asBool())
    GetAttributes(root, response);
  if(root.get("certainties", false).asBool())
    GetCertainties(root, response);

#ifdef HACK_FOR_JOURNAL_PAPER_USER_STUDY
  int u;
  if((u=root.get("user_study_id", -1).asInt()) >= 0)
    ParseUserStudyResults(u, -1, -1, ((PartModelStructuredSVM*)learner)->GetClasses(), response, 0, NULL, root.get("user_study_start_image", -1).asInt());
#endif

  return true;
}

bool PartModelStructuredLearnerRpc::GetClasses(const Json::Value& root, Json::Value& response) {
  if(!classesJSON) {
    ComputeClasses();
  }
  response["classes"] = *classesJSON;
  return true;
}

bool PartModelStructuredLearnerRpc::GetQuestions(const Json::Value& root, Json::Value& response) {
  if(!questionsJSON) {
    ComputeQuestions();
  }
  response["questions"] = *questionsJSON;
  return true;
}

bool PartModelStructuredLearnerRpc::GetParts(const Json::Value& root, Json::Value& response) {
  if(!partsJSON) {
    ComputeParts();
  }
  response["parts"] = *partsJSON;
  return true;
}

bool PartModelStructuredLearnerRpc::GetPoses(const Json::Value& root, Json::Value& response) {
  if(!posesJSON) {
    ComputePoses();
  }
  response["poses"] = *posesJSON;
  return true;
}

bool PartModelStructuredLearnerRpc::GetAttributes(const Json::Value& root, Json::Value& response) {
  if(!attributesJSON) {
    ComputeAttributes();
  }
  response["attributes"] = *attributesJSON;
  return true;
}

bool PartModelStructuredLearnerRpc::GetCertainties(const Json::Value& root, Json::Value& response) {
  if(!certaintiesJSON) {
    ComputeCertainties();
  }
  response["certainties"] = *certaintiesJSON;
  return true;
}

bool PartModelStructuredLearnerRpc::ComputeClasses() {
  Lock();

  Classes *classes = ((PartModelStructuredSVM*)learner)->GetClasses();
  Json::Value a(Json::arrayValue);
  int id;
  for(int i = 0; i < classes->NumClasses(); i++) {
    Json::Value c(Json::objectValue);
    Json::Value img(Json::arrayValue);
    ObjectClass *cl = classes->GetClass(i);
    assert(i == cl->Id());

    for(int j = 0; j < cl->NumExemplarImages(); j++)
      img.append(std::string(cl->GetExemplarImageName(j)));

    //if(cl->GetWikipediaUrl()) c["wikipedia_article"] = std::string(cl->GetWikipediaUrl());
    if(sscanf(cl->Name(), "%03d.", &id)) c["class_name"] = std::string(cl->Name()+4);
    else c["class_name"] = std::string(cl->Name());
    c["class_images"] = img;
    c["class_id"] = i;
    a[i] = c;
  }
  classesJSON = new Json::Value(a);

  Unlock();

  return true;
}

bool PartModelStructuredLearnerRpc::ComputeQuestions() {
  Lock();

  Classes *classes = ((PartModelStructuredSVM*)learner)->GetClasses();
  Json::Value a(Json::arrayValue);
  for(int i = 0; i < classes->NumQuestions(); i++) {
    Json::Value q(Json::objectValue);
    Question *qu = classes->GetQuestion(i);
    assert(i == qu->Id());

    q["question_id"] = i;
    q["question_text"] = std::string(qu->GetText());
    q["type"] = std::string(qu->Type());

    if(!strcmp(qu->Type(), "binary")) {
      BinaryAttributeQuestion *qb = (BinaryAttributeQuestion*)qu;
      Attribute *a = qb->GetAttribute();
      assert(a);
      q["attribute_id"] = a->Id();
      if(a->PropertyName()) q["attribute_property"] = std::string(a->PropertyName());
      if(a->ValueName()) q["attribute_value"] = std::string(a->ValueName());
      ObjectPart *p = a->Part();
      if(a->GetVisualizationImageName()) q["attribute_visualization"] = std::string(a->GetVisualizationImageName());
      if(p) {
        q["part_id"] = p->Id();
        q["part_name"] = p->Name();
        if(p->GetVisualizationImageName()) q["part_visualization"] = std::string(p->GetVisualizationImageName());
      }
      if(qb->GetVisualizationImageName())  q["part_visualization"] = std::string(qb->GetVisualizationImageName());
    } else if(!strcmp(qu->Type(), "multiple_choice")) {
      MultipleChoiceAttributeQuestion *qm = (MultipleChoiceAttributeQuestion*)qu;
      Json::Value choices(Json::arrayValue);
      ObjectPart *p = NULL;
      for(int i = 0; i < qm->NumChoices(); i++) {
        Attribute *a = classes->GetAttribute(qm->GetChoice(i));
        assert(a);
        Json::Value v(Json::objectValue);
        v["attribute_id"] = a->Id();
        //assert(i == 0 || p == a->Part());
        if(a->GetVisualizationImageName()) v["attribute_visualization"] = std::string(a->GetVisualizationImageName());
        if(a->ValueName()) v["attribute_value"] = std::string(a->ValueName());
        if(i == 0) {
          if(a->PropertyName()) q["attribute_property"] = std::string(a->PropertyName());
          p = a->Part();
          if(p) {
            q["part_id"] = p->Id();
            q["part_name"] = p->Name();
            if(p->GetVisualizationImageName()) q["part_visualization"] = std::string(p->GetVisualizationImageName());
          }
        }
        choices[i] = v;
      }
      q["choices"] = choices;
      if(qm->GetVisualizationImageName())  q["part_visualization"] = std::string(qm->GetVisualizationImageName());
    } else if(!strcmp(qu->Type(), "batch")) {
      BatchQuestion *qm = (BatchQuestion*)qu;
      Json::Value choices(Json::arrayValue);
      ObjectPart *p = NULL;
      for(int i = 0; i < qm->NumQuestions(); i++) {
        Question *q_sub = qm->GetQuestion(i);
        assert(!strcmp(q_sub->Type(), "binary"));
        Attribute *a = ((BinaryAttributeQuestion*)q_sub)->GetAttribute();
        assert(a);
        Json::Value v(Json::objectValue);
        v["attribute_id"] = a->Id();
        if(a->ValueName()) v["attribute_value"] = std::string(a->ValueName());
        //assert(i == 0 || p == a->Part());
        if(a->GetVisualizationImageName()) v["attribute_visualization"] = std::string(a->GetVisualizationImageName());
        if(i == 0) {
          if(a->PropertyName()) q["attribute_property"] = std::string(a->PropertyName());
          p = a->Part();
          if(p) {
            q["part_id"] = p->Id();
            q["part_name"] = p->Name();
            if(p->GetVisualizationImageName()) q["part_visualization"] = std::string(p->GetVisualizationImageName());
          }
        }
        choices[i] = v;
      }
      q["choices"] = choices;
      if(qm->GetVisualizationImageName())  q["part_visualization"] = std::string(qm->GetVisualizationImageName());
    } else if(!strcmp(qu->Type(), "part_click")) {
      ClickQuestion *qc = (ClickQuestion*)qu;
      ObjectPart *p = qc->GetPart();
      assert(p);
      q["part_id"] = p->Id();
      q["part_name"] = p->Name();
      if(p->GetVisualizationImageName()) q["part_visualization"] = std::string(p->GetVisualizationImageName());
      if(qc->GetVisualizationImageName())  q["part_visualization"] = std::string(qc->GetVisualizationImageName());
    }

    a[i] = q;
  }

  questionsJSON = new Json::Value(a);

  Unlock();

  return true;
}

bool PartModelStructuredLearnerRpc::ComputeParts() {
  Lock();

  Classes *classes = ((PartModelStructuredSVM*)learner)->GetClasses();
  Json::Value a(Json::arrayValue);
  for(int i = 0; i < classes->NumParts(); i++) {
    ObjectPart *pa = classes->GetPart(i);
    Json::Value p(Json::objectValue);
    assert(i == pa->Id());
    p["parent_id"] = pa->GetParent() ? pa->GetParent()->Id() : -1;
    p["part_id"] = i;
    if(pa->GetAbbreviation()) p["abbreviation"] = pa->GetAbbreviation();
    p["part_name"] = std::string(pa->Name());
    if(pa->GetVisualizationImageName())
      p["part_visualization"] = std::string(pa->GetVisualizationImageName());
    a[i] = p;
    Json::Value poses;
    for(int j = 0; j < pa->NumPoses(); j++) 
      poses[j] = pa->GetPose(j)->Id();
    p["poses"] = poses;
  }
  partsJSON = new Json::Value(a);

  Unlock();

  return true;
}


bool PartModelStructuredLearnerRpc::ComputePoses() {
  Lock();

  Classes *classes = ((PartModelStructuredSVM*)learner)->GetClasses();
  Json::Value a(Json::arrayValue);
  for(int i = 0; i < classes->NumPoses(); i++) {
    ObjectPose *po = classes->GetPose(i);
    Json::Value p(Json::objectValue);
    assert(i == po->Id());
    p["pose_id"] = i;
    p["pose_name"] = std::string(po->Name());
    if(po->GetVisualizationImageName())
      p["pose_visualization"] = std::string(po->GetVisualizationImageName());
    a[i] = p;
  }
  posesJSON = new Json::Value(a);

  Unlock();

  return true;
}

bool PartModelStructuredLearnerRpc::ComputeAttributes() {
  Lock();

  Classes *classes = ((PartModelStructuredSVM*)learner)->GetClasses();
  Json::Value a(Json::arrayValue);
  for(int i = 0; i < classes->NumAttributes(); i++) {
    Attribute *at = classes->GetAttribute(i);
    Json::Value p(Json::objectValue);
    assert(i == at->Id());
    p["attribute_id"] = i;
    p["attribute_name"] = std::string(at->Name());
    if(at->PropertyName()) p["attribute_property"] = std::string(at->PropertyName());
    if(at->ValueName()) p["attribute_value"] = std::string(at->ValueName());
    if(at->GetVisualizationImageName())
      p["attribute_visualization"] = std::string(at->GetVisualizationImageName());
    a[i] = p;
  }
  attributesJSON = new Json::Value(a);

  Unlock();

  return true;
}

bool PartModelStructuredLearnerRpc::ComputeCertainties() {
  Lock();

  Classes *classes = ((PartModelStructuredSVM*)learner)->GetClasses();
  Json::Value a(Json::arrayValue);
  for(int i = 0; i < classes->NumCertainties(); i++) {
    a[i] = std::string(classes->GetCertainty(i));
  }
  certaintiesJSON = new Json::Value(a);

  Unlock();

  return true;
}



PartLocalizedStructuredData *PartModelStructuredLearnerRpc::FindNearestExampleByPose(PartLocation *locs, int restrictClass, int w, int h) {
  float bestDist = INFINITY;
  PartLocalizedStructuredData *retval = NULL;
  float **group_pts = Create2DArray<float>(classes->NumParts(), classes->NumParts()*3);
  assert(all_group_points);

  
  for(int j = 0; j < classes->NumParts(); j++) 
    if(classes->GetPart(j)->NumParts())
      dataset->ComputeExampleGroupOffsets(locs, j, group_pts[j], NULL, nonVisibleCost, w, h);

  for(int i = 0, ii = 0; i < dataset->NumExamples(); i++) {
    for(int o = 0; o < dataset->GetExampleLabel(i)->NumObjects(); o++, ii++) {
      if(restrictClass >= 0 && dataset->GetExampleLabel(i)->GetObject(o)->GetClass()->Id() != restrictClass)
	continue;
      float dist = 0;
      for(int j = 0; j < classes->NumParts(); j++) {
	ObjectPart *part = classes->GetPart(j);
	if(part->NumParts())
	  dist += PointDistanceL2_sqr<float>(group_pts[j], all_group_points[j][ii], 3*part->NumParts());	
      }
      if(dist < bestDist) {
	bestDist = dist;
	retval = dataset->GetExampleData(i);
      }
    }
  }
  free(group_pts);

  return retval;
}

void PartModelStructuredLearnerRpc::BuildDatasetPoseCache() {
  all_group_points = (float***)malloc(sizeof(float**)*classes->NumParts());
  memset(all_group_points, 0, sizeof(float**)*classes->NumParts());
 
  for(int j = 0; j < classes->NumParts(); j++) 
    all_group_points[j] = dataset->ComputeGroupOffsets(j, nonVisibleCost, useMirroredPoses);

  /*
  FILE *fout = fopen("CUB_200_2011_data/train_offsets.txt", "w");
  for(int i = 0, ii = 0; i < dataset->NumExamples(); i++) {
    for(int o = 0; o < dataset->GetExampleLabel(i)->NumObjects(); o++, ii++) {
      char fname[1000];
      ExtractFilename(dataset->GetExampleData(i)->GetProcess(classes)->ImageName(), fname);
      fprintf(fout, "%s", fname);
      for(int j = 0; j < classes->NumParts(); j++) {
	ObjectPart *part = classes->GetPart(j);
	if(part->NumParts()) {
	  for(int k = 0; k < 3*part->NumParts(); k++) {
	    fprintf(fout, " %f", all_group_points[j][ii][k]);
	  }
	}	
      }
      fprintf(fout, "\n");
    }
  }
  fclose(fout);
  */
}
