#include "20q_server.h"
#include "pose.h"
#include "class.h"


class Visual20qData : public InteractivePartsData {
  QuestionAskingSession *session_20q;
public:
  Visual20qData() : InteractivePartsData() { session_20q = NULL; }
  void SetSession(QuestionAskingSession *s) { session_20q = s; }
  QuestionAskingSession *Get20QSession() { return session_20q; }
  virtual ~Visual20qData() { if(session_20q) delete session_20q; session_20q = NULL; }
};

class Visual20q : public PartModelStructuredSVM {
public:
  StructuredData *NewStructuredData() { return new Visual20qData; }
};



Visual20qRpc::Visual20qRpc() : PartModelStructuredLearnerRpc(new Visual20q) {
  classesJSON = partsJSON = posesJSON = attributesJSON = certaintiesJSON = questionsJSON = NULL;
  train = false;
  runServer = true;
}

void Visual20qRpc::parse_command_line_arguments(int argc, const char **argv) {
  PartModelStructuredLearnerRpc::parse_command_line_arguments(argc, argv);
}

void Visual20qRpc::AddMethods() {
  PartModelStructuredLearnerRpc::AddMethods();

  if(!server || !learner || !((Visual20q*)learner)->GetClasses()) {
    fprintf(stderr, "USAGE: ./20q_server.out -P 8086 -c data_pose/classes.txt\n  where data_pose/classes.txt is the output of train_localized_v20q.out");
  } else {
    Json::Value preprocess_parameters, preprocess_returns;
    preprocess_parameters["session_id"] = "Session id returned by new_session().";
    preprocess_parameters["x"] = "String encoding of an example in the format of PartModelStructuredData::read() (the image file name).";
    preprocess_parameters["num_classes"] = "The number of classes to return";
    preprocess_parameters["show_parts"] = "If true, return the most likely part configuration";
    preprocess_parameters["question_select_method"] = "Optional parameter, defining the criterion used to predict which question to ask.  Can be one of 'information_gain' (select most informative question), 'time' (select question that minimize the expected to to identify the true class), 'random' (select a random question)";
    preprocess_parameters["debug"] = "Optional parameter, if true generate extra debug information in the form of an html page in the sessions directory";
    preprocess_parameters["debug_num_class_print"] = "Optional parameter, the number of top ranked classes to show in the html debug page";
    preprocess_parameters["debug_probability_maps"] = "Optional parameter, if true generate image probability maps of where we think parts are located";
    preprocess_parameters["debug_click_probability_maps"] = "Optional parameter, if true generate image probability maps of where we think the user would click on for a part click question";
    preprocess_parameters["debug_num_samples"] = "Optional parameter, if true generate images depicting random part location samples used by internal algorithms";
    preprocess_parameters["debug_question_entropies"] = "Optional parameter, if true display the expected information gain for each question";
    preprocess_parameters["debug_max_likelihood_solution"] = "Optional parameter, if true generate an image of the maximum likelihood part locations";
    preprocess_parameters["disable_click"] = "Optional parameter, if true disable questions asking the user to click on the location of a part";
    preprocess_parameters["disable_binary"] = "Optional parameter, if true disable binary yes/no questions";
    preprocess_parameters["disable_multiple"] = "Optional parameter, if true disable multiple choice and multi-select questions";
    preprocess_parameters["disable_computer_vision"] = "Optional parameter, if true disable computer vision algorithms";
    preprocess_returns["top_classes"] = "An array of the top ranked classes, of size num_classes.  Each entry contains a pair class_id,prob";
    preprocess_returns["parts"] = "A string encoding of the predicted most likely part locations in the format of PartModelStructuredLabel::read() (only return if show_parts=true)";
    preprocess_returns["parts_score"] = "The score of the predicted part locations (only return if show_classes=true)";
    preprocess_returns["session_id"] = "A string encoding of the session id.  The client should pass this as a parameter to all future accesses to x";
    server->RegisterMethod(new JsonRpcMethod<Visual20qRpc>(this, &Visual20qRpc::Preprocess, "initialize_20q", "Preprocess an image, predicting a ranking of the most likely classes and the most likely part configuration", preprocess_parameters, preprocess_returns));

    
    Json::Value answer_question_parameters;
    answer_question_parameters["session_id"] = "Session id returned by new_session().";
    answer_question_parameters["question_id"] = "The id of the question being answered";
    answer_question_parameters["answer"] = "0 or 1 for binary questions, an array of 0 or 1 values for multi-select questions, or a struct with values x,y,scale,orientation,pose for part click questions";
    answer_question_parameters["certainty"] = "The certainty (confidence level) of the user's response";
    answer_question_parameters["response_time"] = "The time in seconds of the user's response";
    answer_question_parameters["num_classes"] = "The number of classes to return, in the ranked list of most likely classes";
    answer_question_parameters["show_parts"] = "If true, return the most likely part configuration";
    server->RegisterMethod(new JsonRpcMethod<Visual20qRpc>(this, &Visual20qRpc::AnswerQuestion, "answer_question", "Answer a question and get back the new class probabilities", answer_question_parameters, preprocess_returns));

    Json::Value verify_class_parameters;
    verify_class_parameters["session_id"] = "Session id returned by new_session().";
    verify_class_parameters["class_id"] = "The id of the class being labeled";
    verify_class_parameters["answer"] = "0 or 1, where 1 indicates the class is the true class";
    verify_class_parameters["num_classes"] = "The number of classes to return, in the ranked list of most likely classes";
    verify_class_parameters["show_parts"] = "If true, return the most likely part configuration";
    server->RegisterMethod(new JsonRpcMethod<Visual20qRpc>(this, &Visual20qRpc::VerifyClass, "verify_class", "Manually specify whether or not a particular class is the true class", verify_class_parameters, preprocess_returns));

    Json::Value next_question, next_question_parameters, next_question_returns;
    next_question_parameters["session_id"] = "Session id returned by new_session().";
    next_question_returns["question_id"] = "The id of the question to ask";
    next_question_returns["session_id"] = "A string encoding of the session id.  The client should pass this as a parameter to all future accesses to x";
    next_question["returns"] = next_question_returns;
    server->RegisterMethod(new JsonRpcMethod<Visual20qRpc>(this, &Visual20qRpc::NextQuestion, "next_question", "Predict which question to pose to a human user, such that the true class can be identified as quickly as possible (e.g. predict the most informative question)", next_question_parameters, next_question_returns));

    
  }
}

void Visual20qRpc::ReturnClassesIfNecessary(QuestionAskingSession *session, const Json::Value& root, Json::Value& response) {
  int num_classes = root.get("num_classes", 0).asInt();
  PartLocation *locs = NULL;
  int w, h;
  if(root.get("show_parts", false).asBool() || (num_classes&&all_group_points)) {
    ImageProcess *process = session->GetProcess();
    locs = process->ExtractPartLocations();
    w = process->Image()->width;
    h = process->Image()->height;
  }

  if(num_classes) {
    ClassProb *probs = session->GetClassProbs(session->GetNumQuestionsAsked());
    Json::Value a(Json::arrayValue);
    for(int i = 0; i < num_classes; i++) {
      Json::Value c(Json::objectValue);
      c["class_id"] = probs[i].classID;
      c["prob"] = probs[i].prob;
      if(all_group_points && dataset) c["img_src"] = FindNearestExampleByPose(locs, probs[i].classID, w, h)->GetImageName();
      a.append(c);
    }
    response["top_classes"] = a;
  }
  if(root.get("show_parts", false).asBool()) {
    ImageProcess *process = session->GetProcess();
    PartLocalizedStructuredData *x = (PartLocalizedStructuredData*)learner->NewStructuredData();
    x->SetImageName(process->ImageName());
    x->SetSize(process->Image()->width, process->Image()->height);
    PartLocalizedStructuredLabel *m_ybar = new PartLocalizedStructuredLabel(x);
    m_ybar->SetPartLocations(locs);
    response["parts"] = m_ybar->save(learner);
    if(!isnan(locs[0].GetScore())) response["parts_score"] = locs[0].GetScore();
    delete m_ybar;
    delete x;
  } else if(locs)
    delete [] locs;
}

// {"jsonrpc":"2.0","method":"verify_class","session_id":"1305135286","class_id":157,"answer":1,"num_classes":4}
bool Visual20qRpc::VerifyClass(const Json::Value& r, Json::Value& response) {
  char errStr[1000];
  int sess_ind = -1;
  if((sess_ind=FindSession(r, response)) < 0) {
    JSON_ERROR("session_id parameter is invalid", -1);
  }
  QuestionAskingSession *session = ((Visual20qData*)sessions[sess_ind].example->x)->Get20QSession();
  if(!session) {
    JSON_ERROR("session is invalid", sess_ind);
  }

  Classes *classes = ((Visual20q*)learner)->GetClasses();
  int class_id = r.get("class_id", -1).asInt();
  if(class_id < 0 || class_id >= classes->NumClasses()) {
    sprintf(errStr, "Invalid class_id %d in VerifyClass\n", class_id);
    JSON_ERROR(errStr, sess_ind);
  }

  int answer = r.get("answer", -1).asInt();
  if(answer != 0 && answer != 1) {
    sprintf(errStr, "Invalid anser '%d' for class_id %d in VerifyClass.  Must be 0 or 1\n", answer, class_id);
    JSON_ERROR(errStr, sess_ind);
  }

  session->VerifyClass(class_id, answer);
  if(answer) sessions[sess_ind].finished = true;

#ifdef HACK_FOR_JOURNAL_PAPER_USER_STUDY
  int u;
  if(answer && (u=r.get("user_study_id", -1).asInt()) >= 0)
    ParseUserStudyResults(u, r.get("user_study_image_id", -1).asInt(), class_id, classes, response, time(NULL)-sessions[sess_ind].timestamp_start, session);
#endif

  ReturnClassesIfNecessary(session, r, response);
  UnlockSession(sess_ind);

  return true;
}

// {"jsonrpc":"2.0","method":"answer_question","session_id":"1305135286","question_id":157,"answer":1,"response_time":1,"certainty":"definitely","num_classes":4}
bool Visual20qRpc::AnswerQuestion(const Json::Value& r, Json::Value& response) {
  char errStr[1000];
  int sess_ind = -1;
  if((sess_ind=FindSession(r, response)) < 0) {
    JSON_ERROR("session_id parameter is invalid", -1);
  }
  QuestionAskingSession *session =  ((Visual20qData*)sessions[sess_ind].example->x)->Get20QSession();
  Classes *classes = ((Visual20q*)learner)->GetClasses();

  // Get question definition from the question_id
  int question_id = r.get("question_id", -1).asInt();
  if(question_id < 0 || question_id >= classes->NumQuestions()) {
    sprintf(errStr, "Invalid question id %d in AnswerQuestion", question_id);
    JSON_ERROR(errStr, sess_ind);
  }
  Question *q = classes->GetQuestion(question_id);

  if(!strcmp(q->Type(), "binary")) {  // yes/no question
    AttributeAnswer a;
    a.answer = r.get("answer", -1).asInt();
    char cert[1000]; strcpy(cert, r.get("certainty", "").asString().c_str());
    a.certainty = classes->FindCertainty(cert);
    a.responseTimeSec = (float)r.get("response_time", 0).asDouble();

    if(a.answer != 0 && a.answer != 1) { sprintf(errStr, "Invalid answer %d in AnswerQuestion\n", a.answer);  JSON_ERROR(errStr, sess_ind); }
    if(a.certainty < 0) { sprintf(errStr, "Invalid certainty %s\n", cert);  JSON_ERROR(errStr, sess_ind); }
    session->FinalizeAnswer(q, &a);
  } else if(!strcmp(q->Type(), "multiple_choice")) {  // multiple choice question
    AttributeAnswer a;
    char cert[1000]; strcpy(cert, r.get("certainty", "UTF-8").asString().c_str());
    a.certainty = classes->FindCertainty(cert);
    if(a.certainty < 0) { sprintf(errStr, "Invalid certainty %s\n", cert);  JSON_ERROR(errStr, sess_ind); }
    a.responseTimeSec = (float)r.get("response_time", 0).asDouble();

    a.answer = r.get("answer", -1).asInt();  // typically one can just provide the index of the selected choice

    if(r.isMember("answers")) {
      // Alternatively, one can provide an array of answers (same format as batch question), where exactly one entry should be 1
      const Json::Value answers = r["answers"];

      MultipleChoiceAttributeQuestion *mq = (MultipleChoiceAttributeQuestion*)q;
      if((int)answers.size() != mq->NumChoices()) {
        sprintf(errStr, "Number of answers does not match question %d %d\n", mq->NumChoices(), (int)answers.size());
        JSON_ERROR(errStr, sess_ind);
      }
      for(int i = 0; i < (int)answers.size(); i++) {
        int j = answers[(Json::UInt)i].asInt();
        if(j != 0 && j != 1) { sprintf(errStr, "Invalid answer %d\n", j);  JSON_ERROR(errStr, sess_ind); }
        if(j) a.answer = i;
      }
      //if(a.answer < 0) { sprintf(errStr, "Invalid multiple choice answer\n");  JSON_ERROR(errStr, sess_ind); }
    }
    session->FinalizeAnswer(q, &a);
  } else if(!strcmp(q->Type(), "batch")) {  // multi-select question (multiple choice, where the user can select more than 1)
    AttributeAnswer a;
    char cert[1000]; strcpy(cert, r.get("certainty", "").asString().c_str());
    a.certainty = classes->FindCertainty(cert);
    if(a.certainty < 0) { sprintf(errStr, "Invalid certainty %s\n", cert);  JSON_ERROR(errStr, sess_ind); }
    a.responseTimeSec = (float)r.get("response_time", 0).asDouble();

    BatchQuestion *bq = (BatchQuestion*)q;
    const Json::Value answers = r["answers"];
    if((int)answers.size() != bq->NumQuestions()) {
      sprintf(errStr, "Number of answers does not match question %d %d\n", bq->NumQuestions(), (int)answers.size());
      JSON_ERROR(errStr, sess_ind);
    }
    AttributeAnswer ans[10000];
    for(int i = 0; i < (int)answers.size(); i++) {
      a.answer = answers[(Json::UInt)i].asInt();
      if(a.answer != 0 && a.answer != 1) { sprintf(errStr, "Invalid answer %d\n", a.answer);  JSON_ERROR(errStr, sess_ind); }
      //Question *q_sub = bq->GetQuestion(i);
      //session->FinalizeAnswer(q_sub, &a);
      ans[i] = a;
    }
    session->FinalizeAnswer(bq, ans);
  } else if(!strcmp(q->Type(), "part_click")) {  // click on the location of a part
    PartLocation l(session->GetProcess()->GetPartInst(((ClickQuestion*)q)->GetPart()->Id()), session->GetProcess()->Image()->width, session->GetProcess()->Image()->height);
    if(!l.load(r)) { JSON_ERROR("Error parsing part click result\n", sess_ind); }
    l.ComputeDetectionLocations();

    if(!r.get("visible", true).asBool() || r.get("visible", 1).asInt()==0) {
      ObjectPart *part = ((ClickQuestion*)q)->GetPart();
      for(int i = 0; i < part->NumPoses(); i++) {
        if(part->GetPose(i)->IsNotVisible()) {
          l.SetDetectionLocation(LATENT, LATENT, LATENT, LATENT, i, LATENT, LATENT);
	}
      }
    }
    session->FinalizeAnswer(q, &l);
  }

  ReturnClassesIfNecessary(session, r, response);

  UnlockSession(sess_ind);

  return true;
}

bool Visual20qRpc::NextQuestion(const Json::Value& root, Json::Value& response) {
  int sess_ind = -1;
  if((sess_ind=FindSession(root, response)) < 0) {
    JSON_ERROR("session_id parameter is invalid", -1);
  }
  QuestionAskingSession *session =  ((Visual20qData*)sessions[sess_ind].example->x)->Get20QSession();

  int q = session->SelectNextQuestion();
  response["question_id"] = q;

  UnlockSession(sess_ind);

  return true;
}

bool Visual20qRpc::Preprocess(const Json::Value& root, Json::Value& response) {
  if(!StructuredLearnerRpc::InitializeSession(root, response))
    return false;

  char errStr[1000];
  int sess_ind = FindSession(root, response);
  if(sess_ind < 0) {
    JSON_ERROR("session_id parameter is invalid", -1);
  }
  char debugDir[1000];
  sprintf(debugDir, "%s/%s", session_dir, sessions[sess_ind].id);
  
  Classes *classes = ((Visual20q*)learner)->GetClasses();
  Visual20qData *x = (Visual20qData*)sessions[sess_ind].example->x;
  IplImage *img = cvLoadImage(x->GetImageName());
  if(!img) {
    sprintf(errStr, "Couldn't open image %s", x->GetImageName());
    JSON_ERROR(errStr, sess_ind);
  }
  x->SetSize(img->width, img->height);
  ImageProcess *process = new ImageProcess(classes, x->GetImageName(), IM_MAXIMUM_LIKELIHOOD, true, true, true);
  x->SetProcess(process);
  process->Features()->SetImage(img);
  QuestionSelectMethod method = QS_INFORMATION_GAIN;
  char meth[1000]; strcpy(meth, root.get("question_select_method", "UTF-8").asString().c_str());
  if(!strcmp(meth, "information_gain")) method = QS_INFORMATION_GAIN;
  else if(!strcmp(meth, "time")) method = QS_TIME_REDUCTION;
  else if(!strcmp(meth, "random")) method = QS_RANDOM;
  else { sprintf(errStr, "question_select_method must be one of 'information_gain', 'time', 'random'"); JSON_ERROR(errStr, sess_ind); }

  bool debug = root.get("debug", false).asBool();
  int debugNumClassPrint = root.get("debug_num_class_print", 10).asInt();
  bool debugProbabilityMaps = root.get("debug_probability_maps", true).asBool();
  bool debugClickProbabilityMaps = root.get("debug_click_probability_maps", false).asBool();
  int debugNumSamples = root.get("debug_num_samples", 0).asInt();
  bool debugQuestionEntropies = root.get("debug_question_entropies", false).asBool();
  bool debugMaxLikelihoodSolution = root.get("debug_max_likelihood_solution", false).asBool();
  bool debugKeepBigImage = root.get("debug_keep_big_image", false).asBool();
  QuestionAskingSession *session = new QuestionAskingSession(process, NULL, true, method, debug ? debugDir : NULL,
                        debugNumClassPrint, debugProbabilityMaps, debugClickProbabilityMaps, debugNumSamples,
							     debugQuestionEntropies, debugMaxLikelihoodSolution, debugKeepBigImage);
  session->SetDebugName(sessions[sess_ind].id);

  bool disableClick = root.get("disable_click", false).asBool();
  bool disableBinary = root.get("disable_binary", false).asBool();
  bool disableMultiple = root.get("disable_multiple", false).asBool();
  bool disableComputerVision = root.get("disable_computer_vision", false).asBool();
  session->DisableQuestions(disableClick, disableBinary, disableMultiple, disableComputerVision);

  ((Visual20qData*)sessions[sess_ind].example->x)->SetSession(session);

  session->Preprocess();
  ReturnClassesIfNecessary(session, root, response);
  UnlockSession(sess_ind);

  return true;
}

int TaxonomicalLoss(Classes *classes, ObjectClass *pred, ObjectClass *gt) {
  char species[400]; strcpy(species, pred->GetMeta("Latin"));
  char genus[400]; strcpy(genus, species);
  char *ptr = strstr(genus, " "); if(!ptr) { fprintf(stderr, "%s %s\n", pred->Name(), species); } assert(ptr); *ptr = '\0';
  char family[400]; strcpy(family, pred->GetMeta("Family"));
  char order[400]; strcpy(order, pred->GetMeta("Order"));
    
  char species2[400]; strcpy(species2, gt->GetMeta("Latin"));
  char genus2[400]; strcpy(genus2, species2);
  char *ptr2 = strstr(genus2, " "); assert(ptr2); *ptr2 = '\0';
  char family2[400]; strcpy(family2, gt->GetMeta("Family"));
  char order2[400]; strcpy(order2, gt->GetMeta("Order"));

  if(pred == gt)
    return 0;
  else if(!strcmp(genus, genus2))
    return 1;
  else if(!strcmp(family, family2))
    return 2;
  else if(!strcmp(order, order2))
    return 3;
  else
    return 4;
}

#define TIME_FACTOR 45
#define SERVER "sbranson.no-ip.org"
int *g_user_images[5000] = {NULL};
float *g_user_losses[5000] = {NULL};
bool *g_user_is_control[5000] = {NULL};

bool ParseUserStudyResults(int user, int image, int predClass, Classes *classes, Json::Value& response, float sessionTime, QuestionAskingSession *session, int setFound) {
  if(user >= 5000) return false;
  char line[1000], fname[1000], fname2[1000], imgName[1000], url[1000];
  bool retval = false;
  sprintf(fname, "user_study/user_images/%d.txt", user);
  FILE *fin = fopen(fname, "r");
  if(!fin) {
    fprintf(stderr, "Couldn't find %s\n", fname);
    return false;
  }
  int *image_order = NULL;
  char type[200][50];
  int num = 0, numImages = 0, i, found = -1;
  int images[1000];
  float loss=0;
  bool isControl = false;

  while(fgets(line, 999, fin)) {
    if(sscanf(line, "%d %s %s", &i, type[num], imgName) == 3 ) {
      char *ptr = strstr(imgName, "/");
      if(ptr) {
	*ptr = '\0';
	if(i == image && predClass >= 0) {
	  ObjectClass *trueClass = classes->FindClass(imgName);
	  int taxoLoss = TaxonomicalLoss(classes, classes->GetClass(predClass), trueClass);
	  loss = my_min(5, taxoLoss + sessionTime/TIME_FACTOR);
	  response["taxonomicalLoss"] = taxoLoss;
	  response["sessionTime"] = sessionTime;
	  response["loss"] = loss;
	  response["predictedClass"] = predClass;
	  response["trueClass"] = trueClass->Id();
	  response["timeFactor"] = TIME_FACTOR;
	  response["type"] = type[num];
	  retval = true;
	  isControl = !strcmp(type[num], "control");
	  if(session) {
	    char str[1000];
	    sprintf(str, "<h1>Predicted %s when the true class is %s</h1><font color=#0000ff><br><h2>Loss was %f, with a taxonomical loss of %f in %f seconds</h2></font>\n", classes->GetClass(predClass)->Name(), trueClass->Name(), (float)my_min(5, taxoLoss + sessionTime/TIME_FACTOR), (float)taxoLoss, (float)sessionTime); 
	    session->DebugReplaceString("Class is unknown", str);
	  }
	  sprintf(fname2, "score_logs/%d.txt", user);
	  FILE *fout = fopen(fname2, "a");
	  if(fout) {
	    fprintf(fout, "image %d, loss %f, taxo_loss %f, time %f, pred_class %d, true_class %d\n", image, (float)loss, (float)taxoLoss, (float)sessionTime, predClass, trueClass->Id());
	    fclose(fout);
	  }
	}
	if(strcmp(type[num], "whatbird")) 
	  images[numImages++] = i;
	num++;
      }
    }
  }
  fclose(fin);


  if(!g_user_images[user]) {
    g_user_images[user] = RandPerm(numImages);
    g_user_losses[user] = new float[numImages];
    g_user_is_control[user] = new bool[numImages];
    for(int i = 0; i < numImages; i++) {
      g_user_losses[user][i] = 0;
      g_user_is_control[user][i] = false;
    }
  }

  for(int i = 0; i < numImages; i++) {
    if(images[g_user_images[user][i]] == image) {
      found = i;
      g_user_losses[user][found] = loss;
      g_user_is_control[user][found] = isControl;
      break;
    }
  }
  float sumLoss = 0;
  for(int i = 0; i < numImages; i++) 
    if( g_user_is_control[user][i])
      sumLoss +=  g_user_losses[user][i];
  
  sprintf(fname2, "score_logs/%d.txt", user);
  FILE *fout = fopen(fname2, "a");
  if(fout) {
    fprintf(fout, "Total loss was %f\n", (float)sumLoss);
    fclose(fout);
  }

  if(found < numImages-1 || (setFound >= 0 && setFound < numImages)) {
    int ind = found+1;
    if(setFound >= 0) ind = setFound;
    sprintf(url, "http://%s/visipedia/user_study/%d/%s/%d_%d.jpg", SERVER, user, type[images[g_user_images[user][ind]]], user, images[g_user_images[user][ind]]);
    response["nextImage"] = url;
    response["user_study_image_id"] = images[g_user_images[user][ind]];
  }
  response["sumLoss"] = sumLoss;

  return retval;
}
