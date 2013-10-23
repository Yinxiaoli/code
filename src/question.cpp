/* 
Copyright (C) 2010-11 Steve Branson and Catherine Wah

This code may not be redistributed without the consent of the authors.
*/

#define NUM_SAMPLES_CLASS_PROB 10
//#define NUM_SAMPLES_CLASS_PROB 100

#include "question.h"
#include "imageProcess.h"
#include "dataset.h"
#include "classes.h"
#include "class.h"
#include "attribute.h"
#include "pose.h"
#include "spatialModel.h"


QuestionAskingSession::QuestionAskingSession(ImageProcess *p, PartLocalizedStructuredLabelWithUserResponses *responses, bool isInteractive, 
					     QuestionSelectMethod method, const char *debugDir, 
					     int debugNumClassPrint, bool debugProbabilityMaps, 
					     bool debugClickProbabilityMaps, int debugNumSamples, 
					     bool debugQuestionEntropies, bool debugMaxLikelihoodSolution, bool keepBigImage) {
  this->process = p;
  this->classes = p->GetClasses();
  this->responses = responses;
  this->isInteractive = isInteractive;
  this->questionSelectMethod = method;

  process->SetBidirectional(true);
  process->SetComputeClickParts(true);

  this->htmlDebugDir = debugDir ? StringCopy(debugDir) : NULL;
  this->debugNumClassPrint = debugNumClassPrint;
  this->debugProbabilityMaps = debugProbabilityMaps;
  this->debugClickProbabilityMaps = debugClickProbabilityMaps;
  this->debugNumSamples = debugNumSamples;
  this->debugQuestionEntropies = debugQuestionEntropies;
  this->debugMaxLikelihoodSolution = debugMaxLikelihoodSolution;
  this->debugName = NULL;
  this->keepBigImage = keepBigImage;

  numClickQuestionsAsked = 0;
  freeResponses = false;
  elapsedTime = 0;
  questionsAsked = NULL;
  questionResponses = NULL;
  numQuestionsAsked = 0;
  classProbHistory = NULL;
  disableComputerVision = false;

  entropy_to_time_conversion = 5;
  

  probabilitiesChanged = false;
  questionAskedMap = (bool*)malloc(sizeof(bool)*classes->NumQuestions());
  memset(questionAskedMap, 0, sizeof(bool)*classes->NumQuestions());

  classProbs = (double*)malloc(classes->NumClasses()*sizeof(double));
  computerVisionProbs = (double*)malloc(classes->NumClasses()*sizeof(double));
  classPriors = (double*)malloc(classes->NumClasses()*sizeof(double));
  for(int i = 0; i < classes->NumClasses(); i++)
    classProbs[i] = classPriors[i] = 1.0/classes->NumClasses();

  buff = new EntropyBuffers(classes->NumClasses());

  samples = NULL;
  num_samples = NUM_SAMPLES_CLASS_PROB;
  sampleClassProbs = NULL;

  questionInformationGains = (double*)malloc(sizeof(double)*classes->NumQuestions());
  questionTimeReductions = (double*)malloc(sizeof(double)*classes->NumQuestions());
}



QuestionAskingSession::~QuestionAskingSession() {
  if(debugName) free(debugName);
  if(htmlDebugDir) free(htmlDebugDir);
  if(buff) delete buff;
  if(questionAskedMap) free(questionAskedMap);
  if(classProbs) free(classProbs);
  if(computerVisionProbs) free(computerVisionProbs);
  if(questionResponses) {
    if(freeResponses)
      for(int i = 0; i < numQuestionsAsked; i++)
	if(questionResponses[i])
	  questionsAsked[i]->FreeAnswer(questionResponses[i]);
    free(questionResponses);
  }
  if(questionsAsked) free(questionsAsked);
  if(sampleClassProbs) free(sampleClassProbs);
  if(classPriors) free(classPriors);
  if(samples) {
    for(int i = 0; i < num_samples; i++)
      delete [] samples->samples[i];
	if(samples->root_samples) delete [] samples->root_samples;
    free(samples);
  }
  if(classProbHistory) {
    for(int i = 0; i <= numQuestionsAsked; i++)
      if(classProbHistory[i])
	free(classProbHistory[i]);
    free(classProbHistory);
  }
  if(questionInformationGains) free(questionInformationGains);
  if(questionTimeReductions) free(questionTimeReductions);
}

void QuestionAskingSession::DisableQuestions(bool disableClick, bool disableBinary, bool disableMultiple, bool disableComputerVision) {
  for(int i = 0; i < classes->NumQuestions(); i++) {
    Question *q = classes->GetQuestion(i);
    if((disableClick||disableComputerVision) && !strcmp(q->Type(), "part_click"))
      questionAskedMap[i] = true;
    if(disableBinary && !strcmp(q->Type(), "binary"))
      questionAskedMap[i] = true;
    if(disableMultiple && (!strcmp(q->Type(), "multiple_choice") || !strcmp(q->Type(), "batch")))
      questionAskedMap[i] = true;
  }
  this->disableComputerVision = disableComputerVision;
}

// Main loop: continuously pose questions to the user, at each step selecting the question with highest information gain
int QuestionAskingSession::AskAllQuestions(int maxQuestions, bool stopEarly) {
  void *answer;
  freeResponses = true;

  // Initialize class probabilities using computer vision
  if(!disableComputerVision) {
    process->Detect();
    probabilitiesChanged = true;
  }

  ComputeClassProbabilities(classProbs);
  classProbHistory = (ClassProb**)realloc(classProbHistory, sizeof(ClassProb*)*(numQuestionsAsked+3));
  classProbHistory[numQuestionsAsked] = BestClasses(classProbs, classes->NumClasses()); 
  classProbHistory[numQuestionsAsked+1] = classProbHistory[numQuestionsAsked+2] = NULL;

  elapsedTime = 0;
  for(int i = 0; i < maxQuestions && i <= classes->NumQuestions(); i++) {

    if(htmlDebugDir)
      PrintDebugInfo(debugNumClassPrint);

    if(stopEarly && responses) { // Assume perfect user class verification (Method 2 from ECCV'10)
      if(classProbHistory[numQuestionsAsked][0].classID == responses->GetClass()->Id())
	break;
      else
	classPriors[classProbHistory[numQuestionsAsked][0].classID] = 0;
    }

    int qi = SelectNextQuestion();

    probabilitiesChanged = false;
    if(qi != -1) {
      Question *q = classes->GetQuestion(qi);
      if(isInteractive)
        answer = q->GetAnswerFromRealUser(process);
      else
        answer = q->GetAnswerFromTestExample(responses);
	
      double timeSpent = q->FinalizeAnswer(answer, classPriors, this, i+1);
      UpdateQuestionHistory(qi, answer, timeSpent);

      ComputeClassProbabilities(classProbs, probabilitiesChanged);
      classProbHistory = (ClassProb**)realloc(classProbHistory, sizeof(ClassProb*)*(numQuestionsAsked+3));
      classProbHistory[numQuestionsAsked] = BestClasses(classProbs, classes->NumClasses()); 
      classProbHistory[numQuestionsAsked+1] = classProbHistory[numQuestionsAsked+2] = NULL;
    } else if(!stopEarly || 1)
      break;
  }

  if(htmlDebugDir) {
    char html_name[1000];
    const char *baseName = debugName ? debugName : process->Features()->Name();
    sprintf(html_name, "%s/%s.html", htmlDebugDir, baseName);
    FILE *fout = fopen(html_name, "a");
    fprintf(fout, "</table></body></html>");
    fclose(fout);
  }

  return numQuestionsAsked ? classProbHistory[numQuestionsAsked-1][0].classID : -1;
}

void QuestionAskingSession::FinalizeAnswer(Question *q, void *answer) {
  if(q) { 
    probabilitiesChanged = false;

    double timeSpent = q->FinalizeAnswer(answer, classPriors, this, numQuestionsAsked+1);
    UpdateQuestionHistory(q->Id(), answer, timeSpent);
    
    ComputeClassProbabilities(classProbs, probabilitiesChanged);
  } else
    ComputeClassProbabilities(classProbs, true);

  classProbHistory = (ClassProb**)realloc(classProbHistory, sizeof(ClassProb*)*(numQuestionsAsked+3));
  classProbHistory[numQuestionsAsked] = BestClasses(classProbs, classes->NumClasses()); 
  classProbHistory[numQuestionsAsked+1] = classProbHistory[numQuestionsAsked+2] = NULL;

  if(htmlDebugDir)
    PrintDebugInfo(debugNumClassPrint);
}


void QuestionAskingSession::Preprocess() {
  // Initialize class probabilities using computer vision
  if(!disableComputerVision) {
    process->Detect();
    probabilitiesChanged = true;
  }
  FinalizeAnswer(NULL, NULL);
}

void QuestionAskingSession::ComputeClassProbabilities(double *classProbs, bool redrawSamples) {
  int i, j;

  if(disableComputerVision) {
    for(j = 0; j < classes->NumClasses(); j++)
      computerVisionProbs[j] = 1;
  } else if(redrawSamples) {
    // Draw a set of part location samples Theta_1...Theta_K according to the distribution p(Theta|x,u_1...u_T)
    if(samples) {
      for(i = 0; i < num_samples; i++)
        delete [] samples->samples[i];
	  if(samples->root_samples) delete [] samples->root_samples;
      free(samples);
    }
    samples = process->DrawRandomPartLocationSet(num_samples, NULL, true);
    
    if(!sampleClassProbs) {
      // Allocate matrix to store induced class probabilities for each sample
      sampleClassProbs = (double**)malloc(num_samples*(sizeof(double*)+sizeof(double)*classes->NumClasses()));
      double *ptr = (double*)(sampleClassProbs+num_samples);
      for(i = 0; i < num_samples; i++, ptr += classes->NumClasses()) 
	sampleClassProbs[i] = ptr;
    }

    // Compute the class probabilities as the average over samples
    for(j = 0; j < classes->NumClasses(); j++)
      computerVisionProbs[j] = 0;
    for(i = 0; i < num_samples; i++) {
      process->ComputeClassProbabilitiesAtLocation(sampleClassProbs[i], samples->samples[i]);
      for(j = 0; j < classes->NumClasses(); j++)
	computerVisionProbs[j] += sampleClassProbs[i][j];
    }
  }

  if(classPriors) {
    for(j = 0; j < classes->NumClasses(); j++)
      classProbs[j] = computerVisionProbs[j]*classPriors[j];
  } else {
    for(j = 0; j < classes->NumClasses(); j++)
      classProbs[j] = computerVisionProbs[j];
  }

  NormalizeProbabilities(classProbs, classProbs, classes->NumClasses()); 
}

void QuestionAskingSession::PrintSummary(char *htmlStr, int numClasses) {
  char *ptr = htmlStr, *ptr2;
  int numC;
  sprintf(ptr, "<b>True class is %s</b>", responses->GetClass()->Name());
  ptr += strlen(ptr);
  for(int j = -1; j < GetNumQuestionsAsked(); j++) {
    if(j >= 0) {
      sprintf(ptr, "<br><b>Q%d: </b>", j+1); ptr += strlen(ptr);
      GetQuestionAsked(j)->GetResponseHTML(ptr, GetQuestionResponse(j), j, NULL, htmlDebugDir);
      if((ptr2=strstr(ptr, "<h3>")) != NULL) { ptr2[0] = ptr2[1] = ptr2[2] = ptr2[3] = ' '; }
      else if((ptr2=strstr(ptr, "</h3>")) != NULL) { ptr2[0] = ptr2[1] = ptr2[2] = ptr2[3] = ptr2[4] = ' '; }
      strcat(ptr, "<br>");
    } else {
      sprintf(ptr, "<br>Initial Class Predictions: ");
    }
    numC = my_min(my_max(numClasses,10), classes->NumClasses());
    ptr += strlen(ptr);
    for(int i = 0; i < numC; i++) {
      int ind = classProbHistory[j+1][i].classID;
      if(responses && responses->GetClass()->Id() == ind)
	sprintf(ptr, " <b>%s</b>", classes->GetClass(ind)->Name());
      else
	sprintf(ptr, " %s", classes->GetClass(ind)->Name());
      ptr += strlen(ptr);
    }
  }
}
void QuestionAskingSession::DebugReplaceString(const char *match, const char *replace) {
  if(htmlDebugDir && process) {
    const char *baseName = debugName ? debugName : process->Features()->Name();
    char fname[1000];
    sprintf(fname, "%s/%s.html", htmlDebugDir, baseName);
    char *str = ReadStringFile(fname);
    if(str) {
      char *str2 = StringReplace(str, match, replace);
      FILE *fout = fopen(fname, "w");
      if(fout) {
	fprintf(fout, "%s", str2);
	fclose(fout);
      }
      free(str);
      free(str2);
    }
  }
}

void QuestionAskingSession::PrintDebugInfo(int num) {
  char *classStr=(char*)malloc(100000); strcpy(classStr, "");
  if(num) {
    sprintf(classStr, "Top %d classes:", num);
    for(int i = 0; i < num && i < classes->NumClasses(); i++) {
      int ind = classProbHistory[numQuestionsAsked][i].classID;
      if(responses && responses->GetClass()->Id() == ind)
	sprintf(classStr+strlen(classStr), " <b>%s:%f</b>", classes->GetClass(ind)->Name(), classProbs[ind]);
      else
	sprintf(classStr+strlen(classStr), " %s:%f", classes->GetClass(ind)->Name(), classProbs[ind]);
    }
    sprintf(classStr+strlen(classStr), "\n");
  }

  const char *baseName = debugName ? debugName : process->Features()->Name();
  char fname[1000], html_name[1000], qstr[1000], *html=(char*)malloc(200000);
  sprintf(fname, "%s/%s_q%d", htmlDebugDir, baseName, numQuestionsAsked);
  
  sprintf(html_name, "%s/%s.html", htmlDebugDir, baseName);
  FILE *fout;
  if(!numQuestionsAsked) { 
    char fname[1000]; strcpy(fname, process->ImageName());
      sprintf(fname, "%s/%s_gt.png", htmlDebugDir, baseName);
      IplImage *img = cvCloneImage(process->Features()->GetImage());
      if(responses) {
	process->Draw(img, responses->GetPartLocations());
      }
      cvSaveImage(fname, img);
      cvReleaseImage(&img);

    sprintf(fname, "%s_gt.png", baseName);
    
    fout = fopen(html_name, "w"); 
    fprintf(fout, "<html><body><h1>%s</h1>\n<img src=\"%s\"><br>Class is %s<br>\n", 
	    process->Features()->Name(), fname, responses ? responses->GetClass()->Name() : "unknown");

    fprintf(fout, "\n\n<br><table>");
  } else {
    fout = fopen(html_name, "a");

    if(debugQuestionEntropies) {
      fprintf(fout, "\n<tr><td colspan=5>\n");
      for(int i = 0; i < classes->NumQuestions(); i++) {
	if(!questionAskedMap[i]) {
	  fprintf(fout, "<br>Question '%s' expected information gain %f, time reduction %f\n", 
		  classes->GetQuestion(i)->GetText(), (float)questionInformationGains[i], (float)questionTimeReductions[i]);
	}
      }
      fprintf(fout, "</td></tr>\n");
    }
  }
  assert(fout);

  fprintf(fout, "\n<tr><td><br></td></tr><tr><td><h2>Question %d</h2></td></tr><tr>", numQuestionsAsked);
  fprintf(fout, "<td>%s</td>", numQuestionsAsked && questionsAsked[numQuestionsAsked-1] ? questionsAsked[numQuestionsAsked-1]->GetResponseHTML(qstr, questionResponses[numQuestionsAsked-1], numQuestionsAsked, process, htmlDebugDir) : "");

  if(debugMaxLikelihoodSolution && probabilitiesChanged) {
    sprintf(fname, "%s/%s_ml_q%d.png", htmlDebugDir, baseName, numClickQuestionsAsked);
    IplImage *img = cvCloneImage(process->Features()->GetImage());
    process->Draw(img, NULL, CV_RGB(0,0,255), false, false, true, false, true, -1, true);
    process->DrawClicks(img, CV_RGB(255,0,0), false, false, true, false);
    cvSaveImage(fname, img);
    cvReleaseImage(&img);
    sprintf(fname, "%s_ml_q%d.png", baseName, numClickQuestionsAsked);
    fprintf(fout, "<td><center><img src=\"%s\" height=300><br>Max Likelihood Solution</center></td>", fname);
  }

  strcpy(html, "");

  if(debugProbabilityMaps && probabilitiesChanged) {
    sprintf(fname, "%s_%d", baseName, numClickQuestionsAsked);
    process->SaveProbabilityMaps(fname, htmlDebugDir, html, false, true, true, true, keepBigImage);
    //process->SaveProbabilityMaps(fname, htmlDebugDir, html);
  }
  fprintf(fout, "%s\n", html);

  strcpy(html, "");
  if(debugClickProbabilityMaps && probabilitiesChanged) {
    sprintf(fname, "%s_%d_click", baseName, numClickQuestionsAsked);
    process->SaveProbabilityMaps(fname, htmlDebugDir, html, true, true, true, true, keepBigImage);
  }
  fprintf(fout, "%s\n", html);

  
  char sample_html[5000];
  strcpy(sample_html, "");
  if(debugNumSamples && probabilitiesChanged) {
    for(int i = 0; i < classes->NumQuestions(); i++) {
      if(!strcmp(classes->GetQuestion(i)->Type(), "part_click")) {
	PartLocationSampleSet *click_samples = ((ClickQuestion*)classes->GetQuestion(i))->GetClickSamples();
	if(!click_samples) continue;
	IplImage *img = cvCloneImage(process->Features()->GetImage());
	sprintf(fname, "%s/%s_%d_%d_click_samples.png", htmlDebugDir, baseName, numClickQuestionsAsked, i);
	int ind = ((ClickQuestion*)classes->GetQuestion(i))->GetPart()->Id(), j;
	float click_x, click_y, part_x, part_y;
	for(j = 0; j < samples->num_samples; j++) {
      samples->samples[j][ind].GetImageLocation(&part_x, &part_y);
      cvCircle(img, cvPoint(part_x,part_y), 4, CV_RGB(0,0,255));
    }
	for(j = 0; j < click_samples->num_samples; j++) {
      click_samples->samples[j][ind].GetImageLocation(&click_x, &click_y);
      cvCircle(img, cvPoint(click_x,click_y), 4, CV_RGB(255,0,0));
    }
    cvSaveImage(fname, img);
	cvReleaseImage(&img);
	sprintf(fname, "%s_%d_%d_click_samples.png", baseName, numClickQuestionsAsked, i);
	fprintf(fout, "<td><center><img src=\"%s\" height=300><br>%s Samples</center></td>", fname, classes->GetPart(ind)->Name());
      }
    }

    for(int i = 0; i < samples->num_samples && i < debugNumSamples; i++) {
      sprintf(fname, "%s/%s_samples_q%d_%d.png", htmlDebugDir, baseName, numQuestionsAsked, i);
      IplImage *img = cvCloneImage(process->Features()->GetImage());
      ObjectPartInstance *p = process->GetPartInst(samples->root_samples[i].GetPartID());
      process->Draw(img, samples->samples[i], CV_RGB(255,0,0), false, false, false, true, false, -1);
      p->Draw(img, &samples->root_samples[i], CV_RGB(0,255,0), CV_RGB(0,255,0), CV_RGB(0,255,0), 
		    NULL/*p->Model()->Name()*/, true, false);
      cvSaveImage(fname, img);
      cvReleaseImage(&img);
      sprintf(fname, "%s_samples_q%d_%d.png", baseName, numQuestionsAsked, i);
      sprintf(sample_html+strlen(sample_html), "<td><center><img src=\"%s\" height=300><br>Sample %d</center></td>", fname, i);
    }
  }

  fprintf(fout, "%s</tr>\n<tr><td colspan=%d>%s</td></tr>\n", sample_html, classes->NumPoses(), classStr);
  strcpy(sample_html, "");

  fclose(fout);

  free(html);
  free(classStr);
}

void QuestionAskingSession::VerifyClass(int c, bool verify) {
  if(verify) {
    for(int i = 0; i < classes->NumClasses(); i++)
      classPriors[i] = i == c ? 1 : 0;
  } else {
    classPriors[c] = 0;
    NormalizeProbabilities(classPriors, classPriors, classes->NumClasses()); 
  }
  UpdateQuestionHistory(-1, NULL, 0);
  ComputeClassProbabilities(classProbs, probabilitiesChanged);

  classProbHistory = (ClassProb**)realloc(classProbHistory, sizeof(ClassProb*)*(numQuestionsAsked+3));
  classProbHistory[numQuestionsAsked] = BestClasses(classProbs, classes->NumClasses());
  classProbHistory[numQuestionsAsked+1] = classProbHistory[numQuestionsAsked+2] = NULL;
  

  if(htmlDebugDir) {
    char html_name[1000];
    const char *baseName = debugName ? debugName : process->Features()->Name();
    sprintf(html_name, "%s/%s.html", htmlDebugDir, debugName);
    FILE *fout = fopen(html_name, "a");
    if(verify) fprintf(fout, "<tr><td colspan=3><h2>Verified Class %s</h2></tr>\n", classes->GetClass(c)->Name());
    else fprintf(fout, "<tr><td colspan=3><h2>Exclude Class %s</h2></td></tr>\n", classes->GetClass(c)->Name());
    fclose(fout);
  }
}


// Maintain a history of what questions have been asked and what their answers were
void QuestionAskingSession::UpdateQuestionHistory(int qi, void *answer, double timeSpent) {
  Question *q = qi >= 0 ? classes->GetQuestion(qi) : NULL;

  if(qi >= 0)
    questionAskedMap[qi] = true;

  questionsAsked = (Question**)realloc(questionsAsked, sizeof(Question*)*(numQuestionsAsked+1));
  questionsAsked[numQuestionsAsked] = q;

  questionResponses = (void**)realloc(questionResponses, sizeof(void*)*(numQuestionsAsked+1));
  questionResponses[numQuestionsAsked] = answer;

  elapsedTime += timeSpent;

  numQuestionsAsked++;
}

int QuestionAskingSession::SelectNextQuestion() {
  if(questionSelectMethod == QS_RANDOM) {
    int ind=-1, i = 0;
    do {
      ind = rand()%classes->NumQuestions();
      i++;
    } while(questionAskedMap[ind] && i < classes->NumQuestions()*100);
    return ind;
  }

  double entropy = ComputeEntropy(classProbs, classes->NumClasses()), IG, expected_time_reduction;
  double best_IG = -INFINITY;
  double best_time_reduction = -INFINITY;
  int best_time_q = -1, best_IG_q = -1;
  if(g_debug > 1) fprintf(stderr, "Entropy is %f\n", (float)entropy);
  for(int i = 0; i < classes->NumQuestions(); i++) {
    if(!questionAskedMap[i]) {
      double new_entropy = classes->GetQuestion(i)->ComputeExpectedEntropy(classProbs, classPriors, this);
      IG = (entropy-new_entropy);
      expected_time_reduction = (entropy-new_entropy)/classes->GetQuestion(i)->ExpectedSecondsToAnswer(); //(entropy-new_entropy)*entropy_to_time_conversion - classes->GetQuestion(i)->ExpectedSecondsToAnswer();
      if(IG > best_IG) {
        best_IG = IG;
        best_IG_q = i;
      } else if(expected_time_reduction > best_time_reduction) {
        best_time_reduction = expected_time_reduction;
        best_time_q = i;
      }
      if(g_debug > 2) 
	fprintf(stderr, "Question '%s' expected information gain %f, time reduction %f\n", 
		classes->GetQuestion(i)->GetText(), (float)IG, (float)expected_time_reduction);
    } else
      IG = expected_time_reduction = 0;
    questionInformationGains[i] = IG;
    questionTimeReductions[i] = expected_time_reduction;
  }

  if(g_debug > 1) 
    if(best_IG_q >= 0 && best_time_q >= 0)
      fprintf(stderr, "Best information gain '%s' %f, Best time '%s' %f\n", 
	      classes->GetQuestion(best_IG_q)->GetText(), (float)best_IG, 
	      classes->GetQuestion(best_time_q)->GetText(), (float)best_time_reduction);
  
  return questionSelectMethod == QS_TIME_REDUCTION ? best_time_q : best_IG_q;
}

Question::Question(Classes *classes) {
  strcpy(questionText, "");
  numAnswers = 0;
  this->classes = classes;
  expectedSecondsToAnswer = 1;
  id = -1;
  visualization_image_name = NULL;
}

Question::~Question() {
	free(visualization_image_name);
}


double Question::ComputeExpectedEntropy(double *classProbs, double *classPriors, QuestionAskingSession *session) {
  ComputeAnswerProbs(classProbs, classPriors, session);
  return session->GetEntropyBuffers()->ComputeExpectedEntropy();
}

Question *Question::New(const Json::Value &root, Classes *classes) {
  char type[1000]; strcpy(type, root.get("type", "").asString().c_str());
  Question *retval;
  if(!strcmp(type, "binary")) retval = new BinaryAttributeQuestion(classes);
  else if(!strcmp(type, "multiple_choice")) retval = new MultipleChoiceAttributeQuestion(classes);
  else if(!strcmp(type, "batch")) retval = new BatchQuestion(classes);
  else if(!strcmp(type, "part_click")) retval = new ClickQuestion(classes);
  else { fprintf(stderr, "Invalid question type %s\n", type); return NULL; }

  retval->id = root.get("id", -1).asInt();
  retval->expectedSecondsToAnswer = root.get("expectedTimeSeconds", 0).asDouble();
  strcpy(retval->questionText, root.get("text","").asString().c_str());
  retval->visualization_image_name = root.isMember("visualization") ? StringCopy(root["visualization"].asString().c_str()) : NULL;
  if(!retval->Load(root, classes)) { delete retval; return NULL; }

  return retval;
}

void Question::SaveGeneric(Json::Value &root) {
  root["type"] = type;
  root["id"] = id;
  root["expectedTimeSeconds"] = expectedSecondsToAnswer;
  if(questionText) root["text"] = questionText;
  if(visualization_image_name) root["visualization"] = visualization_image_name;
}


int Question::AskCertaintyQuestion() {
  char c[1000];
  while(1) {
    fprintf(stderr, "How certain are you (");
    for(int i = 0; i < classes->NumCertainties(); i++)
      fprintf(stderr, "%s%s", i ? "," : "", classes->GetCertainty(i));
    fprintf(stderr, "): ");
    if(!fscanf(stdin, "%s", c)) return -1;
    for(int i = 0; i < classes->NumCertainties(); i++)
      if(!strcmp(c, classes->GetCertainty(i)))
	return i;
  }
  return -1;
}



BinaryAttributeQuestion::BinaryAttributeQuestion(Classes *classes) : Question(classes) {
  attribute_ind = -1;
  type = "binary";
}

void *BinaryAttributeQuestion::GetAnswerFromRealUser(ImageProcess *process) {
  AttributeAnswer *a = (AttributeAnswer*)malloc(sizeof(AttributeAnswer));
  char r[1000];
  do {
    fprintf(stderr, "%s (yes/no): ", questionText);
    if(!fscanf(stdin, "%s", r)) return NULL;
  } while(strcmp(r, "yes") && strcmp(r, "no"));
  a->answer = !strcmp(r, "yes") ? 1 : 0;

  a->certainty = AskCertaintyQuestion();

  a->responseTimeSec = 0; // TODO: set this
  return a;
}

void *BinaryAttributeQuestion::GetAnswerFromTestExample(PartLocalizedStructuredLabelWithUserResponses *u) {
  AttributeAnswer *a = (AttributeAnswer*)malloc(sizeof(AttributeAnswer));
  *a = u->GetAttribute(0, attribute_ind);
  if(g_debug > 1) fprintf(stderr, "  answer %s (%s) (%f seconds)\n", a->answer ? "yes" : "no", 
			  classes->GetCertainty(a->certainty), a->responseTimeSec);
  return a;
}

char *BinaryAttributeQuestion::GetResponseHTML(char *str, void *answer, int q_num, ImageProcess *p, const char *htmlDebugDir) {
  AttributeAnswer *ans = (AttributeAnswer*)answer;
  sprintf(str, "<h3><font color=\"#000099\">%s?</font> <font color=\"#990000\">%s (%s) (%f seconds) </font></h3>", questionText, ans->answer ? "yes" : "no", 
	  classes->GetCertainty(ans->certainty), ans->responseTimeSec);
  return str;
}

double BinaryAttributeQuestion::FinalizeAnswer(void *answer, double *classPriors, 
					       QuestionAskingSession *session, int q_num) {
  Attribute *a = classes->GetAttribute(attribute_ind);
  AttributeAnswer *ans = (AttributeAnswer*)answer;
  ImageProcess *process = session->GetProcess();
  float *certaintyWeights = classes->GetCertaintyWeights();

  if(a->Part()) {
    ObjectPartInstance *inst = process->GetPartInst(a->Part()->Id());
    inst->SetAttributeAnswer(attribute_ind, ans);
  }

  int i;
  
  //fprintf(stderr, "\n\n%s: ", a->Name());
  for(i = 0; i < classes->NumClasses(); i++) {
      assert(!isnan(classPriors[i]));
      classPriors[i] *= classes->GetClass(i)->GetAttributeUserProbability(attribute_ind, ans->answer, ans->certainty);
      assert(!isnan(classPriors[i]));
    /*fprintf(stderr, " %s=%f,%f", classes->GetClass(i)->Name(),
	    (float)classes->GetClass(i)->GetAttributeUserProbability(attribute_ind, ans->answer, ans->certainty),
	    (float)classPriors[i] );*/
  }
  NormalizeProbabilities(classPriors, classPriors, classes->NumClasses());

  return ans->responseTimeSec;
}


void BinaryAttributeQuestion::SetAttribute(int id) { 
  attribute_ind = id; 
  attr = classes->GetAttribute(id); 
} 

bool BinaryAttributeQuestion::Load(const Json::Value &root, Classes *classes) {
  attribute_ind = root.get("attribute",-1).asInt();
  attr=attribute_ind >= 0 ? classes->GetAttribute(attribute_ind) : NULL;  
  if(!attr) { fprintf(stderr, "Couldn't find attribute for question\n"); return false; }
  attr->SetQuestion(this);
  return true;
}
Json::Value BinaryAttributeQuestion::Save() {
  Json::Value root;
  Question::SaveGeneric(root);
  root["attribute"] = attribute_ind;
  return root;
}


// Given a current distribution of class probabilities p(c|...), compute a new distribution of
// class probabilities p(c|u,...) for every possible value of u (every possible answer to
// this question)
void BinaryAttributeQuestion::ComputeAnswerProbs(double *classProbs, double *classPriors, QuestionAskingSession *session) {
  int i, j, k, ind;
  int numClasses = classes->NumClasses();
  int numCertainties = classes->NumCertainties();
  double pos_p, neg_p;
  EntropyBuffers *buff = session->GetEntropyBuffers();

  numAnswers = numCertainties*2;
  buff->InitializeAndClear(numAnswers);
  for(i = 0; i < numClasses; i++) {
      //float **attributeNegativeUserProbs = classes->GetClass(i)->GetAttributeNegativeUserProbs();
      //float **attributePositiveUserProbs = classes->GetClass(i)->GetAttributePositiveUserProbs();
      if(classProbs) {
	for(j = 0, ind = 0; j < numCertainties; j++, ind+=2) {
	  neg_p = classes->GetClass(i)->GetAttributeUserProbability(attribute_ind, 0, j)*classProbs[i];
	  pos_p = classes->GetClass(i)->GetAttributeUserProbability(attribute_ind, 1, j)*classProbs[i];
	  //neg_p = attributeNegativeUserProbs[j][attribute_ind]*classProbs[i];
	  //pos_p = attributePositiveUserProbs[j][attribute_ind]*classProbs[i]; attributeNegativeUserProbs[j][attribute_ind]
	  buff->answerUnnormalizedClassProbs[ind][i] += neg_p;
	  buff->answerUnnormalizedClassProbs[ind+1][i] += pos_p;
	}
      } else {
	for(j = 0, ind = 0; j < numCertainties; j++, ind+=2) {
	  //neg_p = attributeNegativeUserProbs[j][attribute_ind];
	  //pos_p = attributePositiveUserProbs[j][attribute_ind];
	  neg_p = classes->GetClass(i)->GetAttributeUserProbability(attribute_ind, 0, j);
	  pos_p = classes->GetClass(i)->GetAttributeUserProbability(attribute_ind, 1, j);
	  buff->answerUnnormalizedClassProbs[ind][i] += neg_p;
	  buff->answerUnnormalizedClassProbs[ind+1][i] += pos_p;
	}
      }
  }
  buff->NormalizeAnswerProbabilities();
}



MultipleChoiceAttributeQuestion::MultipleChoiceAttributeQuestion(Classes *classes) : Question(classes) {
  choices = NULL;
  numChoices = 0;
  type = "multiple_choice";
}

MultipleChoiceAttributeQuestion::~MultipleChoiceAttributeQuestion() {
  if(choices) free(choices);
}



void *MultipleChoiceAttributeQuestion::GetAnswerFromRealUser(ImageProcess *process) {
  AttributeAnswer *a = (AttributeAnswer*)malloc(sizeof(AttributeAnswer));
  char r[1000], choiceStr[10000];
  int i;
  strcpy(choiceStr, "");
  while(1) {
    for(i = 0; i < numChoices; i++) {
      if(i) strcat(choiceStr, ",");
      strcat(choiceStr, classes->GetAttribute(choices[i])->Name());
    }
    fprintf(stderr, "%s (%s): ", questionText, choiceStr);
    if(!fscanf(stdin, "%s", r)) return NULL;
    for(i = 0; i < numChoices; i++) { 
      if(!strcmp(classes->GetAttribute(choices[i])->Name(), r)) {
	a->answer = choices[i];
	break;
      }
    }
  } 
  a->certainty = AskCertaintyQuestion();

  a->responseTimeSec = 0; // TODO: set this
  return a;
}

void *MultipleChoiceAttributeQuestion::GetAnswerFromTestExample(PartLocalizedStructuredLabelWithUserResponses *u) {
  AttributeAnswer *a = (AttributeAnswer*)malloc(sizeof(AttributeAnswer));
  int numFound = 0;
  *a = u->GetAttribute(0, choices[0]);
  a->certainty = 0;
  for(int i = 0; i < numChoices; i++) {
    if(u->GetAttribute(0, choices[i]).answer) {
      *a = u->GetAttribute(0, choices[i]);
      a->answer = choices[i];
      numFound++;
    }
  }
  /*if(numFound != 1) {
    int x = 1;
    }*/
  //assert(numFound == 1 || (numFound == 0 && !classes->GetCertaintyWeights()[a->certainty]));
  return a;
}


char *MultipleChoiceAttributeQuestion::GetResponseHTML(char *str, void *answer, int q_num, ImageProcess *p, const char *htmlDebugDir) {
  AttributeAnswer *ans = (AttributeAnswer*)answer;
  sprintf(str, "<h3><font color=\"#000099\">%s?</font> <font color=\"#990000\">%s (%s) (%f seconds) </font></h3>", questionText, 
	  ans->answer >= 0 ? classes->GetAttribute(ans->answer)->Name() : "none", 
	  classes->GetCertainty(ans->certainty), ans->responseTimeSec);
  return str;
}


double MultipleChoiceAttributeQuestion::FinalizeAnswer(void *answer, double *classPriors, QuestionAskingSession *session, int q_num) {
  Attribute *a = classes->GetAttribute(choices[0]);
  AttributeAnswer *ans = (AttributeAnswer*)answer;
  ImageProcess *process = session->GetProcess();
  int i;

  if(a->Part()) {  
    AttributeAnswer ab = *ans;
    ObjectPartInstance *inst = process->GetPartInst(a->Part()->Id());
    for(i = 0; i < numChoices; i++) {
      ab.answer = ans->answer == choices[i] ? 1 : 0;
      inst->SetAttributeAnswer(choices[i], &ab);
    }
  }

  if(ans->answer >= 0) {
    for(i = 0; i < classes->NumClasses(); i++) 
        classPriors[i] *= classes->GetClass(i)->GetAttributeUserProbability(ans->answer, 1, ans->certainty);
      //classPriors[i] *= classes->GetClass(i)->GetAttributePositiveUserProbs()[ans->certainty][ans->answer];
    
    NormalizeProbabilities(classPriors, classPriors, classes->NumClasses());
  }

  //EntropyBuffers *buff = session->GetEntropyBuffers();
  //memcpy(classProbs, buff->answerClassProbs[ans->certainty*numChoices+ans->answer], sizeof(double)*classes->NumClasses());
  return ans->responseTimeSec;
}


bool MultipleChoiceAttributeQuestion::Load(const Json::Value &root, Classes *classes) {
  if(root.isMember("attributes") && root["attributes"].isArray()) {
    for(int i = 0; i < (int)root["attributes"].size(); i++) {
      choices = (int*)realloc(choices, (numChoices+1)*sizeof(int));
      choices[i] = root["attributes"][i].asInt();
      numChoices++;
    }
  }
  return true;
}

Json::Value MultipleChoiceAttributeQuestion::Save() {
  Json::Value root, c;
  Question::SaveGeneric(root);
  for(int i = 0; i < numChoices; i++) 
    c[i] = choices[i];
  root["attributes"] = c;
  return root;
}


// Given a current distribution of class probabilities p(c|...), compute a new distribution of
// class probabilities p(c|u,...) for every possible value of u (every possible answer to
// this question)
void MultipleChoiceAttributeQuestion::ComputeAnswerProbs(double *classProbs, double *classPriors, QuestionAskingSession *session) {
  int i, j, k, ind;
  int numClasses = classes->NumClasses();
  int numCertainties = classes->NumCertainties();
  double p;
  EntropyBuffers *buff = session->GetEntropyBuffers();

  numAnswers = numCertainties*numChoices;
  buff->InitializeAndClear(numAnswers);
  for(i = 0; i < numClasses; i++) {
    //float **attributeUserProbs = classes->GetClass(i)->GetAttributePositiveUserProbs();
    for(j = 0, ind = 0; j < numCertainties; j++) {
      
        if(classProbs) {
          for(k = 0; k < numChoices; k++, ind++) {
            p = classes->GetClass(i)->GetAttributeUserProbability(choices[k], 1, j)*classProbs[i];
	    //p = attributeUserProbs[j][choices[k]]*classProbs[i];
            buff->answerUnnormalizedClassProbs[ind][i] += p;
         }
        } else {
          for(k = 0; k < numChoices; k++, ind++) {
            p = classes->GetClass(i)->GetAttributeUserProbability(choices[k], 1, j);
            //p = attributeUserProbs[j][choices[k]];
            buff->answerUnnormalizedClassProbs[ind][i] += p;
          }
        }
    }
  }
    
  buff->NormalizeAnswerProbabilities();
}





BatchQuestion::BatchQuestion(Classes *classes) : Question(classes) {
  questions = NULL;
  numQuestions = 0;
  type = "batch";
  numSamples = 500;
}

BatchQuestion::~BatchQuestion() {
  if(questions) free(questions);
}



void *BatchQuestion::GetAnswerFromRealUser(ImageProcess *process) {
  AttributeAnswer *a = (AttributeAnswer*)malloc(sizeof(void*)*numQuestions);
  for(int i = 0; i < numQuestions; i++) {
    AttributeAnswer *b = (AttributeAnswer*)questions[i]->GetAnswerFromRealUser(process);
	a[i] = *b;
	questions[i]->FreeAnswer(b);
  }
  return (void*)a;
}

void *BatchQuestion::GetAnswerFromTestExample(PartLocalizedStructuredLabelWithUserResponses *u) {
  AttributeAnswer *a = (AttributeAnswer*)malloc(sizeof(AttributeAnswer)*numQuestions);
  for(int i = 0; i < numQuestions; i++) {
	AttributeAnswer *b = (AttributeAnswer*)questions[i]->GetAnswerFromTestExample(u);
    a[i] = *b;
	questions[i]->FreeAnswer(b);
  }
  return (void*)a;
}

char *BatchQuestion::GetResponseHTML(char *str, void *answer, int q_num, ImageProcess *p, const char *htmlDebugDir) {
  AttributeAnswer *ans = (AttributeAnswer*)answer;
  
  char prefix[10000];
  strcpy(prefix, ((BinaryAttributeQuestion*)questions[0])->GetAttribute()->Name());
  for(int i = 0; i < numQuestions; i++) {
    char *ptr = ((BinaryAttributeQuestion*)questions[i])->GetAttribute()->Name();
    int j = 0;
    while(prefix[j] && ptr[j] == prefix[j])
      j++;
    prefix[j] = '\0';
  }
  int l = strlen(prefix);

  char ansStr[10000];
  strcpy(ansStr, "");
  for(int i = 0; i < numQuestions; i++) {
    if(ans[i].answer) {
      sprintf(ansStr+strlen(ansStr), "%s%s", ansStr[0] ? ", " : "", ((BinaryAttributeQuestion*)questions[i])->GetAttribute()->Name()+l);
    }
  }
  sprintf(str, "<h3><font color=\"#000099\">%s?</font> <font color=\"#990000\">%s (%s) (%f seconds) </font></h3>", questionText, ansStr, 
	  classes->GetCertainty(ans->certainty), ans->responseTimeSec);
  return str;
}

double BatchQuestion::FinalizeAnswer(void *answer, double *classPriors, QuestionAskingSession *session, int q_num) {
  AttributeAnswer *ans = (AttributeAnswer*)answer;
  double rt = 0;

  for(int i = 0; i < numQuestions; i++) {
    rt += questions[i]->FinalizeAnswer(&ans[i], classPriors, session, q_num);
  }

  return rt;
}

bool BatchQuestion::Load(const Json::Value &root, Classes *classes) {
  numQuestions = 0;
  numSamples = root.get("numSamples", 500).asInt(); 
  if(root.isMember("questions") && root["questions"].isArray()) {
    for(int i = 0; i < (int)root["questions"].size(); i++) {
      int qid = root["questions"][i].asInt();
      questions = (Question**)realloc(questions, (numQuestions+1)*sizeof(Question));
      if(qid >= id || qid < 0) { fprintf(stderr, "Invalid Batch Question id %d\n", qid); return false; }
      questions[numQuestions++] = classes->GetQuestion(qid);
    }
  }
  return true;
}

Json::Value BatchQuestion::Save() {
  Json::Value root, c;
  Question::SaveGeneric(root);
  root["numSamples"] = numSamples;
  for(int i = 0; i < numQuestions; i++) 
    c[i] = questions[i]->Id();
  root["questions"] = c;
  return root;
}



// Given a current distribution of class probabilities p(c|...), compute a new distribution of
// class probabilities p(c|u,...) for every possible value of u (every possible answer to
// this question)
void BatchQuestion::ComputeAnswerProbs(double *classProbs, double *classPriors, QuestionAskingSession *session) {
  EntropyBuffers *tmp = session->buff;
  EntropyBuffers **buffs = new EntropyBuffers*[numQuestions];
  float *sums = new float[numQuestions];
  int i, j, k, c;
  int numClasses = session->GetProcess()->GetClasses()->NumClasses();

  // Precompute the class probabilities for the answers to each question
  for(i = 0; i < numQuestions; i++) {
    session->buff = buffs[i] = new EntropyBuffers(classes->NumClasses());
    questions[i]->ComputeAnswerProbs(classProbs, classPriors, session);
    if(i > 0)
      for(j = 0; j < buffs[i]->numAnswers; j++)
	for(c = 0; c < numClasses; c++)
	  buffs[i]->answerClassProbs[j][c] /= classProbs[c];
    sums[i] = 0;
    for(j = 0; j < buffs[i]->numAnswers; j++)
      sums[i] += buffs[i]->answerProbs[j];
  }
  session->buff = tmp;
  
  EntropyBuffers *buff = session->GetEntropyBuffers();
  buff->InitializeAndClear(numSamples);

  // Randomly sample an answer to each question according to their probabilities
  for(i = 0; i < numSamples; i++) {
    for(c = 0; c < numClasses; c++)
      buff->answerUnnormalizedClassProbs[i][c] = 1;
    for(j = 0; j < numQuestions; j++) {
      // Randomly select an answer for each question
      float r = RAND_FLOAT*sums[j];
      float s = 0;
      for(k = 0; k < buffs[j]->numAnswers-1; k++) {
        s += buffs[j]->answerProbs[k];
	if(s >= r)
	  break;
      }
      for(c = 0; c < numClasses; c++)
        buff->answerUnnormalizedClassProbs[i][c] *= buffs[j]->answerClassProbs[k][c];
    }
  }
  buff->NormalizeAnswerProbabilities();
  for(i = 0; i < numSamples; i++)
    buff->answerProbs[i] = 1.0/numSamples;

  for(i = 0; i < numQuestions; i++)
    delete buffs[i];
  delete [] buffs;
  delete [] sums;
}




ClickQuestion::ClickQuestion(Classes *classes) : Question(classes) {
  clickPart = NULL;
  type = "part_click";
  num_samples = 0;
  click_samples = NULL;
}

ClickQuestion::~ClickQuestion() {
  if(click_samples) {
    for(int i = 0; i < click_samples->num_samples; i++)
      delete [] click_samples->samples[i];
	if(click_samples->root_samples) delete [] click_samples->root_samples;
    free(click_samples);
  }
}

void cvMouseHandler(int event, int x, int y, int flags, void* param) {
  PartLocation *l = *(PartLocation**)param, *locs;
  ObjectPartInstance *partClick = *(ObjectPartInstance**)(((PartLocation**)param)+1);
  ObjectPartInstance *part = partClick->GetParent();
  ImageProcess *process = part->Process();
  IplImage *img2 = *(IplImage**)(((ObjectPartInstance**)(l+1))+1);
  FeatureOptions *feat = process->Features();
  IplImage *img = feat->GetImage();
  PartLocation l_tmp(part, img->width, img->height);
  int scale, rot, pose;
  float width, height;
  l->GetDetectionLocation(NULL, NULL, &scale, &rot, &pose, &width, &height); 
  int det_x, det_y, pos;
  

  switch(event){
  case CV_EVENT_MOUSEMOVE:
    feat->ImageLocationToDetectionLocation(x, y, 0, 0, &det_x, &det_y);
    l_tmp.SetDetectionLocation(det_x, det_y, LATENT, LATENT, pose, LATENT, LATENT);
    locs = PartLocation::NewPartLocations(process->GetClasses(), img->width, img->height, process->Features(), false);
    locs[part->Id()] = l_tmp;
    partClick->ExtractPartLocations(locs, &l_tmp, NULL);
    if(img2) cvReleaseImage(&img2);
    img2 = cvCloneImage(img);
    process->Draw(img2, locs);
    cvRotatedRect(img2, cvPoint((int)x,(int)y), (int)width, (int)height, 0, CV_RGB(255,0,0));
    delete [] locs;
    cvShowImage("win1", img2);
    *(IplImage**)(((ImageProcess**)(l+1))+1) = img2;
    break;
  case CV_EVENT_LBUTTONDOWN:
  case CV_EVENT_RBUTTONDOWN:
    pos = LATENT;
    if(event == CV_EVENT_RBUTTONDOWN) {
      for(int i = 0; i < partClick->Model()->NumPoses(); i++)
	if(partClick->GetPose(i)->IsNotVisible())
	  pos = i;
    }
    l->SetImageLocation((float)x, (float)y, LATENT, LATENT, partClick->GetPose(pos)->Model()->Name());
    if(img2) cvReleaseImage(&img2);
    img2 = cvCloneImage(img);
    part->Draw(img2, l);
    cvShowImage("win1", img2);
    *(IplImage**)(((ImageProcess**)(l+1))+1) = img2;
    break;
  case CV_EVENT_LBUTTONUP:
  case CV_EVENT_RBUTTONUP:
    break;
  }
}

void *ClickQuestion::GetAnswerFromRealUser(ImageProcess *process) {
  int part_ind = clickPart->Id();
  PartLocation **a = (PartLocation**)malloc(sizeof(PartLocation*)+sizeof(ImageProcess*)+sizeof(IplImage*));
  PartLocation *ans = *a = new PartLocation(process->GetPartInst(part_ind), process->Image()->width, process->Image()->height);
  *(ObjectPartInstance**)(a+1) = process->GetClickPartInst(part_ind);
  *(IplImage**)(((ObjectPartInstance**)(a+1))+1) = NULL;

  /*int pose = -1;
  if(clickPart->NumPoses() > 1 && 0) {
    do {
      pose = -1;
      fprintf(stderr, "Input pose number ");
      for(int i = 0; i < clickPart->NumPoses(); i++) 
	fprintf(stderr, " %s(%d)", clickPart->GetPose(i)->Name(), i);
      fprintf(stderr, ": ");
      if(!fscanf(stdin, "%d", &pose)) return NULL;
    } while(pose < 0 || pose >= clickPart->NumPoses());
  }*/

  // if(a->pose < 0 || !clickPart->GetPose(a->pose)->IsNotVisible()) {
    fprintf(stderr, "%s\n", questionText);
    cvNamedWindow("win1", CV_WINDOW_AUTOSIZE); 
    cvShowImage("win1", process->Features()->GetImage());
    cvSetMouseCallback("win1",cvMouseHandler,a);
    cvWaitKey(0);
    //a->responseTimeSec = 0; // TODO: set this
    //}
	free(a);
  return ans;
}

void *ClickQuestion::GetAnswerFromTestExample(PartLocalizedStructuredLabelWithUserResponses *u) {
  ImageProcess *process = ((PartLocalizedStructuredData*)u->GetData())->GetProcess(classes);
  PartLocation *l = new PartLocation(process->GetClickPartInst(clickPart->Id()), process->Image()->width, process->Image()->height);
  *l = u->GetPartClickLocation(0, clickPart->Id());
  return l;
}

char *ClickQuestion::GetResponseHTML(char *str, void *answer, int q_num, ImageProcess *process, const char *htmlDebugDir) {
  PartLocation *l = (PartLocation*)answer;
  if(process) {
    ObjectPartInstance *part = process->GetPartInst(clickPart->Id());
    IplImage *img = cvCloneImage(process->Features()->GetImage());
    PartLocation tmp(*l); 
    int x, y, scale, rot, pose;
    tmp.GetDetectionLocation(&x, &y, &scale, &rot, &pose);
    if(pose < 0) tmp.SetDetectionLocation(x, y, scale, rot, 0, LATENT, LATENT); 
    part->Draw(img, &tmp, CV_RGB(255,0,0), CV_RGB(255,0,0), CV_RGB(255,0,0), NULL, true, false);
    char fname[1000];
    assert(htmlDebugDir);
    sprintf(fname, "%s/%s_q%d_click.png", htmlDebugDir, process->Features()->Name(), q_num);
    cvSaveImage(fname, img);
    cvReleaseImage(&img);
    sprintf(str, "<center><img src=\"%s\" width=300><h3><font color=\"#000099\">%s</font> <font color=\"#990000\">(%f seconds)</font></h3></center>", 
	    fname+strlen(htmlDebugDir)+1, questionText, l->GetResponseTime());
  } else {
    float x, y;
    l->GetImageLocation(&x, &y);
    sprintf(str, "<h3><font color=\"#000099\">%s</font> <font color=\"#990000\"> (%d, %d)(%f seconds)</font></h3>", questionText, (int)x, (int)y,
	    l->GetResponseTime());
  }
  return str;
}

double ClickQuestion::FinalizeAnswer(void *answer, double *classProbs, QuestionAskingSession *session, int q_num) {
  PartLocation *l = (PartLocation*)answer;
  ImageProcess *process = session->GetProcess();
  ObjectPartInstance *part = process->GetClickPartInst(clickPart->Id());
  
  if(g_debug > 2) {
    float x, y;
    l->GetImageLocation(&x, &y);
    fprintf(stderr, "  answer %s (%d,%d) (%f seconds)\n", questionText, (int)x, (int)y, l->GetResponseTime());
  }

  part->SetClickPoint(l);
  part->GetParent()->PropagateUnaryMapChanges();
  session->SetProbabilitiesChanged();
  
  return l->GetResponseTime();
}


bool ClickQuestion::Load(const Json::Value &root, Classes *classes) {
  num_samples = root.get("numSamples", 0).asInt(); 
  char part_name[1000];
  strcpy(part_name, root.get("part", "").asString().c_str()); 
  if(!(clickPart = classes->FindClickPart(part_name))) {
    fprintf(stderr, "Error loading click question, bad part\n");
    return false;
  }
  clickPart->SetQuestion(this);
  return true;
}

void ClickQuestion::SetPart(ObjectPart *part) { 
   clickPart = part; 
}

Json::Value ClickQuestion::Save() {
  Json::Value root;
  Question::SaveGeneric(root);
  root["numSamples"] = num_samples;
  if(clickPart) root["part"] = clickPart->Name();
  return root;
}



// Given a current distribution of class probabilities p(c|...), compute a new distribution of
// class probabilities p(c|u,...) for every possible value of u (every possible answer to
// this question)
void ClickQuestion::ComputeAnswerProbs(double *classProbs, double *classPriors, QuestionAskingSession *session) {
  int i, j, k, l;
  EntropyBuffers *buff = session->GetEntropyBuffers();
  ImageProcess *process = session->GetProcess();
  PartLocationSampleSet *samples = session->GetSamples();
  double **samples_probs = session->GetSampleClassProbabilities();
  int part_ind = clickPart->Id();
  ObjectPartInstance *part = process->GetClickPartInst(part_ind);
  ObjectPartInstance *p = process->GetPartInst(part_ind);
  float W[1000], f[1000], s, w;
  float gamma = clickPart->GetGamma();//1;//part->Model()->GetParent()->GetGamma();
  int numClasses = classes->NumClasses();
  int tmp[10];
  int *numChildPoses = tmp;

  //if(click_samples)
  //free(click_samples);
  PartLocationSampleSet *click_samples = process->DrawRandomPartLocationSet(num_samples, part, true);

  numAnswers = click_samples->num_samples;
  buff->InitializeAndClear(numAnswers);

  for(i = 0; i < samples->num_samples; i++) {
    int x, y, scale, rot, pose;
    samples->samples[i][part_ind].GetDetectionLocation(&x, &y, &scale, &rot, &pose);
    samples->samples[i][part_ind].SetDetectionLocation(x, y, scale, rot, pose, LATENT, LATENT);
  }

  // Let samples Theta^1...Theta^K be a set of part location samples drawn from the distribution p(Theta|x,u_1...u_T)
  // Let samples \tilde{theta}_p^1...\tilde{theta}_p^M be a set of click point locations for part p drawn from the 
  // distribution p(\tilde{theta}_p|x,u_1...u_T)
  // For each sample Theta^i_p for the current part p, let \tilde{theta}^i_p a random variable for the 
  // corresponding user click location.  The class probability distribution assuming theta^i_p is the true part location is
  //   p(c|theta^i_p,x,u_1...u_T) \approx \integral_{\tilde{theta}^i_p} \sum_{j=1}^K p(\tilde{theta}^i_p|theta^i_p) p(c|theta^j_p,x,u_1...u_T)
  //                                 = \sum_{j=1}^M p(\tilde{theta}^i_p|theta^i_p) * p(c|theta^j_p,x,u_1...u_T)
  //                                 = \sum_{j=1}^M w_ij * p(c|theta^j_p,x,u_1...u_T)
  //   where w_ij \propto exp(-(theta^i_p-\tilde{theta}^j_p)^2*sigma_click_p^-1))
  // Then the expected entropy induced when receiving a part click response \tilde{theta}_p is
  //  H[p(c|x,u_1...u_T)|\tilde{theta}_p] \approx 1/K \sum_{i=1}^K H[p(c|theta^i_p,x,u_1...u_T)]
  // where
  //  H[p(c|theta^i_p,x,u_1...u_T)] = - sum_c p(c|theta^i_p,x,u_1...u_T) * log(p(c|theta^i_p,x,u_1...u_T))
  for(i = 0; i < click_samples->num_samples; i++) {
    for(k = 0; k < numClasses; k++)
      buff->answerUnnormalizedClassProbs[i][k] = 0;
    int click_x, click_y, click_scale, click_rot, click_pose;
    click_samples->root_samples[i].GetDetectionLocation(&click_x, &click_y, &click_scale, &click_rot, &click_pose);
    click_samples->root_samples[i].SetDetectionLocation(click_x, click_y, click_scale, click_rot, click_pose, LATENT, LATENT);
    for(j = 0; j < samples->num_samples; j++) {
      int p_x, p_y, p_scale, p_rot, p_pose;
      samples->samples[j][part_ind].GetDetectionLocation(&p_x, &p_y, &p_scale, &p_rot, &p_pose);
      assert(click_pose >= 0 && click_pose < p->Model()->NumPoses());
      ObjectPoseInstance *pose = part->GetPose(click_pose);
      ObjectPartPoseTransitionInstance ***trans = pose->GetPosePartTransitions(&numChildPoses);
      int n = 0;
      for(k = 0; k < numChildPoses[0]; k++) {
        for(l = 0; l < trans[0][k]->NumChildPoses(); l++) {
          if(trans[0][k]->GetChildPose(l) == p->GetPose(p_pose)) {
            n = trans[0][k]->GetFeatures(f, &click_samples->root_samples[i], &samples->samples[j][part_ind], W);
          }
        }
      }
      if(n) {
	s = 0; 
	for(k = 0; k < n; k++)
	  s += W[k]*f[k];
	w = exp(gamma*s);

	for(k = 0; k < numClasses; k++) 
	  buff->answerUnnormalizedClassProbs[i][k] += w*samples_probs[j][k];
      } 
    }
    for(k = 0; k < numClasses; k++)
      buff->answerUnnormalizedClassProbs[i][k] *= classPriors[k];

    NormalizeProbabilities(buff->answerUnnormalizedClassProbs[i], buff->answerClassProbs[i], classes->NumClasses());
    buff->answerProbs[i] = 1.0f / click_samples->num_samples;
  }

  for(i = 0; i < click_samples->num_samples; i++) 
    delete [] click_samples->samples[i];
  if(click_samples->root_samples) delete [] click_samples->root_samples;
  free(click_samples);
  click_samples = NULL;
}







char *Question::LoadFromString(const char *str, Classes *classes, Question **q) {
  Question *retval = *q = NULL;
  const char *start = strstr(str, "[!question");
  if(!start) 
    return NULL;

  const char *end = strstr(start, "question!]");
  if(!end)
    return NULL;

  char line[10000], tmp[1000];
  char *ptr = (char*)start;
  while(ptr < end && sgets(line, 999, &ptr)) {
    if(sscanf(line, "type %s", tmp)) {
      if(!strcmp(tmp, "binary")) retval = new BinaryAttributeQuestion(classes);
      else if(!strcmp(tmp, "multiple_choice")) retval = new MultipleChoiceAttributeQuestion(classes);
      else if(!strcmp(tmp, "batch")) retval = new BatchQuestion(classes);
      else if(!strcmp(tmp, "part_click")) retval = new ClickQuestion(classes);
      else assert(0);
    } else if(sscanf(line, "id %d", &retval->id)) {
    } else if(!strncmp(line, "text ", 5)) {
      strcpy(retval->questionText, line+5); chomp(retval->questionText);
    } else if(sscanf(line, "expected_time_seconds %lf", &retval->expectedSecondsToAnswer)) {
    } else if(sscanf(line, "visualization %s", tmp)) {
      retval->visualization_image_name = StringCopy(line+strlen("visualization "));
      chomp(retval->visualization_image_name);
    } else {
      if(retval)
        retval->LoadParameterString(line, classes);
    }
  }

  *q = retval;
  return (char*)end;
}

char *Question::ToString(char *str) {
  char params[10000], extra[1000];
  if(visualization_image_name) sprintf(extra, "visualization %s\n", visualization_image_name);
  else strcpy(extra, "");
  sprintf(str, "[!question\ntype %s\nid %d\ntext %s\nexpected_time_seconds %f\n%s%squestion!]\n", 
	  type, id, questionText, expectedSecondsToAnswer, extra, ToParameterString(params));
  return str;
}



bool BinaryAttributeQuestion::LoadParameterString(const char *line, Classes *classes) {
  bool retval = sscanf(line, "attribute: %d", &attribute_ind) > 0;  
  if(retval) {
    attr=classes->GetAttribute(attribute_ind);
    assert((attr) != NULL);
    attr->SetQuestion(this);
  }
  return retval;
}

char *BinaryAttributeQuestion::ToParameterString(char *str) {
  sprintf(str, "attribute: %d\n", attribute_ind);
  return str;
}


bool MultipleChoiceAttributeQuestion::LoadParameterString(const char *line, Classes *classes) {
  char str[10000];
  strcpy(str, line);
  if(!strncmp(str, "attributes: ", 12)) {
    char *p = str+12, *ptr;
    while((ptr=strtok(p, " ")) != NULL) {
      choices = (int*)realloc(choices, (numChoices+1)*sizeof(int));
      sscanf(ptr, "%d", &choices[numChoices]);
      numChoices++;
      p = NULL;
    }
    return true;
  }
  return false;
}

char *MultipleChoiceAttributeQuestion::ToParameterString(char *str) {
  sprintf(str, "attributes:");
  for(int i = 0; i < numChoices; i++) sprintf(str+strlen(str), " %d", choices[i]);
  strcat(str, "\n");
  return str;
}


bool BatchQuestion::LoadParameterString(const char *line, Classes *classes) { 
  if(sscanf(line, "num_samples: %d", &numSamples))
    return true;

  char str[10000];
  strcpy(str, line);

  int qid;
  if(!strncmp(str, "questions: ", 11)) {
    char *p = str+11, *ptr;
    while((ptr=strtok(p, " ")) != NULL) {
      questions = (Question**)realloc(questions, (numQuestions+1)*sizeof(Question));
      int n = sscanf(ptr, "%d", &qid);
	  assert(n && qid < id && qid >= 0);
      questions[numQuestions++] = classes->GetQuestion(qid);
      p = NULL;
    }
    return true;
  }
  return false;
}

char *BatchQuestion::ToParameterString(char *str) {
  sprintf(str, "questions:");
  for(int i = 0; i < numQuestions; i++) 
    sprintf(str+strlen(str), " %d", questions[i]->Id());
  sprintf(str+strlen(str), "\nnum_samples %d\n", numSamples);
  return str;
}

bool ClickQuestion::LoadParameterString(const char *line, Classes *classes) {
  int p;
  int part_ind;
  p=sscanf(line, "part: %d", &part_ind);
  int n = sscanf(line, "num_samples: %d", &num_samples);
  assert(p || n);
  if(p) {
    clickPart = classes->GetClickPart(part_ind);
    assert((clickPart) != NULL);
    clickPart->SetQuestion(this);
  }
  return true;
}

char *ClickQuestion::ToParameterString(char *str) {
  sprintf(str, "part: %d\nnum_samples: %d\n", clickPart->Id(), num_samples);
  return str;
}
