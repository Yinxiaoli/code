/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#ifndef __QUESTION_H
#define __QUESTION_H

#include "opencvUtil.h"
#include "entropy.h"
#include "attribute.h"
#include "part.h"

class ImageProcess;
class PartLocalizedStructuredLabelWithUserResponses;
class Classes;
class Attribute;
class ObjectPart;
class QuestionAskingSession;
struct _PartLocationSampleSet;
struct _AttributeAnswer;

/**
 * @file question.h
 * @brief Routines and structures used for the visual 20 questions game
 */


/**
 * @enum QuestionSelectMethod
 *
 * @brief A method used to select the next question for the 20 questions game
 */
typedef enum {
  QS_INFORMATION_GAIN, 
  QS_TIME_REDUCTION,
  QS_RANDOM
} QuestionSelectMethod;


/**
 * @class Question
 * @brief Class defining a question that can be posed to a user.  
 *
 * This class can be overriden for different types of questions and user interfaces.  
 */
class Question {
 protected:
  int numAnswers; /**< number of possible answers to the question */

  double expectedSecondsToAnswer; /**< expected number of seconds it will take to answer this question */

  char questionText[1000]; /**< A string to display when posing this question to the user */

  const char *type;  /**< string identifier of the question type (should be set by the child class) */

  int id; /**< The id of this question */

  Classes *classes; /**< The Classes object containing part, pose, class, attribute, and question definitions */

  char *visualization_image_name;  /**< filename of a visualization of this question */

 public:
  /**
   * @brief Constructor
   * @param classes Defines classes, parts, attributes, and questions
   */ 
  Question(Classes *classes);

  virtual ~Question();

  /**
   * @brief Get the filename of the image used to visualize this question, if applicable
   */
  const char *GetVisualizationImageName() { return visualization_image_name; }

  /**
   * @brief Set the filename of the image used to visualize this question, if applicable
   */
  void SetVisualizationImage(const char *vis) { visualization_image_name = StringCopy(vis); }

  /**
   * @brief Get a string encoding of the question to ask the user
   * @return A string encoding of the question to ask the user
   */
  const char *GetText() { return questionText; }

  /**
   * @brief Encode generic fields of a question (things that don't depend on the type of question) as a JSON object
   * @param root Store fields into this JSON encoding 
   */ 
  void SaveGeneric(Json::Value &root);

  /**
   * @brief Encode a Question as a JSON object
   * @return A JSON encoding of this question
   */ 
  virtual Json::Value Save() { Json::Value v; SaveGeneric(v); return v; };


  /**
   * @brief Create and initialize a new question, with type and parameters encoded as a string
   * @param root A JSON encoding of the question
   * @param classes Defines all possible questions, parts, classes, etc.
   * @return True if successful
   */
  virtual bool Load(const Json::Value &root, Classes *classes) { return false; };

  /**
   * @brief Create a new question, where the question definition is encoded in a JSON object
   * @param root A JSON encoding of the question
   * @param classes Defines all possible questions, parts, classes, etc.
   * @return The new question
   */
  static Question *New(const Json::Value &root, Classes *classes);


  /**
   * @brief Get the expected time needed to answer this question
   * @return The expected time needed to answer this question
   */
  double ExpectedSecondsToAnswer() { return expectedSecondsToAnswer; }

  /**
   * @brief Set the expected time needed to answer this question
   * @param d The expected time needed to answer this question
   */
  void SetExpectedSecondsToAnswer(double d) { expectedSecondsToAnswer = d; }

  /**
   * @brief Get the id of this question
   * @return the id of this question
   */
  int Id() { return id; }

  /**
   * @brief Set the id of this question
   * @param id the id of this question
   */
  void SetId(int id) { this->id = id; }


  /**
   * @brief Set the text of this question (written text that could be presented to the user to ask this question)
   * @param t the text of this question
   */
  void SetText(const char *t) { strcpy(questionText, t); }

  /********************** Virtual Functions **************************************************************/

  /**
   * @brief Compute the expected entropy that would result from posing this question. 
   *
   * This is a criterion for the "goodness" or utility of asking this question. 
   * Typically, one should override ComputeAnswerProbs()
   * and leave this function untouched (ComputeAnswerProbs is valid for any type of question that has
   * a discrete choice of possible answers or a sampled subset of possible answers)
   *
   * @param classProbs The current class probabilities
   * @param classPriors The current class probabilities without using computer vision (but includes user responses)
   * @param session The QuestionAskingSession which contains buffers used for entropy calculations
   * @return The expected entropy
   */
  virtual double ComputeExpectedEntropy(double *classProbs, double *classPriors, QuestionAskingSession *session);

  /**
   * @brief Create a list of all possible answers, and compute the class probabilities associated
   * with each possible answer.  
   *
   * @param classProbs The current class probabilities before the question is asked
   * @param classPriors The current class probabilities without using computer vision (but includes user responses)
   * @param session The QuestionAskingSession which contains buffers used for probability calculations
   */ 
  virtual void ComputeAnswerProbs(double *classProbs, double *classPriors, QuestionAskingSession *session) {};

  /**
   * @brief Prompt the user to answer this question
   *
   * @param process Contains the image that could be displayed to the user
   * @return The answer provided by the user (usually an AttributeAnswer or PartLocation)
   */ 
  virtual void *GetAnswerFromRealUser(ImageProcess *process) = 0;

  /**
   * @brief Read out a an answer from a testset
   *
   * @param u The testset example containing all responses
   * @return The answer extracted from the test example (usually an AttributeAnswer or PartLocation)
   */ 
  virtual void *GetAnswerFromTestExample(PartLocalizedStructuredLabelWithUserResponses *u) = 0;

  
  /**
   * @brief Finalize an answer received from the user (update any image processing related data structures)
   *
   * @param answer The answer provided by the user (usually an AttributeAnswer or PartLocation)
   * @param classPriors The current class priors, which may be updated by this function
   * @param session The QuestionAskingSession.  The class and part detection probabilities in this structure
   * may be updated by this function
   * @param q_num The current question number of the sesion
   */ 
  virtual double FinalizeAnswer(void *answer, double *classPriors, QuestionAskingSession *session, int q_num) = 0;
  
  /**
   * @brief Generate an HTML string for visualizing the answer to a question
   *
   * @param str The string into which to store the generated HTML
   * @param answer The answer provided by the user (usually an AttributeAnswer or PartLocation)
   * @param q_num The current question number of the sesion
   * @param process Contains the image that could be displayed to the user
   * @param htmlDebugDir A directory into which to store any generated debug images
   */ 
  virtual char *GetResponseHTML(char *str, void *answer, int q_num, ImageProcess *process, const char *htmlDebugDir) { return NULL; }

  /**
   * @brief Resolve links to parts and attributes
   *
   * @param c The Classes definition
   */ 
  virtual void ResolveLinks(Classes *c) {};

  /**
   * @brief Get the response time of a question answer
   *
   * @param answer The answer provided by the user (usually an AttributeAnswer or PartLocation)
   * @return The time in seconds
   */ 
  virtual double GetResponseTime(void *answer) = 0;

  /**
   * @brief Get a string identifier of the question type
   * @return The string identifier of the question type
   */ 
  virtual const char *Type() = 0;

  /**
   * @brief Free the answer to this question, as returned by GetAnswerFromRealUser() or GetAnswerFromTestExample()
   * @param a the answer to this question
   */
  virtual void FreeAnswer(void *a) { free(a); }

 protected:
  /**
   * @brief Ask a certainty question to a real user
   * @return The id of the certainty level provided by the user
   */
  int AskCertaintyQuestion(); 


 public:
  // deprecated
  /// @cond
  static char *LoadFromString(const char *str, Classes *classes, Question **q);
  char *ToString(char *str);
  virtual bool LoadParameterString(const char *line, Classes *classes) = 0;
  virtual char *ToParameterString(char *str) = 0;
  /// @endcond
};




/**
 * @class BinaryAttributeQuestion
 * @brief A yes/no question pertaining to an attribute that can be posed to a human user 
 */
class BinaryAttributeQuestion : public Question {
  int attribute_ind;  // index of the attribute in classes->attributes for a yes/no question
  Attribute *attr;

public:
  /**
   * @brief Constructor
   * @param classes Defines classes, parts, attributes, and questions
   */ 
  BinaryAttributeQuestion(Classes *classes);


  /**
   * @brief Get the attribute associated with this question
   */ 
  Attribute *GetAttribute() { return attr; }

  /**
   * @brief Set the attribute associated with this question
   * @param id the id in classes->attributes of the attribute associated with this question
   */ 
  void SetAttribute(int id);


  void ComputeAnswerProbs(double *classProbs, double *classPriors, QuestionAskingSession *session);
  void *GetAnswerFromRealUser(ImageProcess *process);
  void *GetAnswerFromTestExample(PartLocalizedStructuredLabelWithUserResponses *u);
  double FinalizeAnswer(void *answer, double *classProbs, QuestionAskingSession *session, int q_num);
  Json::Value Save();
  bool Load(const Json::Value &root, Classes *classes);
  char *GetResponseHTML(char *str, void *answer, int q_num, ImageProcess *process, const char *htmlDebugDir);
  double GetResponseTime(void *answer) { return ((struct _AttributeAnswer*)answer)->responseTimeSec; }
  const char *Type() { return "binary"; }


 public:
  // deprecated
  /// @cond
  bool LoadParameterString(const char *line, Classes *classes);
  char *ToParameterString(char *str);
  /// @endcond
};



/**
 * @class MultipleChoiceAttributeQuestion
 * @brief A multiple choice question pertaining to a multi-valued attribute that can be posed to a human user
 */
class MultipleChoiceAttributeQuestion : public Question {
  int *choices;  // array of numChoices indices into classes->attributes for a multiple choice question
  int numChoices;

public:

  /**
   * @brief Constructor
   * @param classes Defines classes, parts, attributes, and questions
   */ 
  MultipleChoiceAttributeQuestion(Classes *classes);

  /**
   * @brief Destructor
   */ 
  ~MultipleChoiceAttributeQuestion();

  /**
   * @brief Get the number of choices 
   */
  int NumChoices() { return numChoices; }

  /**
   * @brief Get the ith attribute choice
   * @param i the index into choices
   * @return an index into classes->attributes
   */
  int GetChoice(int i) { assert(i >= 0 && i < numChoices); return choices[i]; }

  /**
   * @brief Add a new binary attribute to the list of choices
   * @param id an index into classes->attributes defining which attribute to add
   */
  void AddAttribute(int id) { choices = (int*)realloc(choices, sizeof(int)*(numChoices+1)); choices[numChoices++] = id; }

  void ComputeAnswerProbs(double *classProbs, double *classPriors, QuestionAskingSession *session);
  void *GetAnswerFromRealUser(ImageProcess *process);
  void *GetAnswerFromTestExample(PartLocalizedStructuredLabelWithUserResponses *u);
  double FinalizeAnswer(void *answer, double *classProbs, QuestionAskingSession *session, int q_num);
  Json::Value Save();
  bool Load(const Json::Value &root, Classes *classes);
  char *GetResponseHTML(char *str, void *answer, int q_num, ImageProcess *process, const char *htmlDebugDir);
  double GetResponseTime(void *answer) { return ((struct _AttributeAnswer*)answer)->responseTimeSec; }
  const char *Type() { return "multiple_choice"; }
  
 public:
  // deprecated
  /// @cond
  bool LoadParameterString(const char *line, Classes *classes);
  char *ToParameterString(char *str);
  /// @endcond
};

/**
 * @class BatchQuestion
 * @brief A set of questions that can be pooled together and asked at once (presumably because they have something
 * in common that makes answering them all at once more efficient)
 */
class BatchQuestion : public Question {
  Question **questions;
  int numQuestions;
  int numSamples;

public:

  /**
   * @brief Constructor
   * @param classes Defines classes, parts, attributes, and questions
   */
  BatchQuestion(Classes *classes);

  /**
   * @brief Destructor
   */
  ~BatchQuestion();

  /**
   * @brief Add a question to this batch grouping of questions
   * @param q the question to be added
   */
  void AddQuestion(Question *q) { questions = (Question**)realloc(questions, sizeof(Question*)*(numQuestions+1)); questions[numQuestions++] = q; }

  /**
   * @brief Get the number of questions in this batch grouping of questions
   */
  int NumQuestions() { return numQuestions; }

  /**
   * @brief Get the ith question in this batch grouping of questions
   * @param i The index of the question to get
   */
  Question *GetQuestion(int i) { assert(i >= 0 && i < numQuestions); return questions[i]; }

  void ComputeAnswerProbs(double *classProbs, double *classPriors, QuestionAskingSession *session);
  void *GetAnswerFromRealUser(ImageProcess *process);
  void *GetAnswerFromTestExample(PartLocalizedStructuredLabelWithUserResponses *u);
  double FinalizeAnswer(void *answer, double *classProbs, QuestionAskingSession *session, int q_num);
  Json::Value Save();
  bool Load(const Json::Value &root, Classes *classes);
  char *GetResponseHTML(char *str, void *answer, int q_num, ImageProcess *process, const char *htmlDebugDir);
  double GetResponseTime(void *answer) { return ((struct _AttributeAnswer*)answer)->responseTimeSec; }
  const char *Type() { return "batch"; }

 public:
  // deprecated
  /// @cond
  bool LoadParameterString(const char *line, Classes *classes);
  char *ToParameterString(char *str);
  /// @endcond
};




//class BatchAttributeQuestion : public Question {
//};



/**
 * @class ClickQuestion
 *
 * @brief A question asking a human user to click on the location of a part
 */
class ClickQuestion : public Question {
  int num_samples;  // Number of sample part locations used to approximate expected information gain
  ObjectPart *clickPart;
  PartLocationSampleSet *click_samples;
  
public:
  /**
   * @brief Constructor
   * @param classes Defines classes, parts, attributes, and questions
   */ 
  ClickQuestion(Classes *classes);
  ~ClickQuestion();

  /**
   * @brief Get the part associated with this question
   */
  ObjectPart *GetPart() { return clickPart; }

  /**
   * @brief Set the part associated with this question
   * @param part the part associated with this question
   */
  void SetPart(ObjectPart *part);

  /**
   * @brief Set the number of samples used to approximate expected entropy computations
   * @param s the number of samples used to approximate expected entropy computations
   */
  void SetNumSamples(int s) { num_samples = s; }

  /**
   * @brief Get the set of samples used to approximate expected entropy computations
   */
  PartLocationSampleSet *GetClickSamples() { return click_samples; }


  void ComputeAnswerProbs(double *classProbs, double *classPriors, QuestionAskingSession *session);
  void *GetAnswerFromRealUser(ImageProcess *process);
  void *GetAnswerFromTestExample(PartLocalizedStructuredLabelWithUserResponses *u);
  double FinalizeAnswer(void *answer, double *classProbs, QuestionAskingSession *session, int q_num);
  Json::Value Save();
  bool Load(const Json::Value &root, Classes *classes);
  char *GetResponseHTML(char *str, void *answer, int q_num, ImageProcess *process, const char *htmlDebugDir);
  double GetResponseTime(void *answer) { return ((PartLocation*)answer)->GetResponseTime(); }
  const char *Type() { return "part_click"; }
  void FreeAnswer(void *a) { delete (PartLocation*)a; }

 public:
  // deprecated
  /// @cond
  bool LoadParameterString(const char *line, Classes *classes);
  char *ToParameterString(char *str);
  /// @endcond
};


/**
 * @class QuestionAskingSession
 *
 * @brief A user session in which a series of questions can be asked to try to identify the true class of an object
 */
class QuestionAskingSession {
protected:
  Classes *classes; /**< Contains definitions of all classes, parts, and attributes */
  ImageProcess *process; /**< Contains the image we are looking at and all data structures used by computer vision */
  PartLocalizedStructuredLabelWithUserResponses *responses; /**< When evaluating performance on a testset, use this to lookup question answers */

  char *htmlDebugDir;  /**< If non-null, save an html visualization of the questions asked, probability maps, and part location samples */
  
private:
  Question **questionsAsked; /**< The list of questions asked */
  void **questionResponses; /**< The received response to each question */
  int numQuestionsAsked; /**< The number of questions asked */

  bool isInteractive; /**< If true, get responses from a real user.  Otherwise, get responses from a test example */

  double elapsedTime; /**< Duration of this session so far (in seconds) */

  bool *questionAskedMap; /**< A lookup table of size classes->numQuestions specifying which questions have been asked */


  double *computerVisionProbs;
  double *classProbs; /**< The current class probabilities, combining computer vision and the history of responses */
  double *classPriors; /**< Additional term to multiply into class probabilities, which is assumed to be independent from computer vision estimates */

  ClassProb **classProbHistory; /**< A numQuestionsAskedXnumClasses array of sorted class probabilities after each question asked */

  EntropyBuffers *buff; /**< Used to compute information gain for selecting the next question */

  double entropy_to_time_conversion; /**< Constant that converts entropy into expected amount of human time (in seconds) needed to classify an image */
   
  QuestionSelectMethod questionSelectMethod; /**< Method used to select the next question */

  struct _PartLocationSampleSet *samples; /**< A set of random part locations used to estimate class probabilities */
  int num_samples; /**< The number of random part locations to draw in order to estimate class probabilities */
  double **sampleClassProbs; /**< A num_samplesXnumClasses array of class probabilities for each sample */

  bool disableComputerVision; /**< If true, do the 20 questions game without computer vision */

  int debugNumClassPrint;
  bool debugProbabilityMaps, debugClickProbabilityMaps;
  int debugNumSamples;
  bool probabilitiesChanged;  
  bool keepBigImage;  

  double *questionInformationGains, *questionTimeReductions;
  bool debugQuestionEntropies;
  bool debugMaxLikelihoodSolution;

  bool freeResponses;

  int numClickQuestionsAsked;

  char *debugName;



 public:
  /**
   * @brief Constructor
   * 
   * @param p Contains the image we are looking at and all data structures used by computer vision
   * @param responses If non-NULL, when evaluating performance on a testset, use this to lookup question answers
   * @param isInteractive If true, get responses from a real user.  Otherwise, get responses from a test example
   * @param method Method used to select the next question
   * @param debugDir If non-null save an html visualization of the questions asked, probability maps, and part location samples
   * @param debugNumClassPrint Print probabilities for the top debugNumClassPrint classes
   * @param debugProbabilityMaps Save probability maps after each question (can lead to very big files)
   * @param debugClickProbabilityMaps Save probability maps after each question (can lead to very big files)
   * @param debugNumSamples Save sampled part locations maps after each question (can lead to VERY VERY big files)
   * @param debugQuestionEntropies Print entropies for all questions
   * @param debugMaxLikelihoodSolution Draw the current max likelihood solution after every question
   */
  QuestionAskingSession(ImageProcess *p, PartLocalizedStructuredLabelWithUserResponses *responses, bool isInteractive, 
			QuestionSelectMethod method=QS_INFORMATION_GAIN, const char *debugDir=NULL,
			int debugNumClassPrint=10, bool debugProbabilityMaps=true, 
			bool debugClickProbabilityMaps=false, int debugNumSamples=0, 
			bool debugQuestionEntropies=false, bool debugMaxLikelihoodSolution=false, bool keepBigImage=true);

  /**
   * @brief Destructor
   */
  ~QuestionAskingSession();

  /**
   * @brief Run the 20 questions game (ask all questions in a loop)
   * 
   * @param maxQuestions The maximum number of questions to ask
   * @param stopEarly Normally set this to false (corresponds to Method 1 in ECCV'10).  Set this to true if you assume the user will stop the 20 questions game early if the true class is the top ranked class (corresponds to Method 2 in ECCV'10).
   * @return The id of the predicted class
   */
  int AskAllQuestions(int maxQuestions=60, bool stopEarly=false);

  /**
   * @brief Disable certain types of questions
   *
   * @param disableClick If true, disable part click questions
   * @param disableBinary If true, disable binary questions
   * @param disableMultiple If true, disable multiple choice questions
   * @param disableComputerVision If true, disable all use of computer vision (do the 20 questions game without computer vision)
   */
  void DisableQuestions(bool disableClick, bool disableBinary, bool disableMultiple, bool disableComputerVision);
/**
   * @brief Get the duration of this session so far (in seconds)
   * @return The duration of this session so far (in seconds)
   */
  double GetElapsedTime() { return elapsedTime; }

  /**
   * @brief Get the qth question asked
   * @param q The index of the question you want to get
   * @return The qth question asked
   */
  Question *GetQuestionAsked(int q) { return questionsAsked[q]; }

  /**
   * @brief Get the total number of questions asked
   * @return The total number of questions asked
   */
  int GetNumQuestionsAsked() { return numQuestionsAsked; }

  /**
   * @brief Get the class probabilities after asking q questions
   * @param q The index of the question you want to get probabilities for
   * @return A numClasses array of soreted class probabilities
   */
  ClassProb *GetClassProbs(int q) { return classProbHistory[q]; }

  /**
   * @brief Get the qth question response
   * @param q The index of the question response you want to get
   * @return The qth question response (usually an AttributeAnswer or PartLocation)
   */
  void *GetQuestionResponse(int q) { return questionResponses[q]; }

  /**
   * @brief Get the class probabilities associated with each part location sample
   * @return A num_samplesXnumClasses array of class probabilities for each sample
   */ 
  double **GetSampleClassProbabilities() { return sampleClassProbs; }

  /**
   * @brief Get the processing container for the image we are looking at and all data structures used by computer vision
   * @return The processing container
   */
  ImageProcess *GetProcess() { return process; }

  
  /**
   * @brief When a question is finalized, they can signal that the probability maps for parts have been altered (for debugging purposes)
   */
  void SetProbabilitiesChanged() { probabilitiesChanged = true; numClickQuestionsAsked++;}

  /**
   * @brief Proprocess an image (run initial computer vision algorithms)
   */
  void Preprocess();

  /**
   * @brief Submit the answer to a question, updating the class/part probabilities and updating any cache tables
   * @param q The question we are submitting an answer to
   * @param answer The answer to the question (usually an AttributeAnswer or PartLocation)
   */
  void FinalizeAnswer(Question *q, void *answer);

  /**
   * @brief Find the index of the predicted best question to ask next
   */
  int SelectNextQuestion();

  /**
   * @brief Verify whether or not a particular class is the true class
   * @param c Index into classes of the class we want to verify
   * @param verify If true, mark class c as the true class, setting all other class probabilities to 0.  If false,
   *  set the probability of class c to 0
   */ 
  void VerifyClass(int c, bool verify);

  /**
   * @brief Set the base name of the file (not directory) in which debug information will be written
   * @param f the base name
   */
  void SetDebugName(const char *f) { debugName = StringCopy(f); }

  /**
   * @brief Print a short summary of the questions asked in this session to html
   * @param htmlStr a string into which the generated html is written
   * @param numClasses show the top numClasses with highest probability after each question
   */
  void PrintSummary(char *htmlStr, int numClasses);

  void DebugReplaceString(const char *match, const char *replace);

private:
  /**
   * @brief Get the structure storing buffers for computing expected entropy
   * @return The structure storing buffers for computing expected entropy
   */
  EntropyBuffers *GetEntropyBuffers() { return buff; }

  /**
   * @brief Get the set of part location samples most recently used to estimate class probabilities
   * @return The set of part location samples most recently used to estimate class probabilities
   */
  struct _PartLocationSampleSet *GetSamples() { return samples; }

  
  /**
   * @brief Estimate the class probabilities given all user input so far
   * @param classProbs A numClasses array into which we store class probabilities
   * @param redrawSamples If true, randomly sample candidate part locations for approximating probabilities
   */ 
  void ComputeClassProbabilities(double *classProbs, bool redrawSamples=true);

  void UpdateQuestionHistory(int questionInd, void *answer, double timeSpent);
  void PrintDebugInfo(int num);

  friend class Question;
  friend class BinaryAttributeQuestion;
  friend class MultipleChoiceAttributeQuestion;
  friend class BatchQuestion;
  friend class ClickQuestion;
};



#endif
