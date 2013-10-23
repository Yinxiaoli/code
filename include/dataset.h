/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#ifndef __DATASET_H
#define __DATASET_H

#include "question.h"
#include "attribute.h"
#include "part.h"
#include "imageProcess.h"
#include "structured_svm_partmodel.h"
#include "structured_svm_multi_object.h"

/**
 * @file dataset.h
 * @brief Routines for training and testing on a dataset
 */


class ImageProcess;
class Classes;
class ObjectClass;
class InteractiveLabelingSession;
class UserResponses;
class MultipleUserResponses;
class PartLocalizedStructuredLabelWithUserResponses;
class MultiObjectLabelWithUserResponses;
struct _PartLocation;
struct _AttributeAnswer;

/// @cond
int PartLocationsPoseCmp(const void *v1, const void *v2);
int PartLocationsGTPoseCmp(const void *v1, const void *v2);
int PartLocationsClassCmp(const void *v1, const void *v2);
int PartLocationsScoreCmp(const void *v1, const void *v2);
int PartLocationsAspectRatioCmp(const void *v1, const void *v2);
int ExampleLossCmp(const void *v1, const void *v2);
/// @endcond

typedef int (*DatasetSortFunc)(const void *, const void *);

/**
 * @class Dataset
 * 
 * @brief A dataset of images and their associated annotations 
 *
 * Each example in the dataset can be associated with part localization and class labels as well as user click responses and attribute question answers
 */
class Dataset {
  StructuredDataset *examples;  /**< container object for all examples and their labels */

  Classes *classes; /**< Pointer to a Classes object, which defines the set of possible classes, parts, and attributes */

 public:
  /**
   * @brief Constructor
   * 
   * @param classes If non-null, the classes definition file for the dataset is set to this parameter (instead of loading it from disk)
   */
  Dataset(Classes *classes = NULL) {
    examples = NULL;
    this->classes = classes;
  }

  void SetClasses(Classes *cl) { classes = cl; }

  /**
   * @brief Destructor
   */
  ~Dataset() {
    if(examples) 
      delete examples;
  }

  /**
   * @brief Randomize the order of examples in this dataset
   */
  void Randomize() { if(examples) examples->Randomize(); }

  /**
   * @brief Append a new example to the array of dataset examples
   * @param e The example to be added.  This function does not copy e; it just stores a pointer
   */
  void AddExample(StructuredExample *e) {
    if(!examples) examples = new StructuredDataset();
    examples->AddExample(e);
  }

  /**
   * @brief  Read in a dataset from file
   *
   * @param fname The file name where the list of images and labels for the datset is stored
   */
  bool Load(const char *fname = NULL);
  
  /**
   * @brief  Save a dataset to file
   *
   * @param fname The file name where the list of images and labels for the datset is stored
   */
  bool Save(const char *fname = NULL);

  /**
   * @brief Get the number of examples in the dataset
   *
   * @return The number of examples in the dataset
   */
  int NumExamples() { return examples ? examples->num_examples : 0; }

  /**
   * @brief Get an example from the dataset
   *
   * @param i The index of the example in the dataset
   * @return The ith example in the dataset
   */
  PartLocalizedStructuredData *GetExampleData(int i) { 
    assert(i >= 0 && i < examples->num_examples); 
    return (PartLocalizedStructuredData*)examples->examples[i]->x; 
  }
  StructuredExample *GetExample(int i) { 
    assert(i >= 0 && i < examples->num_examples); 
    return examples->examples[i]; 
  }

  /**
   * @brief Get the label of an example from the dataset
   *
   * @param i The index of the example in the dataset
   * @return The label of the ith example in the dataset
   */
  MultiObjectLabelWithUserResponses *GetExampleLabel(int i) { 
    assert(i >= 0 && i < examples->num_examples); 
    return (MultiObjectLabelWithUserResponses*)examples->examples[i]->y; 
  }

  MultiObjectLabelWithUserResponses *GetExampleLatentLabel(int i) { 
    assert(i >= 0 && i < examples->num_examples); 
    return (MultiObjectLabelWithUserResponses*)examples->examples[i]->y_latent; 
  }

  /**
   * @brief Get the class definition used by the dataset
   *
   * @return the class definition used by the dataset
   */
  Classes *GetClasses() { return classes; }
  
  /**
   * @brief Create a new dataset that copies a subset of examples from this dataset
   * @param ids An array of size NumExamples().  An ith example will be included in the returned subset if ids[i]==id
   * @param id The id of the subset
   *
   * @return the new dataset
   */
  Dataset *ExtractSubset(int *ids, int id, bool inPlace=false);
  Dataset *Copy() { return ExtractSubset(NULL, 0); }


  /**
   * @brief Sort the order of training examples in this dataset
   * @param comparator function used to compare two examples
   *
   * @return the new dataset
   */
  void Sort(DatasetSortFunc comparator=PartLocationsClassCmp);


  /**
   * @brief Assign the orientation of each example to 0
   */
  void AssignZeroOrientations();

  /**
   * @brief Assign the scale of each example using its bounding box
   * @param minWidth The minimum bounding box width, such that all boxes smaller than this have scale 0
   */
  void ApplyScaleConversion(float minWidth);


  /**
   * @brief Cluster part locations into mixture components, where clusterings are based on the relative offset between parent and child parts and we cluster each parent grouping separately
   * @param numPoses the number of mixture components
   * @param nonVisibleCost a constant used to tradeoff spatial distances between parts and a cost for a part being non-visible
   * @return A classes->NumParts() X numPoses X classes->NumParts()*3 array of cluster centers
   */
  float **ClusterPosesByOffsets(int par, int numPoses, float nonVisibleCost, bool useMirroredPoses);


  /**
   * @brief Convert the part locations for each part into a Euclidean pose-normalized feature space
   * @param nonVisibleCost a constant used to tradeoff spatial distances between parts and a cost for a part being non-visible
   * @return A classes->NumParts() X NumExamples() X classes->NumParts()*3 array, where for each example and each parent part, we compute a parent->NumParts()*3 vector of part of parent-child part offsets x,y,visible.
   */
  float **ComputeGroupOffsets(int par, float nonVisibleCost, bool useMirroredPoses);
  void ComputeExampleGroupOffsets(PartLocation *locs, int par, float *group_pts, float *group_pts_mirrored, 
				  float nonVisibleCost, int w, int h);

  /**
   * @brief Assign poses of all parts in each example to the pose of nearest cluster center
   * @param numPoses the number of mixture components
   * @param nonVisibleCost a constant used to tradeoff spatial distances between parts and a cost for a part being non-visible
   * @param centers A classes->NumParts() X numPoses X classes->NumParts()*3 array of cluster centers returned by ClusterPoses()
   */
  void AssignPosesByOffsets(int par, float **centers, int numPoses, float nonVisibleCost, bool assignChildren);

  float ***ClusterPosesBySegmentationMasks(int numPoses, int part_width, int part_height, bool useMirroredPoses, const char *maskDir = NULL);
  void AssignPosesBySegmentationMasks(float ***centers, int numPoses, int part_width, int part_height, bool assignParentParts=true);
  float ***ComputeSegmentationMasks(int part_width, int part_height, bool useMirroredPoses, int *num);

  /**
   * @brief Learn probability distributions for how users answer attribute questions
   * 
   * Let \f$ u_a \f$  be random a variable of how people answer an attribute questions with certainty \f$ r_a \f$.  
   * We learn binomial distributions distributions \f$ p(u_a,r_a|c) \f$  with beta priors
   *
   * @param beta_class A beta prior with the probability distribution of \f$ p(u_a|c) \f$
   * @param beta_cert  An array of beta priors for each possible certainty value with the probability distribution of \f$ p(u_a,r_a) \f$
   */
  void LearnClassAttributeUserProbabilities(float beta_class, float *beta_cert);

  /**
   * @brief Learn probability distributions for how users answer part click questions
   * 
   * Let \f$ u_p \f$  be random a variable of where a person clicks for part p, with ground truth 
   * part location \f$ \theta_p \f$
   * We want to learn a probability distributions as a Gaussian distribution
   * \f[ u_p-\theta_p \sim \mathcal{N}(\mu_p^{click},\Sigma_p^{click}) \f] 
   */
  void LearnUserClickProbabilities();

  /**
   * @brief Use cross-validation to learn parameters that transform attribute and class detection scores to probabilities
   * 
   * Parameters are learned to maximize the log-likelihood of the validation set.
   * Each attribute a has a probability according to a sigmoid function \f[ p(a|x) \propto \exp\{\gamma_a f_a(x)\} \f] 
   * We learn \f$ \gamma_a \f$ given that we already have learned a function \f$ f_a(x) \f$ to compute attribute detection scores
   * (see TrainAttributes()) 
   *
   * For each class c, we learn a parameter \f$ \gamma_c \f$, where each class has a probability 
   * \f[ p(c|x) \propto \exp\{\gamma_c \sum_{a \in \mathcal{A}_c} f_a(x)\} \f] where \f$ \mathcal{A}_c \f$ is a set of attributes
   * for c.  We assume already trained attribute detectors \f$ f_a(x) \f$ jointly to minimize misclassification of classes
   *
   * @param numEx Limit the number of examples used for training to numEx (useful if the dataset has more examples than we need)
   */
  void LearnClassAttributeDetectionParameters(int numEx = 1000000);

  /**
   * @brief Use cross-validation to learn parameters that transform part detection scores to probabilities
   * 
   * Parameters \f$ \gamma_p \f$ are learned for each part to maximize the log-likelihood of the validation set.
   * Each part p has a probability according to an exponential function \f[ p(\theta_p|x) \propto \exp\{\gamma_p f_p(\theta_p,x)\} \f]
   * We assume we have already learned a function \f$ f_p(\theta_p,x) \f$ (see TrainDetectors())
   * 
   * Secondly, we learn parameters \f$ \gamma_p^{click} \f$ to put user click probabilities onto the same scale as detection
   * scores.  \f[ p(\theta_p|x,\theta_p^{click}) \propto \exp\{\gamma_p f_p(\theta_p,x) + \gamma_p^{click} \log(p(\theta_p|u_p)) \} \f]
   * where we assume \f$ p(\theta_p|u_p) \f$ has already been learn (see LearnUserClickProbabilities())
   *
   * @param numEx Limit the number of examples used for training to numEx (useful if the dataset has more examples than we need)
   */
  void LearnPartDetectionParameters(int numEx);


  /**
   * @brief Learn a codebook using k-means
   * 
   * @param dictionaryOutFile The file name to output the learned dictionary
   * @param featName A string identifier of the base feature, e.g. "HOG", "RGB", or "CIE"
   * @param w The width of the template for extracting a feature descriptor.  For example, for SIFT w=h=4, baseFeature="HOG"
   * @param h The height of the template for extracting a feature descriptor.  
   * @param numWords The number of words in the dictionary
   * @param maxImages The maximum number of images to use to construct the training set 
   * @param ptsPerImage The number of interest points to extract from each image
   * @param tree_depth If non-zero, use hierarchical k-means with this depth instead of regular k-means
   */
  void BuildCodebook(const char *dictionaryOutFile, const char *featName, 
		     int w, int h, int numWords, int maxImages, int ptsPerImage, int tree_depth=0, int resize_image_width=0);

  /**
   * @brief Learn a codebook for fisher feature encodings by learning a gaussian mixture model (EM)
   * 
   * @param dictionaryOutFile The file name to output the learned dictionary
   * @param featName A string identifier of the base feature, e.g. "HOG", "RGB", or "CIE"
   * @param w The width of the template for extracting a feature descriptor.  For example, for SIFT w=h=4, baseFeature="HOG"
   * @param h The height of the template for extracting a feature descriptor.  
   * @param numWords The number of words in the dictionary
   * @param maxImages The maximum number of images to use to construct the training set 
   * @param ptsPerImage The number of interest points to extract from each image
   * @param pcaDims Reduced descriptor dimensionality for the fisher encoding
   */
  void BuildFisherCodebook(const char *dictionaryOutFile, const char *featName, 
		     int w, int h, int numWords, int maxImages, int ptsPerImage, int pcaDims=64, int resize_image_width=0);

  /**
   * @brief Run the 20 questions game with part clicks on a testset
   * 
   * @param maxQuestions The maximum number of questions to ask.  If you want to measure results using human labor (in seconds), set maxQuestions=maxTime/timeInterval
   * @param timeInterval Set this to 0 if you want to measure results by the number of questions asked. If you want to measure results using human labor (in seconds), set this to some time interval (e.g. 1 second), such that classification accuracy will be plotted for time values spaced at that interval
   * @param isInteractive Normally set to false. If true, queries the question answers from the user instead of reading them from the test set
   * @param stopEarly Normally set this to false (corresponds to Method 1 in ECCV'10).  Set this to true if you assume the user will stop the 20 questions game early if the true class is the top ranked class (corresponds to Method 2 in ECCV'10).
   * @param isCorrectWindow Set this to 1 to measure regular classification accuracy.  Set to some value isCorrectWindow>1 to assume success if the true class is in the top isCorrectWindow ranked classes
   * @param method The method used to select the next question to ask
   * @param disableClick Set to true to disable asking click questions
   * @param disableBinary Set to true to disable asking binary questions
   * @param disableMultiple Set to true to disable asking multiple choice questions
   * @param disableComputerVision Set to true to disable using computer vision
   * @param accuracy An array of size maxQuestions.  If non-null,  the classification accuracy for a given number of questions is outputed into this array
   * @param perQuestionConfusionMatrix A 3D array of size maxQuestionsXnumClassesXnumclasses.  If non-null,  the confusion matrix for a given number of questions is outputed into this array
   * @param perQuestionPredictions A 2D array of size numExamplesXmaxQuestions.  If non-null, the indices of the top ranked class for each example and each question number outputed into this array
   * @param perQuestionClassProbabilities A 3D array of size numExamplesXmaxQuestionsXnumClasses.  If non-null, the probability of each class for every example and every number of questions asked is outputed into this array
   * @param questionsAsked A 2D array of size numExamplesXmaxQuestions.  If non-null, the indices of each question asked for each example is outputed into this array
   * @param debugDir If non-null, generate an HTML visualization for each test image of the questions asked, and the evolution of class probabilities and
   *  part location probability maps as the user answers questions
   * @param debugNumClassPrint Print probabilities for the top debugNumClassPrint classes
   * @param debugProbabilityMaps Save probability maps after each question (can lead to very big files)
   * @param debugClickProbabilityMaps Save probability maps after each question (can lead to very big files)
   * @param debugNumSamples Save sampled part locations maps after each question (can lead to VERY VERY big files)
   * @param debugQuestionEntropies Print entropies for all questions
   * @param debugMaxLikelihoodSolution Draw the current max likelihood solution after every question
   * @param matlabProgressOut If non-null, periodically saves progress of all results to matlab file (for debugging while experiment is in progress)
   * @param responseTimes a numImagesXnumQuestions array of times in seconds needed to answer each question, set by this function
   * @param numQuestionsAsked a numImages array storing the number of questions that were asked to correctly identify the true class
   */
  void EvaluateTestset20Q(int maxQuestions, double timeInterval, bool isInteractive, bool stopEarly, int isCorrectWindow, 
			  QuestionSelectMethod method, double *accuracy, double ***perQuestionConfusionMatrix,
			  int **perQuestionPredictions, double ***perQuestionClassProbabilities, 
			  int **questionsAsked, int **responseTimes, int *numQuestionsAsked, int *gtClasses, 
			  bool disableClick=false, bool disableBinary=false, 
			  bool disableMultiple=false, bool disableComputerVision=false, 
                          bool disableCertainty=false, const char *debugDir = NULL,
			  int debugNumClassPrint=10, bool debugProbabilityMaps=false, 
			  bool debugClickProbabilityMaps=false, int debugNumSamples=0, bool debugQuestionEntropies=false,
			  bool debugMaxLikelihoodSolution=false, const char *matlabProgressOut=NULL);

  /**
   * @brief Run part detectors to predict the location of each part.  The predicted part locations are stored
   * into the part locations field of each dataset example (over-writing any existing values)
   * @param evaluatePartDetection If true, runs part detector on image
   * @param evaluateClassification If true, evaluates multi-class classifier.  If evaluatePartDetection is also on, it evaluates the classifiers
   * on the predicted location.  Otherwise it does it on the ground truth
   * @param imagesDirOut If non-null, store an image visualization of the predicted bounding box of each part for each image
   * @param predictedClasses A numImages array that will store indices of the predicted maximum likelihood classes
   * @param trueClasses A numImages array that will store indices of the ground truth classes
   * @param classScores A numImagesXnumClasses array that will store classification scores of each class
   * @param localizationLoss A numImages array that will store the part localization loss (predicted vs. groundtruth 
   *    bounding box area of intersection divided by area of union, summed over each part)
   * @param predictedLocations A numImagesXnumPartsX5 array of predicted locations for each part, where each entry
   *    encodes the (x,y,scale,orientation,pose) of the ith part
   * @param trueLocations A numImagesXnumPartsX5 array of ground truth locations for each part, where each entry
   *    encodes the (x,y,scale,orientation,pose) of the ith part
   * @param predictedLocationScores A numImages array that will store the part localization score of the predicted location
   * @param matlabProgressOut If non-null, periodically saves progress of all results to matlab file (for debugging while experiment is in progress)
   * @param localizationLossComparedToUsers If non-null, extract a numImagesXnumParts array storing the loss associated with each predicted part location, as compared to MTurk responses
   * 
   * @return The average loss
   * 
   */
  float EvaluateTestset(bool evaluatePartDetection, bool evaluateClassification, const char *imagesDirOut = NULL, 
			int *predictedClasses=NULL, int *trueClasses=NULL, double **classScores=NULL, 
			double *localizationLoss=NULL, int ***predictedLocations=NULL, int ***trueLocations=NULL, 
			double *predictedLocationScores=NULL, const char *matlabProgressOut=NULL, 
			double **localizationLossComparedToUsers = NULL);
  

  /**
   * @brief Interactively label a testset, using the current part detector to speedup annotation
   * @param maxDrag The maximum number of parts a user can move
   * @param stopThresh When simulating users, the user is assumed to accept a part location when it is within stopThresh 
   *        standard deviations of their labeled location (standard deviation as measured from a validation set)
   * @param debugDir If non-null, generate an HTML visualization for each test image.  The html will be stored to 
   *                 debugDir/index.html, with lots of images in debugDir
   * @param debugImages If true, display a visualization of the sequence of corrected parts
   * @param debugProbabilityMaps If true, display the evolution of part location probability maps as the user answers questions.
   * @param aveLoss a maxDrag array into which the average loss (number of parts outside stopThresh) is stored
   * @param partLoss a numExamplesXmaxDrag array into which the loss (number of parts outside stopThresh) is stored for each example
   * @param perDraggedPredictions a numExamplesXnumPartsX5 array into which the predicted location (x,y,scale,
   * @param partsDragged a numExamplesXmaxDrag array into which the sequence of corrected parts for each image is stored
   * @param dragTimes a numExamplesXmaxDrag array into which the time to correct each part for each image is stored
   * @param matlabProgressOut If non-null, periodically saves progress of all results to matlab file (for debugging while experiment is in progress)
   *
   */
  void EvaluateTestsetInteractive(int maxDrag, float stopThresh, const char *debugDir=NULL, bool debugImages=false,
				  bool debugProbabilityMaps=false, 
				  double *aveLoss=NULL, double **partLoss=NULL, int ***perDraggedPredictions=NULL, 
				  int **partsDragged=NULL, double **dragTimes=NULL, const char *matlabProgressOut=NULL);


  void LearnPartMinimumSpanningTree(float nonVisbleCost);

  void BuildConfusionMatrices(const char *debugDir, const char *fname, int **perQuestionPred, int **perQuestionGT, 
				     int num_conf, const char *title, const char *header, const char **imageNames, const char **linkNames);
  void BuildGalleries(const char *dir, const char *fname, const char *title, const char *header, DatasetSortFunc *sort_funcs, const char **labels, int num_sort);

  void BuildCroppedImages(const char *srcString, const char *dstString);

  // deprecated
  /// @cond
  bool LoadOld(const char *fname, bool bidirectional, bool computeClickParts);
  bool SaveOld(const char *fname);
  /// @endcond
};






/**
 * @class UserResponses
 *
 * @brief One particular user's responses to a set of questions about an object.  In particular, answers to questions
 * about whether or not an object has an attribute, and answers to questions about where parts are located
 */
class UserResponses {
  PartLocation *partClickLocations;            /**< A numPartClickUsersXnumParts array of all part click point locations */
  struct _AttributeAnswer *attributeResponses;  /**< A numAttributes array of answers to all attribute questions */
  PartLocalizedStructuredData *x;

 public:
  /**
   * @brief Create a new UserResponses object
   * @param x the data example this set of user responses is associated with
   */
  UserResponses(StructuredData *x) {
    partClickLocations = NULL;
    attributeResponses = NULL;
    this->x = (PartLocalizedStructuredData*)x;
  }

  ~UserResponses() { 
    if(partClickLocations) delete [] partClickLocations;
    if(attributeResponses) free(attributeResponses);
    partClickLocations = NULL;
    attributeResponses = NULL;
  }

  /**
   * @brief Load a UserResponse from a JSON encoding
   * @param root JSON encoding of this set of user responses
   * @param c object defining the definition of all parts, classes, attributes, and questions
   */
  bool load(const Json::Value &root, Classes *c) {
    if(partClickLocations) free(partClickLocations);
    if(attributeResponses) free(attributeResponses);
    partClickLocations = NULL;
    attributeResponses = NULL;
    if(root.isMember("part_locations") && !(partClickLocations=c->LoadPartLocations(root["part_locations"], x->Width(), x->Width(), true)))
      return false;
    if(root.isMember("attribute_responses") && !(attributeResponses=c->LoadAttributeResponses(root["attribute_responses"])))
      return false;
    return true;
  }

  /**
   * @brief Save a UserResponse to a JSON encoding
   * @param c object defining the definition of all parts, classes, attributes, and questions
   * @return JSON encoding of this set of user responses
   */
  Json::Value save(Classes *c) {
    Json::Value root;
    if(partClickLocations) root["part_locations"] = c->SavePartLocations(partClickLocations);
    if(attributeResponses) root["attribute_responses"] = c->SaveAttributeResponses(attributeResponses);
    return root;
  }

  /**
   * @brief Set the array of part click repsonses
   * @param locs the array of part click repsonses
   */
  void SetPartLocations(PartLocation *locs) {
    partClickLocations = locs;
  }

  friend class MultipleUserResponses;
};

/**
 * @class MultipleUserResponses
 *
 * @brief A set of one or more users responses to a set of questions about an object.  

 * In particular, answers to questions about whether or not an object has an attribute, and answers to 
 * questions about where parts are located.  These answers could be different from user to user to differences
 * in human perception or human error
 */
class MultipleUserResponses {
  UserResponses **users;    /**< An numUsers array of UserResponses, where each UserResponses stores one person's answers to a set of questions */
  int numUsers;            /**< The number of users */

  friend class Dataset;
public:
  MultipleUserResponses() { users = NULL; numUsers = 0; }

  ~MultipleUserResponses() {
    for(int i = 0; i < numUsers; i++) 
      delete users[i];
    free(users);
  }

  /**
   * @brief Add a new user to this set of user responses
   * @param u the new user
   */
  void AddUser(UserResponses *u) {
    users = (UserResponses**)realloc(users, sizeof(UserResponses*)*(numUsers+1));
    users[numUsers++] = u;
  }
  
  /**
   * @brief Load a MultipleUserResponses from a JSON encoding
   * @param root JSON encoding of this set of user responses
   * @param c object defining the definition of all parts, classes, attributes, and questions
   * @param x the example this set of user responses is associated with
   */
  bool load(const Json::Value &root, Classes *c, StructuredData *x) {
    if(!root.isArray()) return false;
    numUsers = 0;
    for(int i = 0; i < (int)root.size(); i++) {
      AddUser(new UserResponses(x));
      if(!users[i]->load(root[i], c))
	return false;
    }
    return true;
  }

  /**
   * @brief Save a MultipleUserResponses to a JSON encoding
   * @param c object defining the definition of all parts, classes, attributes, and questions
   */
  Json::Value save(Classes *c) {
    Json::Value root;
    for(int i = 0; i < numUsers; i++)
      root[i] = users[i]->save(c);
    return root;
  }


  /**
   * @brief Get the number of users
   */
  int NumUsers() { return numUsers; }

  /**
   * @brief Get the answers to all attribute questions
   * @param user the index of the user to get question answers for
   * @return An array of classes->numAttributes attribute question answers
   */
  struct _AttributeAnswer *GetAttributes(int user) { return users[user]->attributeResponses; }

  /**
   * @brief Set the answers to all attribute questions
   * @param user the index of the user that is answering all questions
   * @param a An array of classes->numAttributes attribute question answers
   */
  void SetAttributes(int user, struct _AttributeAnswer *a) { 
    users[user]->attributeResponses = a; 
  }

  /**
   * @brief Get the answer to the ith attribute question
   * @param user the index of the user to get question answers for
   * @param i The index into classes->attributes that we want to get
   * @return The answer to the ith attribute question
   */
  struct _AttributeAnswer GetAttribute(int user, int i) { return users[user]->attributeResponses[i]; }

  /**
   * @brief Get the clicked locations of all parts in the image. 
   * @param user the index of the user to get part click responses for
   * @return An array of classes->numParts part locations
   */
  PartLocation *GetPartClickLocations(int user) { 
    return users[user]->partClickLocations; 
  }

  /**
   * @brief Set the clicked locations of all parts in the image. 
   * @param user the index of the user to set part click responses for
   * @param locs An array of classes->numParts part locations
   */
  void SetPartClickLocations(int user, PartLocation *locs) { 
    users[user]->partClickLocations = locs; 
  }

  /**
   * @brief Get the clicked location of the ith part in the image.  While GetPartLocation() gets ground truth locations, GetPartClickLocation() gets user supplied click locations
   * @param user the index of the user to get question answers for
   * @param i The index into classes->clickParts that we want to get
   * @return The location of the ith click part in the image
   */
  PartLocation GetPartClickLocation(int user, int i) { 
    return users[user]->partClickLocations[i]; 
  }
};



/**
 * @class PartLocalizedStructuredLabelWithUserResponses
 *
 * @brief For a given object, stores the ground truth class and part locations of an object and
 * a set of one or more users responses to a set of questions about an object.  
 */
class PartLocalizedStructuredLabelWithUserResponses : public PartLocalizedStructuredLabel, public MultipleUserResponses {
 public:


 /**
  * @brief Create a new PartLocalizedStructuredLabelWithUserResponses 
  * @param x the PartLocalizedStructuredData associated with this label
  */
  PartLocalizedStructuredLabelWithUserResponses(StructuredData *x) : PartLocalizedStructuredLabel(x) {}

  /**
   * @brief Load a PartLocalizedStructuredLabelWithUserResponses from a JSON encoding
   * @param root JSON encoding of this set of user responses
   * @param s pointer to an object for the learner
   */
  bool load(const Json::Value &root, StructuredSVM *s) { 
    bool retval = PartLocalizedStructuredLabel::load(root, s) && 
      (!root.isMember("users") || MultipleUserResponses::load(root["users"], classes, x)); 
    if(!root.isMember("users") && root.isMember("objects") && 
       !MultipleUserResponses::load(root["objects"][Json::UInt(0)]["users"], classes, x)) 
      return false;
    return retval;
  }

  /**
   * @brief Save a PartLocalizedStructuredLabelWithUserResponses into a JSON encoding
   * @param s pointer to an object for the learner
   * @return JSON encoding of this set of user responses
   */
  Json::Value save(StructuredSVM *s) { 
    Json::Value root = PartLocalizedStructuredLabel::save(s);
    /*if(!s->GetCompactLabels())*/ root["users"] = MultipleUserResponses::save(classes);
    return root;
  }
};

/**
 * @class MultiObjectLabelWithUserResponses
 * @brief Stores a label y for a training example for a deformable part model.  This is a
 * list of 
 */
class MultiObjectLabelWithUserResponses : public MultiObjectLabel {
 public:
 /**
  * @brief Create a new PartLocalizedStructuredLabelWithUserResponses 
  * @param x the PartLocalizedStructuredData associated with this label
  */
  MultiObjectLabelWithUserResponses(StructuredData *x) : MultiObjectLabel(x) {}

  /**
   * @brief Create a new PartLocalizedStructuredLabel
   */
  PartLocalizedStructuredLabel *NewObject() { return new PartLocalizedStructuredLabelWithUserResponses(x); }
  
  /**
   * @brief Get the ith object in this image
   * @param i The index of the object to get
   */
  PartLocalizedStructuredLabelWithUserResponses *GetObject(int i) { 
    return (PartLocalizedStructuredLabelWithUserResponses*)MultiObjectLabel::GetObject(i); 
  }
};


/// @cond
void SaveMatlab20Q(const char *matfileOut, int maxQuestions, int numClasses, int numExamples, 
		   double *accuracy, double ***perQuestionConfusionMatrix, int **perQuestionPredictions, 
		   double ***perQuestionClassProbabilities, int **questionsAsked, int **responseTimes, int *numQuestionsAsked, int *gtClasses);
void SaveMatlabTest(const char *matfileOut, int numClasses, int numExamples, int numParts,
		    int *predictedClasses, int *trueClasses, double **classScores, double *localizationLoss,
		    int ***predictedLocations, int ***trueLocations, double *predictedLocationScores,
		    double **localizationLossComparedToUsers);
void SaveMatlabInteractive(const char *matfileOut, int maxDrag, int numExamples, double *aveLoss, double **partLoss, 
			   int ***perDraggedPredictions, int **partsDragged, double **dragTimes);
/// @endcond

#endif
