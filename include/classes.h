/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#ifndef __CLASSES_H__
#define __CLASSES_H__


#include "feature.h"

/**
 * @file classes.h
 * @brief Defines a supercategory and collection of possible classes, parts, pose, 
 * and attributes
 */

class ObjectClass;
class ObjectPart;
class ObjectPose;
class ObjectPartInstance;
class ObjectPoseInstance;
class Attribute;
class FeatureOptions;
class ImageProcess;
class ObjectPartInstanceConstraints;
class ObjectPartPoseTransition;
class Question;
class PartLocation;
class FeatureDictionary;
class FisherFeatureDictionary;

/**
 * @brief Software version number that should be stored into configuration or output files to keep track of changes to file formats as the software version changes
 */
#define API_VERSION "0.0"


/**
 * @enum ScaleOrientationMethod
 *
 * @brief A method used to simplify pictorial structure inference in the presence of multiple scales, orientations, and poses
 */
typedef enum {
  SO_PARENT_CHILD_SAME_SCALE_ORIENTATION=0,  /**< child and parent parts must have same scale/orientation */
  SO_SCALE_ORIENTATION_NO_COST,            /**< child and parent parts can be scaled/rotated independently w/ no cost */
  SO_SCALE_ORIENTATION_ARBITRARY_COST      /**< spatial cost function depends on relative scale/orientation of child and parent parts */
} ScaleOrientationMethod;


/**
 * @enum PartDetectionLossType
 *
 * @brief Different possible loss functions for training part models
 */
typedef enum {
  LOSS_BOUNDING_BOX_AREA_UNION_OVER_INTERSECTION, /**< Set the loss using a VOC bounding box area of intersection divided by area of union criterion, similar to what's in Blaschko&Lampert 2008.  Bounding box is just for the main object's bounding box. */
  LOSS_PART_AVERAGE_AREA_UNION_OVER_INTERSECTION, /**< Area of intersection divided by area of union criterion evaluated for each part and then averaged */
  LOSS_USER_STANDARD_DEVIATIONS, /**< The average number of standard deviations each part is off by, where a standard deviation is computed for each part independently using user click experiments */
  LOSS_NUM_INCORRECT, /**< For each part, a location is correct if it is within NUM_DEVIATIONS_BEFORE_INCORRECT standard deviations */
  LOSS_DETECTION, /**< For a negative image (no object present), the detection score should be < -1, for a positive image, it should be > 1 */
} PartDetectionLossType;

/**
 * @class Classes
 *
 * @brief Represents a supercategory (e.g. "bird") or collection of possible classes, parts, pose, 
 * and attributes.  It also defines different ways of getting user input about those classes, parts,
 * poses, and attributes
 *
 * This includes defintions of:
 *   1) classes (ObjectClass), which are different types of classes (e.g. "Blue Jay")
 *   2) Parts (ObjectPart) (e.g. wing, beak, belly)
 *   3) Poses (ObjectPose), which are different views of a particular part
 *   4) Attributes (Attribute), e.g. (striped, blue, duck-like shape)
 *
 * It also includes definitions of different ways of getting user input, including:
 *   1) Click parts (click ObjectPart), which is a random variable associated with a user's perception of the location of an ObjectPart
 *   2) Click poses (click ObjectPose), which is a random variable associated with a user's perception of the location of an ObjectPose
 *   3) Questions (Question), which is a definition of a possible question we can pose to a user
 *   4) Certainty levels, which is the certainty of a response provided by a user
 *
 * Lastly, it includes some basic definitions of parameters and features used by different detectors
 */
class Classes {

 public:

  /**
   * @brief Constructor
   * @param s Defines an acceleration method for dealing with parts of different scales/orientations
   */
  Classes(ScaleOrientationMethod s=SO_PARENT_CHILD_SAME_SCALE_ORIENTATION);

  /**
   * @brief Destructor
   */
  ~Classes();

  /**
   * @brief Load a Classes object from disk
   * @param fname The file name of the Classes definition on disk
   * @return true on success
   */
  bool Load(const char *fname);

  /**
   * @brief Save a Classes object to disk
   * @param fname The file name of the where to write the Classes definition.  If NULL, use the file name from a previous call to Load()
   * @return true on success
   */
  bool Save(const char *fname);

  /**
   * @brief Load a Classes object from a JSON object
   * @param root A JSON encoding of this Classes object
   * @return true if successful
   */
  bool Load(const Json::Value &root);

  /**
   * @brief Encode the Classes definition as a JSON object
   * @return The JSON encoding of the Classes
   */
  Json::Value Save();

  /**
   * @brief Get the name of the file this Classes structure was loaded from
   * @return The name of the file this Classes structure was loaded from
   */
  const char *GetFileName() { return fname; }

  /**
   * @brief Get the number of possible classes
   * @return The number of possible classes
   */
  int NumClasses() { return numClasses; }

  /**
   * @brief Get the ith ObjectClass
   * @param i The index of the ObjectClass we are searching for
   * @return The ith ObjectClass
   */
  ObjectClass *GetClass(int i) { return classes[i]; }
  
  /**
   * @brief Find the Class with the given name
   * @param name The name of the Class we are searching for
   * @return A pointer to the Class with the given name
   */
  ObjectClass *FindClass(const char *name);

  /**
   * @brief Get the number of possible parts
   * @return The number of possible parts
   */
  int NumParts() { return numParts; }

  /**
   * @brief Get the ith ObjectPart
   * @param i The index of the ObjectPart we are searching for
   * @return The ith ObjectPart
   */
  ObjectPart *GetPart(int i) { return parts[i]; }

  /**
   * @brief Find the ObjectPart with the given id
   * @param name The name of the ObjectPart we are searching for
   * @return A pointer to the ObjectPart with the given name
   */
  ObjectPart *FindPart(const char *name);

  /**
   * @brief Get the number of possible poses
   * @return The number of possible poses
   */
  int NumPoses() { return numPoses; }

  /**
   * @brief Get the ith ObjectPose
   * @param i The index of the ObjectPose we are searching for
   * @return The ith ObjectPose
   */
  ObjectPose *GetPose(int i) { return poses[i]; }

  /**
   * @brief Find the ObjectPose with the given id
   * @param name The name of the ObjectPose we are searching for
   * @return A pointer to the ObjectPose with the given name
   */
  ObjectPose *FindPose(const char *name);

  /**
   * @brief Get the number of possible attributes
   * @return The number of possible attributes
   */
  int NumAttributes() { return numAttributes; }

  /**
   * @brief Get the ith Attribute
   * @param i The index of the Attribute we are searching for
   * @return The ith Attribute
   */
  Attribute *GetAttribute(int i) { return attributes[i]; }

  /**
   * @brief Find the Attribute with the given id
   * @param name The name of the Attribute we are searching for
   * @return A pointer to the Attribute with the given name
   */
  Attribute *FindAttribute(const char *name);

  /**
   * @brief Get the number of possible certainty levels
   * @return The number of possible certainty levels
   */
  int NumCertainties() { return numCertainties; }

  /**
   * @brief Get the name of the ith possible certainty level
   * @param i The index of the certainty we are searching for
   * @return The name of the ith possible certainty level
   */
  const char *GetCertainty(int i) { return certainties[i]; }

  

  /**
   * @brief Get the array of certainty weights, which is used as a multiplicative scale to likelihoods
   *        when encoding the confidence of attribute question responses
   * @return the array of certainty weights (of size numCertainties)
   */
  float *GetCertaintyWeights() { return certaintyWeights; }

  
  /**
   * @brief Set the array of certainty weights, which is used as a multiplicative scale to likelihoods
   *        when encoding the confidence of attribute question responses
   * @param w the array of certainty weights (of size numCertainties)
   */
  void SetCertaintyWeights(float *w) { 
    certaintyWeights = (float*)realloc(certaintyWeights, sizeof(float)*numCertainties); 
    memcpy(certaintyWeights,w,sizeof(float)*numCertainties);  
  }

  /**
   * @brief Find the id of the certainty level with the given name
   * @param name The name of the certainty level we are searching for
   * @return A pointer to the certainty level with the given name
   */
  int FindCertainty(const char *name);


  /**
   * @brief Get parameters describing the number of scales, orientations, and spatial granularities of detectors
   * @return a FeatureParams object describing basic shared parameters of detectors
   */
  FeatureParams *GetFeatureParams() { return &params; }


  /**
   * @brief Get a codebook (dictionary)
   * @param i The index of the codebook 
   * @return The ith codebook
   */
  FeatureDictionary *GetCodebook(int i) { return codebooks[i]; }

  FeatureDictionary *FindCodebook(const char *baseName);

  /**
   * @brief Get a the number of codebooks (dictionaries)
   * @return The number of codebooks
   */
  int NumCodebooks() { return numCodebooks; }

  /**
   * @brief Add a new codebook (dictionary)
   * @param c The codebook to add
   */
  void AddCodebook(FeatureDictionary *c) { 
    codebooks=(FeatureDictionary**)realloc(codebooks, sizeof(FeatureDictionary*)*(numCodebooks+1));
    codebooks[numCodebooks++] = c; 
  }

  
  /**
   * @brief Get a codebook (dictionary) for fisher feature encodings
   * @param i The index of the codebook 
   * @return The ith codebook
   */
  FisherFeatureDictionary *GetFisherCodebook(int i) { return fisherCodebooks[i]; }

  FisherFeatureDictionary *FindFisherCodebook(const char *baseName);
  

  /**
   * @brief Get a the number of codebooks (dictionaries) for fisher feature encodings
   * @return The number of codebooks
   */
  int NumFisherCodebooks() { return numFisherCodebooks; }

  /**
   * @brief Add a new codebook (dictionary) for fisher feature encodings
   * @param c The codebook to add
   */
  void AddFisherCodebook(FisherFeatureDictionary *c) { 
    fisherCodebooks=(FisherFeatureDictionary**)realloc(fisherCodebooks, sizeof(FisherFeatureDictionary*)*(numFisherCodebooks+1));
    fisherCodebooks[numFisherCodebooks++] = c; 
  }

  void RemoveCodebooks();

  int *Subset(int *subset, int num);

  /**
   * @brief Get the number of possible questions
   * @return The number of possible questions
   */
  int NumQuestions() { return numQuestions; }

  /**
   * @brief Get the ith Question
   * @param i The index of the Question we are searching for
   * @return The ith Question
   */
  Question *GetQuestion(int i) { return questions[i]; }


  /**
   * @brief Get the number of possible click parts
   * @return The number of possible click parts
   */
  int NumClickParts() { return numClickParts; }

  /**
   * @brief Get the ith click ObjectPart
   * @param i The index of the click ObjectPart we are searching for
   * @return The ith click ObjectPart
   */
  ObjectPart *GetClickPart(int i) { return clickParts[i]; }

  /**
   * @brief Find the click ObjectPart with the given id
   * @param name The name of the click ObjectPart we are searching for
   * @return A pointer to the click ObjectPart with the given name
   */
  ObjectPart *FindClickPart(const char *name);

  /**
   * @brief Get the number of possible click poses
   * @return The number of possible click poses
   */
  int NumClickPoses() { return numClickPoses; }

  /**
   * @brief Get the ith click ObjectPose
   * @param i The index of the click ObjectPose we are searching for
   * @return The ith ObjectPose
   */
  ObjectPose *GetClickPose(int i) { return clickPoses[i]; }

  /**
   * @brief Find the click ObjectPose with the given id
   * @param name The name of the click ObjectPose we are searching for
   * @return A pointer to the click ObjectPose with the given name
   */
  ObjectPose *FindClickPose(const char *name);

  ObjectPartPoseTransition *GetSpatialTransition(int i) { return spatialTransitions[i]; }
  int NumSpatialTransitions() { return numSpatialTransitions; }

  void OnModelChanged();

  /**
   * @brief Get the absolute scale factor corresponding to the ith scale index
   * @param i The scale index.  When i=0, the scale should be 1. When i>0 the scale is an exponential function of i
   * @return The absolute scale factor corresponding to the ith scale index
   */
  float Scale(int i) { return pow(params.hogParams.subsamplePower, i-params.scaleOffset); }

  /**
   * @brief Get the discrete scale index corresponding to an absolute scale factor
   * @param scale The scale factor
   * @return The scale index
   */
  int ScaleInd(float scale) { return (int)(LOG_B(scale,params.hogParams.subsamplePower)+params.scaleOffset + .5); }

  /**
   * @brief Get the rotation factor in radians corresponding to the ith orientation index
   * @param i The orientation index
   * @return The rotation factor in radians corresponding to the ith orientation index
   */
  float Rotation(int i) { return (float)(i*2*M_PI/params.numOrientations); }

  /**
   * @brief Get the rotation factor in radians corresponding to the ith orientation index
   * @param rot The rotation factor in radians corresponding to the ith orientation index
   * @return The orientation index
   */
  int RotationInd(float rot) { 
    while(rot < 0) rot += (float)(2*M_PI); 
    while(rot >= 2*M_PI) rot -= (float)(2*M_PI); 
    return (int)(rot*params.numOrientations/2/M_PI); 
  }
  

  /**
   * @brief Get the spatial granularity (downsampling of detection score image width compared to the original image width)
   * @return The spatial granularity
   */
  int SpatialGranularity() { return params.spatialGranularity; }

  /**
   * @brief Get the RotationInfo defining an affine transformation that would be applied to an image such that sliding window responses are computable as convolutions
   * @param rot the rotation of the detection image
   * @param scale the scale of the detection image
   * @param width the width of the original image
   * @param height the height of the original image
   */
  RotationInfo GetRotationInfo(int rot, int scale, int width, int height) {
    return ::GetRotationInfo(width, height, Rotation(rot), 1.0f/SpatialGranularity()/Scale(my_max(params.scaleOffset,scale)));
  };


  /**
   * @brief Convert a location in the original image to a location in a detection score map
   *
   * A detection image is an image of detection scores that may be a rotated/scaled version of the original image
   *
   * @param x The x-coordinate in the original image
   * @param y The y-coordinate in the original image
   * @param scale The scale index of the detection score map
   * @param rot The orientation index of the detection score map
   * @param width The image width
   * @param height The image height
   * @param xx A pointer to the x-coordinate in the detection score map (set by this function)
   * @param yy A pointer to the y-coordinate in the detection score map (set by this function)
   */
  void ImageLocationToDetectionLocation(float x, float y, int scale, int rot, int width, int height, int *xx, int *yy); 

  /**
   * @brief Convert a location in a detection score map to a location in the original image
   *
   * A detection image is an image of detection scores that may be a rotated/scaled version of the original image
   *
   * @param x The x-coordinate in the detection score map
   * @param y The y-coordinate in the detection score map
   * @param scale The scale index of the detection score map
   * @param rot The orientation index of the detection score map
   * @param width The image width
   * @param height The image height
   * @param xx A pointer to the x-coordinate in the original image (set by this function)
   * @param yy A pointer to the y-coordinate in the original image (set by this function)
   */
  void DetectionLocationToImageLocation(int x, int y, int scale, int rot, int width, int height, float *xx, float *yy); 
  

  /**
   * @brief Convert detection coordinates from one scale/orientation to another
   *
   * @param x The x-coordinate in the original image
   * @param y The y-coordinate in the original image
   * @param srcScale The scale of the srcImg
   * @param srcRot The orientation of the srcImg
   * @param dstScale The scale of the destination image
   * @param dstRot The orientation of the destionation image
   * @param width The image width
   * @param height The image height
   * @param xx The x-coordinate in the destination image
   * @param yy The y-coordinate in the destination image
   * @return A dynamically allocated image which is a copy of srcImg in the new coordinate system
   */
  void ConvertDetectionCoordinates(float x, float y, int srcScale, int srcRot, int dstScale, int dstRot, int width, int height, float *xx, float *yy);

  /**
   * @brief Return whether or not we should scale attribute detection scores independently
   * for each attribtue when computing class probabilities
   *
   * When attribute detectors were trained independently, this should be true.  When
   * trained jointly, this should be false
   */
  bool ScaleAttributesIndependently() { return scaleAttributesIndependently; }

  /**
   * @brief Get the gamma probability used to convert class detection scores to probabilities
   *
   * log(p(c|x)) is proportional to gamma_class times a class's sum attribute scores
   */
  float GetClassGamma() { return gamma_class; }

  /**
   * @brief Get the scale factor that was applied to click detection scores to tradeoff computer vision and user click information
   */
  float GetClickGamma() { return gamma_click; }

  /**
   * @brief Set the scale factor to be applied to click detection scores to tradeoff computer vision and user click information
   */
  void SetClickGamma(float g) { gamma_click=g; }

  /**
   * @brief Set the gamma probability used to convert class detection scores to probabilities
   *
   * log(p(c|x)) is proportional to gamma_class times a class's sum attribute scores
   * @param gamma_class
   */
  void SetClassGamma(float gamma_class) { this->gamma_class = gamma_class; }

  /**
   * @brief Return true if parts already have the set of allowable pose transitions computed
   */
  bool HasPartPoseTransitions();

  /**
   * @brief Return true if parts already have the set of allowable pose transitions from click location to ground truth location is computed
   */
  bool HasClickPartPoseTransitions();

  /**
   * @brief Get an array of all weights for part and attribute detectors
   * @param w the array in which to store the extracted weights
   * @param getPartFeatures If true, extract part detector weights
   * @param getAttributeFeatures If true, extract attribute detector weights
   * @return The number of extracted weights
   */
  int GetWeights(double *w, bool getPartFeatures, bool getAttributeFeatures);

  /**
   * @brief Get an array of all weights for part and attribute detectors
   * @param w the array in which to store the extracted weights
   * @param getPartFeatures If true, extract part detector weights
   * @param getAttributeFeatures If true, extract attribute detector weights
   * @return The number of extracted weights
   */
  int GetWeights(float *w, bool getPartFeatures, bool getAttributeFeatures);

  /**
   * @brief Set the weights for part and attribute detectors 
   * @param w the array in which to extract the weights.  It should have the same ordering as GetWeights()
   * @param setPartFeatures If true, set part detector weights
   * @param setAttributeFeatures If true, set attribute detector weights
   */
  void SetWeights(double *w, bool setPartFeatures, bool setAttributeFeatures);

  /**
   * @brief Set the weights for part and attribute detectors 
   * @param w the array in which to extract the weights.  It should have the same ordering as GetWeights()
   * @param setPartFeatures If true, set part detector weights
   * @param setAttributeFeatures If true, set attribute detector weights
   */
  void SetWeights(float *w, bool setPartFeatures, bool setAttributeFeatures);

  /**
   * @brief Get the number of weights for part and attribute detectors 
   * @param getPartFeatures If true, set part detector weights
   * @param getAttributeFeatures If true, set attribute detector weights
   * @return The number of weights
   */
  int NumWeights(bool getPartFeatures, bool getAttributeFeatures);


  int GetWeightOffsets(int **pose_offsets, int **spatial_offsets);

  /**
   * @brief Get an upper bound on the squared l2 norm of the feature space
   * @param getPartFeatures If true, set part detector weights
   * @param getAttributeFeatures If true, set attribute detector weights
   */
  float MaxFeatureSumSqr(bool getPartFeatures, bool getAttributeFeatures);

  /**
   * @brief Get model weight constraints defining which model weights are learned, regularized, or constrained to be positive or negative
   * @param wc An array of size NumWeights(getPartFeatures, getAttributeFeatures), where a positive value indicates a weight must be positive and a negative value indicates it must be negative
   * @param learn_weights An array of size NumWeights(getPartFeatures, getAttributeFeatures) indicating which weights are learned
   * @param regularize An array of size NumWeights(getPartFeatures, getAttributeFeatures) indicating which weights are regularized
   * @param getPartFeatures If true, include part detector weights
   * @param getAttributeFeatures If true, include attribute detector weights
   */
  int GetWeightContraints(int *wc, bool *learn_weights, bool *regularize, bool getPartFeatures, bool getAttributeFeatures);
  
  /**
   * @brief Set whether or not we should scale attribute detection scores independently
   * for each attribute when computing class probabilities
   *
   * When attribute detectors were trained independently, this should be true.  When
   * trained jointly, this should be false
   */
  void SetScaleAttributesIndependently(bool s) { scaleAttributesIndependently = s; }

  /**
   * @brief Add a new ObjectClass
   * @param cl The new ObjectClass
   */
  void AddClass(ObjectClass *cl);

  /**
   * @brief Add a new ObjectPart
   * @param part The new ObjectPart
   */
  void AddPart(ObjectPart *part);

  /**
   * @brief Add a new ObjectPose
   * @param pose The new ObjectPose
   */
  void AddPose(ObjectPose *pose);

  /**
   * @brief Add a new Attribute
   * @param attribute The new Attribute
   */
  void AddAttribute(Attribute *attribute);

  /**
   * @brief Add a new click ObjectPart
   * @param part The new click ObjectPart
   */
  void AddClickPart(ObjectPart *part);

  /**
   * @brief Add a new click ObjectPose
   * @param pose The new click ObjectPose
   */
  void AddClickPose(ObjectPose *pose);

  /**
   * @brief Add a new certainty level
   * @param c The new certainty level
   */
  void AddCertainty(const char *c);

  /**
   * @brief Add a new Question
   * @param q The new Question
   */
  void AddQuestion(Question *q);

  /**
   * @brief Add a new ObjectPartPoseTransition
   * @param t The new ObjectPartPoseTransition
   */
  void AddSpatialTransition(ObjectPartPoseTransition *t);


  /**
   * @brief Add a new click ObjectPart (and associated click ObjectPoses) for a corresponding ObjectPart
   * @param part The corresponding ObjectPart
   */
  ObjectPart *CreateClickPart(ObjectPart *part);

  /**
   * @brief Add a new click ObjectPose for a corresponding ObjectPose
   * @param pose The corresponding ObjectPose
   */
  ObjectPose *CreateClickPose(ObjectPose *pose);

  /**
   * @brief Create a click ObjectPart and click ObjectPose for every ObjectPart and ObjectPose
   */
  void CreateClickParts();

  /**
   * @brief Set parameters describing the number of scales, orientations, and spatial granularities of detectors
   * @param f a FeatureParams object describing basic shared parameters of detectors
   */
  void SetFeatureParams(FeatureParams *f) { params=*f; }

  /**
   * @brief Sort all parts such that moving through the list of parts in order corresponds to a depth first traversal of the tree
   */
  void TopologicallySortParts(int *inds=NULL);


  /**
   * @brief Build the list of allowable pose transitions between all parent and child parts
   * @param d A dataset of training examples with labelled parts
   * @param minExamples The minimum numbr of examples in the training set for a child/parent part pose transition to be valid
   * @param computeClickParts If true, learn Guassian parameters for the offset between parts and user click locations
   * @param computeParts If true, learn Guassian parameters for the offset between child and parent part locations
   */
  void AddSpatialTransitions(Dataset *d, int minExamples, bool computeClickParts=true, bool computeParts=false, bool combinePoses=false);

  /**
   * @brief Manually compute the spatial weights between parent and child parts.  Assumes AddSpatialTransitions() has been called
   * prior to this function.  Usually, this is function is only used for computeClickParts=true, as the spatial weights of parts
   * are learned jointly with the appearance parameters
   * @param d A dataset of training examples with labelled parts
   * @param computeClickParts If true, learn Guassian parameters for the offset between parts and user click locations
   * @param computeParts If true, learn Guassian parameters for the offset between child and parent part locations
   */ 
  void ComputePartPoseTransitionWeights(Dataset *d, bool computeClickParts=true, bool computeParts=false);

  /**
   * @brief Get the method defining whether or not we allow child parts to rotate/scale independently from the parent according to some cost
   */
  ScaleOrientationMethod GetScaleOrientationMethod() { return scaleOrientationMethod; }

  /**
   * @brief Load an array defining the location of each part from a JSON encoding
   * @param root a JSON encoding of an array of part locations
   * @param image_width the width of the image
   * @param image_height the height of the image
   * @param isClick if true, the part locations correspond to part click locations by a person (as opposed to ground truth)
   * @return an array of NumParts() PartLocation objects
   */
  PartLocation *LoadPartLocations(const Json::Value &root, int image_width, int image_height, bool isClick);

  /**
   * @brief Save an array defining the location of each part into a JSON encoding
   * @param loc An array of NumParts() PartLocation objects
   * @return a JSON encoding of an array of NumParts() part locations
   */
  Json::Value SavePartLocations(PartLocation *loc);

  /**
   * @brief Load an array defining the response to each attribute question from a JSON encoding
   * @param root a JSON encoding of an array of attribute answers
   * @return an array of NumAttributes() AttributeAnswer objects
   */
  struct _AttributeAnswer *LoadAttributeResponses(const Json::Value &root);

  /**
   * @brief Save an array defining the response to each attribute question into a JSON encoding
   * @param r an array of NumAttributes() AttributeAnswer objects
   * @return a JSON encoding of an array of attribute answers
   */
  Json::Value SaveAttributeResponses(struct _AttributeAnswer *r);

  /**
   * @brief Get the loss method used for training part detectors
   */
  PartDetectionLossType GetDetectionLossMethod() { return detectionLoss; }

  /**
   * @brief Get the loss method used for training part detectors
   * @param m The method used for loss computation
   */
  void SetDetectionLossMethod(PartDetectionLossType m) { detectionLoss=m; }

  /**
   * @brief Get the loss method used for training part detectors from a string encoding
   * @param l a string encoding of the loss method
   */
  static PartDetectionLossType DetectionLossMethodFromString(const char *l);


  /**
   * @brief Convert the loss method used for training part detectors into a string encoding
   * @param m The method used for loss computation
   * @param l a string encoding of the loss method, written by this function
   */
  static void DetectionLossMethodToString(PartDetectionLossType m, char *l);

  /**
   * @brief Returns whether or not our learner supports detecting multiple objects per image
   */
  bool SupportsMultiObjectDetection() { return supportsMultiObjectDetection; }
  
  /**
   * @brief Set whether or not our learner supports detecting multiple objects per image
   * @param b If true, we support multi-object detection
   */
  void SetSupportsMultiObjectDetection(bool b) { supportsMultiObjectDetection = b; }

  bool AllAttributesHaveSaveFeatures() { return allAttributesHaveSaveFeatures; }
  int NumFeatureTypes() { return numFeatureTypes; }
  SiftParams *GetFeatureType(int i) { assert(i >= 0 && i < numFeatureTypes); return features+i; }
  void AddDefaultFeatures();
  void SetImageScale(float s) { imageScale = s; }
  float GetImageScale() { return imageScale; }
  SiftParams *Feature(const char *n);

  FeatureWindow *FeatureWindows() { return featureWindows; }
  int NumFeatureWindows() { return numFeatureWindows; }
  int NumWindowFeatures() { int n = 0; for(int i = 0; i < numFeatureWindows; i++) n += featureWindows[i].dim; return n; }

  void SetFeatureWindows(FeatureWindow *f, int num) {
    numFeatureWindows = 0;
    AddFeatureWindows(f, num);
  }
  void AddFeatureWindows(FeatureWindow *f, int num) {
    featureWindows = (FeatureWindow*)realloc(featureWindows, sizeof(FeatureWindow)*(numFeatureWindows+num));
    memcpy(featureWindows+numFeatureWindows, f, sizeof(FeatureWindow)*num);
    for(int i = 0; i < num; i++) 
      featureWindows[numFeatureWindows+i].name = f[i].name ? StringCopy(f[i].name) : NULL;
    numFeatureWindows += num;
  }

 private:
  /**
   * Resolve all pointers between class, parts, attribtues, etc.
   */
  void ResolveLinks();

  void TopologicallySortParts(ObjectPart *part);

 private:
  FeatureParams params; /**< Defines the number of scales, orientations, and spatial granularity of detectors */
  SiftParams *features;
  int numFeatureTypes;
  float imageScale;

  FILE *binaryWeightFile;
  
  ObjectClass **classes; /**< List of all possible object classes */
  int numClasses; /**< Number of object classes */

  
  Attribute **attributes; /**< List of all attributes */
  int numAttributes; /**< Number of attributes */

  ObjectPart **parts; /**< The set of all possible objects or object parts in this image */
  int numParts; /**< Number of parts */

  ObjectPose **poses; /**< The set of all possible poses */
  int numPoses; /**< Number of poses */

  ObjectPartPoseTransition **spatialTransitions;  /**< the set of all spatial relationships between pairs of parts */
  int numSpatialTransitions;   /**< number of spatial transitions */
  
  ObjectPart **clickParts; /**< Used to store latent or observed variables associated with test time user input clicking a particular part location */
  int numClickParts; /**< Number of click parts */
  ObjectPose **clickPoses; /**< Used to store latent or observed variables associated with test time user input clicking a particular part location */
  int numClickPoses; /**< Number of click poses */

  
  char **certainties; /**< The set of all certainty responses to attribute questions, e.g. "probably", "guessing", "definitely" */
  int numCertainties; /**< Number of certainty levels */
  float *certaintyWeights; 

  Question **questions; /**< The set of possible questions we can pose to the user */
  int numQuestions; /**< Number of questions */

  int *poseOffsets, *spatialOffsets, numOffsets;

  float gamma_class;  /**< log(p(c|x)) is proportional to gamma_class times a class's sum attribute scores */
  float gamma_click;  /**< scale factor that was applied to click detection scores to tradeoff computer vision and user click information */
  bool scaleAttributesIndependently;  /**< if true, then scale each attribute detection score before adding to the class score */

  FeatureDictionary **codebooks;  /**< The set of all possible dictionaries used for bag of words or histogram features */
  int numCodebooks; /**< the number of dictionaries */

  FisherFeatureDictionary **fisherCodebooks;  /**< The set of all possible dictionaries used for fisher encoding features */
  int numFisherCodebooks; /**< the number of dictionaries for fisher encodings */

  char fname[1000];  /**< file this class structure was loaded from */

  ScaleOrientationMethod scaleOrientationMethod; /**< Defines an acceleration method for dealing with parts of different scales/orientations */

  float *attributeMeans, **attributeCovariances, **classAttributeMeans, ***classAttributeCovariances;

  // Temporary data for topological sort of parts
  int currPart;
  ObjectPart **newPartsArray;

  PartDetectionLossType detectionLoss;  /**< Defines the method use for part detection loss */

  bool supportsMultiObjectDetection;  /**< True if we can (learn to) detect multiple objects in the same image */

  FeatureWindow *featureWindows; /**< A set of feature types and window definitions defining how to apply this attribute detector */
  int numFeatureWindows;         /**< The number of different feature types and window definitions */

  bool allAttributesHaveSaveFeatures;

 public:
  // deprecated
  /// @cond
  char *LoadFromString(char *str);
  char *ToString(char *str);
  bool SaveOld(const char *fname);
  bool LoadOld(const char *fname);
  void ResolveLinksOld() ;
  FILE *GetBinaryWeightsFile() { return binaryWeightFile; }
  /// @endcond
  
};


/**
 * @brief Set a variable (e.g., a part location) to this constant to indicate a variable is latent
 */
#define LATENT -1000000

/**
 * @brief Test if a variable (e.g., a part location) is latent
 */
#define IS_LATENT(v) (v==LATENT)

/**
 * @brief Test if a PartLocation is latent
 */ 
#define IS_PART_LATENT(loc) (IS_LATENT(loc->x) && IS_LATENT(loc->y) && IS_LATENT(loc->scale) && IS_LATENT(loc->rotation) && IS_LATENT(loc->pose))

#endif

