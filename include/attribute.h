/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#ifndef __ATTRIBUTE_H
#define __ATTRIBUTE_H


#include "feature.h"

/**
 * @file attribute.h
 * @brief Definition of attribute models and routines for detecting attributes in images
 */

class ImageProcess;
class ObjectPart;
class ObjectPartInstance;
class Classes;
class Question;
class PartLocation;

/**
 * @struct _AttributeAnswer
 *
 * @brief An answer to a binary or multiple choice attribute question
 */
typedef struct _AttributeAnswer {
  int answer;    /**< 0/1 for yes/no question or attribute index for multiple choice */
  int certainty; /**< index for certainty level */
  float responseTimeSec; /**< time in seconds that it took to answer a question */
} AttributeAnswer;

/**
 * @class Attribute
 * 
 * @brief A binary attribute (e.g. blue belly, cone-shaped beak).  An attribute can be associated with a detector on low-level visual features
 *
 */
class Attribute {
  char *name; /**< The name of this attribute */
  int id; /**< The id of this attribute */

  char *visualization_image;  /**< image name of visualization */
  char *value_name;  /**< e.g., "blue" for blue belly */
  char *property_name; /**< e.g., "belly color" for blue belly */

  FeatureWindow *features; /**< A set of feature types and window definitions defining how to apply this attribute detector */
  int numFeatureTypes; /**< The number of different feature types and window definitions */

  float *weights;  /**< A set of weights used by the detector */

  ObjectPart *part; /**< Non-null iff this attribute is an attribute of a particular part */
  char *part_name; /**< Part name if this is an attribute of a particular part */

  float gamma;  /**< log(p(a|x)) is proportional to gamma times the detection score */

  Question *binaryQuestion;  /**< Question posable to a human user pertaining to this Attribute */

  Classes *classes; /**< The Classes definition, defining all classes, parts, poses, and attributes */

public:

  /**
   * @brief Constructor
   * 
   * @param name If non-null, the name of this attribute
   */
  Attribute(const char *name=NULL);


  /**
   * @brief Destructor
   */
  ~Attribute();

  /**
   * @brief Encode the definition of this attribute as a JSON object
   * 
   * @return A JSON encoding of the attirbute definition
   */
  Json::Value Save();

  /**
   * @brief Load definition of this attribtue from a JSON object
   * 
   * @param root A JSON encoding of this class
   * @return True if successful
   */
  bool Load(const Json::Value &root);

  /**
   * @brief Get the id of this attribute
   *
   * @return The id of this attribute
   */
  int Id() { return id; }


  /**
   * @brief Get the name of this attribute
   *
   * @return The name of this attribute
   */
  char *Name() { return name; }

  /**
   * @brief Get the part associated with this attribute (if one exists)
   *
   * @return The part associated with this attribute (or NULL if none exists)
   */
  ObjectPart *Part() { return part; }

  /**
   * @brief Set the part associated with this attribute
   * @param p The part
   */
  void SetPart(ObjectPart *p) { part=p; }

  /**
   * @brief Get the number of elements in the attribute detector's vector of weights
   * 
   * @return The number of weights
   */
  int NumWeights() { int n = 0; for(int i = 0; i < numFeatureTypes; i++) { n+= features[i].dim; } return n; }

  /**
   * @brief Get an upper bound on the squared l2 norm of the feature space
   */
  float MaxFeatureSumSqr();

  /**
   * @brief Get the detection weights for this attribute
   * 
   * @param w An array of size NumWeights() into which the detection weights are copied
   * @return The number of weights
   */
  int GetWeights(double *w);

  /**
   * @brief Get the detection weights for this attribute
   * 
   * @param w An array of size NumWeights() into which the detection weights are copied
   * @return The number of weights
   */
  int GetWeights(float *w);

  /**
   * @brief Get the detection weights for this attribute and a specific feature type
   *
   * @param w An array of size NumWeights() into which the detection weights are copied
   * @param feat_name The name of the feature you want to extract
   * @return The number of weights
   */
  int GetWeights(float *w, const char *feat_name);

  /**
   * @brief Get the detection weights for this attribute
   * 
   * @return An array of NumWeights() weights
   */
  float *Weights() { return weights; }

  /**
   * @brief Set the detection weights for this attribute
   * 
   * @param w An array of size NumWeights() from which the detection weights are copied
   */
  void SetWeights(double *w);

  /**
   * @brief Set the detection weights for this attribute
   * 
   * @param w An array of size NumWeights() from which the detection weights are copied
   */
  void SetWeights(float *w);



  //void TrainFromTemplate(ImageProcess *p, int x, int y, int w, int h, const char *fname);

  /**
   * @brief Get the number of different feature types used by this attribute detector
   * @return The number of different feature types used by this attribute detector
   */
  int NumFeatureTypes() { return numFeatureTypes; }

  /**
   * @brief Get the ith feature type used by this attribute detector
   * @return The ith feature type used by this attribute detector
   */
  FeatureWindow *Feature(int i) { return i >= 0 && i < numFeatureTypes ? (features+i) : NULL; }

  /**
   * @brief Get the list of feature types used by this attribute detector
   * @return The list of feature types used by this attribute detector
   */
  FeatureWindow *Features() { return features; }

  /**
   * @brief Set the feature types associated with this attribute
   * @param f an array of num sliding window feature definitions
   * @param num the number of feature types
   */
  void SetFeatures(FeatureWindow *f, int num) {
    if(weights) {
      free(weights);
      weights = NULL;
    }
    features = (FeatureWindow*)realloc(features, sizeof(FeatureWindow)*num);
    memcpy(features, f, sizeof(FeatureWindow)*num);
    for(int i = 0; i < num; i++) features[i].name = f[i].name ? StringCopy(f[i].name) : NULL;
    numFeatureTypes = num;
  }

  /**
   * @brief Get the width of the sliding window detector for this attribute
   * @param i If supplied, gets the width of the ith feature template
   * @return The width of the sliding window detector for this attribute
   */
  int Width(int i=0);

  /**
   * @brief Get the height of the sliding window detector for this attribute
   * @param i If supplied, gets the height of the ith feature template
   * @return The height of the sliding window detector for this attribute
   */
  int Height(int i=0);

  /**
   * @brief Get the parameter used to convert the attribute detection score to a probability 
   *
   * It is assumed the attribute probability is of the form 
   * \f$ p(a|x) \propto exp\{ \gamma_a f_a(x) \} \f$, where
   * \f$ f_a(x) \f$ is the attribtue detection score learned by TrainAttributes()
   *
   * @return The parameter \f$ \gamma_a \f$
   */
  float GetGamma() { return gamma; }

  /**
   * @brief Get the question associated with this attribute (if one exists)
   * @return The question associated with this attribute (or NULL if none exists)
   */
  Question *GetQuestion() { return binaryQuestion; }

  /**
   * @brief Set the parameter used to convert the attribute detection score to a probability 
   *
   * It is assumed the attribute probability is of the form 
   * \f$ p(a|x) \propto exp\{ \gamma_a f_a(x) \} \f$, where
   * \f$ f_a(x) \f$ is the attribtue detection score learned by TrainAttributes()
   *
   * @param g The parameter \f$ \gamma_a \f$
   */
  void SetGamma(float g) { gamma = g; }

  /**
   * @brief Set the question associated with this attribute (if one exists)
   * @param q The question associated with this attribute (or NULL if none exists)
   */
  void SetQuestion(Question *q) { binaryQuestion = q; }

  /**
   * @brief Set this attributes pointer to the Classes definition
   * @param c Pointer to Classes definition object
   */
  void SetClasses(Classes *c) { classes = c; }

  /**
   * @brief Get the image name of visualization
   */
  const char *GetVisualizationImageName() { return visualization_image; }

  /**
   * @brief Get the attribute value name, e.g., "blue" for blue belly
   */
  const char *ValueName() { return value_name; }

  /**
   * @brief Get the attribute propert name, e.g., "belly color" for blue belly
   */
  const char *PropertyName() { return property_name; }

  /**
   * @brief Set the filename of the image used to visualize this attribute
   * @param v the filename of the image used to visualize this attribute
   */
  void SetVisualizationImage(const char *v) { visualization_image=StringCopy(v); }

  /**
   * @brief If this attribute is a binary attribute in some grouping defined by a multi-value attribute, we can store that information here 
   * @param p The attribute property name (name of the multi-value attribute)
   * @param v The attribute value name for this binary attribute
   */
  void SetPropertyNames(const char *p, const char *v) { property_name=StringCopy(p); value_name=StringCopy(v); }


 private:
  bool ResolveLinks(Classes *c);
  void SetId(int i) { id = i; }

  friend class Classes;

 public:
  //deprecated
  /// @cond
  char *LoadFromString(char *str);
  char *ToString(char *str);
  bool ResolveLinksOld(Classes *c);
  const char *LoadWeightString(const char *str, float *w, int num);
  void WeightString(char *str, float *w, int num);
  int part_id;
  /// @endcond
};

/**
 * @class AttributeInstance
 * 
 * @brief An instance of an attribute in a particular image
 *
 */
class AttributeInstance {
  Attribute *model; /**< the Attribute model used by this instance */

  IplImage ***responses; /**< a numScalesXnumOrientations array of detection scores for pixel location */

  ImageProcess *process; /**< pointer to the processing object used to compute responses */

  float *custom_weights; /**< Override the model weights */

  bool isFlipped;

 public:

  /**
   * @brief Constructor
   * 
   * @param a The Attribute model this instance is associated with 
   * @param p The structure used to compute image features
   */
  AttributeInstance(Attribute *a, ImageProcess *p, bool isFlipped = false);

  /**
   * @brief Destructor
   */
  ~AttributeInstance();

  /**
   * @brief Run the sliding window attribute detector in the image
   * 
   * @return A numScalesXnumOrientations array of detection scores at each pixel
   */
  IplImage ***Detect(ObjectPose *pose=NULL);

  /**
   * @brief Get the attribute detection scores for this image (calls Detect() if necessary)
   * 
   * @return A numScalesXnumOrientations array of detection scores at each pixel
   */
  IplImage ***GetResponses(ObjectPose *pose=NULL);

  /**
   * @brief Get the width of the sliding window detector for this attribute
   * @return The width of the sliding window detector for this attribute
   */
  int Width();

  /**
   * @brief Get the height of the sliding window detector for this attribute
   * @return The height of the sliding window detector for this attribute
   */
  int Height();

  bool IsFlipped() { return isFlipped; }

  /**
   * @brief Free cached sliding window detection scores
   */ 
  void Clear();

  /**
   * @brief Get the Attribute model associated with this attribute
   * @return The Attribute model associated with this attribute
   */
  Attribute *Model() { return model; }

  /**
   * @brief Get the ObjectPart associated with this attribute (if one exists)
   * @return The ObjectPart associated with this attribute
   */
  ObjectPartInstance *Part();
  
  /**
   * @brief Get the features for this attribute detector at a particular location
   * 
   * @param f An array of size model->NumWeights() into which the extracted features are copied
   * @param loc The location at which to extract features
   * @param has_attribute if non-NULL, an array of size classes->numAttributes. has_attribute[id] is multiplied times the features
   * @param feat_name if non-NULL, only gets features that are of this type of feature
   * @return The number of features extracted
   */
  int GetFeatures(float *f, PartLocation *loc, float *has_attribute=NULL, const char *feat_name=NULL, FeatureWindow *features=NULL, int numFeatures=0);

  int GetFeatures(float *f, PartLocation *loc, FeatureWindow *features, int numFeatures) {
    return GetFeatures(f, loc, NULL, NULL, features, numFeatures);
  }

  /**
   * @brief Set custom detection weights for this attribute instance, which override those of the attribute models
   * @param w An array of detection weights, with the same ordering as Model()->GetWeights()
   */
  void SetCustomWeights(float *w) { custom_weights = w; }

  /**
   * @brief Get the custom detection weights for this attribute instance, which override those of the attribute model
   * @return An array of detection weights, with the same ordering as Model()->GetWeights()
   */
  float *GetCustomWeights() { return custom_weights; }

  /**
   * @brief Get the log likelihood or detection score at location loc
   * 
   * @param loc The location of the part associated with the part
   * @param attributeWeight A value of 1 indicates attribute presence and -1 indicates absence of this attribute.  A value in between indicates soft membership.
   * @param useGamma If true, use gamma to scale the detection score into a log likelihood
   */ 
  double GetLogLikelihoodAtLocation(PartLocation *loc, float attributeWeight, bool useGamma=true, float *features=NULL);

  /**
   * @brief Get the location of the highest scoring detection response
   *
   * @param best_x If non-NULL the x-coordinate of the max location is stored into best_x 
   * @param best_y If non-NULL the y-coordinate of the max location is stored into best_y 
   * @param best_scale If non-NULL the scale of the max location is stored into best_scale 
   * @param best_rot If non-NULL the rotation of the max location is stored into best_rot
   * @param responseMap If non-NULL the detection score at every pixel location that merges all scales/orientations (by taking the max) is stored into responseMap
   * @param bestIndMap If non-NULL, a 2D image of the scale and orientation corresponding to the max score in responseMap is stored into bestIndMap
   * @return The score at the highest scoring location
   */
  float GetMaxResponse(int *best_x=NULL, int *best_y=NULL, int *best_scale=NULL, int *best_rot=NULL, 
		       IplImage **responseMap=NULL, IplImage **bestIndMap=NULL);

  /**
   * @brief Helper routine for debugging/inspecting attribute detection results.  This is useful to verify that inference is working correctly.
   * @param w A classes->NumWeights() array, which should have been extracted using Model()->GetWeights()
   * @param f A classes->NumWeights() array, which should have been extracted using GetFeatures()
   * @param print_weights If true, print out weights and features in w and f
   * @param f_gt If non-null, a classes->NumWeights() array of features for the ground truth part locations
   */
  int Debug(float *w, float *f, bool print_weights, float *f_gt=NULL);
};


/// @cond
float GetMaxResponse(IplImage ***responses, int numScales, int numOrientations, 
		     int *best_x=NULL, int *best_y=NULL, int *best_scale=NULL, int *best_rot=NULL, 
		     IplImage **reponseMap=NULL, IplImage **bestIndMap=NULL);
/// @endcond

#endif
