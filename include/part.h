/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#ifndef __PART_H
#define __PART_H

#include "feature.h"

/**
 * @file part.h
 * @brief Definition of part models and routines for detecting parts in images
 */

class ObjectPart;
class ObjectPose;
class ObjectPartInstance;
class ObjectPoseInstance;
class Attribute;
class FeatureOptions;
class ImageProcess;
class Classes;
class Attribute;
class Question;
class ObjectPartPoseTransition;

#define BIAS_FEATURE 10

typedef struct {
  bool useBiasTerm;
  bool useScalePrior;
  bool useOrientationPrior;
  int numSpatialLocationLevels;
  float *weights;
} StaticPartFeatures;

/**
 * @class PartLocation
 * 
 * @brief Predicted location and pose of a particular part 
 */
class PartLocation {
 public:
  PartLocation();

  /**
   * @brief Create a new PartLocation as a copy of another PartLocation
   * @param p the PartLocation to copy
   */
  PartLocation(const PartLocation &p);

  /**
   * @brief Create a new PartLocation as an unitialized label for a particular part
   * @param part the part that this location corresponds to
   * @param image_width the width of the image that this location will label
   * @param image_height the height of the image that this location will label
   */
  PartLocation(ObjectPartInstance *part, int image_width, int image_height);

  ~PartLocation();
  
  /**
   * @brief Clear all memory associated with this PartLocation
   */
  void Clear();

  /**
   * @brief Copy another PartLocation into this one
   * @param p the PartLocation to copy
   */
  void Copy(const PartLocation &p);

  /**
   * @brief Get a pointer to the object defining the set of parts and poses
   */
  Classes *GetClasses() { return classes; }

  void SetFeat(FeatureOptions *fe) { feat = fe; }

  /**
   * @brief Allocate an array of PartLocations, which will be used to annotate the location of each part in an image but is otherwise uninitialized
   * @param classes a pointer to the object defining the set of parts and poses
   * @param image_width the width of the image that this location will label
   * @param image_height the height of the image that this location will label
   * @param feat object used for computing features in this image
   * @param isClick if true, will correspond to click locations of a particular person, as opposed to ground truth locations
   */
  static PartLocation *NewPartLocations(Classes *classes, int image_width, int image_height, FeatureOptions *feat, bool isClick);
  static PartLocation *CopyPartLocations(PartLocation *locs);

  static PartLocation *FlippedPartLocations(PartLocation *locs);

  /**
   * @brief Get the location of this part in pixel coordinates
   * @param x the x-pixel location of the center of the part in the image
   * @param y the y-pixel location of the center of the part in the image
   * @param scale the scale of this part
   * @param rot the orientation of this part in radians
   * @param pose the name of the pose of this part
   * @param width the width of the image
   * @param height the height of the image
   */
  void GetImageLocation(float *x, float *y, float *scale=NULL, float *rot=NULL, 
			const char **pose=NULL, float *width=NULL, float *height=NULL);

  /**
   * @brief Set the location of this part in pixel coordinates
   * @param x the x-pixel location of the center of the part in the image
   * @param y the y-pixel location of the center of the part in the image
   * @param scale the scale of this part
   * @param rot the orientation of this part in radians
   * @param pose the name of the pose of this part
   */
  void SetImageLocation(float x, float y, float scale, float rot, const char *pose);


  /**
   * @brief Get the location of this part in detection coordinates (the location in the detection score maps)
   * @param x the x-coordinate of the center of the part in the image in detection coordinates
   * @param y the y-coordinate location of the center of the part in the image in detection coordinates
   * @param scale the index of the scale of this part
   * @param rot the index of the orientation of this part
   * @param pose the index of the pose of this part
   * @param width the width of the image
   * @param height the height of the image
   * @param dx the offset between parent and child parts (x_parent+mu_x-x_child)
   * @param dy the offset between parent and child parts (y_parent+mu_y-y_child)
   */
  void GetDetectionLocation(int *x, int *y, int *scale=NULL, int *rot=NULL, 
			    int *pose=NULL, float *width=NULL, float *height=NULL, int *dx=NULL, int *dy=NULL);


  /**
   * @brief Set the location of this part in detection coordinates (the location in the detection score maps)
   * @param x the x-coordinate of the center of the part in the image in detection coordinates
   * @param y the y-coordinate location of the center of the part in the image in detection coordinates
   * @param scale the index of the scale of this part
   * @param rot the index of the orientation of this part
   * @param pose the index of the pose of this part
   * @param dx the offset between parent and child parts (x_parent+mu_x-x_child)
   * @param dy the offset between parent and child parts (y_parent+mu_y-y_child)
   */
  void SetDetectionLocation(int x, int y, int scale, int rot, int pose, int dx, int dy);

  /**
   * @brief Get the image size
   * @param w the returned image width in pixels
   * @param h the returned image height in pixels
   */
  void GetImageSize(int *w, int *h) { *w = image_width;  *h = image_height; }

  /**
   * @brief Get the id of the part that this PartLocation labels
   */
  int GetPartID();

  /**
   * @brief Get the part that this PartLocation will label
   */
  void SetPart(ObjectPart *part);

  /**
   * @brief Load an ObjectPart object from a JSON encoding
   * @param root A JSON encoding of this part
   * @return True if successful
   */
  bool load(const Json::Value &root);

  /**
   * @brief Save a definition of this part to a JSON object
   * 
   * @return A JSON encoding of this part
   */
  Json::Value save();

  /**
   * @brief Compute detection locations from image locations
   */
  bool ComputeDetectionLocations();

  /**
   * @brief Compute image locations from detection locations
   */
  bool ComputeImageLocations();

  /**
   * @brief Get the time needed for a part click question
   */
  float GetResponseTime() { return responseTimeSec; }

  /**
   * @brief Set the time needed for a part click question
   */
  void SetResponseTime(float f) { responseTimeSec = f; }

  /**
   * @brief Get the detection score at this location
   */
  float GetScore() { return score; }

  /**
   * @brief Set the detection score at this location
   */
  void SetScore(float f) { score = f; }

  /**
   * @brief Check if this part location is fully latent/unspecified
   */
  bool IsLatent();

  /**
   * @brief Copy another PartLocation into this one
   * @param v the PartLocation to copy
   */
  PartLocation &operator=(const PartLocation &v) {
    this->Copy(v); 
    return *this;
  }

  bool IsValid() { return hasDetectionCoords || hasImageCoords; }

  /**
   * @brief Create an unitialized PartLocation
   * @param classes a pointer to the object defining the set of parts and poses
   * @param image_width the width of the image that this location will label
   * @param image_height the height of the image that this location will label
   * @param feat object used for computing features in this image
   */
  void Init(Classes *classes, int image_width, int image_height, FeatureOptions *feat);

  bool IsVisible() { return visible; }

private:

  bool hasDetectionCoords;   // If true, then the detection coordinates x_det, y_det, scale_det, rotation_det, dx, dy are valid 
  bool hasImageCoords;       // If true, then the image coordinates x_img, y_img, scale_img, rotation_img are valid

  bool visible;
  bool isClick;
  Classes *classes;
  int image_width, image_height;
  FeatureOptions *feat;

  float x_img;  /**< The x location of this part in the image */
  float y_img;  /**< The y location of this part in the image */
  float scale_img;     /**< The scale of this part */
  float rotation_img;  /**< The rotation of this part in radians */
  char *partName; /**< The name of this part */
  char *poseName;  /**< The name of the pose of this part  */

  int x_det, y_det; /**< The location of this part in the image in terms of the detection response map indices */
  int scale_det;  /**< The scale of this part (an index into the feature scales) */
  int rotation_det; /**< The rotation of this part (an index into the feature orientations) */
  int dx; /**< The x offset of this part in the image in terms of the parent part and detection response map indices */
  int dy; /**< The y offset of this part in the image in terms of the parent part and detection response map indices */ 
  int partID; /**< The id of this part */
  int poseID;  /**< The pose of this part (an index into part->poses) */

  float width;  /**< The width of the bounding box of this part */
  float height; /**< The width of the bounding box of this part */

  bool flipHorizontal; /**< If true, this part is flipped horizontally from the canonical part */
  bool flipVertical; /**< If true, this part is flipped vertically from the canonical part */
  
  float responseTimeSec; /**< For a click part question, the amount of time spent by the user */

  float score; /**< The detection score at this point */
};


/**
 * @struct _PartLocationSampleSet
 * 
 * @brief A collection of multiple assignments to all part locations of an object.  This
 *   is used when randomly sampling multiple sets of part locations
 */
typedef struct _PartLocationSampleSet {
  PartLocation **samples;  /**< A num_samplesXnum_parts array of PartLocation samples */
  int num_samples;  /**< The total number of samples */
  ObjectPart *root_part;  /**< The root part that was used to guide the random sampling process (or NULL if no part was used) */
  PartLocation *root_samples;  /**< A num_samples array of the root part location for each sample */
} PartLocationSampleSet;



/**
 * @class ObjectPart
 * 
 * @brief An object or a portion of an object, e.g. beak, belly, head.  A part is associated with a set of possible poses.
 *
 * Parts can be defined hierarchically.  For example person->head+torso+arms+legs, head->eyes+mouth+nose, etc.
 * Parts can have a collection of possible poses, which are different possible views or configurations of the part. 
 * A part can have attributes Attribute like shiny, red, has_metal
 */
class ObjectPart {
  char *name; /**< The name of this part, e.g. "beak" */
  int id; /**< The id of this part */

  ObjectPart *parent;  /**< The parent of this part. The parts should be connected in some hierarchical structure  */
  char *parent_name; /**< The name of the parent part */

  ObjectPart *flipped;
  char *flipped_name;

  char *visualization_image;
  char *abbreviation;

  StaticPartFeatures staticFeatures;

  /**
   * @brief The set of possible poses for this part
   * 
   * An object or part can have a bunch of possible poses (part models).
   * Each pose can have a list of subparts with some spatial relationship
   */
  ObjectPose **poses; 
  int numPoses; /**< The number of possible poses for this part */
  char **pose_names; /**< The names of each possible pose of this part */

  /**
   * @brief The child parts of this part.
   *
   * An object or part can have a bunch of possible poses (part models).
   * Each pose can have a list of subparts with some spatial relationship
   */
  ObjectPart **parts;
  int numParts; /**< The number of child parts of this part */
  char **part_names; /**< The names of each child of this part */

  Classes *classes; /**< The Classes definition, defining all classes, parts, poses, and attributes */

  bool isClick; /**< For asking the user to click the location of a part, the click location can be treated as a latent variable */

  Attribute **attributes; /**< List of attributes associated with this part */
  int numAttributes; /**< Number of attributes associated with this part */

  FeatureOptions *feat;

  unsigned int color;

  // Temporary storage used for topological sort
  int classInd;

  
  float gamma; /**< Multiply this times the log-likelihood in the process of converting part detection scores to probabilities */

  Question *question; /**< The part click question associated with this part */

  ObjectPartPoseTransition ****childPartPoseTransitions; /**< A numPosesXnumPartsXnumChildPartPoseTransitions[i][j] array of spatial models for every allowable transition between parent and child poses */
  int **numChildPartPoseTransitions; /**< A numPosesXnumParts array of the number of allowable transitions between parent and child poses */

 public:

  /**
   * @brief Constructor
   * @param name The name of this part, e.g. "beak"
   * @param isClick Set to true if this is a latent variable associated with a user clicking on a part (instead of an actual part) 
   * @param abbrev the abbreviation for this part (one or two letters that is used to label a part in a part tree diagram)
   */
  ObjectPart(const char *name=NULL, bool isClick=false, const char *abbrev=NULL);

  /**
   * @brief Destructor
   */
  ~ObjectPart();

  /**
   * @brief Get the abbreviation for this part (one or two letters that is used to label a part in a part tree diagram)
   */
  char *GetAbbreviation() { return abbreviation; }

  /**
   * @brief Save a definition of this part to a JSON object
   * 
   * @return A JSON encoding of this part
   */
  Json::Value Save();

  /**
   * @brief Load an ObjectPart object from a JSON encoding
   * @param root A JSON encoding of this part
   * @return True if successful
   */
  bool Load(const Json::Value &root);

  /**
   * @brief Get the name of this part
   * @return the name of this part
   */
  char *Name() { return name; }

  /**
   * @brief Get the id of this part
   * @return the id of this part
   */
  int Id() { return id; }

  /**
   * @brief Check if this ObjectPart variable is a latent variable associated with a user 
   *   clicking on a part (instead of an actual part
   * @return true if this ObjectPart variable is a latent variable associated with a user 
   *   clicking on a part
   */
  bool IsClick() { return isClick; }

  /**
   * @brief Get the parent ObjectPart of this part. The parts should be connected in some hierarchical structure
   * @return The parent of this part
   */ 
  ObjectPart *GetParent() { return parent; }

  ObjectPart *GetFlipped() { return flipped; }
  void SetFlipped(ObjectPart *f) { flipped = f; }

  /**
   * @brief Get the number of child parts of this part
   * @return the number of child parts of this part
   */
  int NumParts() { return numParts; }

  /**
   * @brief Get the ith child ObjectPart of this part
   * @param i The index of the ObjectPart we are searching for
   * @return The ith ObjectPart
   */
  ObjectPart *GetPart(int i) { return parts ? parts[i] : NULL; }

  /**
   * @brief Get the number of possible poses of this part
   * @return The number of possible poses of this part
   */
  int NumPoses() { return numPoses; }

  /**
   * @brief Get the ith ObjectPose of this part
   * @param i The index of the ObjectPose we are searching for
   * @return The ith ObjectPose
   */
  ObjectPose *GetPose(int i) { return poses[i]; }
  
  /**
   * @brief Get the number of possible attributes that are associated with this part
   * @return The number of possible attributes
   */
  int NumAttributes() { return numAttributes; }

  /**
   * @brief Get the ith Attribute that is associated with this part
   * @param i The index of the Attribute we are searching for
   * @return The ith Attribute
   */
  Attribute *GetAttribute(int i) { return attributes[i]; }

  /**
   * @brief Get the click point Question associated with this part
   * @return The click point Question associated with this part
   */
  Question *GetQuestion() { return question; }

  /**
   * @brief Get the parameter to convert part detection scores to probabilities
   * 
   * Given a part detection scoring function \f$ f(\theta_p,x)\f$ (see TrainDetectors()),
   *   the detection score is converted to a probability \f$ p(\theta_p|x) \propto \exp\{\gamma_p f(\theta_p,x)\} \f$
   *
   * @return The parameter \f$ \gamma_p \f$
   */
  float GetGamma() { return gamma; } 


  /**
   * @brief Add a new ObjectPose to the list of possible poses for this part
   * @param pose The new ObjectPose
   */
  void AddPose(ObjectPose *pose);

  /**
   * @brief Add a new ObjectPart to the list of child parts this part
   * @param part The new ObjectPart
   */
  void AddPart(ObjectPart *part);

  /**
   * @brief Add a new Attribute that can be associated with this part
   * @param a The new Attribute
   */
  void AddAttribute(Attribute *a) {
    attributes = (Attribute**)realloc(attributes, sizeof(Attribute*)*(numAttributes+1));
    attributes[numAttributes++] = a;
  }

  /**
   * @brief Set the click point Question associated with this part
   * @param q The click point Question associated with this part
   */
  void SetQuestion(Question *q) { question = q; }


  /**
   * @brief Set the parameter to convert part detection scores to probabilities
   *
   * Given a part detection scoring function \f$ f(\theta_p,x)\f$ (see TrainDetectors()),
   *   the detection score is converted to a probability \f$ p(\theta_p|x) \propto \exp\{\gamma_p f(\theta_p,x)\} \f$
   *
   * @param gamma The parameter \f$ \gamma_p \f$. 
   */
  void SetGamma(float gamma) { this->gamma = gamma; }

  /**
   * @brief Get the weights associated with the spatial model of this part
   * @param w A vector of weights that is set by this function
   * @return w The number of weights extracted
   */
  int GetStaticWeights(float *w) { memcpy(w, staticFeatures.weights, sizeof(float)*NumStaticWeights()); return NumStaticWeights(); } 

  /**
   * @brief Set the weights associated with the spatial model of this part
   * @param w A vector of weights from which the part's weights are copied from
   */
  void SetStaticWeights(float *w);

  /**
   * @brief Get the number of weights associated with the spatial model of this part
   * @return w The number of weights 
   */
  int NumStaticWeights();

  /**
   * @brief Get an upper bound on the squared l2 norm of the feature space
   */
  float MaxFeatureSumSqr();

  /**
   * @brief Get information about which weight parameters should be learned, constrained to be positive or negative,
   *  or regularized
   * @param wc a NumWeights() array set by this function, where a value of 1 indicates a weight must be positive, -1 indicates it
   * must be negative, and 0 indicates no constraint
   * @param learn_weights A NumWeights() array set by this function, where a value of true indicates a weight parameter should be learned
   * @param regularize A NumWeights() array set by this function, where a value of true indicates a weight parameter should be regularized
   */
  int GetStaticWeightConstraints(int *wc, bool *learn_weights, bool *regularize);
  
  /**
   * @brief Get the index of a pose in this part's list of poses
   * @param p The pose we are searching for
   * @return The pose index
   */
  int PoseInd(ObjectPose *p) { for(int i = 0; i <numPoses; i++) { if(poses[i] == p) return i; } return -1; }

  /**
   * @brief Get the index of a part in this part's list of child parts 
   * @param p The part we are searching for
   * @return The part index
   */
  int PartInd(ObjectPart *p) { for(int i = 0; i <= numParts; i++) { if(parts[i] == p) return i; } return -1; }

  /**
   * @brief Get the set of spatial models defining the spatial relationship between this part
   * and all possible neighboring parts and poses.
   * @param numChildPoses A pointer to a numPosesXnumParts array defining the number of allowable child poses for every possible
   *  child part and pose of this part.  The pointer is set by this function
   * @return A numPosesXnumPartsX(*numChildPoses)[i][j] array of part/pose transition models
   */
  ObjectPartPoseTransition ****GetPosePartTransitions(int ***numChildPoses) { *numChildPoses = numChildPartPoseTransitions; return childPartPoseTransitions; }
  
  /**
   * @brief Get pointer to Classes object
   */
  Classes *GetClasses() { return classes; }

  /**
   * @brief Get the image name of visualization
   */
  const char *GetVisualizationImageName() { return visualization_image; }

  /**
   * @brief Set the image name of visualization
   */
  void SetVisualizationImage(const char *v) { visualization_image=StringCopy(v); }

  StaticPartFeatures GetStaticPartFeatures() { return staticFeatures; }

  bool LoadStaticFeatures(const Json::Value &root);
  Json::Value SaveStaticFeatures();

  unsigned int GetColor() { return color; }
  void SetColor(unsigned int c) { color = c; }

private:
  /**
   * @brief Resolve pointers to other objects, parts, or attributes
   *
   * Typically, this is called after all classes, parts, and attribute definitions have been loaded
   * 
   * @param classes Classes definition object
   * @return true on success
   */
  bool ResolveLinks(Classes *c);


  /**
   * @brief Set the parent ObjectPart of this part. The parts should be connected in some hierarchical structure
   * @param The parent of this part
   */  
  void SetParent(ObjectPart *p) { 
    parent = p; 
    parts = (ObjectPart**)realloc(parts, sizeof(ObjectPart*)*(numParts+1));
    parts[numParts] = parent;
  }

  /**
   * @brief Set the id of this part
   * @param i The new id of this part
   */
  void SetId(int i) { id = i; }

  void AddSpatialTransition(ObjectPartPoseTransition *t);
  
  void AddSpatialTransitions(PartLocation *locs, PartLocation *l=NULL, bool computeSigma=false, bool combinePoses=false);
  void NormalizePartPoseTransitions(int minExamples, bool computeSigma=false);

  int GetClassInd() { return classInd; }
  void SetClassInd(int i) { classInd = i; }


  int DebugWeights(float *w);

  friend class Classes;


  // deprecated
  char *LoadFromString(char *str, char **ptrStart);
  char *ToString(char *str);
  bool ResolveLinksOld(Classes *c);
  char *transition_string;
  int parent_id;
  int *pose_ids, *part_ids;
};



/**
 * @class ObjectPartInstance
 * 
 * @brief An instance of a part occurring in a particular image
 */
class ObjectPartInstance {
  ObjectPart *model; /**< Defines the part/pose model for this part */

  ImageProcess *process;  /**< pointer to the processing object used to compute feature responses */  

  ObjectPartInstance **parts; /**< Array of size model->numParts of subparts for this part */

  ObjectPoseInstance **poses;  /**< Array of size model->numPoses of poses for this part */

  ObjectPartInstance *clickPart; /**< Used if this part has a variable associated with a user click location */

  bool isObserved; /**< For a click part, true if the user has already clicked */   
  bool isAnchor;  /**< true if a user has dragged this part during interactive labeling */

  PartLocation theta; /**< The location of this part in the image */

  PartLocation ground_truth_loc;

  IplImage ****pose_responses; /**< A numPosesXnumScalesXnumOrientations array of detection response maps of this part */ 
  IplImage ***responses; /**< A numScalesXnumOrientations array of detection response maps of this part, taking the max of responses over pose */

  float maxScore;
  float delta;  /**< Scale the loglikelihood by gamma, add delta, and take the exp() to convert to a probability */
  float lower_bound; /**< A lower bound on the log likelihood, to avoid assigning any pixel location too low of a probability */

  IplImage ****cumulativeSumMaps; /**< A numPosesXnumScalesXnumOrientations array of cumulative sum maps of the part's probability map, used for drawing random samples */
  float ***cumulativeSums;  /**< A numPosesXnumScalesXnumOrientations array of cumulative sums of the part's probability map, used for drawing random samples */
  float cumulativeSumTotal; /**< The total cumulative sum of all maps */

  PartLocation **cumulativeSumsNMS_keypoints;
  int cumulativeSumsNMS_numKeypoints;
  float cumulativeSumsNMS_sum;

  float *customStaticWeights;

 public:
  /**
   * @brief Constructor
   * @param m The model of this part
   * @param p pointer to the processing object used to compute feature responses 
   */
  ObjectPartInstance(ObjectPart *m, ImageProcess *p);

  /**
   * @brief Destructor
   */
  ~ObjectPartInstance();


  /**
   * @brief Get a pointer to the processing object used to compute feature responses 
   * @return A pointer to the processing object used to compute feature responses 
   */
  ImageProcess *Process() { return process; }

  /**
   * @brief Get the part model associated with this part instance
   * @return The part model associated with this part instance
   */
  ObjectPart *Model() { return model; }

 /**
  * @brief Get the id of this part
  * @return the id of this part
  */
  int Id() { return model->Id(); }

  /**
   * @brief Get the detection score at a particular part location
   * @param loc the location at which to get the detection score
   * @return the detection score
   */
  float GetScoreAt(PartLocation *loc) {
    int x, y, scale, rot, pose;
    loc->GetDetectionLocation(&x, &y, &scale, &rot, &pose);
    return ((float*)(pose_responses[pose][scale][rot]->imageData + pose_responses[pose][scale][rot]->widthStep*y))[x];
  }

  /**
  * @brief Get the part location with maximum likelihood (assumes) Detect(true) has been called
  * @return the id of this part
  */
  PartLocation GetBestLoc() { assert(responses[0][0]); return theta; }

  /**
   * @brief Get the ith pose instance that can be associated with this part
   * @param i The index of the pose we want to get
   * @return The ith pose instance associated with this part
   */
  ObjectPoseInstance *GetPose(int i) { return poses[i]; }

  /**
   * @brief Find the pose instance with a particular id
   * @param id The id of the pose we want to get
   * @return The pose
   */
  ObjectPoseInstance *FindPose(int id);

  /**
   * @brief Find any pose of this part that is not the non-visible pose
   */
  ObjectPoseInstance *GetVisiblePose();

  /**
   * @brief Get the non-visible pose for this part
   */
  ObjectPoseInstance *GetNotVisiblePose();

  /**
   * @brief Get the ith child of this part
   * @param i The index of the part we are searching for
   * @return The ith part
   */
  ObjectPartInstance *GetPart(int i) { return parts ? parts[i] : NULL; }


  /**
   * @brief Get the parent of this part. The parts should be connected in some hierarchical structure
   * @return The parent of this part
   */  
  ObjectPartInstance *GetParent();

  
  /**
   * @brief Compute the detection scores of this part at every location in the image
   * @return A numScalesXnumOrientations array of probability maps
   */
  IplImage ***Detect(bool isRoot);

  /**
   * @brief Get the detection score of this part at a particular scale and rotation
   *
   * Calls Detect() if necessary
   *
   * @param s The scale
   * @param r The rotation
   */
  IplImage *GetResponse(int s, int r);

  /**
   * @brief Get the detection score of this part at a particular location
   *
   * Assumes Detect() has already been called
   *
   * @param loc Defines the pixel location, pose, scale, and orientation of where we want to get the detection score
   * @return The detection score (log likelihood)
   */
  float GetLogLikelihoodAtLocation(PartLocation *loc);

  int GetLocalizedFeatures(float *f, PartLocation *loc);


  /**
   * @brief Get the location of this part with maximum score in the image (assumes Detect() has been called first)
   * @return The location of this part with maximum score in the image
   */
  PartLocation *GetPartLocation() { return &theta; }

  /**
   * @brief Get the obsered location (set with SetLocation()) if one exists
   */
  PartLocation *GetObservedLocation() { return isObserved ? &theta : NULL; }

  /**
   * @brief Extract the maximum likelihood part locations for all parts in the object
   *
   * Assumes Detect() has already been called. 
   *
   * @param locs an array of size numParts, all values of which are set using this function
   * @param l if non-NULL, extract the solution given that this part must be in possition l (otherwise use the maximum scoring position of this part)
   * @param par GetPartLocation() recursively descends through all neighboring parts.  If par is non-NULL, we avoid descending through par
   */
  void ExtractPartLocations(PartLocation *locs, PartLocation *l=NULL, ObjectPartInstance *par=NULL);

  /**
   * @brief Clear memory buffers created from a previous call to Detect()
   * @brief clearResponses If true, clears detection response scores 
   * @brief clearPose If true, clears buffers associated with pose detection and spatial models
   */
  void Clear(bool clearResponses=true, bool clearPose=true);

  /**
   * @brief Clear cumulative sum buffers
   */
  void ClearCumulativeSums();

  /**
   * @brief Draw a bounding box visualization of this part
   * @param img The image we want to draw into
   * @param l The location of the part in the image (if NULL, uses the part in GetPartLocation())
   * @param color The color used to draw the circle for this part
   * @param color2 The color used to draw the outer circle for this part
   * @param color3 The color used to draw the text label
   * @param str If non-null draw this string along with the bounding box
   * @param labelPoint If true, draw a circle depicting the center of the part
   * @param labelRect If true, draw a rectangle for the bounding box of the part
   * @param zoom Scale the numbers in l by this factor before drawing into the image
   */
  void Draw(IplImage *img, PartLocation *l=NULL, CvScalar color=CV_RGB(0,0,200), CvScalar color2=CV_RGB(0,0,200), CvScalar color3=CV_RGB(0,0,200), const char *str=NULL, 
	    bool labelPoint=false, bool labelRect=true, float zoom=1);

  /**
   * @brief Check if this part is a click part variable and the user has already clicked somewhere
   */
  bool IsObserved() { return isObserved; }

  /**
   * @brief Check if this part is a click part variable and the user has already clicked somewhere
   */
  bool IsAnchor() { return isAnchor; }

  PartLocation **GreedyNonMaximalSuppression(int num_samples, double min_score, int *num, double suppressWidth=1, 
					     double suppressScale=2, double suppressOrientation=2*M_PI, 
					     bool suppressAllPoses=false, bool getDiversePoses=false);
  PartLocation **GreedyNonMaximalSuppressionByBoundingBox(int num_samples, double min_score, int *num);
  
  /**
   * @brief Associate a latent variable for clicking on this part 
   * @param c The click ObjectPartInstance
   */
  void SetClickPart(ObjectPartInstance *c) { clickPart = c; }

  /**
   * @brief Compute the cumulative sum of the detection probability map for this part
   *
   * Assumes Detect() has already been called.  This is used for randomly sampling part locations
   *
   * @param recompute If true, always recomputes the cumulative sums, even if they already exist from a previous call
   * @return The cumulative sum
   */
  float ComputeCumulativeSums(bool recompute = false);

  /**
   * @brief Extract interest points that are local maxima of the detection score, then compute the cumulative sum 
   *        of the detection probability map for this part
   *
   * Assumes Detect() has already been called.  This is used for randomly sampling part locations
   *
   * @param recompute If true, always recomputes the cumulative sums, even if they already exist from a previous call
   * @return The cumulative sum
   */
  float ComputeCumulativeSumsNMS(bool recompute = false);

  /**
   * @brief Draw a random part location sample according to the current probability distribution
   * 
   * Part location is drawn according to the current probability distribution over locations of this part
   * \f$ p(\theta_p|x,U_t) \f$.  Calls ComputeCumulativeSums() if necessary
   *
   * @return Returns a random part location
   */
  PartLocation DrawRandomPartLocation(int numTries=5);

  /**
   * @brief Draw a random part location sample according to the current probability distribution, restricted to interest
   * points that are local maxima of the part detection score
   * 
   * Part location is drawn according to the current probability distribution over locations of this part
   * \f$ p(\theta_p|x,U_t) \f$.  Calls ComputeCumulativeSums() if necessary
   *
   * @return Returns a random part location
   */
  PartLocation DrawRandomPartLocationNMS(int numTries=5);

  void DrawRandomPartLocationsWithoutDetector(PartLocation *locs, ObjectPartInstance *parent, int pad);

  /**
   * @brief Updates the class log likelihoods using the attribute detection scores associated with this part
   *
   * @param classLogLikelihoods A numClasses array of class log likelihoods
   * @param theta_p The location of this part where we want to evaluate attribute detectors
   * @param useGamma If true, scale attribute detection scores independently
   */
  void UpdateClassLogLikelihoodAtLocation(double *classLogLikelihoods, PartLocation *theta_p, bool useGamma=true);

  /**
   * @brief Get the parameter to normalize a log likeihood parameter such that the sum over pixel locations is 1
   *
   * A detection score \f$ f(\theta_p,x) \f$ is converted to a probability as \f[ p(\theta_p|x) = \exp\{\gamma_p f(\theta_p,x) + \delta\} \f]
   * Whereas the parameter \f$ \gamma_p \f$ is a learned parameter associated with the model, \f$ \delta \f$ is a per image constant
   * that is computed numerically by summing the probability maps using ComputeCumulativeSums().  Thus it is assumed ComputeCumulativeSums()
   * has been called prior to calling this function
   *
   * @return The parameter \f$ \delta \f$ 
   */ 
  float GetDelta() { return delta; }

  /**
   * @brief Get the lower bound on the detection score, which is used to avoid assigning any pixel location too low of a probability
   *
   * The detection score is thresholded as \f$ f'(\theta_p,x)=\min(f'(\theta_p,x),l) \f$ for lower bound l
   * @return The lower bound l
   */
  float GetLowerBound() { return lower_bound; }

  /**
   * @brief Get the maximum score for this part (computed during ComputeCumulativeSums())
   */
  float GetMaxScore() { return maxScore; }

  /**
   * @brief Assign a value to the latent variable for clicking on this part 
   *
   * When this is called, the click variable becomes observed.  This causes the probability map of this
   * part to change (e.g. a new term is factored into the unary potential for this part)
   *
   * @param l The location where the user clicked
   */
  void SetClickPoint(PartLocation *l, bool useMultiObject=false);

  /**
   * @brief Assign a value to the latent variable for the location of this part 
   *
   * When this is called, the probability map of this part is changed (e.g. a new term is factored 
   * into the unary potential for this part)
   *
   * @param l The location where the user clicked
   * @param detect If true, update the detection maps
   */
  void SetLocation(PartLocation *l, bool detect=true, bool useMultiObject=false);


  /**
   * @brief Set the answer of an attribute question associated with this part
   *
   * Setting the attribute answer should change the distribution over the location of this part to favor
   * locations where an attribute detector fires.  This change can be factored into the unary potential for this part
   *
   * @param attribute_ind The index of the attribute in classes->attributes
   * @param a The user supplied answer to an attribute question
   */
  void SetAttributeAnswer(int attribute_ind, struct _AttributeAnswer *a);

  /**
   * @brief Propagate changes to the unary potential of this part to other parts in the tree structure
   *
   * If the unary potential of a part changes (e.g. due to SetClickPoint() or SetAttributeAnswer()), we
   * can propagate those changes to other neighboring parts efficiently using this function, while avoiding
   * having to do most of the expensive operations
   */
  void PropagateUnaryMapChanges(ObjectPartInstance *pre=NULL);

  /**
   * @brief Add a part localization loss term into the unary potential for this part
   *
   * This is used when training using SVM struct.  It allows us to do inference over part locations while
   * factoring in a loss term.  The loss used is a soft loss based on the percent intersection of the 
   * predicted bounding box and ground truth bounding box (the loss used in the Blaschko CVPR paper)
   *
   * @param gt_loc The ground truth location of the part
   * @param l The maximum loss (loss associated with a predicted bounding box that doesn't intersect the true one)
   *
   */
  void SetLoss(PartLocation *gt_loc, float l=1);

  /**
   * @brief Get the loss associated with predicting a part location pred_loc when the true location is gt_loc
   *
   * Assumes SetLoss() has already been called
   *
   * @param pred_loc The predicted location of the part
   *
   */
  float GetLoss(PartLocation *pred_loc);

  /**
   * @brief Get the features associated with the spatial model of this part
   * @param f A vector of weights that is set by this function
   * @param loc If non-NULL, override the part location for this part in locs
   * @return The number of features extracted
   */
  int GetStaticFeatures(float *f, PartLocation *loc=NULL);

  /**
   * @brief Update (add in) the features associated with the appearance of this part
   *
   * We add features (instead of set them) because multiple parts might share the same appearance model
   *
   * @param f A vector of weights that is set by this function
   * @param locs A classes->numParts vector of part locations
   * @param loc If non-NULL, override the part location for this part in locs
   * @param poseOffsets a classes->numPoses array of offsets into f, defining which components of f are used for each pose
   */
  void UpdateSpatialFeatures(float *f, PartLocation *locs, int *spatialOffsets, PartLocation *loc=NULL);

  void UpdatePoseFeatures(float *f, PartLocation *locs, int *poseOffsets, PartLocation *loc=NULL);

  /**
   * @brief Set custom detection weights for this part instance, which override those of the part models
   * @param w An array of detection weights, with the same ordering as Model()->GetStaticWeights()
   */
  void SetCustomStaticWeights(float *w);

  /**
   * @brief Get the number of child parts of this part
   * @return the number of child parts of this part
   */
  int NumParts() { return model->NumParts(); }

  /**
   * @brief Get the index of a pose in this part's list of poses
   * @param p The pose we are searching for
   * @return The pose index
   */
  int PoseInd(ObjectPoseInstance *p) { for(int i = 0; i < model->NumPoses(); i++) { if(poses[i] == p) return i; } return -1; }

  /**
   * @brief Get the index of a part in this part's list of child parts 
   * @param p The part we are searching for
   * @return The part index
   */
  int PartInd(ObjectPartInstance *p) { for(int i = 0; i <= NumParts(); i++) { if(parts[i] == p) return i; } return -1; }


  /**
   * @brief Set whether or not to use loss during inference
   * @param u If true, do inference using a loss term
   */
  void UseLoss(bool u);

  /**
   * @brief Resolve pointers to other objects, parts, or attributes
   *
   * Typically, this is called after all classes, parts, and attribute definitions have been loaded
   */
  void ResolveLinks();
  void ResolveParentLinks();

  /**
   * @brief Helper routine for debugging/inspecting pose detection results.  This is useful to verify that inference is working correctly.
   * @param w A classes->NumWeights() array, which should have been extracted using Model()->GetWeights()
   * @param f A classes->NumWeights() array, which should have been extracted using GetFeatures()
   * @param poseOffsets a classes->NumPoses() array of offsets into w for the appearance weights of each pose
   * @param locs a numParts array of part locations, which should be the same locations passed to GetFeatures()
   * @param debug_scores If true, print out debugging information using cached detection scores for each part, pose, and spatial model, and compare them to what is expected from <w,f>
   * @param print_weights If true, print out weights and features in w and f
   * @param f_gt If non-null, a classes->NumWeights() array for ground truth features, which should have been extracted using GetFeatures()
   */
  void Debug(float *w, float *f, int *poseOffsets, int *spatialOffsets, PartLocation *locs, bool debug_scores, bool print_weights, float *f_gt=NULL);
  
  /**
   * @brief Helper routine for debugging/inspecting pose detection results.  This is useful to verify that inference is working correctly.
   * @param w A classes->NumWeights() array, which should have been extracted using Model()->GetWeights()
   * @param f A classes->NumWeights() array, which should have been extracted using GetFeatures()
   * @param locs a numParts array of part locations, which should be the same locations passed to GetFeatures()
   * @param debug_scores If true, print out debugging information using cached detection scores for each part, pose, and spatial model, and compare them to what is expected from <w,f>
   * @param print_weights If true, print out weights and features in w and f
   * @param f_gt If non-null, a classes->NumWeights() array for ground truth features, which should have been extracted using GetFeatures()
   * @return The number of sptial weights
   */
  int DebugStatic(float *w, float *f, PartLocation *locs, bool debug_scores, bool print_weights, float *f_gt=NULL);

  /**
   * @brief Zero out detection scores in a box of size (loc->width*s,loc->height*s) around loc->x,loc->y.  
   *   Used for a greedy non-max-suppression-like algorithm
   * @param loc Location of where to zero out detection scores
   * @param s Fraction of the bounding box in which to zero out
   */
  void ZeroScoreMap(PartLocation loc, float s);


  /**
   * @brief Get the bounding box around the highest scoring part location
   * @param rect Store the bounding box into this parameter
   * @param locs An array of part locations (of size classes->numParts)
   * @param includeThisPart If true include this part when extracting the bounding box (otherwise extracts the bounding box over
   * just subparts)
   */
  void GetBoundingBox(CvRect *rect, PartLocation *locs, bool includeThisPart=false, float width_scale=.5f);

  /**
   * @brief Get an array of bounding boxes and their associated score for every possible position of this part
   * @param includeThisPart If true include this part when extracting the bounding box (otherwise extracts the bounding box over
   * just subparts)
   * @param num a pointer to the number of bounding boxes (widthXheight), returned by this function
   * @return A widthXheight array of bounding boxes
   */
  CvRectScore *GetBoundingBoxes(int *num, bool includeThisPart=true, bool storePartLocations=false, bool computeBoundingFromParts=true);

  void PredictLatentValues(PartLocation *l, bool useUnaryExtra=true);

   /**
   * @brief Extract the cached score at a given location (computed using dynamic programming)
   * @param l The part location in the image
   */
  float ScoreAt(PartLocation *l);

  /**
   * @brief Get the number of standard deviations a predicted part location differs from the true location, where
   * the standard deviation is measured from user click statistics for each part (the click part)
   *
   * @param pred_loc The predicted location of the part
   * @param maxDev The maximum loss (number of standard deviations).  This is set when the visibility of a part is
   * predicted incorrectly
   */
  float GetUserLoss(PartLocation *pred_loc, float maxDev=5);

  float GetStaticScore(int x, int y, int scale, int rot);
  void AddStaticFeaturesToDetectionMap(IplImage *response, int scale, int rot);
  void SanityCheckDynamicProgramming(PartLocation *gt_locs);
private:
  void FreeResponses();
};



#endif

