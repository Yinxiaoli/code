/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#ifndef __POSE_H
#define __POSE_H



/// @cond
#define CIRCLE_RADIUS 9
#define CIRCLE_RADIUS2 7
#define CIRCLE_RADIUS3 9
/// @endcond

#define MAX_DEVIATIONS 5
#define NUM_DEVIATIONS_BEFORE_INCORRECT 1.5

/**
 * @file pose.h
 * @brief Definition of pose models and routines for detecting attributes in images
 */

#include "feature.h"
#include "part.h"

class ObjectPart;
class ObjectPose;
class Attribute;
class ObjectPartInstance;
class ObjectPoseInstance;
class AttributeInstance;
class FeatureOptions;
class ImageProcess;
class Classes;
class ObjectPartPoseTransitionInstance;
class MultipleUserResponses;

/**
 * @class ObjectPose
 * 
 * @brief A particular view or configuration of an object part, e.g. "beak side view left", "beak frontal view".  A pose is associated with a sliding window detector.
 *
 * A part can have multiple possible poses. We allow per part pose changes, e.g. the body could be in frontal view while the head
 * is in side view
 */
class ObjectPose {
  char *name; /**< The name of this pose */
  int id; /**< The id of this pose */

  bool isClick; /**< For asking the user to click the location of a part, the click location can be treated as a latent variable */

  Attribute *appearanceModel; /**<  An appearance model used for detecting this part/pose. */
  Classes *classes;

  char *visualization_image;

  bool isNotVisible;
  ObjectPose *flipped;
  char *flippedName;
  bool isFlipped;

  IplImage *segmentation;
  char *segmentation_name;


 public:
  /**
   * @brief Constructor
   * @param name The name of the pose
   * @param isClick true if this pose corresponds to a latent variable of where we think the user will click (instead of an actual pose)
   */
  ObjectPose(const char *name=NULL,  bool isClick=false);

  /**
   * @brief Destructor
   */
  ~ObjectPose();

  /**
   * @brief Save a definition of this pose to a JSON object
   * 
   * @return A JSON encoding of this pose
   */
  Json::Value Save();

  /**
   * @brief Load an ObjectPose object from a JSON encoding
   * @param root A JSON encoding of this pose
   * @return True if successful
   */
  bool Load(const Json::Value &root);

  /**
   * @brief Check if this is a variable for where we think a user will click instead of an actual pose
   * @return true if this is a click variable
   */
  bool IsClick() { return isClick; }

  bool IsFlipped() { return isFlipped; }
  ObjectPose *GetFlipped() { return flipped; }
  Classes *GetClasses() { return classes; }
  void SetClasses(Classes *c) { classes = c; }

  void SetSegmentation(const char *s) { 
    if(segmentation_name) free(segmentation_name);
    if(segmentation) cvReleaseImage(&segmentation);
    segmentation_name = StringCopy(s); 
  }

  IplImage *GetSegmentationMask() { 
    if(!segmentation_name) return NULL;
    else if(!segmentation) segmentation = cvLoadImage(segmentation_name, CV_LOAD_IMAGE_GRAYSCALE);
    return segmentation;
  }

  /**
   * @brief Get the id of this pose
   * @return The id of this pose
   */
  int Id() { return id; }

  /**
   * @brief Get the name of this pose
   * @return The name of this pose
   */
  char *Name() { return name; }

  void SetName(const char *n) { if(name) { StringFree(name); } name = StringCopy(n); }

  /**
   * @brief Get the model defining the appearance/detection of this pose
   * @return The appearance model
   */
  Attribute *Appearance() { return appearanceModel; }
  
  /**
   * @brief Set the model defining the appearance/detection of this pose
   * @param a The appearance model
   */
  void SetAppearanceModel(Attribute *a) { appearanceModel = a; }

  /**
   * @brief Get an array of all weights for the appearance parameters of the detector
   * @param w the array in which to store the extracted weights
   * @return The number of extracted weights
   */
  int GetWeights(float *w);

  /**
   * @brief Set the weights for the appearance parameters of the detector
   * @param w the array from which to extract the weights
   */
  void SetWeights(float *w);

  /**
   * @brief Get the number of weights for the appearance parameters of the detector
   * @return The number of weights
   */
  int NumWeights();

  /**
   * @brief Get an upper bound on the squared l2 norm of the feature space
   */
  float MaxFeatureSumSqr();

  /**
   * @brief Check if this is a special pose corresponding to a part not being visible
   * @return True if this pose is not visible
   */
  bool IsNotVisible() { return isNotVisible; }

  /**
   * @brief Get the image name of visualization
   */
  const char *GetVisualizationImageName() { return visualization_image; }

  ObjectPose *FlippedCopy();

private:
  bool ResolveLinks(Classes *classes);
  void SetId(int i) { id = i; }

  friend class Classes;

  
  // deprecated
  char *LoadFromString(char *str, char **ptrStart);
  char *ToString(char *str);
  bool ResolveLinksOld(Classes *c);

};



/**
 * @class ObjectPoseInstance
 * 
 * @brief An instance of an object in a particular pose in a particular image
 *
 */
class ObjectPoseInstance {
  ObjectPose *pose;  /**< The model for this pose instance */
  AttributeInstance *appearance;  /**< Defines the appearance model for detection of this pose */
  ObjectPartInstance *part;  /**< The part this pose instance refers to */

  IplImage ***responses;  /**< A numScalesXnumOrientations array of detection response maps */
  IplImage ***unaryExtra;  /**< A numScalesXnumOrientations array of miscellaneous scores to add into the unary potential of this pose */
  IplImage ****childPartScores;  /**< A (namParts+1)XnumScalesXnumOrientations array of response scores of all neighboring parts, which integrates out the location of each adjacent part for all possible locations of the parent part */
  IplImage ****childPartBestPoseIndices;  /**< A (namParts+1)XnumScalesXnumOrientations array of indices defining the optimal location of each child part for every location of this part (used for reading out the optimal solution at the end of dynamic programming) */

  IplImage ***losses;  /**< A numScalesXnumOrientations array of loss maps with respect to the ground truth position of this part (used for loss-sensitive training of part detectors) */
  IplImage *maxResponse;  /**< A map of the maximum response, which takes the max over all scales and orientation in 'responses' */
  IplImage *maxResponseInds;  /**< A map of scale and orientation indices for each pixel in 'maxResponse' */
  int best_x;  /**< The x-location of the pixel with maximum detection response */
  int best_y;  /**< The y-location of the pixel with maximum detection response */
  int best_scale;  /**< The scale of the maximum detection response */
  int best_rot;  /**< The orientation of the maximum detection response */
  float best_response;  /**< The maximum detection response */

  bool useLoss;  /**< If true, does inference with respect to the loss term defined in 'losses' */

  double minScale, maxScale;

  PartLocation ground_truth_loc;
  float max_loss;
  IplImage *loss_buff;

  
  ImageProcess *process; /**< pointer to the processing object used to compute responses */

  /**
   * A numPartsXnumChildPartPoseTransitions[i] array of all possible part/pose spatial transition
   * scores where the parent part/pose is this pose
   */
  ObjectPartPoseTransitionInstance ***childPartPoseTransitions; /**< A (part->numParts+1)XnumChildPartPoseTransitions[i] array of sptial models for each allowable pose of each neighboring part */
  int *numChildPartPoseTransitions; /**< A (part->numParts+1) array of the number of possible poses of each neighboring part */
  bool **dontDeleteChildPartPoseTransitions;

 public:

  /**
   * @brief Constructor
   * @param pose The model for this pose instance
   * @param part The part this pose instance refers to
   * @param p A pointer to the processing object used to compute responses
   */
  ObjectPoseInstance(ObjectPose *pose, ObjectPartInstance *part, ImageProcess *p);

  /**
   * @brief Destructor
   */
  ~ObjectPoseInstance();


  /**
   * @brief Get the part isntance this pose is associated with
   * @return The part instance this pose is associated with
   */
  ObjectPartInstance *Part() { return part; }

  /**
   * @brief Get the pose model for this pose instance
   * @return The pose model for this pose instance
   */
  ObjectPose *Model() { return pose; }

  /**
   * @brief Get the appearance model for detection of this pose
   * @return The appearance model of this pose
   */
  AttributeInstance *Appearance() { return appearance; }

  /**
   * @brief Evaluate a pose detector, which combines the appearance detector of this pose with the detection score of all child parts
   * @param parent If non-NULL, when doing inference acts as the parent of this node (the score integrates out all neighboring parts except parent).  This parameter can be set to any valid neighbor of this part.
   * @return A numScalesXnumOrientations array of detection score maps
   */
  IplImage ***Detect(ObjectPartInstance *parent);

  /**
   * @brief Get the detection response map at a particular scale/orientation.  Assumes Detect() has already been called
   * @param s The scale of interest
   * @param r The orientation of interest
   * @param part If non-NULL, treats this parameter as the parent of this part (the score combines the scores of all neighboring parts except parent).  This parameter can be set to any valid neighbor of this part).  
   * @param tmp A ptr to the return detection response map image.  This is set only if the response is dynamically allocated.
   * @return The detection response map
   */
  IplImage *GetResponse(int s, int r, ObjectPartInstance *part=NULL, IplImage **tmp=NULL);

  /**
   * @brief Draw a bounding box visualization of this pose
   * @param img The image we want to draw into
   * @param l The location of the part in the image 
   * @param color The color used to draw the circle for this part
   * @param color2 The color used to draw the outer circle for this part
   * @param color3 The color used to draw the text label
   * @param str If non-null draw this string along with the bounding box
   * @param key If non-null draw this string inside the point label
   * @param labelPoint If true, draw a circle depicting the center of the part
   * @param labelRect If true, draw a rectangle for the bounding box of the part
   * @param zoom Scale the numbers in l by this factor before drawing into the image
   */
  void Draw(IplImage *img, PartLocation *l, CvScalar color=CV_RGB(0,0,200), CvScalar color2=CV_RGB(0,0,200), CvScalar color3=CV_RGB(0,0,200), const char *str=NULL, 
	    const char *key=NULL, bool labelPoint=false, bool labelRect=true, float zoom=1);

  /**
   * @brief Clears all cached memory associated with detector responses.  If buffers are left uncleared, calls to Detect() will use cached responses
   * @param clearAppearance If true, clears memory associated with the appearance detection score of this pose (otherwise, just clears memory associated
   * @param clearBuffers If true, clears memory associated with buffers for dynamic programming
   * @param clearResponses If true, clears memory associated with detection scores
   *  with doing inference over neighboring parts)
   */
  void Clear(bool clearAppearance=true, bool clearBuffers=true, bool clearResponses=true, bool clearLoss=true);

  /**
   * @brief Add in a new term into the unary score of this pose (normally this is just the appearance detection score)
   * @param add A numScalesXnumOrientations array of per pixel unary scores
   */
  void AddToUnaryExtra(IplImage ***add, bool useMax=false, float scalar=0);

  /**
   * @brief Clear all extra terms added into the unary score of this pose
   */
  void FreeUnaryExtra();

  /**
   * @brief Get the term added to the unary score of this pose, if applicable
   */
  IplImage ***GetUnaryExtra() { return unaryExtra; }

  /**
   * @brief Compute a loss score map of the loss at every possible pixel location with respect to the ground truth position of this part
   *
   * The loss used is the normalized intersection of the ground truth and predicted bounding boxes (intersection divided by union)
   *
   * @param gt_loc The ground truth location of the part
   * @param maxLoss The maximum loss 
   */
  void SetLoss(PartLocation *gt_loc, float maxLoss);

  /**
   * @brief Get the loss of a particular pixel location with respect to the ground truth position of this part.  Assumes SetLoss() has been called beforehand
   *
   * @param pred_loc The predicted location of the part
   * @return The loss of pred_loc with respect to the ground truth location
   */
  float GetLoss(PartLocation *pred_loc);


  /**
   * @brief Set whether or not the loss term is used during inference
   *
   * @param u If true, does inference with respect to the loss term
   */
  void UseLoss(bool u) { useLoss = u; }

  /**
   * @brief Get the features associated with the appearance of this pose
   * @param f A vector of weights that is set by this function
   * @param locs A classes->numParts vector of part locations
   * @param loc If non-NULL, override the part location for this part in locs
   * @return The number of features extracted
   */
  int GetFeatures(float *f, PartLocation *locs, PartLocation *loc=NULL);  

  /**
   * @brief Set custom detection weights for this pose instance, which override those of the pose models
   * @param w An array of detection weights, with the same ordering as Model()->GetWeights()
   */
  void SetCustomWeights(float *w, int *poseOffsets, int *spatialOffsets);

  /**
   * @brief Get the features associated with the spatial model of this pose
   * @param f A vector of weights that is set by this function
   * @param locs A classes->numParts vector of part locations
   * @param loc If non-NULL, override the part location for this part in locs
   */
  void UpdateTransitionFeatures(float *f, PartLocation *locs, PartLocation *loc=NULL, int *offsets=NULL);

  /**
   * @brief Save a visualization of a set of probability maps for this pose.  Saves to dir/fname_poseName_scale_rot.png for all scale/rotations
   * @param fname The base filename of all files stored
   * @param dir The directory into which to store files
   * @param html If non-NULL, generates an html string for viewing the probability maps
   * @param mergeResults If true, merge probability maps across different scales, orientations
   */
  IplImage *SaveProbabilityMaps(const char *fname=NULL, const char *dir=NULL, char *html=NULL, bool mergeResults=true);

  /**
   * @brief Invalidates the score with respect to a neighboring part, such that if Detect() is called again, it will reintegrate out that neighboring part again.  This is used to efficiently update the inference result when the unary score of a neighboring part changes
   * @param neighbor The neighboring part
   */
  void InvalidateChildScores(ObjectPartInstance *neighbor);
 
  /**
   * @brief Extract the maximum likelihood part locations for all parts in the object
   *
   * Assumes Detect() has already been called on the root part. 
   *
   * @param locs an array of size numParts, all values of which are set using this function
   * @param l if non-NULL, extract the solution given that this part must be in possition l (otherwise use the maximum scoring position of this part)
   * @param par ExtractPartLocation() recursively descends through all neighboring parts.  If par is non-NULL, we avoid descending through par
   */
  void ExtractPartLocations(PartLocation *locs, PartLocation *l, ObjectPartInstance *par);

  /**
   * @brief Get the set of spatial models defining the spatial relationship between this part
   * and all possible neighboring parts and poses.
   * @param numChildPoses A pointer to a part->numParts array defining the number of allowable child poses for every possible
   *  child part.  The pointer is set by this function
   * @return A part->numPartsX(*numChildPoses)[i] array of part/pose transition models
   */
  ObjectPartPoseTransitionInstance ***GetPosePartTransitions(int **numChildPoses) { *numChildPoses = numChildPartPoseTransitions; return childPartPoseTransitions; }

  /**
   * @brief get the number of spatial transitions for this pose
   * @return a part->NumParts() array, where each entry has the number of spatial transitions going to a particular child part
   */
  int *NumChildPartPoseTransitions() { return numChildPartPoseTransitions; }

  /**
   * @brief Initialize the locations of the children of this part to the default offsets relative to this part
   * @param locs An array of numParts part locations
   */ 
  void SetPartLocationsAtIdealOffset(PartLocation *locs);


  /**
   * @brief Resolve pointers to other objects, parts, or attributes
   *
   * Typically, this is called after all classes, parts, and attribute definitions have been loaded
   */
  void ResolveLinks();
  void ResolveParentLinks();

  /**
   * @brief Get a map of the maximum response (detection score) for every pixel, which maximizes over all scales/orientations
   * @param parent If non-NULL, the score with respect to the parent is subtracted out
   * @param tmp If the return value was dynamically allocated, *tmp is set
   * @return The image of maximum response
   */
  IplImage *GetMaxResponse(ObjectPartInstance *parent, IplImage **tmp);

  /**
   * @brief Get the scale and orientation with maximum response for a particular pixel location (assumes GetMaxResponse() has already been called)
   * @param x The x-pixel location
   * @param y The y-pixel location
   * @param scale The scale index (set by this function)
   * @param rot The orientation index (set by this function)
   */
  void GetScaleRot(int x, int y, int *scale, int *rot);

  /**
   * @brief Associate a latent variable for clicking on this part 
   * @param l The click ObjectPartInstance
   */
  void SetClickPoint(PartLocation *l, bool useMultiObject=false);

  /**
   * @brief Assign a value to the latent variable for the location of this part 
   *
   * When this is called, the probability map of this part is changed (e.g. a new term is factored 
   * into the unary potential for this part)
   *
   * @param l The location where the user clicked
   */
  void SetLocation(PartLocation *l, bool useMultiObject=false);

  /**
   * @brief Check if this is a special pose corresponding to a part not being visible
   * @return True if this pose is not visible
   */
  bool IsNotVisible() { return pose->IsNotVisible(); }
 
  /**
   * @brief Helper routine for debugging/inspecting pose detection results.  This is useful to verify that inference is working correctly.
   * @param w A classes->NumWeights() array, which should have been extracted using Model()->GetWeights()
   * @param f A classes->NumWeights() array, which should have been extracted using GetFeatures()
   * @param locs a numParts array of part locations, which should be the same locations passed to GetFeatures()
   * @param debug_scores If true, print out debugging information using cached detection scores for each part, pose, and spatial model, and compare them to what is expected from <w,f>
   * @param print_weights If true, print out weights and features in w and f
   * @param f_gt If non-null, a classes->NumWeights() array for ground truth features, which should have been extracted using GetFeatures()
   * @return The number of spatial weights
   */ 
  void DebugSpatial(float *w, float *f, int *spatial_offsets, PartLocation *locs, bool debug_scores, bool print_weights, float *f_gt=NULL);

  /**
   * @brief Helper routine for debugging/inspecting pose detection results.  This is useful to verify that inference is working correctly.
   * @param w A classes->NumWeights() array, which should have been extracted using Model()->GetWeights()
   * @param f A classes->NumWeights() array, which should have been extracted using GetFeatures()
   * @param loc The location of this pose's part. The same location passed to GetFeatures()
   * @param debug_scores If true, print out debugging information using cached detection scores for each part, pose, and spatial model, and compare them to what is expected from <w,f>
   * @param print_weights If true, print out weights and features in w and f
   * @param f_gt If non-null, a classes->NumWeights() array for ground truth features, which should have been extracted using GetFeatures()
   * @return The number of sptial weights
   */
  int DebugAppearance(float *w, float *f, PartLocation *loc, bool debug_scores, bool print_weights, float *f_gt=NULL);

  /**
   * @brief Draw a visualization of the spatial model portion of a part model. 
   * @param img The image to be updated by this call
   * @param locs a numParts array of part locations
   */
  float VisualizeSpatialModel(IplImage *img, PartLocation *locs);

   /**
   * @brief Extract the cached score at a given location (computed using dynamic programming)
   * @param l The part location in the image
   */
  float ScoreAt(PartLocation *l);


  /**
   * @brief Special handling for poses in the config file but are never used in the ground truth part annotations 
   */
  bool IsInvalid();
  
  void ComputeLoss(PartLocation *gt_loc, float maxLoss, bool useMultiObject=false);
  void ClearLoss();
  void RestrictLocationsToAgreeWithAtLeastOneUser(MultipleUserResponses *users, int radius);
  PartLocation GetBestLoc();

  void SanityCheckDynamicProgramming(PartLocation *gt_locs);

private: 
  void FreePoseScaleRotArray(IplImage ****best_offsets);
  void AllocateCacheTables();
};



void MaxWithIndex(IplImage *currMax, IplImage *img, IplImage *currMaxInds, int newInd);



#endif

