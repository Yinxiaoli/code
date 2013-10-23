/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#ifndef __IMAGE_PROCESS_H__
#define __IMAGE_PROCESS_H__

#include "feature.h"
#include "classes.h"
#include "class.h"


struct _PartLocation;
class ObjectPart;
class ObjectPose;
class Attribute;
class ObjectPartInstance;
class ObjectPoseInstance;
class AttributeInstance;
class FeatureOptions;
class ImageProcess;
class PartLocation;
class MultiObjectLabel;
class MultipleUserResponses;

/**
 * @file imageProcess.h
 * @brief Routines to compute and cache features and part and attribute detection responses for a given image
 */


/**
 * @enum InferenceMethod
 *
 * @brief A method used for inference (e.g. maximum likelihood or MAP)
 */
typedef enum {
  IM_MAXIMUM_LIKELIHOOD,  /**< find the maximum likelihood location of each part (uses distance transform operation) */
  IM_POSTERIOR            /**< compute the posterior probabilities of each part location (uses convolution operation) */
} InferenceMethod;

/**
 * @class ImageProcess
 *
 * @brief Class to store precomputed features and part and attribute detection responses for a given image
 */
class ImageProcess {
  bool computeBidirectional;  /**< If true, when running Detect(), compute the maximum likelihood location of all parts conditioned on setting any one part in any one location  */
  bool computeClickParts; /**< If true, compute the log likelihoods on where we think the user will click */
  bool useLoss; /**< If true, Run detection inference with a loss term incorporated */
  

  FeatureOptions *feat; /**< Data structure to store cached features or feature computation options*/


  ObjectPartInstance **partInstances; /**< Storage of all instance detection info for all possible objects or object parts in this image */

  AttributeInstance **attributeInstances; /**< Storage of all instance detection info for all possible attributes in this image */

  ObjectPartInstance **partClickInstances; /**< Storage of log likelihood scores for where we think the user will click on a part */

  IplImage ****attribute_scores;  /**< A cache of the detection scores of each attribute (size numAttributesXnumScalesXnumOrientatons) */
  IplImage ****part_scores;  /**< A cache of the detection scores of each part (size numPartsXnumScalesXnumOrientatons) */
  IplImage ****part_click_scores;  /**< A cache of the detection scores of each click part (size numPartsXnumScalesXnumOrientatons) */

  Classes *classes; /**< Defines all possible classes, parts, and attribtues */

  ScaleOrientationMethod scaleOrientationMethod; /**< Defines an acceleration method for dealing with parts of different scales/orientations */
  InferenceMethod inferenceMethod; /**< A method of inerence (maximum likelihood or MAP) */

  char *imgName;  /**< The file name of the image */

  int nthreads;  /**< The number of threads to use for within image parallelization */
  bool use_within_image_parallelization;  /**< If true, parallelize operations over this image */

  float *custom_weights;  /**< can be used to override model weights */

 public:

  /**
   * @brief Get the number of threads to use for parallel processing of this image
   */
  int NumThreads() { return use_within_image_parallelization ? nthreads : 1; }
  
  /**
   * @brief Set whether or not we should use parallel processing for this particular image
   * @param p true if we want to use parallel processing
   */
  void UseParallelization(bool p) { use_within_image_parallelization = p; }

  /**
   * @brief Constructor
   * @param classes Defines all possible classes, parts, and attributes
   * @param imgName The name of the image being processed
   * @param m A method of inerence (maximum likelihood or MAP)
   * @param bidirectional If true, when running Detect(), compute the maximum likelihood location of all parts conditioned on setting any one part in any one location
   * @param computeClickParts If true, when running Detect(), compute the log likelihoods on where we think the user will click
   * @param parallelize If true, parallelizes operations on this particular image.  Typically, when classifying an image interactively this should be true (for maximum speed), but when processing an entire dataset in bulk this should be false (in this case parallelization should occur over multiple images at the same time)
   */
  ImageProcess(Classes *classes, const char *imgName, 
	       InferenceMethod m=IM_MAXIMUM_LIKELIHOOD, bool bidirectional=false, bool computeClickParts=false, bool parallelize=true);

  /**
   * @brief Destructor
   */
  ~ImageProcess();

  /**
   * @brief Get the file name of the image
   * @return The file name of the image
   */
  char *ImageName() { return imgName; }

  /**
   * @brief Get the image
   * @return The image
   */
  IplImage *Image() { return feat->GetImage(); }

  /**
   * @brief Get the acceleration method for dealing with parts of different scales/orientations
   * @return The acceleration method for dealing with parts of different scales/orientations
   */
  ScaleOrientationMethod GetScaleOrientationMethod() { return scaleOrientationMethod; }

  /**
   * @brief Get the method of inference (maximum likelihood or MAP)
   * @return The method of inerence (maximum likelihood or MAP)
   */
  InferenceMethod GetInferenceMethod() { return inferenceMethod; }

  /**
   * @brief Extract features for all part and attribute detectors
   * @param feat An array into which we store extracted features
   * @param locs An array of classes->numParts part locations (defines where we extract features)
   * @param has_attribute An array of classes->numAttributes of attribute memberships (should have values between -1 and 1)
   * @param getPartFeatures If true, extract part detection features
   * @param getAttributeFeatures If true, extract attribute features
   */
  int GetFeatures(double *feat, PartLocation *locs, float *has_attribute, bool getPartFeatures, bool getAttributeFeatures);

  /**
   * @brief Extract features for all part and attribute detectors
   * @param feat An array into which we store extracted features
   * @param locs An array of classes->numParts part locations (defines where we extract features)
   * @param has_attribute An array of classes->numAttributes of attribute memberships (should have values between -1 and 1)
   * @param getPartFeatures If true, extract part detection features
   * @param getAttributeFeatures If true, extract attribute features
   * @return The number of extracted features
   */
  int GetFeatures(float *feat, PartLocation *locs, float *has_attribute, bool getPartFeatures, bool getAttributeFeatures);

  int UpdatePartFeatures(float *f, PartLocation *locs);
  void SetMultiObjectLoss(MultiObjectLabel *y, double *partLosses);
  float GetMultiObjectLoss(MultiObjectLabel *y);


  /**
   * @brief Set custom detection weights for this image, which override those of the part models.  
   * @param w An array of detection weights, with the same ordering as Model()->GetWeights()
   * @param setPartWeights If true, set part detection weights
   * @param setAttributeWeights If true, set attribute weights
   */
  void SetCustomWeights(float *w, bool setPartWeights, bool setAttributeWeights);

  /**
   * @brief Get custom weights that override the weights for the detection model
   * @return A pointer to a vector of custom weights, or NULL of no such weights exist
   */
  float *GetCustomWeights() { return custom_weights; }

  /**
   * @brief Run inference to detect parts in the image
   * @return The maximum detection response
   */
  float Detect();

  /**
   * @brief Evaluate all attribute detectors
   */
  void DetectAttributes();

  /**
   * @brief Evaluate all part detectors
   * @param secondPass If true, we are calling DetectParts() a second time, this time propagating detection scores from the root of the tree down to the children
   */
  void DetectParts(bool secondPass);

  /**
   * @brief Compute log likelihoods of where we think the user will click
   */
  void DetectClickParts();

  /**
   * @brief Accessor for computeClickParts. If it is true, we compute the log likelihoods on where we think the user will click when Detect() is called
   */
  bool ComputeClickParts() { return computeClickParts; }

  /**
   * @brief Set computeClickParts. If it is true, we compute the log likelihoods on where we think the user will click when Detect() is called
   * @param b value to set computeClickParts
   */
  void SetComputeClickParts(bool b) { computeClickParts = b; }

  /**
   * @brief If true, this image processor is set to run dynamic programming in both directions up and down the part tree,
   * which is used to speedup computations for interactive labeling 
   */
  bool IsBidirectional() { return computeBidirectional; }

  /**
   * @brief Set isBidirectional.  If true, this image processor is set to run dynamic programming in both directions up and down the part tree, which is used to speedup computations for interactive labeling 
   * @param b the value to set isBidirectional
   */
  void SetBidirectional(bool b) { computeBidirectional = b; }

  /**
   * @brief Set isMultithreaded.  If true, use within image parallelization when running detection on a particular image (usually we don't use this)
   * @param b the value to set isMultithreaded
   */
  void SetMultithreaded(bool b) { use_within_image_parallelization = b; }

  /**
   * @brief Get a part instance
   * @param id The id of the part we want to get
   * @return The part
   */
  ObjectPartInstance *GetPartInst(int id) { return partInstances[id]; }

  /**
   * @brief Get a attribute instance
   * @param id The id of the attribute we want to get
   * @return The attribute
   */
  AttributeInstance *GetAttributeInst(int id) { return attributeInstances[id]; }

  /**
   * @brief Get a click part instance
   * @param id The id of the click part we want to get
   * @return The click part
   */
  ObjectPartInstance *GetClickPartInst(int id) { return partClickInstances[id]; }

  /**
   * @brief Get the collection of feature computation objects for this image
   * @return The collection of feature computation objects for this image
   */
  FeatureOptions *Features() { return feat; }

  /**
   * @brief Get the object defining all possible classes, parts, and attributes
   * @return The object defining all possible classes, parts, and attributes
   */
  Classes *GetClasses() { return classes; }

  /**
   * @brief Clear memory for all cached detection responses and features
   * @param clearBuffers If true, clears all memory buffers associated with buffers for inference and extracting the max likelihood solution
   * @param clearFeatures If true, clears memory associated with precomputed features
   * @param clearFeaturesFull If false and clearFeatures=true, only partially clears precomputed features
   * @param clearScores If true, clears part and pose detection score maps
   */
  void Clear(bool clearBuffers=true, bool clearFeatures=true, bool clearFeaturesFull=true, bool clearScores=true);

  /**
   * @brief Add a part localization loss term, which can be used during inference
   * @param locs A numParts array of ground truth part locations
   * @param partLosses A numParts of maximum loss terms for each part
   */
  void SetLossImages(PartLocation *locs, double *partLosses=NULL);

  /**
   * @brief Set whether or not to use loss during inference
   * @param u If true, do inference using a loss term
   */
  void UseLoss(bool u);

  /**
   * @brief Get the part localization loss term.  Assumes SetLossImages() has already been called
   * @param pred_locs A numParts array of predicted part locations
   * @return The loss associated with predicting pred_locs when the true location is gt_locs
   */
  float ComputeLoss(PartLocation *pred_locs);


  /**
   * @brief Draw bounding box visualizations for a set of predicted part locations
   * @param img The image we want to draw into
   * @param locs A numParts array of part locations
   * @param color The color of the bounding boxes
   * @param labelParts Draw the name of the part along with the bounding box
   * @param mixColors Draw each part with a different color
   * @param labelPoint If true, draw a circle depicting the center of the part
   * @param labelRect If true, draw a rectangle for the bounding box of the part
   * @param showAnchors If true, highlight parts that have been user-verified in a different color
   * @param showTree If true, draw a visualization of the part tree
   * @param selectedPart The id of the part that is being moved with the mouse (for showing an interactive GUI)
   */ 
  void Draw(IplImage *img, PartLocation *locs=NULL, CvScalar color=CV_RGB(0,0,200), bool labelParts=false, bool mixColors=false, 
	    bool labelPoint=true, bool labelRect=false, bool showAnchors=true, int selectedPart=-1, bool showTree=true);

  /**
   * @brief Draw bounding box visualizations for a set of part click locations
   * @param img The image we want to draw into
   * @param color The color of the bounding boxes
   * @param labelParts Draw the name of the part along with the bounding box
   * @param mixColors Draw each part with a different color
   * @param labelPoint If true, draw a circle depicting the center of the part
   * @param labelRect If true, draw a rectangle for the bounding box of the part
   */ 
  void DrawClicks(IplImage *img, CvScalar color=CV_RGB(0,0,200), bool labelParts=false, bool mixColors=false, 
		  bool labelPoint=false, bool labelRect=true);

  /**
   * @brief Save a visualization of the current probability maps
   * @param fname A prefix of the file names to which we will store probability maps
   * @param dir The directory into which we will store probability maps
   * @param html If non-NULL, generate an HTML string for visualizing the probability maps
   * @param isClick If true, save click probability maps
   * @param mergeResults If true, merge probability maps across different scales, orientations
   * @param generateHeatMaps If true, generate images of probability maps overlayed on the original image as a
   * red heat map (results in large files)
   * @param mergePoses If true, merge probability maps across different poses
   * @param keepBigImage If true, when generating heat maps, keep the full sized image
   */
  void SaveProbabilityMaps(const char *fname, const char *dir=NULL, char *html=NULL, bool isClick=false, bool mergeResults=true, bool generateHeatMaps=false, bool mergePoses=false, bool keepBigImage=false);

  /**
   * @brief Save a visualization of the current click part probability maps
   * @param fname A prefix of the file names to which we will store probability maps
   * @param dir The directory into which we will store probability maps
   * @param html If non-NULL, generate an HTML string for visualizing the probability maps
   */
  void SaveClickProbabilityMaps(const char *fname, const char *dir, char *html);

  /**
   * @brief Compute the class log likelihoods at a particular set of part locations
   * @param classLogLikelihoods A numClasses array of extracted log likelihoods
   * @param locs A numParts array of part locations where we want to evaluate class detectors
   * @param useGamma If true, apply a correction term to convert class detection scores to probabilities
   */
  void ComputeClassLogLikelihoodsAtLocation(double *classLogLikelihoods, PartLocation *locs, bool useGamma=true);


  /**
   * @brief Extract a set of part locations for the maximum likelihood solution (ssumes Detect() has already been called)
   * @param locs An array of numParts part locations extracted by this function
   * @return If locs is null, a dynamically allocated array of numParts part locations (must be deallocated using free())
   */
  PartLocation *ExtractPartLocations(PartLocation *locs=NULL);

  /**
   * @brief Estimate class probabilities as the average over a set of sample part locations
   * @param classProbs An array of numClasses class probabilities computed by this function
   * @param locs A set of part location samples
   */
  void ComputeClassProbabilitiesFromSamples(double *classProbs, struct _PartLocationSampleSet *locs);

  /**
   * @brief Estimate class probabilities for the image as a whole
   * @param classProbs An array of numClasses class probabilities computed by this function
   */
  void ComputeClassProbabilities(double *classProbs);
  
  /**
   * @brief Estimate class probabilities at a particular set of part locations
   * @param classProbs An array of numClasses class probabilities computed by this function
   * @param locs A numParts array defining a single set of part locations
   */
  void ComputeClassProbabilitiesAtLocation(double *classProbs, PartLocation *locs);

  /**
   * @brief Extract random samples of part locations according to the current probability distribution
   * @param numSamples The number of samples to extract
   * @param root_part If non-NULL, random samples are drawn starting from the root_part's probability map
   * @param useNMS If true, consider only part locations left after non-max suppression
   */
  struct _PartLocationSampleSet *DrawRandomPartLocationSet(int numSamples, ObjectPartInstance *root_part=NULL, bool useNMS=true);

  /**
   * @brief Resolve pointers to parts, poses, and attributes
   */
  void ResolveLinks();

  /**
   * @brief Print a string representation of a set of part locations (for debug output)
   * @param locs A numParts array of part locations
   * @param str The output string
   * @return str
   */
  char *PrintPartLocations(PartLocation *locs, char *str);


  /**
   * @brief Helper routine for debugging/inspecting part detection results.  This is useful to verify that inference is working correctly.
   * @param w A classes->NumWeights() array, which should have been extracted using Classes::GetWeights()
   * @param f A classes->NumWeights() array, which should have been extracted using GetFeatures()
   * @param locs a numParts array of part locations, which should be the same locations passed to GetFeatures()
   * @param debug_scores If true, print out debugging information using cached detection scores for each part, pose, and spatial model, and compare them to what is expected from <w,f>
   * @param print_weights If true, print out weights and features in w and f
   * @param getPartFeatures If true, the weight vector w contains part detection weights
   * @param getAttributeFeatures If true, the weight vector w contains attribute detector weights
   * @param f_gt If non-ULL, a classes->NumWeights() array, which should have been extracted using GetFeatures() on ground truth part locations
   */
  void Debug(float *w, float *f, PartLocation *locs,  bool debug_scores, bool print_weights,
	     bool getPartFeatures, bool getAttributeFeatures, float *f_gt = NULL);


  /**
   * @brief Visualize features extracted with respect to a particular assignment to all part locations
   * @param locs a numParts array of part locations
   * @param fname_prefix A prefix for the file name of where to store images (an image per feature type will be generated)
   */
  void VisualizeFeatures(PartLocation *locs, const char *fname_prefix);

  /**
   * @brief Visualize the learned weights for part detectors
   * @param locs a numParts array of part locations.  The visualization is drawn on top of the specified part locations, and thus 
   *   only visualizes the model for a particular assignment to pose values.
   * @param fname_prefix A prefix for the file name of where to store images (an image per feature type will be generated)
   */
  void VisualizePartModel(PartLocation *locs, const char *fname_prefix);

  /**
   * @brief Visualize the learned weights for each attribute detector
   * @param fname_prefix A prefix for the file name of where to store images (an image per feature type will be generated)
   */
  void VisualizeAttributeModels(const char *fname_prefix);

  /**
   * @brief Get the root part of the part tree
   */
  ObjectPartInstance *GetRootPart() { return partInstances[classes->NumParts()-1]; }
  
  void RestrictLocationsToAgreeWithAtLeastOneUser(MultipleUserResponses *users, int radius=30);

  int GetLocalizedFeatures(float *f, PartLocation *locs, FeatureWindow *featureWindows=NULL, int numWindows=0, int *partInds=NULL);

  // Get features from the full image, possibly in a spatial pyramid
  int GetImageFeatures(float *f, FeatureWindow *featureWindows=NULL, int numWindows=0);

  // deprecated
  /// @cond
  char *PrintPartLocations(struct _PartLocation *locs, char *str);
  struct _PartLocation *LoadPartLocations(const char *str);
  bool LoadPartLocation(const char *str, struct _PartLocation *loc);
  void SanityCheckDynamicProgramming(PartLocation *gt_locs);
  /// @endcond

 private:
  void CreateInstances();

  void VisualizeSpatialModel(PartLocation *locs, const char *fname_prefix, char *html);
};

/// @cond
extern float *g_w, *g_f;
/// @endcond

#endif

