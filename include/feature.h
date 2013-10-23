/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#ifndef __FEATURE_H
#define __FEATURE_H

#include "util.h"
#include "opencvUtil.h"

/**
 * @file feature.h
 * @brief Structures and methods for extraction of different types of features that can be evaluated in a sliding window fashion
 */


extern int g_debug; /**< If greater than 1, adds some excessive print statements in various parts of the detection code */

class FeatureOptions;
class Classes;
class Dataset;
class Attribute;
class PartLocation;
class AttributeInstance;
class ObjectPose;


// Define to normalize templates by the dimensionality of the template
#define NORMALIZE_TEMPLATES 1


/**
 * @struct SiftParams
 *
 * @brief Parameters for a sift or HOG descriptor, which is of dimension numBlocksXnumBlocksXnumBins
 */
typedef struct {
  char name[200], type[200];
  int dims;            /**< total dimensionality of the SIFT descriptor (numBlocks*numBlocks*numBins) */
  int subsample;       /**< reduce the image in size by a factor of subsamplePower^subsample  */
  float subsamplePower;/**< scale interval between levels of the pyramid */
  float rot;           /**< rotate the image by rot radians */
  int maxScales;       /**< number of possible scales for a keypoint */
  int maxOrientations; /**< number of possible orientations for a keypoint */
  int smoothWidth;     /**< size in pixels of a gaussian smoothing kernel (or 0 for no smoothing) */
  int numBins;         /**< number of orientation bins.  If maxOrientations>numBins, then HOG will be computed
                          at n=maxOrientations/numBins times after rotating the image by 360/n */
  bool combineOpposite;/**< merge orientations 180 degrees apart into same bin  */
  int cellWidth;       /**< per-orientation histograms are computed by summing gradients in a cellWidthXcellWidth rectangle (in pixels) */
  int numBlocks;       /**< a descriptor concatenates histograms from numBlocksXnumBlocksY cells */
  int numBlocksX;      /**< a descriptor concatenates histograms from numBlocksXnumBlocksY cells */
  int numBlocksY;      /**< a descriptor concatenates histograms from numBlocksXnumBlocksY cells */
  bool normalize;      /**< locally normalize the descriptor (can be 0 or 1), as in Dalal Triggs' HOG */
  float normMax;       /**< constant used for local normalization */
  float normEps;       /**< constant used for local normalization */
  bool softHistogram;  /**< If true, aggregate orientations into a spatial cell using a gaussian filter rather than a box filter */
} SiftParams;

/**
 * @struct FeatureParams
 * 
 * @brief Defines detection/feature parameters, e.g. the number of scales, orientations, and spatial granularity
 *
 */
typedef struct {
  // 
  SiftParams hogParams; /**< Parameters to HOG descriptor computation, like the number orientation bins */

  int numScales; /**< Optionally compute HOG images over more than one scale */
  
  int numOrientations; /**< Optionally compute HOG images over more than one orientation */

  int spatialGranularity; /**< If greater than 1, compute feature descriptors over a coarse grid instead of at every pixel location */

  int scaleOffset;  /**< Normally 0.  If non-zero, shifts the scale indices by this number, such that scales less than scaleOffset go below the threshold of extracting features */
} FeatureParams;


/**
 * @struct FeatureWindow
 * 
 * @brief Defines a window or bounding box region for evaluating a specific type of low level feature
 *
 * Features like SIFT, spatial pyramids, multi-resolution HOG can be constructed using multiple FeatureWindows
 */
typedef struct {
  char *name;  /**< The name of the feature (should be the same as SlidingWindowFeature->Name()) */
  int w;  /**< The bounding box width for applying the detector */
  int h;  /**< The bounding box height for applying the detector */
  int dx;  /**< The x-offset for applying the bounding box offset (useful if we our detector is made of multiple spatial grids) */
  int dy;  /**< The y-offset for applying the bounding box offset (useful if we our detector is made of multiple spatial grids) */
  int scale;  /**< The bounding box scale index */
  int orientation;  /**< The boundinx box orientation index */
  int poseInd;  /**< An index into this part's list of poses.  Features are zero'd unless that pose is selected.  If poseInd=-1, then
		   features are used no matter what */
  int dim;  /**< The total number of features in the detector */
} FeatureWindow;

inline bool IsSameFeatureWindow(FeatureWindow &w1, FeatureWindow &w2) {
  return !strcmp(w1.name,w2.name) && w1.w == w2.w && w1.h == w2.h && w1.dx == w2.dx && 
    w1.dy == w2.dy && w1.scale == w2.scale && w1.orientation == w2.orientation &&
    w1.poseInd == w2.poseInd && w1.dim == w2.dim;
}

bool LoadFeatureWindow(FeatureWindow *f, Json::Value fe);
Json::Value SaveFeatureWindow(FeatureWindow *f);


/**
 * @class SlidingWindowFeature
 *
 * @brief A virtual class defining a simple low-level feature that can be evaluated as a sliding window detector.
 * 
 * Different low-level features must inherit from this class
 */
class SlidingWindowFeature {
protected:
  char *name;  /**< The name of the feature, e.g. "HOG" */
  FeatureOptions *fo;  /**< A pointer to the FeatureOptions container object */
  IplImage *****featureImages;  /**< A numXnumXnumOrientationsXnumScales array of detection responses  */
  IplImage *****featureImages_flip; 
  SiftParams params;  /**< A set of parameters defining the number of scales, orientations, granularity, and dimensionality of the feature  */

  friend class FeatureOptions;
public:
  /**
   * @brief Constructor
   * @param fo A pointer to the FeatureOptions object that contains the image and all feature definitions 
   */
  SlidingWindowFeature(FeatureOptions *fo);

  /**
   * @brief Destructor
   */
  virtual ~SlidingWindowFeature();

  /**
   * @brief Densely compute a detector response at every pixel location, scale, and orientation
   * @param weights A vector of wXhXparams.numBins weights applied to the feature responses
   * @param w The width of the sliding window detector
   * @param h The height of the sliding window detector
   * @return A numScalesXnumOrientations array of detection responses
   */
  virtual IplImage ***SlidingWindowDetect(float *weights, int w, int h, bool flip, ObjectPose *pose) = 0;

  /**
   * @brief Extract features at a particular pixel location
   * @param f A vector of wXhXparams.numBins into which the extracted features are stored
   * @param w The width of the sliding window detector
   * @param h The height of the sliding window detector
   * @param loc Defines the x,y,scale,orientation at whcih to extract features
   * @return The number of features extracted
   */
  virtual int GetFeaturesAtLocation(float *f, int w, int h, int feat_scale, PartLocation *loc, bool flip) = 0;

  /**
   * @brief Get the feature space dimensionality of a particular bounding box size
   * @param w The width of the sliding window detector
   * @param h The height of the sliding window detector
   * @return The feature space dimensionality
   */
  virtual int NumFeatures(int w, int h) = 0;

  /**
   * @brief Create a visualization image of the feature space, drawing the visualization over the specified part locations
   * @param classes The class and part definition file
   * @param locs A numParts array of part locations
   * @param visualizeWeights Visualize the weights of the model instead of the features in this image
   * @param attr Visualize the features/weights specific attribute (instead of the features/weights of a part detector)
   * @return An allocated image of the visualization image of the feature space
   */
  virtual IplImage *Visualize(Classes *classes, PartLocation *locs, bool visualizeWeights=false, AttributeInstance *attr=NULL) { return NULL; }

  /**
   * @brief Create a visualization image of the feature space
   * @param f An array of features
   * @param w The width of the sliding window, at which f was extracted
   * @param h The height of the sliding window, at which f was extracted
   * @return An allocated image of the visualization image of the feature space
   */
  virtual IplImage *Visualize(float *f, int w, int h, float mi=0, float ma = 0) { return NULL; }

  /**
   * @brief Free all memory for cached feature responses
   */
  virtual void Clear(bool full=true);

  /**
   * @brief Get an upper bound on the squared l2 norm of the feature space
   * @param w The width of the sliding window detector
   * @param h The height of the sliding window detector
   */
  virtual float MaxFeatureSumSqr(int w, int h) = 0;


  /**
   * @brief Get the name of the feature
   * @return The name of the feature, e.g. "HOG"
   */
  const char *Name() { return name; }

  /**
   * @brief Get parameters of the feature space
   * @return The parameters of the feature space
   */ 
  SiftParams *Params() { return &params; }

  /**
   * @brief Get the number of different scales at which we evaluate detectors
   * @return The number of different scales at which we evaluate detectors
   */
  int NumScales() { return params.maxScales; }

  /**
   * @brief Get a text description of this feature type
   * @param str The string into which the description is stored
   * @return str
   */
  virtual const char *Description(char *str) { strcpy(str, ""); return str; }
};

/**
 * @class TemplateFeature
 *
 * @brief A sliding window detector consisting of a template of weights in some feature space
 */
class TemplateFeature : public SlidingWindowFeature {
protected:

public:
  /**
   * @brief Constructor
   * @param fo A pointer to the FeatureOptions object that contains the image and all feature definitions 
   * @param p Parameters defining the spatial granularity of the template grid and dimensionality of the feature space
   */
  TemplateFeature(FeatureOptions *fo, SiftParams p);

  virtual ~TemplateFeature();
  virtual void Clear(bool full=true);

  /**
   * @brief Densely compute this feature response at every pixel location, scale, and orientation
   * @return A numXnumXnumOrientationsXnumScales array of detection responses, where num is the spatial resolution of the detector
   */
  virtual IplImage *****PrecomputeFeatures(bool flip) = 0;
  virtual float MaxFeatureSumSqr(int w, int h) = 0;
  virtual IplImage ***SlidingWindowDetect(float *weights, int w, int h, bool flip, ObjectPose *pose);
  virtual int GetFeaturesAtLocation(float *f, int w, int h, int feat_scale, PartLocation *loc, bool flip) {
    return GetFeaturesAtLocation(f, w, h, feat_scale, loc, flip, 0, 0);
  }
  virtual int GetFeaturesAtLocation(float *f, int w, int h, int feat_scale, PartLocation *loc, bool flip, int ox, int oy);
  virtual int NumFeatures(int w, int h) { return params.numBins*w*h; }
};



/**
 * @class FeatureOptions
 *
 * @brief Stores precomputed features for a given image
 */
class FeatureOptions {
  IplImage *img; /**< The image being processed */

  char *name; /**< A string identifier of this image, usually something similar to the image file name */

  SlidingWindowFeature **features; /**< A set of different possible feature types that can be used to process the image */
  int numFeatureTypes; /**< The number of different possible feature types that can be used to process the image */

  RotationInfo **rotations; /** a numScalesXnumOrientations array defining coordinate system transformations for rotated/scaled detection responses */
  RotationInfo **rotations_base;

  int nthreads;  /**< The number of threads to use for within image parallelization */

  char *imgName;
  
  char *segmentationName;
  IplImage ***segmentations;
  
  float imageScale;

public:
  FeatureParams params; /**< Basic parameters defining the number of scales, orientations, and spatial granularity of the features and detectors */

  /**
   * @brief Constructor
   * @param imgName The file name of the image to be processed
   * @param p Basic parameters defining the number of scales, orientations, and spatial granularity of the features and detectors
   * @param name A string identifier of this image, usually something similar to the image file name
   * @param classes pointer to a Classes object, which contains feature definitions
   */
  FeatureOptions(const char *imgName, FeatureParams *p, const char *name=NULL, Classes *classes=NULL);

  /**
   * @brief Destructor
   */
  ~FeatureOptions();

  /**
   * @brief Get a string identifier of this image, usually something similar to the image file name
   * @return A string identifier of this image, usually something similar to the image file name
   */
  const char *Name() { return name ? name : "untitled"; }

  
  /**
   * @brief Free memory for a numScalesxnumOrientations array of detection responses
   * @param buff A pointer to a numScalesxnumOrientations array of detection responses.  buff is set to NULL by this function
   * @param numScales the number of scales in the array of detection responses
   */
  void ReleaseResponses(IplImage ****buff, int numScales=-1);

  /**
   * @brief Evaluate a sliding window detector assembled from different types of features
   * @param weights A vector of detector weights.  The dimensionality is the sum of the feature dimensionalities of the different FeatureWindows
   * @param feats An array of num FeatureWindows.  Each FeatureWindow defines a type of low-level feature as well as a scale, orientation, and spatial 
   *   offset of a detector
   * @param num The number of feature types
   * @return A numScalesxnumOrientations array of detection responses
   */
  IplImage ***SlidingWindowDetect(float *weights, FeatureWindow *feats, int num, bool flip, ObjectPose *pose);

  
  /**
   * @brief Get the original image
   * @return the image
   */
  IplImage *GetImage(int image_resize_width=0) { 
    if(!img) {
      img = cvLoadImage(imgName);
      assert(img != NULL); 
      if(imageScale != 1 || image_resize_width > 0) {
	if(image_resize_width > 0) 
	  imageScale = image_resize_width/(float)my_max(img->width,img->height);
	IplImage *img_scaled = cvCreateImage(cvSize(my_round(img->width*imageScale),my_round(img->height*imageScale)), img->depth, img->nChannels);
	cvResize(img, img_scaled);
	cvReleaseImage(&img);
	img = img_scaled;
      }
      params.numScales = (int)LOG_B(my_min(img->width/2,img->height/2)/params.spatialGranularity, params.hogParams.subsamplePower) + params.scaleOffset;
    }
    return img; 
  }
  const char *GetImageName() { return imgName; }

  /**
   * @brief Get the original image
   * @return the image
   */
  void SetImage(IplImage *im) { 
    if(img) cvReleaseImage(&img);
    img=im; 
    params.numScales = (int)LOG_B(my_min(img->width/2,img->height/2)/params.spatialGranularity, params.hogParams.subsamplePower) + params.scaleOffset;
  }

  void SetSegmentationName(char *str) { segmentationName = str; }
  IplImage *GetSegmentation(int scale, int rot);
  void GetSegmentationMask(PartLocation *loc, float *f, int w, int h, bool flip=false, unsigned int color=0);

  void SetImageScale(float s) { imageScale = s; }

  /**
   * @brief Get the number of different scales at which we evaluate detectors
   * @return The number of different scales at which we evaluate detectors
   */
  int NumScales() { return params.numScales; }
 
  /**
   * @brief Get the number of different orientations at which we evaluate detectors
   * @return The number of different orientations at which we evaluate detectors
   */
  int NumOrientations() { return params.numOrientations; }

  /**
   * @brief Get the spatial granularity of detection responses. Typically, this should be equal to the cell width  
   *
   * If equal to 1, detection response images are the same resolution as the original image.  If bigger than 1, the 
   * detection response width and height is lower by the spatial granularity.  
   *
   * @return The spatial granularity of detection responses.
   */
  int SpatialGranularity() { return params.spatialGranularity; }

  /**
   * @brief Get the cell width of feature responses.  In other words, features are computed on a grid spaced at intervals of cell width
   *
   * If this is bigger than the spatial granularity, features are computed for multiple offsets into the image and then combined together
   * to produce a detection response of higher dimensionality
   *
   * @return The cell width of feature responses
   */
  int CellWidth() { return params.hogParams.cellWidth; }

  /**
   * @brief Get the scale offset, where all scales below this offset are too small to compute features.  This is typically used where
   *  some feature is computed over a pyramid of levels, such that the higher levels of the pyramid are computable but the lower levels are not
   */
  int ScaleOffset() { return params.scaleOffset; }

  /**
   * @brief Get the basic parameters defining the number of scales, orientations, and spatial granularity of the features and detectors
   * @return Basic parameters defining the number of scales, orientations, and spatial granularity of the features and detectors
   */
  FeatureParams *GetParams() { return &params; }

  /**
   * @brief Get the absolute scale factor corresponding to the ith scale index
   * @param i The scale index.  When i=0, the scale should be 1. When i>0 the scale is an exponential function of i
   * @return The absolute scale factor corresponding to the ith scale index
   */
  float Scale(int i) { return pow(params.hogParams.subsamplePower, i-params.scaleOffset); }

  /**
   * @brief Get the rotation factor in radians corresponding to the ith orientation index
   * @param i The orientation index
   * @return The rotation factor in radians corresponding to the ith orientation index
   */
  float Rotation(int i) { return (float)(i*2*M_PI/params.numOrientations); }

  /**
   * @brief Get a SlidingWindowFeature used to compute a certain type of base feature (e.g. "HOG") in this image
   * @param n The name of the base feature (e.g. "HOG")
   * @return The base feature
   */
  SlidingWindowFeature *Feature(const char *n);

  /**
   * @brief Register a new type of feature used to compute a certain type of base feature (e.g. "HOG") in this image
   * @param f The feature to register
   */
  void RegisterFeature(SlidingWindowFeature *f);

  /**
   * @brief Free memory for all precomputed feature responses
   */
  void Clear(bool full=true);

  /**
   * @brief Create a detection image copy that is the transformation from one scale/orientation coordinate system to another
   *
   * A detection image is an image of detection scores that may be a rotated/scaled version of the original image
   *
   * @param srcImg The srcImg to be copied
   * @param srcScale The scale of the srcImg
   * @param srcRot The orientation of the srcImg
   * @param dstScale The scale of the returned image
   * @param dstRot The orientation of the returned image
   * @param d For pixels in the returned image that are out of bounds in the source image, fill the returned image with this value
   * @param inds If non-null, get the x-y indices into the src image for each pixel in the dst image
   * @return A dynamically allocated image which is a copy of srcImg in the new coordinate system
   */
  IplImage *ConvertDetectionImageCoordinates(IplImage *srcImg, int srcScale, int srcRot, int dstScale, int dstRot, float d, IplImage *inds=NULL); 

 /**
  * @brief Convert detection coordinates from one scale/orientation to another
  *
  * @param x The x-coordinate in the original image
  * @param y The y-coordinate in the original image
  * @param srcScale The scale of the srcImg
  * @param srcRot The orientation of the srcImg
  * @param dstScale The scale of the destination image
  * @param dstRot The orientation of the destionation image
  * @param xx The x-coordinate in the destination image
  * @param yy The y-coordinate in the destination image
  * @return A dynamically allocated image which is a copy of srcImg in the new coordinate system
  */
  void ConvertDetectionCoordinates(float x, float y, int srcScale, int srcRot, int dstScale, int dstRot, float *xx, float *yy);

  /**
   * @brief Convert a location in the original image to a location in a detection score map
   *
   * A detection image is an image of detection scores that may be a rotated/scaled version of the original image
   *
   * @param x The x-coordinate in the original image
   * @param y The y-coordinate in the original image
   * @param scale The scale index of the detection score map
   * @param rot The orientation index of the detection score map
   * @param xx A pointer to the x-coordinate in the detection score map (set by this function)
   * @param yy A pointer to the y-coordinate in the detection score map (set by this function)
   */
  void ImageLocationToDetectionLocation(float x, float y, int scale, int rot, int *xx, int *yy); 

  /**
   * @brief Convert a location in a detection score map to a location in the original image
   *
   * A detection image is an image of detection scores that may be a rotated/scaled version of the original image
   *
   * @param x The x-coordinate in the detection score map
   * @param y The y-coordinate in the detection score map
   * @param scale The scale index of the detection score map
   * @param rot The orientation index of the detection score map
   * @param xx A pointer to the x-coordinate in the original image (set by this function)
   * @param yy A pointer to the y-coordinate in the original image (set by this function)
   */
  void DetectionLocationToImageLocation(int x, int y, int scale, int rot, float *xx, float *yy); 

  /**
   * @brief Get the width and height of a detection score map
   *
   * A detection image is an image of detection scores that may be a rotated/scaled version of the original image
   *
   * @param scale The scale index of the detection score map
   * @param rot The orientation index of the detection score map
   * @param w A pointer to the width (set by this function)
   * @param h A pointer to the height (set by this function)
   */
  void GetDetectionImageSize(int *w, int *h, int scale, int rot);

  /**
   * @brief Get the RotationInfo defining an affine transformation that would be applied to an image such that sliding window responses are computable as convolutions
   * @param rot the rotation of the detection image
   * @param scale the scale of the detection image
   */
  RotationInfo GetRotationInfo(int rot, int scale) {
    GetImage();
    if(!rotations) BuildRotationInfo();
    return rotations[scale][rot];
    //return ::GetRotationInfo(img->width, img->height, Rotation(rot), 1.0f/SpatialGranularity()/Scale(my_max(params.scaleOffset,scale)));
  };

  /**
   * @brief Get the number of threads to use for within image parallelization
   */
  int NumThreads() { return nthreads; }
  
  /**
   * @brief Set the number of threads to use for within image parallelization
   */
  void SetNumThreads(int n) { nthreads=n; }

  
  /**
   * @brief Create a visualization image of the feature space, drawing the visualization over the specified part locations for all available feature types
   * @param classes The class and part definition file
   * @param locs A numParts array of part locations
   * @param fname_prefix The prefix of the file name at which to store generated images 
   * @param html If non-null HTML code for visualizing this feature is written to this string
   * @param visualizeWeights Visualize the weights of the model instead of the features in this image
   * @param attr Visualize the features/weights specific attribute (instead of the features/weights of a part detector)
   * @return html
   */
  const char *Visualize(Classes *classes, PartLocation *locs, const char *fname_prefix, char *html=NULL, bool visualizeWeights=false, AttributeInstance *attr=NULL);

  /**
   * @brief Get a pointer to the ith feature type available for this image
   * @param i The index of the feature type we want to get
   */
  SlidingWindowFeature *GetFeatureType(int i) { return features[i]; }

  /**
   * @brief Get the number of feature types available for this image
   */
  int NumFeatureTypes() { return numFeatureTypes; }


  
private:
  void BuildRotationInfo();

};

FeatureWindow *SpatialPyramidFeature(const char *featName, int *numWindows, int numLevels, int dim, 
				     int gridWidthX=2, int gridWidthY=2, int startAtLevel=0, 
				     int w=0, int h=0, int scale=0, int rot=0, int pose=-1);

#endif

