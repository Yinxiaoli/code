/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#ifndef __COLOR_H
#define __COLOR_H

#include "hog.h"
#include "feature.h"

/**
 * @file color.h
 * @brief Routines to compute sliding window RGB and CIE color features
 */


/**
 * @brief Precompute color descriptors for an entire image at many different scales and rotations.
 *        This function is a preprocessing step, which should be called before ScoreHOG()
 * 
 * @param img The source RGB image
 * @param params2 HOG parameters (see StringToSiftParams())
 * @param spatialGranularity If greater than 1, compute HOG descriptors over a coarse grid
 *            instead of at every pixel location
 * @param numOrientations If greater than 1, compute HOG descriptors for many different angles.
 *            In this case, it should be at least params->numBins to work properly
 * @param numScales If greater than 1, compute HOG descriptors for many different scales
 * @return An array of cached HOG images, which should be passed as input to ScoreHOG()
 */
IplImage *****PrecomputeColor(IplImage *img, SiftParams *params2, int spatialGranularity, int numOrientations, int numScales);

/**
 * @brief Does something similar to PrecomputeColor(), except that it computes HOG just at one orientation, then computes HOG images
 *   at other orientations using PrecomputeColorFastFinish2(), by rotating that image and shifting its channels
 * 
 * @param img The source RGB image
 * @param params2 HOG parameters (see StringToSiftParams())
 * @return An array of cached color images, which should be passed as input to PrecomputeColorFastFinish2()
 */
IplImage **PrecomputeColorFastBegin2(IplImage *img, SiftParams *params2);

/**
 * @brief Does something similar to PrecomputeColor(), except that it computes HOG just at one orientation PrecomputeColorFastBegin2(), then computes HOG images
 *   at other orientations using PrecomputeColorFastFinish2(), by rotating that image and shifting its channels
 * 
 * @param img The source RGB image
 * @param hogImages the output of PrecomputeColorFastBegin2()
 * @param params2 HOG parameters (see StringToSiftParams())
 * @param spatialGranularity If greater than 1, compute HOG descriptors over a coarse grid
 *            instead of at every pixel location
 * @param rotations If greater than 1, compute HOG descriptors for many different angles.
 *            In this case, it should be at least params->numBins to work properly
 * @param scales If greater than 1, compute HOG descriptors for many different scales
 * @return An array of cached HOG images, which should be passed as input to ScoreHOG()
 */
IplImage *****PrecomputeColorFastFinish2(IplImage **hogImages, IplImage *img, SiftParams *params2, int spatialGranularity, int rotations, int scales,bool flip);


/**
 * @brief Create an image visualization of a color descriptor
 * @param descriptor the input color descriptor, in the format returned by ComputeColorDescriptor()
 * @param numX  the width (in cells) of the descriptor
 * @param numY  the height (in cells) of the descriptor
 * @param cellWidth The width of each spatial bin in pixels
 * @return a visualization of the color image
 */
IplImage *VisualizeColorDescriptor(float *descriptor, int numX, int numY, int cellWidth);




/**
 * @class ColorTemplateFeature
 *
 * @brief A sliding window detector, consisting of a template of weights on a grid in pixel space
 */
class ColorTemplateFeature : public TemplateFeature {
protected:
  IplImage *colorImg;  /**< The image in the appropriate colorspace */

public:
  /**
   * @brief Constructor
   * @param fo A pointer to the FeatureOptions object that contains the image and all feature definitions 
   * @param p Parameters defining the spatial granularity of the template grid and dimensionality of the feature space
   */
  ColorTemplateFeature(FeatureOptions *fo, SiftParams p);
  ~ColorTemplateFeature();

  /**
   * @brief Convert the original image into a different color space
   */
  virtual void ComputeColorImage() = 0;

  /**
   * @brief Convert pixels in this colorspace back to rgb color
   * @param tmp a wXhX3 array of pixels in the original colorspace
   * @param dst a wXhX3 array of pixels in BGR color written by this function
   * @param w The width of the image
   * @param h The height of the image
   */
  virtual void ConvertToRGB(float *tmp, float *dst, int w, int h) { memcpy(dst,tmp,w*h*3*sizeof(float)); }

  float MaxFeatureSumSqr(int w, int h) { return w*h*3; }  

  IplImage *****PrecomputeFeatures(bool flip);
  virtual void Clear(bool full=true);  
  IplImage *Visualize(Classes *classes, PartLocation *locs, bool visualizeWeights=false, AttributeInstance *attr=NULL);
  IplImage *Visualize(float *f, int w, int h, float mi=0, float ma=0);
};

/**
 * @class RGBTemplateFeature
 *
 * @brief A sliding window detector, consisting of a template of weights on a grid in RGB space
 */
class RGBTemplateFeature : public ColorTemplateFeature {
 public:
  /**
   * @brief Constructor
   * @param fo A pointer to the FeatureOptions object that contains the image and all feature definitions 
   * @param p Parameters defining the spatial granularity of the template grid and dimensionality of the feature space
   */
  RGBTemplateFeature(FeatureOptions *fo, SiftParams p);
  void ComputeColorImage();
  const char *Description(char *str) { sprintf(str, "Raw RGB pixels in a template"); return str; }
};

/**
 * @class CIETemplateFeature
 *
 * @brief A sliding window detector, consisting of a template of weights on a grid in CIE color space
 */
class CIETemplateFeature : public ColorTemplateFeature {
 public:
  /**
   * @brief Constructor
   * @param fo A pointer to the FeatureOptions object that contains the image and all feature definitions 
   * @param p Parameters defining the spatial granularity of the template grid and dimensionality of the feature space
   */
  CIETemplateFeature(FeatureOptions *fo, SiftParams p);
  void ComputeColorImage();
  void ConvertToRGB(float *tmp, float *dst, int w, int h);
  const char *Description(char *str) { sprintf(str, "Raw pixels in CIELab color in a template"); return str; }
};



/**
 * @class ColorMaskFeature
 *
 * @brief A sliding window detector based on a segmentation mask, which scores high when the foreground and background of the
 * mask come from different color distributions
 */
class ColorMaskFeature : public SlidingWindowFeature {
protected:
  IplImage ****features, ****features_flip;  
  IplImage *****colorImages, *****colorImages_flip; 
  int numPoses;

public:
  /**
   * @brief Constructor
   * @param fo A pointer to the FeatureOptions object that contains the image and all feature definitions 
   */
  ColorMaskFeature(FeatureOptions *fo, SiftParams p);

  virtual IplImage ***SlidingWindowDetect(float *weights, int numX, int numY, bool flip, ObjectPose *pose);
  virtual IplImage ***PrecomputeFeatures(ObjectPose *pose, bool flip);
  virtual void Clear(bool full=true);  
  virtual int GetFeaturesAtLocation(float *f, int w, int h, int feat_scale, PartLocation *loc, bool flip);
  virtual int NumFeatures(int w, int h) { return 1; }
  virtual float MaxFeatureSumSqr(int w, int h) { return 10; }
};


#endif
