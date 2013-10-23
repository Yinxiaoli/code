/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#ifndef __SIFT_H
#define __SIFT_H

#include "opencvUtil.h"
#include "feature.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

/**
 * @file hog.h
 * @brief Routines to compute HOG and SIFT
 */

/**
 * @struct SiftCoord
 *
 * @brief An interest point
 */
typedef struct {
  int x;   /**< The x-location of the interest point in the image */
  int y;   /**< The y-location of the interest point in the image */
  int rot;   /**< The orientation index of the interest point in the image */
  int scale;   /**< The scale index of the interest point in the image */
} SiftCoord;


/**
 * @brief Computes a HOG or SIFT descriptor at a particular (x,y) location using the image precomputed 
 * using ComputeSiftImage()
 * 
 * @param siftImg  the image precomputed using ComputeSiftImage()
 * @param descriptor  storage for the outputed descriptor
 * @param x the x-pixel location for the center of the top-left cell of the descriptor
 * @param y the y-pixel location for the center of the top-left cell of the descriptor
 * @param numX  the width (in cells) of the descriptor
 * @param numY  the height (in cells) of the descriptor
 * @param p  the sift parameters for the descriptor
 */
int ComputeHOGDescriptor(IplImage *siftImg, float *descriptor, int x, int y, int numX, int numY, SiftParams *p);


/**
 * @brief Compute sift descriptors for a bunch of interest points in a given image
 * @param img source image (either 8-bit grayscale or 24-bit RGB)
 * @param pts an array of interest points (x,y,scale,orientation)
 * @param numPts the number of interest points in the array pts
 * @param params the sift parameters for the descriptor
 * @param descriptors the output buffer to store the computed descriptors.  The buffer will have size numPtsXparams->dims.  
 * @return If descriptors is NULL, an array for descriptors will be dynamically allocated and returned
 */
float *ComputeSiftDescriptors(IplImage *img, SiftCoord *pts, int numPts, SiftParams *params, float *descriptors = NULL);


/**
 * @brief Computes a wXhXnumBins image of SIFT cell histograms for every pixel location in the image.  
 *        A SIFT descriptor can be quickly computed by querying numBlocksXnumBlocks pixels from this image. 
 * @param src source image (either 8-bit grayscale or 24-bit RGB)
 * @param params the parameters for computing SIFT or HOG
 * @return a numBins channel floating point image which stores the histogram summing the gradient magnitudes 
 *   over a cellWidthXcellWidth grid centered at each pixel for each orientation bin.
 */
IplImage *ComputeSiftImage(IplImage *src, SiftParams *params);



/**
 * @brief Rotate a sift or HOG descriptor by intervals of (360/numBins) degrees, by permuting the descriptor bins
 * @param descriptorIn input sift descriptor
 * @param descriptorOut output (rotated) sift descriptor
 * @param rot rotate the sift descriptor by rot*(360/numBins) degrees
 * @param params parameters describing the sift descriptor
 */
void RotateSiftDescriptor(float *descriptorIn, float *descriptorOut, int rot, SiftParams *params, bool flip);


/**
 * @brief Create an image visualization of a HOG descriptor
 * @param descriptor the input HOG descriptor, in the format returned by ComputeHOGDescriptor()
 * @param numX  the width (in cells) of the descriptor
 * @param numY  the height (in cells) of the descriptor
 * @param cellWidth The width of each spatial bin in pixels
 * @param numBins params->numBins
 * @param combineOpposite params->combineOpposite
 * @return an image visualization of the HOG descriptor
 */
IplImage *VisualizeHOGDescriptor(float *descriptor, int numX, int numY, int cellWidth, int numBins, bool combineOpposite, float mi=0, float ma=0);


/**
 * @brief Create an image visualization of a HOG descriptor
 * @param descriptor the input HOG descriptor, in the format returned by ComputeHOGDescriptor()
 * @param params parameters describing the HOG descriptor
 * @return a visualization of the HOG image
 */
IplImage *VisualizeHOGDescriptor(float *descriptor, SiftParams *params);



/**
 * @brief Compute sift descriptor for an interest points in a given image
 * @param siftImg the image precomputed using ComputeSiftImage()
 * @param descriptor the returned SIFT descriptor, an array of size p->numBlocks*p->numBlocks*p->numBins
 * @param x the x-coordinate at which to extract the SIFT descriptor
 * @param y the y-coordinate at which to extract the SIFT descriptor
 * @param p the sift parameters for the descriptor
 */
inline void ComputeSiftDescriptor(IplImage *siftImg, float *descriptor, int x, int y, SiftParams *p) {
  int offset = (1-p->numBlocks)*p->cellWidth/2;
  ComputeHOGDescriptor(siftImg, descriptor, x+offset, y+offset, p->numBlocks, p->numBlocks, p);
}



/**
 * @brief Initialize SIFT parameters.  Default parameter values are something similar to what is usually used for SIFT.  The SIFT descriptor will be numBlocksXnumBlocksXnumBins.  
 * @param p parameters describing the SIFT descriptor, which are written by this function
 * @param numScales total number of possible scales
 * @param numRotPerBin The total number of orientations will be equal to numBins*numRotPerBin.  Usually this will be 1.
 * @param numBins The number of orientation bins to histogram over
 * @param numBlocks The number of spatial bins in each direction
 * @param cellWidth The width of each spatial bin in pixels
 * @param smoothWidth Gaussian kernel width used for downsampling
 * @param combineOpposite If true, merge orientation bins that differ by 180 degrees into a single bin
 * @param normalize If true, do local normalization of HOG descriptors, similar to what Dalal and Triggs do
 * @param siftNormMax Constant used for normalization
 * @param siftNormEps Constant used for normalization
 * @param subsamplePower The amount in which to downsample the image for each scale
 */
void InitSiftParams(SiftParams *p, int numScales=1, int numRotPerBin=1, int numBins=8, int numBlocks=4, int cellWidth=8, int smoothWidth=5, bool combineOpposite=false, bool normalize=false, float siftNormMax=.2f, float siftNormEps=1.1920929e-7, float subsamplePower=1.3, bool softHistogram=true);



/**
 * @brief Initialize HOG parameters.  Default parameter values are something similar to what is usually used for HOG.  The HOG descriptor 
 *        will be numBlocksXnumBlocksXnumBins.  
 * @param p parameters describing the HOG descriptor, which are written by this function
 * @param numScales total number of possible scales
 * @param numRotPerBin The total number of orientations will be equal to numBins*numRotPerBin.  Usually this will be 1.
 * @param numBins The number of orientation bins to histogram over
 * @param numBlocks The number of spatial bins in each direction
 * @param cellWidth The width of each spatial bin in pixels
 * @param smoothWidth Gaussian kernel width used for downsampling
 * @param combineOpposite If true, merge orientation bins that differ by 180 degrees into a single bin
 * @param normalize If true, do local normalization of HOG descriptors, similar to what Dalal and Triggs do
 * @param siftNormMax Constant used for normalization
 * @param siftNormEps Constant used for normalization
 * @param subsamplePower The amount in which to downsample the image for each scale
 */
void InitHOGParams(SiftParams *p, int numScales=1, int numRotPerBin=1, int numBins=9, int numBlocks=4, int cellWidth=8, int smoothWidth=0, bool combineOpposite=true, bool normalize=true, float siftNormMax=.2f, float siftNormEps=16, float subsamplePower=1.3, bool softHistogram=false);

void InitLBPParams(SiftParams *p, int numScales=1, int numRotPerBin=1, int numBlocks=4, int cellWidth=8, int smoothWidth=0, bool normalize=true, float siftNormMax=.2f, float siftNormEps=2, float subsamplePower=1.3);

/**
 * @brief Create a JSON representation of these SIFT parameters 
 * @param params A pointer to the parameters object
 * @return a JSON encoding of the parameters
 */
Json::Value SiftParamsToJSON(SiftParams *params);


/**
 * @brief Extract SIFT parameters from a JSON encoding
 * @param root a JSON encoding of the parameters
 * @return the extracted parameters object
 */
SiftParams SiftParamsFromJSON(Json::Value root);

/// @cond
void StringToSiftParams(char *str, SiftParams *params);
void SiftParamsToString(SiftParams *params, char *str);
/// @endcond


/**
 * @brief Precompute HOG descriptors for an entire image at many different scales and rotations.
 *        This function is a preprocessing step, which should be called before ScoreHOG()
 * 
 * @param img The source RGB image
 * @param params HOG parameters (see StringToSiftParams())
 * @param spatialGranularity If greater than 1, compute HOG descriptors over a coarse grid
 *            instead of at every pixel location
 * @param rotations If greater than 1, compute HOG descriptors for many different angles.
 *            In this case, it should be at least params->numBins to work properly
 * @param scales If greater than 1, compute HOG descriptors for many different scales
 * @return An array of cached HOG images, which should be passed as input to ScoreHOG()
 */
IplImage *****PrecomputeHOG(IplImage *img, SiftParams *params, int spatialGranularity, int rotations, int scales);

/**
 * @brief Does something similar to PrecomputeHOG(), except that it computes HOG just at one orientation, then computes HOG images
 *   at other orientations using PrecomputeHOGFastFinish2(), by rotating that image and shifting its channels
 * 
 * @param img The source RGB image
 * @param params2 HOG parameters (see StringToSiftParams())
 * @return An array of cached HOG images, which should be passed as input to PrecomputeHOGFastFinish2()
 */
IplImage **PrecomputeHOGFastBegin2(IplImage *img, SiftParams *params2);

/**
 * @brief Does something similar to PrecomputeHOG(), except that it computes HOG just at one orientation PrecomputeHOGFastBegin2(), then computes HOG images
 *   at other orientations using PrecomputeHOGFastFinish2(), by rotating that image and shifting its channels
 * 
 * @param img The source RGB image
 * @param hogImages the output of PrecomputeHOGFastBegin2()
 * @param params2 HOG parameters (see StringToSiftParams())
 * @param spatialGranularity If greater than 1, compute HOG descriptors over a coarse grid
 *            instead of at every pixel location
 * @param rotations If greater than 1, compute HOG descriptors for many different angles.
 *            In this case, it should be at least params->numBins to work properly
 * @param scales If greater than 1, compute HOG descriptors for many different scales
 * @return An array of cached HOG images, which should be passed as input to ScoreHOG()
 */
IplImage *****PrecomputeHOGFastFinish2(IplImage **hogImages, IplImage *img, SiftParams *params2, int spatialGranularity, int rotations, int scales, bool flip);



/**
 * @brief Compute the response of a part template to an image in HOG space at different resolutions/scales.
 *   It is assumed PrecomputeHOG() was called before calling this function
 * 
 * @param cached - The output of PrecomputeHOG()
 * @param templ A template image with params->numBins channels per pixels. The response is the dot
 *            product of templ and the HOG image
 * @param spatialGranularity - If greater than 1, compute HOG descriptors over a coarse grid
 *            instead of at every pixel location
 * @param srcImg The original image
 * @param scales If greater than 1, compute HOG desriptors for many different scales
 * @param rotations If greater than 1, compute HOG descriptors for many different angles.
 *            In this case, it should be at least params->numBins to work properly
 * @param params HOG parameters (see StringToSiftParams())
 * @param nthreads The number of threads to use (usually should be 1)
 * 
 * @return A scalesXrotations array of template response images
 */
IplImage ***ScoreHOG(IplImage *****cached, IplImage *srcImg, IplImage *templ, SiftParams *params, int spatialGranularity, int rotations, int scales, int nthreads=1);



/**
 * @class HOGTemplateFeature
 *
 * @brief A sliding window HOG detector, consisting of a template of weights on a grid in HOG space
 */
class HOGTemplateFeature : public TemplateFeature {
  IplImage **denseFeatures;

 public:
  /**
   * @brief Constructor
   * @param fo A pointer to the FeatureOptions object that contains the image and all feature definitions 
   * @param p Parameters defining the spatial granularity of the template grid and dimensionality of the feature space
   */
  HOGTemplateFeature(FeatureOptions *fo, SiftParams p);
  IplImage *****PrecomputeFeatures(bool flip);
  virtual IplImage *Visualize(Classes *classes, PartLocation *locs, bool visualizeWeights=false, AttributeInstance *attr=NULL);
  virtual IplImage *Visualize(float *f, int w, int h, float mi=0, float ma=0);
  float MaxFeatureSumSqr(int w, int h) { return w*h/2.0f; }  // HOG should be normalized such that the sum of squares in each 2X2 template is 1
  virtual void Clear(bool full=true);  

  virtual const char *Description(char *str) { sprintf(str, "Histogram of Oriented Gradients features in a template"); return str; }
};


class LBPTemplateFeature : public HOGTemplateFeature {
 public:
  LBPTemplateFeature(FeatureOptions *fo, SiftParams p) : HOGTemplateFeature(fo, p) {}
  virtual const char *Description(char *str) { sprintf(str, "Histogrammed LBP codes in a template"); return str; }
};

void SiftNormalize(float *desc, int d, float siftNormMax, float siftNormEps);


#endif
