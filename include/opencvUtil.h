/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#ifndef __OPENCV_UTIL
#define __OPENCV_UTIL

//#include "util.h"
//#include "cv.h"
//#include "highgui.h"

//#include <opencv2\opencv.hpp>
//#include <opencv\cv.h>
//#include <opencv\cxcore.h>
#include "C:\opencv\build\include\opencv2\opencv.hpp"
#include "C:\opencv\build\include\opencv\cv.h"
#include "C:\opencv\build\include\opencv\cxcore.h"


#include "../online_structured_svm\include\util.h"
using namespace cv;

//#define EXTRA_DEBUG

/**
 * @file opencvUtil.h
 * @brief Helper routines for things I often do on OpenCV images but couldn't find an OpenCV function for
 */

/**
 * @struct RotationInfo
 * @brief Stores an affine matrix used for rotating/scaling detection responses
 *
 * When rotating an image, the size of the image may need to change to keep all pixels
 * visible.  Computes a 3X2 affine matrix that maps each pixel in an image to a pixel
 * in the rotated image.  See function GetRotationInfo() for details.
 */
typedef struct {
  float mat[6];  /**< A 3X2 affine matrix that maps each pixel in an image to a pixel in the rotated image */
  float invMat[6];  /**< A 3X2 affine matrix that maps each pixel in the rotated image to a pixel in the image */
  int minX;  /**< minX,maxX,minY,maxY define a tightly cropped box in the affine transformed image */
  int maxX;  /**< minX,maxX,minY,maxY define a tightly cropped box in the affine transformed image */
  int minY;  /**< minX,maxX,minY,maxY define a tightly cropped box in the affine transformed image */
  int maxY;  /**< minX,maxX,minY,maxY define a tightly cropped box in the affine transformed image */
  float rot; /**< The rotation in radians used to generate the affine matrix */
  CvPoint2D32f center; /**< The rotation occurred around this point */
  int width;  /**< The original width of the image */
  int height;  /**< The original height of the image */
} RotationInfo;


/**
 * @struct CvRectScore
 * @brief Stores a bounding box and a detection score
 */
typedef struct {
  float score;  /**< the detection score */
  CvRect rect;  /**< a bounding box */
  void *data;   /**< user-defined custom data */
  int det_x, det_y, scale_ind, rot_ind;
} CvRectScore;

#define DEFAULT_OVERLAP .5  /**< Default percent overlap of bounding boxes for greedy NMS */

/**
 * @brief Draw a rotated rectangle into an image 
 *
 * @param img The image into which the rectangle is drawn
 * @param centerPt the x,y pixel location of the center of the rectangle
 * @param width the width of the rectangle in pixels
 * @param height the height of the rectangle in pixels
 * @param rot The rotation in radians
 * @param color The color used to draw the rectangle, in the format of CV_RGB(r,g,b)
 * @param thickness THe thickness of the rectangle in pixels
 */
void cvRotatedRect(IplImage *img, CvPoint centerPt, int width, int height, float rot, CvScalar color, int thickness=1);

/**
 * @brief Apply a 2D affine transformation to a point [x,y] = [a*x1+b*y1+tx, c*x1+d*y1+ty]
 *
 * @param mat a 6 dimensional vector encoding an affine transformation matrix: mat=[a,b,tx,c,d,ty]
 * @param x1 the x-coordinate of the source point  
 * @param y1 the y-coordinate of the source point 
 * @param x the x-coordinate of the affine transformed point, an output of this function
 * @param y the y-coordinate of the affine transformed point, an output of this function
 */
inline void AffineTransformPoint(float *mat, float x1, float y1, float *x, float *y) {
  *x = mat[0]*x1 + mat[1]*y1 + mat[2];
  *y = mat[3]*x1 + mat[4]*y1 + mat[5];
}

/**
 * @brief Apply a 2D affine transformation to a point [x,y] = [a*x1+b*y1+tx, c*x1+d*y1+ty]
 *
 * @param mat a 2X3 encoding an affine transformation matrix: mat=[a,b,tx;c,d,ty]
 * @param x1 the x-coordinate of the source point  
 * @param y1 the y-coordinate of the source point 
 * @param x the x-coordinate of the affine transformed point, an output of this function
 * @param y the y-coordinate of the affine transformed point, an output of this function
 */
inline void AffineTransformPoint(CvMat *mat, float x1, float y1, float *x, float *y) {
  *x = (float)(cvGet2D(mat,0,0).val[0]*x1 + cvGet2D(mat,0,1).val[0]*y1 + cvGet2D(mat,0,2).val[0]);
  *y = (float)(cvGet2D(mat,1,0).val[0]*x1 + cvGet2D(mat,1,1).val[0]*y1 + cvGet2D(mat,1,2).val[0]);
}

// d is the affine matrix that is the result of applying affine matrix m followed by affine matrix n
/**
 * @brief Compute an affine matrix that combines 2 consecutive affine transformations: d is the affine matrix that is the result of applying affine matrix m followed by affine matrix n
 *
 * @param m a 6 dimensional vector encoding an affine transformation matrix: m=[a,b,tx,c,d,ty]
 * @param n a 6 dimensional vector encoding an affine transformation matrix: n=[a,b,tx,c,d,ty]
 * @param d a 6 dimensional vector encoding an affine transformation matrix: n=[a,b,tx,c,d,ty], the output of this function
 */
inline void MultiplyAffineMatrices(float *m, float *n, float *d) {
  d[0] = n[0]*m[0]+n[1]*m[3];   d[1] = n[0]*m[1]+n[1]*m[4];   d[2] = n[0]*m[2]+n[1]*m[5]+n[2];
  d[3] = n[3]*m[0]+n[4]*m[3];   d[4] = n[3]*m[1]+n[4]*m[4];   d[5] = n[3]*m[2]+n[4]*m[5]+n[5];
}

/**
 * @brief Compute affine matrices for computing a rotated version of an image
 *
 * A tightly cropped rotated image can be computed as:
 *  RotationInfo r = GetRotationInfo(srcImage->width, srcImage->height, rot);
 *  IplImage *rotImg = cvCreateImage(cvSize((r.maxX-r.minX), (r.maxY-r.minY)), srcImage->depth, srcImage->nChannels);
 *  CvMat affineMat = cvMat(2, 3, CV_32FC1, r.mat);
 *  cvWarpAffineMultiChannel(srcImg, rotImg, &affineMat);
 * It can then be rotated back to the source image as
 *  CvMat invAffineMat = cvMat(2, 3, CV_32FC1, r.invMat);
 *  cvWarpAffineMultiChannel(rotImg, srcImg, &invAffineMat);
 *
 * @param width The source image width
 * @param height The source image height
 * @param rot The rotation in radians
 * @param scale The scale change
 * @return A structure defining affine matrices for warping srcImg to a rotated image
 */
RotationInfo GetRotationInfo(int width, int height, float rot, float scale=1.0f);

/**
 * @brief Extract a particular color channel from a color image
 * @param src the input image, a widthXheightXnChannels image
 * @param dst the destination image, a widthXheightXnum image into which num color channels from src are extracted
 * @param c the first color channel to extract
 * @param num the number of color channels to extract.  For example, if src is an RGB image, c=1, and num=2, then dst will store the G and B channels extracted from src
 */
void cvExtractChannel(IplImage *src, IplImage *dst, int c, int num=1);

/**
 * @brief Set a particular color channel of a color image from another image
 * @param src the input image, a widthXheightXnChannelsSrc image
 * @param dst the destination image, a widthXheightXnChannelsDst image into which num color channels from src are extracted
 * @param c the 1st color channel in dst into which pixels from src are stored into
 * @param cSrc the 1st color channel in src from which pixels from src are extracted 
 * @param num the number of color channels to extract from src.  For example, if src and dst are both RGB images, cSrc=0, c=1, and num=2, then the R and G channels from src will be stored into the G and B channels in dst
 */
void cvSetChannel(IplImage *dst, IplImage *src, int c, int cSrc = 0, int num = 1);

/**
 * @brief Helper function to do template matching, where the template and image can be multiple channels, and the response image is the sum of the template matching scores for each channel
 * @param img the input image, a widthXheightXnChannels image
 * @param templ the template image, a wXhXnChannels image
 * @param score the image into which the template response is stored, a widthXheightX1 image, the output of this function
 * @param method The template matching method (see documentation for cvMatchTemplate())
 */
void cvMatchTemplateMultiChannel(IplImage *img, IplImage *templ, IplImage *score, int method);

/**
 * @brief Helper function to apply an affine transformation to a multi-channel image (that may be over 4 channels)
 * @param src the input image
 * @param dst the output image
 * @param mat a 6 dimensional vector encoding an affine transformation matrix: m=[a,b,tx,c,d,ty]
 * @param interpolation the interpolation method, see documentation for cvWarpAffine()
 * @param fillval value to set outlier pixels
 */
void cvWarpAffineMultiChannel(IplImage *src, IplImage *dst, CvMat *mat, 
			      int interpolation=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, CvScalar fillval=cvScalarAll(0));


/**
 * @brief Helper function to do convolution, where the filter and image can be multiple channels, and the response image is the sum of the filter responses for each channel
 * @param img the input image, a widthXheightXnChannels image
 * @param templ the filter image, a wXhXnChannels image
 * @param score the image into which the filter response is stored, a widthXheightX1 image, the output of this function
 */
void cvFilter2DMultiChannel(IplImage *img, IplImage *score, IplImage *templ);

/**
 * @brief Draw a portion of an image into another image
 * @param src the source image
 * @param to the destination image
 * @param from the rectangle defining the pixels to extract from src
 * @param toX the x-coordinate of the pixel location of upper-left corner of the rectangle into the destination image
 * @param toY the y-coordinate of the pixel location of upper-left corner of the rectangle into the destination image
 */
void DrawImageIntoImage(IplImage *src, CvRect from, IplImage *to, int toX, int toY);

/**
 * @brief Normalize an image to be on the range [0,255],  and convert to 8-bit.  Finds the min and max pixel values in the image and maps those to 0 and 255 respectively.  Usually, this function is used to convert a floating point image into a visualizable image
 * @param img the source image
 * @param mi_p if non-null, outputs the minimum pixel value into mi_p
 * @param ma_p if non-null, outputs the maximum pixel value into ma_p
 * @return an 8 bit image
 */
IplImage *MinMaxNormImage(IplImage *img, double *mi_p=NULL, double *ma_p=NULL, double mi=0, double ma=0);


/**
 * @brief Draw a portion of an image into another image, taking the max of the input and output images at each pixel
 * @param src the source image
 * @param to the destination image
 * @param from the rectangle defining the pixels to extract from src
 * @param toX the x-coordinate of the pixel location of upper-left corner of the rectangle into the destination image
 * @param toY the y-coordinate of the pixel location of upper-left corner of the rectangle into the destination image
 */
void DrawImageIntoImageMax(IplImage *src, CvRect from, IplImage *to, int toX, int toY);

/**
 * @brief Compute the cumulative sum of an image.  This is often used in conjunction with cvFindCumulativeSumLocation(sumImg,sum*RandFloat(),pt), which effectively randomly selects a pixel location according to its probability.  If gamma is non-zero, img is assumed to be a likelihood map (instead of a probability map), where the conversion to a probability is exp{gamma*max(mi,likelihood)+delta}
 * @param img the source image
 * @param sumImg the destination image, assumed to be the same width and height as img
 * @param sum a pointer into which the total cumulative sum is stored 
 * @param gamma if non-zero, interpret img as a likelihood, where probability=exp{gamma*max(mi,likelihood)+delta}
 * @param delta if gamma is non-zero, interpret img as a likelihood, where probability=exp{gamma*max(mi,likelihood)+delta}
 * @param mi if non-zero, assume there is a lowerbound of mi on any pixel in img
 */
IplImage *cvCumulativeSum(IplImage *img, float *sum, IplImage *sumImg=NULL, float gamma=0, float delta=0, float mi=0);

/**
 * @brief Compute the cumulative sum of an image.  This is often used in conjunction with cvFindCumulativeSumLocation(sumImg,sum*RandFloat(),pt), which effectively randomly selects a pixel location according to its probability.  If gamma is non-zero, img is assumed to be a likelihood map (instead of a probability map), where the conversion to a probability is exp{gamma*max(mi,likelihood)+delta}
 * @param img the source image
 * @param sumImg the destination image, assumed to be the same width and height as img
 * @param sum a pointer into which the total cumulative sum is stored 
 * @param gamma if non-zero, interpret img as a likelihood, where probability=exp{gamma*max(mi,likelihood)+delta}
 * @param delta if gamma is non-zero, interpret img as a likelihood, where probability=exp{gamma*max(mi,likelihood)+delta}
 * @param mi if non-zero, assume there is a lowerbound of mi on any pixel in img
 */
IplImage *cvCumulativeSum(IplImage *img, double *sum, IplImage *sumImg=NULL, double gamma=0, double delta=0, double mi=0);

/**
 * @brief Compute the cumulative sum of an image.  This is often used in conjunction with cvFindCumulativeSumLocation(sumImg,sum*RandFloat(),pt), which effectively randomly selects a pixel location according to its probability.  If gamma is non-zero, img is assumed to be a likelihood map (instead of a probability map), where the conversion to a probability is exp{gamma*max(mi,likelihood)+delta}
 * @param img the source image
 * @param sumImg the destination image, assumed to be the same width and height as img
 * @param sum a pointer into which the total cumulative sum is stored 
 * @param gamma if non-zero, interpret img as a likelihood, where probability=exp{gamma*max(mi,likelihood)+delta}
 * @param delta if gamma is non-zero, interpret img as a likelihood, where probability=exp{gamma*max(mi,likelihood)+delta}
 * @param mi if non-zero, assume there is a lowerbound of mi on any pixel in img
 */
IplImage *cvCumulativeSum(IplImage *img, int *sum, IplImage *sumImg=NULL, int gamma=0, int delta=0, int mi=0);


/**
 * @brief Find the 1st pixel for which the cumulative sum exceeds a target value
 * @param sumImg the output of cvCumulativeSum()
 * @param target the target value
 * @param pt the pixel location returned by this function
 * @return false if it failed to find any pixel within the specified target cumulative sum
 */
bool cvFindCumulativeSumLocation(IplImage *sumImg, float target, CvPoint *pt);

void FlipImage(IplImage *src);

/**
 * @brief Find the 1st pixel for which the cumulative sum exceeds a target value
 * @param sumImg the output of cvCumulativeSum()
 * @param target the target value
 * @param pt the pixel location returned by this function
 * @return false if it failed to find any pixel within the specified target cumulative sum
 */
bool cvFindCumulativeSumLocation(IplImage *sumImg, double target, CvPoint *pt);


/**
 * @brief Find the 1st pixel for which the cumulative sum exceeds a target value
 * @param sumImg the output of cvCumulativeSum()
 * @param target the target value
 * @param pt the pixel location returned by this function
 * @return false if it failed to find any pixel within the specified target cumulative sum
 */
bool cvFindCumulativeSumLocation(IplImage *sumImg, int target, CvPoint *pt);


/**
 * @brief Apply a distance transformation, as defined in Felzenszwalb's paper "Distance Transforms of Sampled Functions"
 *
 * A distance transform densely solves for each pixel \f$ \forall x',\forall y', score[x',y'] = \max_{x,y} \left( img[x,y] + wx(x'+\mu_x-x) + wy(y'+\mu_y-y) + wxx(x'+\mu_x-x)^2 + wyy(y'+\mu_y-y)^2 \right) \f$.  It is used for dynamic programming based maximum likelihood inference for pictorial structures, where img stores the detection response of a child part and *score stores the corresponding computed detection response of a parent part that factors in a spring cost wxx,wyy,wx,wy.  The runtime is linear in the number of pixel locations (a standard dynamic programming algorithm takes quadratic time in the number of pixel locations)
 *
 * @param img the detection response of the child part
 * @param wxx a weight for a quadratic penalty in the x-direction
 * @param wyy a weight for a quadratic penalty in the y-direction
 * @param wx a weight for a linear penalty in the x-direction
 * @param wy a weight for a linear penalty in the y-direction
 * @param mu_x the ideal x-offset between the parent and child parts
 * @param mu_y the ideal y-offset between the parent and child parts
 * @param max_change_x Can optionally set a maximum value for (x'+mu_x-x), such that if (x'+mu_x-x) > max_change_x, then the parent score is -infinity
 * @param max_change_y Can optionally set a maximum value for (y'+mu_y-y), such that if (y'+mu_y-y) > max_change_y, then the parent score is -infinity
 * @param best_child_locs a pointer to a widthXheightX2 image that is allocated and outputed by this function.  The image stores the corresponding x,y location of the child part for each location in the parent image
 * @param score a pointer to an image of scores for each parent location, the output of this function
 */
void DistanceTransform(IplImage *img, float wxx, float wyy, float wx, float wy, int mu_x, int mu_y, IplImage **best_child_locs, IplImage **score, int max_change_x, int max_change_y);

/**
 * @brief Apply a distance transformation, as defined in Felzenszwalb's paper "Distance Transforms of Sampled Functions", except this function does the distance transform in scale and orientation space.  
 *
 * A distance transform densely solves for each pixel \f$ \forall scale', \forall rot', \forall x',\forall y', score[x',y'] = \max_{scale,rot,x,y} \left( scores[scale,rot,x,y] + wx(x'+\mu_x-x) + wy(y'+\mu_y-y) + ws(scale'+\mu_{scale}-scale) + wo(rot'+\mu_{rot}-rot) + wxx(x'+\mu_x-x)^2 + wyy(y'+\mu_y-y)^2 + wss(scale'+\mu_{scale}-scale)^2 + woo(rot'+\mu_{rot}-rot)^2 \right) \f$.  It is used for dynamic programming based maximum likelihood inference for pictorial structures, where img stores the detection response of a child part and the output image stores the corresponding computed detection response of a parent part that factors in a spring cost wss,woo,wxx,wyy,ws,wo,wx,wy.  The function DistanceTransform() is assumed to have been called for each score map scores[scale][rot] prior to making this call.  The runtime is linear in the number of pixel locations, scales, and orientations (a standard dynamic programming algorithm takes quadratic time in the number of pixel locations, scales, and orientations).  The fact that orientation wraps around between 0 and 2 pi is correctly handled by this function
 *
 * @param scores A num_scalesXnum_orientations array of scores, where x and y have already been optimized over using DistanceTransform()
 * @param best_child_scales A pointer to a num_scalesXnum_orientations array into which optimal scale indices for the child part is stored
 * @param best_child_orientations A pointer to a num_scalesXnum_orientations array into which optimal orientation indices for the child part is stored
 * @param num_scales the number of discrete scales
 * @param num_orientations the number of discrete orientations
 * @param wss a weight for a quadratic penalty in the scale-direction
 * @param woo a weight for a quadratic penalty in the offset-direction
 * @param ws a weight for a linear penalty in the scale-direction
 * @param wo a weight for a linear penalty in the orientation-direction
 * @param mu_s the ideal scale-offset between the parent and child parts
 * @param mu_o the ideal orientation-offset between the parent and child parts
 * @param max_change_scale Can optionally set a maximum value for (scale'+mu_{scale}-scale), such that if (scale'+mu_{scale}-scale) > max_change_scale, then the parent score is -infinity
 * @param max_change_ori Can optionally set a maximum value for (rot'+mu_{rot}-rot), such that if (rot'+mu_{rot}-rot) > max_change_ori, then the parent score is -infinity
 * @return a num_scalesXnum_orientations array of parent score images
 */
IplImage ***DistanceTransformScaleOrientation(IplImage ***scores,  IplImage ****best_child_scales, IplImage ****best_child_orientations,
	               int num_scales, int num_orientations, int mu_s, int mu_o, float wss, float woo, float ws, float wo, int max_change_scale, int max_change_ori);

/**
 * @brief Copies the source 2D array into interior of destination array and makes a border of the specified type around the copied area, where the images can be multi-channel.
 * @param src the input image
 * @param dst the destination image
 * @param offset Coordinates of the top-left corner of the destination image rectangle where the source image is copied. Size of the rectangle matches the source image size/ROI size
 * @param bordertype Type of the border to create around the copied source image rectangle (see documentation for cvMatchTemplate())
 * @param value Value of the border pixels if bordertype=CONSTANT
 */
void cvCopyMakeBorderMultiChannel(IplImage* src, IplImage* dst, CvPoint offset,
				  int bordertype, CvScalar value);



/**
 * @brief Helper function to apply an affine transformation to a multi-channel image (that may be over 4 channels)
 * @param src the input image
 * @param dst the output image
 * @param inds a 2-channel image with the same dimensions as dst that stores the pixel location in src for each pixel in dst, an output of this function
 * @param mat a 6 dimensional vector encoding an affine transformation matrix: m=[a,b,tx,c,d,ty]
 * @param fillVal value to set outlier pixels
 */
void cvWarpAffineMultiChannelCustomGetIndices(IplImage *src, IplImage *dst, IplImage *inds, float *mat, double fillVal);


/**
 * @brief Helper function to apply an affine transformation to a multi-channel image (that may be over 4 channels).  This function is implemented manually (instead of calling the opencv implementation), such that the exact behavior is predictable
 * @param src the input image
 * @param dst the output image
 * @param mat a 6 dimensional vector encoding an affine transformation matrix: m=[a,b,tx,c,d,ty]
 * @param fillVal value to set outlier pixels
 */
void cvWarpAffineMultiChannelCustom(IplImage *src, IplImage *dst, float *mat, double fillVal);

/**
 * @brief Implements 2 kinds of non-maximal suppression: 1) regular non-maximal suppression and 2) greedy "non-maximal-suppression", where we greedily select the bounding box with highest score, then invalidate all overlapping boxes
 * @param boxes An array of num bounding boxes, before non-max suppression.  This array is overwritten by the array of bounding boxes after non-max suppresison.   Should include one bounding box per pixel location in the correct order when w and h are defined
 * @param num the number of bounding boxes before non-max suppression
 * @param overlap If -1, use regular non-max suppresison.  Otherwise, do greedy non-max suppression, which suppress bounding boxes for which the percent area of overap with a higher scoring bounding box is greater than this parameter.
 * @param w If non-zero, boxes should include one bounding box for each pixel location in a wXh image
 * @param h If non-zero, boxes should include one bounding box for each pixel location in a wXh image
 */
int NonMaximalSuppression(CvRectScore *boxes, int num, float overlap=-1, int w=0, int h=0);


/**
 * @brief Solve for the value t in the interval [s,e] that maximizes a custom function using line search
 * @param s the minimum value of t
 * @param e the maximum value of t
 * @param p the parameter that will be passed to the function (*eval)()
 * @param eval a custom function that ouputs the score at a particular value of t: eval(t, p)
 * @param acc target accuracy, where we can optionally terminate early
 */
template <class T> 
inline T line_search(T s, T e, const void *p, T (*eval)(T, const void *), T acc) {
  T seeds[1000];
  T seed_vals[1000], d=0;
  int n = 0, i;
  while(s+d < e) {
    seeds[n] = s+d;
    seed_vals[n++] = (*eval)(s+d, p);
    d = d ? d*2 : (e-s)/1024;
  }
  seeds[n] = e; 
  T bestV = (T)INFINITY;
  int best = 0;
  for(i = 0; i < n; i++) {
    if(seed_vals[i] < bestV) { bestV = seed_vals[i];  best = i; }
  }
  s = best ? seeds[best-1] : s;
  e = n ? seeds[best+1] : e;

  T sv = (*eval)(s, p), ev = (*eval)(e, p), mv, m, sm2, sv2, em2, ev2;
  int t = 0;
  mv = sv;
  while(t < 3 || my_abs(sv-ev) > acc) {
    m = (s+e)/2;
    mv = (*eval)(m, p); 
    //fprintf(stderr, "%f %f\n", m, mv);
    if(mv <= sv && mv <= ev) {
      sm2 = (s+m)/2;
      sv2 = (*eval)(sm2, p);
      em2 = (e+m)/2;
      ev2 = (*eval)(em2, p);
      if(!(sv2 <= sv && ev2 <= ev)) {
        fprintf(stderr, "line_search non_convex error s=%f sv=%f sv2=%f, e=%f ev=%f ev2=%f\n", s, sv, sv2, e, ev, ev2);
        break; 
      }
      if(sm2 > em2) { s = sm2; sv = sv2; }
      else { e = em2; ev = ev2; }
    } else if(sv > ev) {
      s = m; sv = mv;
    } else {
      e = m; ev = mv;
    } 
	if(t++ >= 100) { fprintf(stderr, "line_search error s=%f sv=%f, e=%f ev=%f\n", s, sv, e, ev); break; }
  }
  //fprintf(stderr, "line_search %g %f\n", m, mv);
  return m;
}

void MakeThumbnail(IplImage *img, int thumb_width, int thumb_height, const char *outFileName);

#endif
