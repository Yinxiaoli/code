/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#include "hog.h"
#include "part.h"
#include "classes.h"
#include "attribute.h"
#include "pose.h"
#include "distance.h"

#include <stdio.h>
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>

void LBPComputeCodes(IplImage *img, IplImage *weights, IplImage *codes);

#define BORDER 5

#define SHIFT_ROTATION


// Allocate multi-dimensional array for storing HOG images
IplImage *****AllocatePrecomputeHOGBuffers(int rotations, int maxScales, int num) {
  IplImage *****retval = (IplImage*****)malloc(sizeof(IplImage****)*rotations);
  for(int k = 0; k < rotations; k++) {
    retval[k] = (IplImage****)malloc(sizeof(IplImage***)*maxScales);
    for(int i = 0; i < maxScales; i++) {
      retval[k][i] = (IplImage***)malloc(sizeof(IplImage**)*num);
      for(int l=0; l < num; l++) {
        retval[k][i][l] = (IplImage**)malloc(sizeof(IplImage*)*num);
        for(int m=0; m < num; m++) {
          retval[k][i][l][m] = NULL;
        }
      }
    }
  }
  return retval;
}

// Beginning with an image of dense HOG desciptors, Extract a numXnum array of images, each of which is a 
// (non-dense) HOG descriptor image of size (siftImage->width/params->cellWidth)X(siftImage->height/params->cellWidth), 
// taken at a different offset from 0 to num-1.  This is done such that a HOG template can be applied to
// produce a detection response of resolution greater than (siftImage->width/params->cellWidth)
void ExtractHOGOffsetImages(IplImage *siftImage, IplImage ***retval, IplImage *img, SiftParams *params, 
			    RotationInfo *r, int rotations, int spatialGranularity) {
  int x, y, l, m;
  int num = params->cellWidth/spatialGranularity;
  int numBlocksX = siftImage->width/params->cellWidth;
  int numBlocksY = siftImage->height/params->cellWidth;
  int rx = (r->maxX-r->minX) - numBlocksX*num;
  int ry = (r->maxY-r->minY) - numBlocksY*num;
  assert(rx <= num+3 && ry <= num+3 && rx >= 0 && ry >= 0);

  // Extract HOG images (sample pixels from the dense HOG image based on the HOG cell width)
  for(x = (params->cellWidth%(spatialGranularity+1))/2, l=0; l < num; x += spatialGranularity, l++) {
    params->numBlocksX = numBlocksX + (l >= rx ? 0 : 1);
    for(y = (params->cellWidth%(spatialGranularity+1))/2, m=0; m < num; y += spatialGranularity, m++) {
      params->numBlocksY = numBlocksY + (m >= ry ? 0 : 1);
      params->dims = params->numBlocksX*params->numBlocksY*params->numBins;
      IplImage *tmpImg = cvCreateImage(cvSize(params->numBlocksX, params->numBlocksY), IPL_DEPTH_32F, params->numBins);
      cvZero(tmpImg);
      float *descriptor = (float*)tmpImg->imageData;
      assert(tmpImg->widthStep == (int)(tmpImg->width*sizeof(float)*params->numBins));
      ComputeHOGDescriptor(siftImage, descriptor, x, y, params->numBlocksX, params->numBlocksY, params);
      retval[l][m] = tmpImg;
    }
  }
}


IplImage *****PrecomputeHOG(IplImage *img, SiftParams *params2, int spatialGranularity, int rotations, int scales) {
  //int num = params2->cellWidth/spatialGranularity;
  int num = my_max(1,params2->cellWidth/spatialGranularity);
  IplImage *****retval = AllocatePrecomputeHOGBuffers(rotations, params2->maxScales, num);

  // compute descriptors, ordered by scale and orientation
#ifdef WITHIN_IMAGE_PARALELLIZE
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
#endif
  for(int j = 0; j < rotations; j++) {
    SiftParams tmpP = *params2;
    SiftParams *params = &tmpP;
    int i;
    RotationInfo r;
  
    params->rot = 0;
    float rot = (float)(j*2*M_PI/rotations);
    r = GetRotationInfo(img->width, img->height, rot);
    IplImage *rotImage = cvCreateImage(cvSize(r.maxX-r.minX, r.maxY-r.minY), IPL_DEPTH_8U, img->nChannels);
    cvZero(rotImage);
    CvMat affineMat = cvMat(2, 3, CV_32FC1, r.mat);
    cvWarpAffineMultiChannel(img, rotImage, &affineMat);
	  
    for(i = 0; i < params->maxScales; i++) {
      params->subsample = i;
      IplImage *siftImage = ComputeSiftImage(rotImage, params);
      RotationInfo r = GetRotationInfo(img->width, img->height, (float)(j*2*M_PI/rotations), 1.0f/pow(params->subsamplePower, i)/spatialGranularity);
      ExtractHOGOffsetImages(siftImage, retval[j][i], img, params, &r, rotations, spatialGranularity);
      cvReleaseImage(&siftImage);
    }
    cvReleaseImage(&rotImage);
  }

  return retval;
}


IplImage **PrecomputeHOGFastBegin2(IplImage *img, SiftParams *params2) {
  IplImage **retval = (IplImage**)malloc(sizeof(IplImage*)*params2->maxScales);
  
  for(int i = 0; i < params2->maxScales; i++) {
    SiftParams params = *params2;
    params.subsample = i;
    retval[i] = ComputeSiftImage(img, &params);  // Densely compute HOG at every pixel location
  }
  return retval;
}

  
IplImage *****PrecomputeHOGFastFinish2(IplImage **hogImages, IplImage *img, SiftParams *params2, int spatialGranularity, int rotations, int scales, bool flip) {
  //int num = params2->cellWidth/spatialGranularity;
  int num = my_max(1,params2->cellWidth/spatialGranularity);
  IplImage *****retval = AllocatePrecomputeHOGBuffers(rotations, params2->maxScales, num);
  int nh = params2->combineOpposite ? 2 : 1;
  int nb = params2->numBins*nh;
  for(int j = 0; j < rotations; j++) {
    int bins[100];
    int rot = (nb-j*nh)%params2->numBins;
    for(int i = 0; i < params2->numBins; i++) 
      bins[i] = (i+rot)%params2->numBins;
    if(flip) {
      int nb2 = params2->numBins/(params2->combineOpposite ? 1 : 2);
      for(int i = 0; i < params2->numBins; i++) 
	bins[i] = nb2-1-bins[i] + (!params2->combineOpposite && bins[i] >= nb2 ? params2->numBins : 0);
    }


    for(int i = 0; i < params2->maxScales; i++) {
      float rot = (float)(j*2*M_PI/rotations);
      RotationInfo r = GetRotationInfo(img->width, img->height, rot, 1.0f/pow(params2->subsamplePower, i)/spatialGranularity);
      int numBlocksX = (r.maxX-r.minX)/num;
      int numBlocksY = (r.maxY-r.minY)/num;
      int rx = (r.maxX-r.minX) - numBlocksX*num;
      int ry = (r.maxY-r.minY) - numBlocksY*num;
      assert(rx <= num+3 && ry <= num+3 && rx >= 0 && ry >= 0);

      for(int x = 0; x < num; x++) {
      	for(int y = 0; y < num; y++) {
	  // Extract a rotated version of the HOG images, by sampling pixels on a grid from the dense HOG image, 
	  // while applying an affine warp (for rotation and scale)
	  int w = numBlocksX + (x >= rx ? 0 : 1);
	  int h = numBlocksY + (y >= ry ? 0 : 1);
	  int k, l, m, nc = hogImages[i]->nChannels, ix, iy, sx, sy;
	  float xx, yy, *ptr, *srcPtr;
	  char *ptr2;
	  RotationInfo r2 = GetRotationInfo(hogImages[i]->width, hogImages[i]->height, (float)(j*2*M_PI/rotations), 1.0f);
	  float *mat = r2.invMat;
	  float dx = mat[0]*spatialGranularity/*params2->cellWidth*/, dy = mat[3]*spatialGranularity/*params2->cellWidth*/;
	  retval[j][i][x][y] = cvCreateImage(cvSize(w, h), hogImages[i]->depth, hogImages[i]->nChannels);
	  for(k = 0, sy = y, ptr2=retval[j][i][x][y]->imageData; k < retval[j][i][x][y]->height; k++, ptr2 += retval[j][i][x][y]->widthStep, sy += params2->cellWidth) {
	    AffineTransformPoint(mat, x, sy, &xx, &yy);  xx += .5f; yy += .5f;
	    for(l = 0, sx = x, ptr=(float*)ptr2; l < retval[j][i][x][y]->width; l++, ptr += nc, xx += dx, yy += dy, sx += params2->cellWidth) {
	      iy = ((int)yy);  ix = ((int)xx);
	      if(ix >= 0 && iy >= 0 && ix < hogImages[i]->width && iy < hogImages[i]->height) {
		srcPtr = ((float*)(hogImages[i]->imageData+hogImages[i]->widthStep*iy))+(ix*nc);
		for(m = 0; m < nc; m++)  // Shift the orientation channel bins to correct for the orientation
		  ptr[bins[m]] = srcPtr[m];
	      } else {
		for(m = 0; m < nc; m++) 
		  ptr[m] = 0;
	      }
	    }
	  }
	  if(flip)
	    FlipImage(retval[j][i][x][y]);
	}
      }
    }
  }
  return retval;
}





IplImage ***ScoreHOG(IplImage *****cached, IplImage *srcImg, IplImage *templ, SiftParams *sparams, int spatialGranularity, int rotations, int scales, int nthreads) {
  IplImage ***retval = (IplImage***)malloc(sizeof(IplImage**)*scales + sizeof(IplImage*)*scales*rotations);
  for(int i = 0; i < sparams->maxScales; i++) {
    retval[i] = ((IplImage**)(retval+scales))+i*rotations;
    memset(retval[i], 0, rotations*sizeof(IplImage*));
  }

  int l_scales[1000], l_rotations[1000];
  int num = 0;
  for(int j = 0; j < rotations; j++) {
    for(int i = 0; i < scales; i++) {
      l_scales[num] = i; 
      l_rotations[num++] = j;
    }
  }

  // compute descriptors, ordered by scale and orientation
  //#ifdef WITHIN_IMAGE_PARALELLIZE
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(nthreads)
#endif
    //#endif
  for(int k = 0; k < num; k++) {
    int i, j, l, m, n, o, num, ind2;
    SiftParams params = *sparams;
    j = l_rotations[k];
    i = l_scales[k];
    params.rot = 0;
    
    params.subsample = i;

    num = (params.cellWidth+spatialGranularity-1)/spatialGranularity;
    RotationInfo r;
    IplImage *tmpScore2 = NULL;
    for(l=0; l < num; l++) {
      for(m=0; m < num; m++) {
	params.numBlocksX = cached[j][i][l][m]->width;
	params.numBlocksY = cached[j][i][l][m]->height;
        
	// Match the HOG template at each pixel location in the image
	IplImage *tmpScore = NULL;
	//if(params.numBlocksX > templ->width) {
	  params.dims = params.numBlocksX*params.numBlocksY*params.numBins;
	  //tmpScore = cvCreateImage(cvSize(params.numBlocksX-templ->width+1, params.numBlocksY-templ->height+1), IPL_DEPTH_32F, 1);
	  //cvMatchTemplateMultiChannel(cached[j][i][l][m], templ, tmpScore, CV_TM_CCORR);
	  tmpScore = cvCreateImage(cvSize(params.numBlocksX+BORDER*2, params.numBlocksY+BORDER*2), IPL_DEPTH_32F, 1);
	  IplImage *src = cached[j][i][l][m];
	  if(BORDER) {
	    src = cvCreateImage(cvSize(params.numBlocksX+BORDER*2, params.numBlocksY+BORDER*2), IPL_DEPTH_32F, cached[j][i][l][m]->nChannels);
	    cvZero(src);
	    DrawImageIntoImage(cached[j][i][l][m], cvRect(0,0,params.numBlocksX,params.numBlocksY), src, BORDER,BORDER);
	  }		    
	  cvFilter2DMultiChannel(src, tmpScore, templ);
	  if(BORDER) {
	    cvReleaseImage(&src);
	    IplImage *tmpScore3 = cvCreateImage(cvSize(params.numBlocksX, params.numBlocksY), IPL_DEPTH_32F, 1);
	    DrawImageIntoImage(tmpScore, cvRect(BORDER,BORDER,tmpScore3->width,tmpScore3->height), tmpScore3, 0, 0);
	    cvReleaseImage(&tmpScore);
	    tmpScore = tmpScore3;
	  }
	  //}

	// If we want a response map with greater resolution than the width of a HOG bin, we compute
	// the response map for many different pixel offsets and interleave them into a single response map
	if(!tmpScore2) {
	  r = GetRotationInfo(srcImg->width, srcImg->height, (float)(j*2*M_PI/rotations), 1.0f/pow(params.subsamplePower, i)/spatialGranularity);
	  tmpScore2 = (r.maxX-r.minX) > 0 && (r.maxY-r.minY) > 0 ? cvCreateImage(cvSize((r.maxX-r.minX), (r.maxY-r.minY)), IPL_DEPTH_32F, 1) : NULL;
	  if(tmpScore2) cvSet(tmpScore2, cvScalar(-10000000000000.0));
	}
	if(tmpScore && tmpScore2) {
	  if(num == 1) {
	    assert(tmpScore->width == tmpScore2->width && tmpScore->height == tmpScore2->height);
	  }

	  assert(l+num*tmpScore->width <= tmpScore2->width && m+num*tmpScore->height <= tmpScore2->height);
	  unsigned char *ptr = (unsigned char*)(tmpScore2->imageData+(m/*+templ->height/2*f*/)*tmpScore2->widthStep) + (l/*+templ->width/2*f*/)*sizeof(float);
	  unsigned char *ptr2 = (unsigned char*)tmpScore->imageData;
	  for(n = 0; n < tmpScore->height; n++, ptr += num*tmpScore2->widthStep, ptr2 += tmpScore->widthStep) {
	    for(o = 0, ind2 = 0; o < tmpScore->width; o++, ind2+=num) {
	      //assert(m+templ->height/2+n*num < tmpScore2->height);
	      //assert(l+templ->width/2+o*num < tmpScore2->width);
	      ((float*)ptr)[ind2] = ((float*)ptr2)[o];
	    }
	  }
	}
	  
	if(tmpScore) cvReleaseImage(&tmpScore);
      }
    }

    retval[i][j] = tmpScore2;

    // Rotate the response map back to the original image dimensions
    // retval[i][j] = UndoRotation(tmpScore2, &r, &params, params.subsample, spatialGranularity); cvReleaseImage(&tmpScore2);
  }

  return retval;
}


void RotateSiftDescriptor(float *descriptorIn, float *descriptorOut, int rot, SiftParams *params);
IplImage *SiftSubsampleImage(IplImage *src, float p, int subsample, bool *freeSrc);
IplImage *SiftSmoothImage(IplImage *src, int smoothWidth);
void SiftComputeGradients(IplImage *img, IplImage *gradMag, IplImage *gradBin, int numBins, int cellWidth, float rot, bool combineOpposite);
IplImage *SiftComputeGradientHistograms(IplImage *gradMag, IplImage *gradBin, int numBins, int cellWidth);
IplImage *SiftComputeGradientHistogramsSoft(IplImage *img, int numBins, int cellWidth, float rot, bool combineOpposite);
void HogNormalize(IplImage *imgNew, int cellWidth, float siftNormEps, float siftNormMax);
void SiftNormalize(float *desc, int d, float siftNormMax);

void cvTransparentLine(IplImage *img, CvPoint2D32f pt1, CvPoint2D32f pt2, float alpha);


/* Computes a wXhXnumBins image of SIFT cell histograms for every pixel location in the image.  
 * A SIFT descriptor can be quickly computed by querying numBlocksXnumBlocks pixels from this image. 
 * Parameters: 
 *   src -- source image (either 8-bit grayscale or 24-bit RGB)
 *   params -- the parameters for computing SIFT or HOG
 * Returns: a numBins channel floating point image which stores the histogram summing the gradient magnitudes 
 *   over a cellWidthXcellWidth grid centered at each pixel for each orientation bin.
 */
IplImage *ComputeSiftImage(IplImage *src, SiftParams *params) {
  // Scale and smooth the source image, if necessary
  bool freeSrc = false;
  src = SiftSubsampleImage(src, params->subsamplePower, params->subsample, &freeSrc);
  if(!src) return NULL;
  IplImage *img = SiftSmoothImage(src, params->smoothWidth);
  if(freeSrc) cvReleaseImage(&src);

  // Compute and quantize image gradients
  IplImage *imgNew = NULL;
  if(!params->softHistogram) {
    IplImage *gradMag = cvCreateImage(cvSize(img->width+params->cellWidth,img->height+params->cellWidth), IPL_DEPTH_32F, 1);
    IplImage *gradBin = cvCreateImage(cvSize(img->width+params->cellWidth,img->height+params->cellWidth), IPL_DEPTH_32S, 1);
    if(!strcmp(params->type, "LBP")) {
      assert(params->numBins == 8 && params->rot == 0);
      LBPComputeCodes(img, gradMag, gradBin);
    } else
      SiftComputeGradients(img, gradMag, gradBin, params->numBins, params->cellWidth, params->rot, params->combineOpposite);
  

    // Compute gradient orientation histograms over a cellWidthXcellWidth block for each pixel location
    imgNew = SiftComputeGradientHistograms(gradMag, gradBin, params->numBins, params->cellWidth);
    cvReleaseImage(&gradMag);
    cvReleaseImage(&gradBin);
  } else 
    imgNew = SiftComputeGradientHistogramsSoft(img, params->numBins, params->cellWidth, params->rot, params->combineOpposite);

  // Locally normalize the HOG histograms
  if(params->normalize) 
    HogNormalize(imgNew, params->cellWidth, params->normEps, params->normMax);
  

  cvReleaseImage(&img);

  return imgNew;
}

void LBPComputeCodes(IplImage *img, IplImage *weights, IplImage *codes) {
  int b1, b2, b3, b4, b5, b6, b7, b8, b12, b34, b56, b78, b1234, b5678, b12345678;
  int hw = (codes->width-img->width)/2;
  unsigned char *codePtr2 = ((unsigned char*)codes->imageData) + codes->widthStep*(hw+1) + (hw)*sizeof(float);
  unsigned char *weightPtr2 = ((unsigned char*)weights->imageData) + weights->widthStep*(hw+1) + (hw)*sizeof(float);
  unsigned char *ptr2 = ((unsigned char*)img->imageData) + img->widthStep;
  float *codePtr, *weightPtr, *ptr, *ptrP, *ptrN;

  cvZero(weights);   
  cvZero(codes);
  for(int y = 1; y < img->height-1; y++, codePtr2 += codes->widthStep, weightPtr2 += weights->widthStep, ptr2 += img->widthStep) {
    codePtr = (float*)codePtr2;
    weightPtr = (float*)weightPtr2;
    ptr = (float*)ptr2;
    ptrP = (float*)(ptr2-img->widthStep);
    ptrN = (float*)(ptr2+img->widthStep);

    for(int x = 1; x < img->width-1; x++) {
      // Test if the center pixel is greater than each of its 8-neighboring pixels
      b1 = ptr[x+1] > ptr[x];  b2 = ptrP[x+1] > ptr[x];  b3 = ptrP[x] > ptr[x];  b4 = ptrP[x-1] > ptr[x];
      b5 = ptr[x-1] > ptr[x];  b6 = ptrN[x-1] > ptr[x];  b7 = ptrN[x] > ptr[x];  b8 = ptrN[x+1] > ptr[x];
      b12 = b1 + b2;  b34 = b3 + b4;  b56 = b5 + b6;  b78 = b7 + b8;
      b1234 = b12 + b34;   b5678 = b56 + b78;
      b12345678 = b1234 + b5678;
      if(b12345678 == 1) {
	// If exactly 1 bit is on, set the LBP word to the index of the on bit
	if(b1234) {
	  if(b12) codePtr[x] = b1 ? 0 : 1;
	  else    codePtr[x] = b3 ? 2 : 3;
	} else {
	  if(b56) codePtr[x] = b5 ? 4 : 5;
	  else    codePtr[x] = b7 ? 6 : 7;
	}
	weightPtr[x] = 1;
      } else
	weightPtr[x] = 0;  // ignore this pixel because it is not one of the allowable 8-codes
    }
  }
}


/*
 * Computes a HOG or SIFT descriptor at a particular (x,y) location using the image precomputed 
 * using ComputeSiftImage()
 * Parameters:
 *   siftImg -- the image precomputed using ComputeSiftImage()
 *   descriptor -- storage for the outputed descriptor
 *   x,y -- the pixel location for the center of the top-left cell of the descriptor
 *   numX,numY -- the width and height (in cells) of the descriptor
 *   p -- the sift parameters for the descriptor
 */
int ComputeHOGDescriptor(IplImage *siftImg, float *descriptor, int x, int y, int numX, int numY, SiftParams *p) {
  int k, ind = 0;
  int dy = siftImg->widthStep*p->cellWidth;
  int dx = p->numBins*p->cellWidth;
  unsigned char *ptr2;
  float *ptr;
  int sx = x, sy = y, ex = x + numX*p->cellWidth, ey = y + numY*p->cellWidth, xx, yy;
  int maxX = siftImg->width-p->cellWidth, maxY = siftImg->height-p->cellWidth;

  while(sx < 0) sx += p->cellWidth;
  while(sy < 0) sy += p->cellWidth;
  ind = 0;
  ptr2 = (unsigned char*)siftImg->imageData + sy*siftImg->widthStep + sx*p->numBins*sizeof(float);
  for(yy = y; yy < ey; yy += p->cellWidth) { 
    for(xx = x, ptr = (float*)ptr2; xx < ex; xx += p->cellWidth) {
      for(k = 0; k < p->numBins; k++) {
	descriptor[ind++] = ptr[k];
      }
      if(xx < maxX && xx >= 0) ptr += dx;
    }
    if(yy < maxY && yy >= 0) ptr2 += dy;
  }

  return ind;
}

/*
 * Computes a HOG image at a particular rotation
 * Parameters:
 *   siftImg -- the image precomputed using ComputeSiftImage()
 *   p -- the sift parameters for the descriptor
 *   rot -- a number between 0 and maxOrientations-1, representing a rotation of 360/rot degrees
 */
IplImage *ComputeRotatedHOGImage(IplImage **siftImgs, SiftParams *params, RotationInfo *r, int rot) {
  int subRot = rot % (params->maxOrientations/params->numBins);
  int binRot = rot / (params->maxOrientations/params->numBins);
  IplImage *siftImg = siftImgs[subRot];
  *r = GetRotationInfo(siftImg->width, siftImg->height, (float)((params->combineOpposite ? M_PI : 2*M_PI)/params->maxOrientations*rot));
  IplImage *retval = cvCreateImage(cvSize((r->maxX-r->minX), (r->maxX-r->minX)), IPL_DEPTH_32F, params->numBins);
  int i, j;
  float xx, yy;
  float dx = r->invMat[0], dy = r->invMat[1];
  int bins[100];
  float *srcPtr, *dstPtr2;
  unsigned char *dstPtr;

  // Cache pointers to source siftImg (assumes memory lookup from a small buffer is faster than float multiplication)
  int w = my_max(siftImg->width,siftImg->height);
  float **yPtr = (float**)malloc(2*w*sizeof(float*) + 2*w*(sizeof(int)));
  int *xOff = (int*)(yPtr + (r->maxY-r->minY+1));
  float **yPtr2 = yPtr+r->minY;
  int ix, iy, k;
  for(i = -w; i < w; i++)
    yPtr2[i] = (i < 0 || i >= siftImg->height) ? NULL : (float*)(siftImg->imageData + i*siftImg->widthStep);
  for(j = -w; j < w; j++)
    xOff[j] = (j < 0 || j >= siftImg->width) ? -1 : j*siftImg->nChannels;
			       
  // Cache bins 
  for(i = 0; i < params->numBins; i++) 
    bins[i] = (i+binRot)%params->numBins;

  for(i = 0, dstPtr = (unsigned char*)retval->imageData; i < retval->height; i++, dstPtr += retval->widthStep) {
    AffineTransformPoint(r->invMat, 0.0f, (float)i, &xx, &yy); xx += .5f; yy += .5f; 
    for(j = 0, dstPtr2 = (float*)dstPtr; j < retval->width; j++, xx += dx, yy += dy) {
      ix = (int)xx;  iy = (int)yy;
      if(yPtr[iy] && xOff[ix] >= 0) {
	srcPtr = yPtr[iy] + xOff[ix];
	for(k = 0; k < params->numBins; k++)
	  dstPtr2[bins[k]] = srcPtr[k];
      }
    }
  }
  return retval;
}


/*
 * Compute sift descriptors for a bunch of interest points in a given image
 * Parameters: 
 *   img -- source image (either 8-bit grayscale or 24-bit RGB)
 *   pts -- an array of interest points (x,y,scale,orientation)
 *   numPts -- the number of interest points in the array pts
 *   params -- the sift parameters for the descriptor
 *   descriptors -- the output buffer to store the computed descriptors.  The buffer will have size
 *     numPtsXparams->dims.  If NULL, will be dynamically allocated and returned
 *   
 */
float *ComputeSiftDescriptors(IplImage *img, SiftCoord *pts, int numPts, SiftParams *params, float *descriptors) {
  IplImage *siftImg = NULL;
  float *ptr;
  int i, j, k;
#ifdef SHIFT_ROTATION
  int numRot = params->maxOrientations/params->numBins;
  // the number of possible interest point orientations must be an even multiple of the number of orientation bins
  assert(numRot*params->numBins == params->maxOrientations);
#else
  int numRot = params->maxOrientations;
#endif
  if(!descriptors) descriptors = (float*)malloc(sizeof(float)*params->dims*numPts);

  // compute descriptors, ordered by scale and orientation
  for(i = 0; i < params->maxScales; i++) {
    params->subsample = i;
    for(j = 0; j < numRot; j++) {
      params->rot = (float)((params->combineOpposite ? M_PI : 2*M_PI)/params->maxOrientations*j);
      if(siftImg) {
	cvReleaseImage(&siftImg);
	siftImg = NULL;
      }
      ptr = descriptors;
      for(k = 0; k < numPts; k++, ptr += params->dims) {
	if(pts[k].scale == i && pts[k].rot%params->numBins == j) {
	    int rShift = pts[k].rot/numRot;
	  if(!siftImg) siftImg = ComputeSiftImage(img, params);
	  ComputeSiftDescriptor(siftImg, ptr, pts[k].x, pts[k].y, params);
	  if(rShift) RotateSiftDescriptor(ptr, ptr, rShift, params, false);
	}
      }
    }
  }
  if(siftImg) 
    cvReleaseImage(&siftImg);
  return descriptors;
}

/*
 * Initialize parameters for a SIFT descriptor
 * Parameters: (defaults to traditional SIFT)
 *  subsample -- current scale of the sift descriptor. Reduce the image in size by a factor of 2^subsample 
 *  rot -- current orientation of the sift descriptor. Rotate each gradient by rot radians before binning
 *  maxScales -- number of possible scales for a keypoint
 *  maxOrientations -- number of possible orientations for a keypoint
 *  smoothWidth -- size in pixels of a gaussian smoothing kernel (or 0 for no smoothing)
 *  numBins -- number of orientation bins
 *  combineOpposite -- merge orientations 180 degrees apart into same bin 
 *  cellWidth -- per-orientation histograms are computed by summing gradients in a cellWidthXcellWidth rectangle (in pixels)
 *  numBlocks -- a descriptor concatenates histograms from numBlocksXnumBlocks cells
 *  normalize -- locally normalize the descriptor (can be 0 or 1), as in Dalal Triggs' HOG
 *  normMax -- constant used for local normalization
 *  float normEps -- constant used for local normalization
 */
void InitSiftParams(SiftParams *params, int numScales, int numRotPerBin, int numBins, int numBlocks, int cellWidth, int smoothWidth, bool combineOpposite, bool normalize, float normMax, float normEps, float subsamplePower, bool softHistogram) { 
  strcpy(params->name, "SIFT");
  strcpy(params->type, "HOG");
  params->maxScales = numScales;
  params->maxOrientations = numRotPerBin*numBins;
  params->subsample = 0;
  params->rot = 0;
  params->smoothWidth = smoothWidth;
  params->numBins = numBins;
  params->combineOpposite = combineOpposite;
  params->cellWidth = cellWidth;
  params->numBlocks = params->numBlocksX = params->numBlocksY = numBlocks;
  params->normalize = normalize;
  params->normMax = normMax;
  params->normEps = normEps;
  params->subsamplePower = subsamplePower;
  params->dims = params->numBlocksX*params->numBlocksY*numBins;
  params->softHistogram = softHistogram;
}

/*
 * Initialize parameters for a SIFT descriptor
 * Parameters: (Same as InitSiftParams(), but with different default parameters)
 */
void InitHOGParams(SiftParams *p, int numScales, int numRotPerBin, int numBins, int numBlocks, int cellWidth, int smoothWidth, bool combineOpposite, bool normalize, float normMax, float normEps, float subsamplePower, bool softHistogram) {
  InitSiftParams(p, numScales, numRotPerBin, numBins, numBlocks, cellWidth, smoothWidth, combineOpposite, normalize, normMax, normEps, subsamplePower, softHistogram);
  strcpy(p->name, "HOG");
  strcpy(p->type, "HOG");
}

void InitLBPParams(SiftParams *p, int numScales, int numRotPerBin, int numBlocks, int cellWidth, int smoothWidth, bool normalize, float normMax, float normEps, float subsamplePower) {
  InitSiftParams(p, numScales, numRotPerBin, 8, numBlocks, cellWidth, smoothWidth, false, normalize, normMax, normEps, subsamplePower, false);
  strcpy(p->name, "LBP");
  strcpy(p->type, "LBP");
}

/*
 * Convert a set of SIFT parameters into string form
 */
void SiftParamsToString(SiftParams *params, char *str) {
  sprintf(str, "numBins %d, cellWidth %d, numBlocks %d, smoothWidth %d, combineOpposite %d, normalize %d, subsample %d, rot %f, maxScales %d, maxOrientations %d, normMax %f, normEps %f, subsamplePower %f, softHistogram %d", params->numBins, params->cellWidth, params->numBlocks, params->smoothWidth, params->combineOpposite, params->normalize, params->subsample, params->rot, params->maxScales, params->maxOrientations, params->normMax, params->normEps, params->subsamplePower, params->softHistogram ? 1 : 0);
}



/*
 * Parse SIFT parameters from a string
 */
void StringToSiftParams(char *str, SiftParams *params) {
  int combineOpposite, normalize, softHistogram;

  InitSiftParams(params);
  sscanf(str, "numBins %d, cellWidth %d, numBlocks %d, smoothWidth %d, combineOpposite %d, normalize %d, subsample %d, rot %f, maxScales %d, maxOrientations %d, normMax %f, normEps %f, subsamplePower %f, softHistogram %d", &params->numBins, &params->cellWidth, &params->numBlocks, &params->smoothWidth, &combineOpposite, &normalize, &params->subsample, &params->rot, &params->maxScales, &params->maxOrientations, &params->normMax, &params->normEps, &params->subsamplePower, &softHistogram);
  params->combineOpposite = combineOpposite > 0;
  params->normalize = normalize > 0;
  params->numBlocksX = params->numBlocksY = params->numBlocks;
  params->dims = params->numBlocksX*params->numBlocksY*params->numBins;
  params->softHistogram = softHistogram > 0;
}

Json::Value SiftParamsToJSON(SiftParams *params) {
  Json::Value obj;
  obj["name"] = params->name;
  obj["type"] = params->type;
  obj["numBins"] = params->numBins;
  obj["cellWidth"] = params->cellWidth;
  obj["numBlocks"] = params->numBlocks;
  obj["numBlocksX"] = params->numBlocksY;
  obj["smoothWidth"] = params->smoothWidth;
  obj["combineOpposite"] = params->combineOpposite;
  obj["normalize"] = params->normalize;
  obj["subsample"] = params->subsample;
  obj["rot"] = params->rot;
  obj["maxScales"] = params->maxScales;
  obj["maxOrientations"] = params->maxOrientations;
  obj["normMax"] = params->normMax;
  obj["normEps"] = params->normEps;
  obj["subsamplePower"] = params->subsamplePower;
  obj["dims"] = params->dims;
  obj["softHistogram"] = params->softHistogram;
  return obj;
}

SiftParams SiftParamsFromJSON(Json::Value root) {
  SiftParams params;
  InitHOGParams(&params);
  strcpy(params.name, root.get("name", "HOG").asString().c_str());
  strcpy(params.type, root.get("type", "HOG").asString().c_str());
  params.numBins = root.get("numBins", params.numBins).asInt();
  params.subsample = root.get("subsample", params.subsample).asInt();
  params.subsamplePower = root.get("subsamplePower", params.subsamplePower).asFloat();
  params.rot = root.get("rot", params.rot).asFloat();
  params.maxScales = root.get("maxScales", params.maxScales).asInt();
  params.maxOrientations = root.get("maxOrientations", params.maxOrientations).asInt();
  params.smoothWidth = root.get("smoothWidth", params.smoothWidth).asInt();
  params.numBins = root.get("numBins", params.numBins).asInt();
  params.combineOpposite = root.get("combineOpposite", params.combineOpposite).asBool();
  params.cellWidth = root.get("cellWidth", params.cellWidth).asInt();
  params.numBlocks = root.get("numBlocks", params.numBlocks).asInt();
  params.numBlocksX = root.get("numBlocksX", params.numBlocks).asInt();
  params.numBlocksY = root.get("numBlocksY", params.numBlocks).asInt();
  params.dims = root.get("dims", params.numBlocksX*params.numBlocksY*params.numBins).asInt();
  params.normalize = root.get("normalize", params.normalize).asBool();
  params.normMax = root.get("normMax", params.normMax).asFloat();
  params.normEps = root.get("normEps", params.normEps).asFloat();
  params.softHistogram = root.get("softHistogram", params.softHistogram).asBool();
  return params;
}

/*
 * Rotate a sift or HOG descriptor by intervals of (360/numBins) degrees, by permuting the descriptor bins
 * Parameters: 
 *   descriptorIn: input sift descriptor
 *   descriptorOut: output (rotated) sift descriptor
 *   rot: rotate the sift descriptor by rot*(360/numBins) degrees
 *   params: parameters describing the sift descriptor
 */
void RotateSiftDescriptor(float *descriptorIn, float *descriptorOut, int rotat, SiftParams *params, bool flip) {
  int bins[100], i, j;
  float *tmp = NULL;
  if(descriptorIn == descriptorOut) {
    tmp = (float*)malloc(sizeof(float)*params->dims);
    memcpy(tmp, descriptorIn, sizeof(float)*params->dims);
    descriptorIn = tmp;
  }

  int nh = params->combineOpposite ? 2 : 1;
  int nb = params->numBins*nh;
  int rot = (nb-rotat*nh)%params->numBins;
  for(i = 0; i < params->numBins; i++) 
    bins[i] = (i+rot)%params->numBins;
  if(flip) {
    int nb2 = params->numBins/(params->combineOpposite ? 1 : 2);
    for(i = 0; i < params->numBins; i++) 
      bins[i] = nb2-1-bins[i] + (!params->combineOpposite && bins[i] >=nb2 ? params->numBins : 0);
  }
  for(i = 0; i < params->numBlocksX*params->numBlocksY; i++, descriptorOut += params->numBins, descriptorIn += params->numBins) 
    for(j = 0; j < params->numBins; j++) 
      descriptorOut[bins[j]] = descriptorIn[j];

  if(tmp) free(tmp);
}

/* 
 * Scale down the image, if necessary
 * Parameters:
 *   src: Input image
 *   subsample: src will be scaled down by a factor of p^subsample 
 *   freeSrc: if subsampling occurred, this is set to true
 * Returns: scaled down image
 */
IplImage *SiftSubsampleImage(IplImage *src, float p, int subsample, bool *freeSrc) {
  IplImage *oldImg = src;
  float scale = pow(p, subsample);

  assert((int)(oldImg->width/scale) > 0 && (int)(oldImg->height/scale) > 0);

  if(subsample && src) {
    while(scale > 2) {
      src = cvCreateImage(cvSize(oldImg->width>>1,oldImg->height>>1), oldImg->depth, oldImg->nChannels);
      cvPyrDown(oldImg, src);
      if(*freeSrc) cvReleaseImage(&oldImg);
      *freeSrc = true;
      oldImg = src;
      scale /= 2;
    }
    
    if((int)(oldImg->width/scale) != oldImg->width || (int)(oldImg->height/scale) != oldImg->height) {
      src = cvCreateImage(cvSize((int)(oldImg->width/scale),(int)(oldImg->height/scale)), oldImg->depth, oldImg->nChannels);
      cvResize(oldImg, src);
      if(*freeSrc) cvReleaseImage(&oldImg);
      *freeSrc = true;
    }
  }
  return src;
}



/*
 * Gaussian smooth the image, if necessary.  Also convert the image to grayscale
 * Parameters: 
 *   img -- source image (either 8-bit grayscale or 24-bit RGB)
 *   smoothWidth -- the width of the Gaussian smoothing kernel (or 0 for no smoothing)
 * Returns: an 8-bit grayscale Gaussian smoothed image
 */
IplImage *SiftSmoothImage(IplImage *src, int smoothWidth) {
  IplImage *img = src->nChannels == 1 ? cvCloneImage(src) : cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 1);
  IplImage *blurred = smoothWidth ? cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_8U, 1) : img;
  if(src->nChannels == 3) cvCvtColor(src, img, CV_BGR2GRAY);
  if(smoothWidth) cvSmooth(img, blurred, CV_GAUSSIAN, smoothWidth, smoothWidth);
  if(blurred != img) cvReleaseImage(&img);

  return blurred;
}

/*
 * Compute the x and y image derivatives.  Use that top compute the gradient magnitude, and
 * assign each pixel location an orientation.  
 * Parameters:
 *   img -- source image (single channel floating point)
 *   gradMag, gradBin -- the outputs of this function
 *   numBins -- the number of orientation bins
 *   cellWidth -- the output images gradMag and gradBin are both padded with cellWidth/2 pixels on all sides
 *   rot -- a rotation to be applied to each orientation bin (an angle shift of the orientation bins)
 *   combineOpposite -- if this is true, angles 180 degrees apart are considered to be the same
 * Outputs: 
 *   gradMag -- stores the gradient magnitude for each pixel location (single channel floating point)
 *   gradBin -- stores the index of the assigned orientation bin for each pixel location (single channel integer)
 *     
 */
void SiftComputeGradients(IplImage *img, IplImage *gradMag, IplImage *gradBin, int numBins, int cellWidth, float rot, bool combineOpposite) {
  cvZero(gradMag);
  cvZero(gradBin);
  int hw = cellWidth/2, i, j, k;
  unsigned char *gPtr2 = (unsigned char*)gradMag->imageData + gradMag->widthStep*(hw+1) + (hw)*sizeof(float);
  float *gPtr;
  unsigned char *iPtr2 = (unsigned char*)gradBin->imageData + gradBin->widthStep*(hw+1) + (hw)*sizeof(int);
  int *iPtr;
  float uu[1000], vv[1000];
  float dx, dy, dot, bestDot, ang;
  float s = (float)(combineOpposite ? M_PI : (2*M_PI));
  for(i = 0; i < numBins; i++) {
    ang = s*(i + (numBins%2)/2.0f) / numBins + rot;
    uu[i] = cos(ang);
    vv[i] = sin(ang);
  }
  if(img->nChannels == 1) {
    unsigned char *ptr2 = (unsigned char*)img->imageData + img->widthStep + 1, *ptr;
    for(i = 1; i < img->height-1; i++, ptr2 += img->widthStep, gPtr2 += gradMag->widthStep, iPtr2 += gradBin->widthStep) {
      for(j = 1, gPtr = (float*)gPtr2, ptr = ptr2, iPtr = (int*)iPtr2; j < img->width-1; j++, ptr++) {
	dx = (float)ptr[1]-(float)ptr[-1];
	dy = (float)ptr[img->widthStep] - (float)ptr[-img->widthStep];
	if(combineOpposite && dy < 0) { dx = -dx; dy = -dy; }
	
	gPtr[j] = sqrt(dx*dx + dy*dy);  // gradient magnitude
	
	// assign to the appropriate orientation bin
	bestDot = 0;
	for(k = 0; k < numBins; k++) {
	  dot = uu[k]*dx + vv[k]*dy;
	if(dot > bestDot) { iPtr[j] = k; bestDot = dot; }
	}
      }
    }
  } else {
    assert(img->nChannels == 3);
    float dx1, dy1, v1, dx2, dy2, v2, dx3, dy3, v3, v; 
    unsigned char *ptr2 = (unsigned char*)img->imageData + img->widthStep + 3, *ptr;
    for(i = 1; i < img->height-1; i++, ptr2 += img->widthStep, gPtr2 += gradMag->widthStep, iPtr2 += gradBin->widthStep) {
      for(j = 1, gPtr = (float*)gPtr2, ptr = ptr2, iPtr = (int*)iPtr2; j < img->width-1; j++, ptr+=3) {
	dx1 = (float)ptr[3]-(float)ptr[-3];
	dy1 = (float)ptr[img->widthStep] - (float)ptr[-img->widthStep];
	v1 = dx1*dx1 + dy1*dy1;
	dx2 = (float)ptr[4]-(float)ptr[-2];
	dy2 = (float)ptr[img->widthStep+1] - (float)ptr[-img->widthStep+1];
	v2 = dx2*dx2 + dy2*dy2;
	dx3 = (float)ptr[5]-(float)ptr[-1];
	dy3 = (float)ptr[img->widthStep+2] - (float)ptr[-img->widthStep+2];
	v3 = dx3*dx3 + dy3*dy3;
	if(v1 > v3) {
	  if(v1 > v2) { dx = dx1; dy = dy1;  v = v1; }
	  else { dx = dx2; dy = dy2;  v = v2; }
	} else {
	  if(v3 > v2) { dx = dx3; dy = dy3;  v = v3; }
	  else { dx = dx2; dy = dy2;  v = v2; }
	}
	if(combineOpposite && dy < 0) { dx = -dx; dy = -dy; }
	
	gPtr[j] = sqrt(v);  // gradient magnitude
	
	// assign to the appropriate orientation bin
	bestDot = 0;
	for(k = 0; k < numBins; k++) {
	  dot = uu[k]*dx + vv[k]*dy;
	  if(dot > bestDot) { iPtr[j] = k; bestDot = dot; }
	}
      }
    }
  }
}

IplImage *SiftComputeGradientHistogramsSoft(IplImage *img, int numBins, int cellWidth, float rot, bool combineOpposite) {
  IplImage *grad = cvCreateImage(cvSize(img->width+cellWidth,img->height+cellWidth), IPL_DEPTH_32F, numBins);
  cvZero(grad);
  int hw = cellWidth/2, i, j, k, last;
  unsigned char *gPtr2 = (unsigned char*)grad->imageData + grad->widthStep*(hw+1) + (hw)*sizeof(float)*grad->nChannels;
  float *gPtr;
  float uu[1000], vv[1000];
  float dx, dy, dot, lastDot, cross, lastCross, ang, m, w;
  float s = (float)(combineOpposite ? M_PI : (2*M_PI));
  for(i = 0; i < numBins; i++) {
    ang = s*(i + (numBins%2)/2.0f) / numBins + rot;
    uu[i] = cos(ang);
    vv[i] = sin(ang);
  }
  if(img->nChannels == 1) {
    unsigned char *ptr2 = (unsigned char*)img->imageData + img->widthStep + 1, *ptr;
    for(i = 1; i < img->height-1; i++, ptr2 += img->widthStep, gPtr2 += grad->widthStep) {
      for(j = 1, gPtr = (float*)gPtr2, ptr = ptr2; j < img->width-1; j++, ptr++, gPtr += grad->nChannels) {
	dx = (float)ptr[1]-(float)ptr[-1];
	dy = (float)ptr[img->widthStep] - (float)ptr[-img->widthStep];
	if(combineOpposite && dy < 0) { dx = -dx; dy = -dy; }
	
	m = sqrt(dx*dx + dy*dy);  // gradient magnitude
	
	// assign as a weighted average between the two closest orientation bins
	lastCross = uu[numBins-1]*dy - vv[numBins-1]*dx;
	last = numBins-1;
	for(k = 0; k < numBins; k++) {
	  cross = uu[k]*dy - vv[k]*dx;
	  if((cross >= 0) != (lastCross >= 0)) {
	    dot = uu[k]*dx + vv[k]*dy;
	    if(dot > 0) {
	      lastDot = uu[last]*dx + vv[last]*dy;
	      w = dot/(dot+lastDot);
	      gPtr[k] = m*w;
	      gPtr[last] = m*(1-w);
	      break;
	    }
	  }
	  last = k;
	  lastCross = cross;
	}
      }
    }
  } else {
    assert(img->nChannels == 3);
    float dx1, dy1, v1, dx2, dy2, v2, dx3, dy3, v3, v; 
    unsigned char *ptr2 = (unsigned char*)img->imageData + img->widthStep + 3, *ptr;
    for(i = 1; i < img->height-1; i++, ptr2 += img->widthStep, gPtr2 += grad->widthStep) {
      for(j = 1, gPtr = (float*)gPtr2, ptr = ptr2; j < img->width-1; j++, ptr+=3, gPtr += grad->nChannels) {
	// Use the color channel with maximum gradient magnitude
	dx1 = (float)ptr[3]-(float)ptr[-3];
	dy1 = (float)ptr[img->widthStep] - (float)ptr[-img->widthStep];
	v1 = dx1*dx1 + dy1*dy1;
	dx2 = (float)ptr[4]-(float)ptr[-2];
	dy2 = (float)ptr[img->widthStep+1] - (float)ptr[-img->widthStep+1];
	v2 = dx2*dx2 + dy2*dy2;
	dx3 = (float)ptr[5]-(float)ptr[-1];
	dy3 = (float)ptr[img->widthStep+2] - (float)ptr[-img->widthStep+2];
	v3 = dx3*dx3 + dy3*dy3;
	if(v1 > v3) {
	  if(v1 > v2) { dx = dx1; dy = dy1;  v = v1; }
	  else { dx = dx2; dy = dy2;  v = v2; }
	} else {
	  if(v3 > v2) { dx = dx3; dy = dy3;  v = v3; }
	  else { dx = dx2; dy = dy2;  v = v2; }
	}
	if(combineOpposite && dy < 0) { dx = -dx; dy = -dy; }
	
	m = sqrt(v);  // gradient magnitude
	
	// assign as a weighted average between the two closest orientation bins
	lastCross = uu[numBins-1]*dy - vv[numBins-1]*dx;
	last = numBins-1;
	for(k = 0; k < numBins; k++) {
	  cross = uu[k]*dy - vv[k]*dx;
	  if((cross >= 0) != (lastCross >= 0)) {
	    dot = uu[k]*dx + vv[k]*dy;
	    lastDot = uu[last]*dx + vv[last]*dy;
	    w = my_abs(dot)/(my_abs(dot)+my_abs(lastDot));
	    gPtr[k] = m*(1-w);
	    gPtr[last] = m*w;
	    break;
	  }
	  last = k;
	  lastCross = cross;
	}
      }
    }
  }

  int smoothWidth = 2*cellWidth-1;
  cvSmooth(grad, grad, CV_GAUSSIAN, smoothWidth, smoothWidth, 2*cellWidth);
  return grad;
}



/* For each pixel, compute a histogram summing the gradient magnitudes over a cellWidthXcellWidth 
 * grid centered at that pixel for each orientation bin.
 * Parameters:
 *   gradMag -- the gradient magnitude for each pixel location (single channel floating point)
 *   gradBin -- the index of the assigned orientation bin for each pixel location (single channel integer)
 *   numBins -- the number of orientation bins
 *   cellWidth -- the width in pixels of each histogram cell
 * Returns: a numBins channel floating point image which stores the histogram summing the gradient magnitudes 
 *   over a cellWidthXcellWidth grid centered at that pixel for each orientation bin.
 */
IplImage *SiftComputeGradientHistograms(IplImage *gradMag, IplImage *gradBin, int numBins, int cellWidth) {
  int i, j, k;
  int hw = cellWidth/2;
  float *gPtr;
  int *iPtr;

  IplImage *imgNew = cvCreateImage(cvSize(gradMag->width-cellWidth,gradMag->height-cellWidth), IPL_DEPTH_32F, numBins);
  float *dstPtr;
  cvZero(imgNew);

  // Compute the sum of a 1XcellWidth vertical strip of pixels for each pixel in the first scan line
  float **vertLineSum = (float**)malloc(gradMag->width*(sizeof(float*)+numBins*sizeof(float)));
  for(i = 0; i < gradMag->width; i++) {
    vertLineSum[i] = ((float*)(vertLineSum+gradMag->width))+i*numBins;
    for(j = 0; j < numBins; j++)
      vertLineSum[i][j] = 0;
    for(j = hw; j < cellWidth; j++)
      vertLineSum[i][((int*)(gradBin->imageData+gradBin->widthStep*j))[i]] += ((float*)(gradMag->imageData+gradMag->widthStep*j))[i];
  }

  // Compute the sum cell response for the pixel location in the upper left corner of the image.
  // This is necessary because all future computations simply add/subtract pixels from the previous
  // cell
  dstPtr = (float*)imgNew->imageData;
  for(j = 0; j < numBins; j++)
    dstPtr[j] = 0;
  for(i = hw; i < cellWidth; i++) 
    for(j = hw; j < cellWidth; j++)
      dstPtr[((int*)(gradBin->imageData+gradBin->widthStep*i))[j]] += ((float*)(gradMag->imageData+gradMag->widthStep*i))[j];

  // In the output image, store the histogram counts for each orientation in a cellWidthXcellWidth
  // cell centered at each pixel.  This can be computed efficiently in a single pass by adding
  // and subtracting the sum in a vertical line strip to the cell response of the previous pixel
  int ySkip = hw*gradMag->widthStep;
  int yiSkip = hw*gradBin->widthStep;
  float *gPtrN, *gPtrP;
  int *iPtrN, *iPtrP;
  for(i = hw; i < gradMag->height-hw; i++) {
    gPtr = ((float*)(gradMag->imageData+gradMag->widthStep*i));
    iPtr = ((int*)(gradBin->imageData+gradBin->widthStep*i));
    gPtrN = (float*)(((unsigned char*)gPtr)+ySkip);
    gPtrP = (float*)(((unsigned char*)gPtr)-ySkip);
    iPtrN = (int*)(((unsigned char*)iPtr)+yiSkip);
    iPtrP = (int*)(((unsigned char*)iPtr)-yiSkip);
    dstPtr = ((float*)(imgNew->imageData+imgNew->widthStep*(i-hw)))+numBins;

    // Update the current cell response using the cell response of the previous x-location
    for(j = hw; j < gradMag->width-hw-1; j++, dstPtr += numBins) 
      for(k = 0; k < numBins; k++) 
	dstPtr[k] = dstPtr[k-numBins] + vertLineSum[j+hw][k] - vertLineSum[j-hw][k];

    // Update the vertical strip sums for the next scan line
    if(i < gradMag->height-hw-1) {
      for(j = 0; j < gradMag->width; j++) {
	vertLineSum[j][iPtrN[j]] += gPtrN[j];
	vertLineSum[j][iPtrP[j]] -= gPtrP[j];
      }
    }

    // Update the cell response for the 1st x-location of the next scan line
    if(i < gradMag->height-hw-1) {
      dstPtr = ((float*)(imgNew->imageData+imgNew->widthStep*(i+1-hw)));
      memcpy(dstPtr, imgNew->imageData+imgNew->widthStep*(i-hw), sizeof(float)*numBins);
      for(j = hw; j < cellWidth; j++) {
	dstPtr[iPtrN[j]] += gPtrN[j];
	dstPtr[iPtrP[j]] -= gPtrP[j];
      }
    }
  }
  free(vertLineSum);
  
  return imgNew;
}



IplImage *SiftComputeGradientHistogramsSubResolution(IplImage *gradMag, IplImage *gradBin, int numBins, int cellWidth, float scale) {
  int i, j, k;
  float hw = ((float)cellWidth*scale)/2.0f;
  int ihw = (int)hw;
  float w2 = hw-(float)ihw;
  float w1 = 1.0f-w2;
  float *gPtr;
  int *iPtr;
  int hcw = cellWidth/2;
  assert(scale < 1);

  IplImage *imgNew = cvCreateImage(cvSize(gradMag->width-cellWidth,gradMag->height-cellWidth), IPL_DEPTH_32F, numBins);
  float *dstPtr;
  cvZero(imgNew);

  // Compute the sum of a 1XcellWidth vertical strip of pixels for each pixel in the first scan line
  float **vertLineSum = (float**)malloc(gradMag->width*(sizeof(float*)+numBins*sizeof(float)));
  for(i = 0; i < gradMag->width; i++) {
    vertLineSum[i] = ((float*)(vertLineSum+gradMag->width))+i*numBins;
    for(j = 0; j < numBins; j++)
      vertLineSum[i][j] = 0;
    for(j = hcw; j < hcw+ihw+1; j++) {
      float w = j == hcw+ihw ? w2 : 1;
      vertLineSum[i][((int*)(gradBin->imageData+gradBin->widthStep*j))[i]] += ((float*)(gradMag->imageData+gradMag->widthStep*j))[i]*w;
    }
  }

  // Compute the sum cell response for the pixel location in the upper left corner of the image.
  // This is necessary because all future computations simply add/subtract pixels from the previous
  // cell
  dstPtr = (float*)imgNew->imageData;
  for(j = 0; j < numBins; j++)
    dstPtr[j] = 0;
  for(i = hcw; i < hcw+ihw+1; i++) {
    float w = i == hcw+ihw ? w2 : 1;
    for(j = hcw; j < hcw+ihw+1; j++) {
      if(j == hcw+ihw) w *= w2;
      dstPtr[((int*)(gradBin->imageData+gradBin->widthStep*i))[j]] += ((float*)(gradMag->imageData+gradMag->widthStep*i))[j]*w;
    }
  }

  // In the output image, store the histogram counts for each orientation in a cellWidthXcellWidth
  // cell centered at each pixel.  This can be computed efficiently in a single pass by adding
  // and subtracting the sum in a vertical line strip to the cell response of the previous pixel
  int ySkip = ihw*gradMag->widthStep, ySkip2 = (ihw+1)*gradMag->widthStep;
  int yiSkip = ihw*gradBin->widthStep, yiSkip2 = (ihw+1)*gradBin->widthStep;
  float *gPtrN, *gPtrP, *gPtrN2, *gPtrP2;
  int *iPtrN, *iPtrP, *iPtrN2, *iPtrP2;
  for(i = hcw; i < gradMag->height-hcw; i++) {
    gPtr = ((float*)(gradMag->imageData+gradMag->widthStep*i));
    iPtr = ((int*)(gradBin->imageData+gradBin->widthStep*i));
    gPtrN = (float*)(((unsigned char*)gPtr)+ySkip);
    gPtrN2 = (float*)(((unsigned char*)gPtr)+ySkip2);
    gPtrP = (float*)(((unsigned char*)gPtr)-ySkip);
    gPtrP2 = (float*)(((unsigned char*)gPtr)-ySkip2);
    iPtrN = (int*)(((unsigned char*)iPtr)+yiSkip);
    iPtrN2 = (int*)(((unsigned char*)iPtr)+yiSkip2);
    iPtrP = (int*)(((unsigned char*)iPtr)-yiSkip);
    iPtrP2 = (int*)(((unsigned char*)iPtr)-yiSkip2);
    dstPtr = ((float*)(imgNew->imageData+imgNew->widthStep*(i-hcw)))+numBins;

    // Update the current cell response using the cell response of the previous x-location
    for(j = hcw; j < gradMag->width-hcw-1; j++, dstPtr += numBins) 
      for(k = 0; k < numBins; k++) 
	dstPtr[k] = dstPtr[k-numBins] + (vertLineSum[j+ihw][k]*w1+vertLineSum[j+ihw+1][k]*w2)
	                              - (vertLineSum[j-ihw][k]*w1+vertLineSum[j-ihw-1][k]*w2);

    // Update the vertical strip sums for the next scan line
    if(i < gradMag->height-hcw-1) {
      for(j = 0; j < gradMag->width; j++) {
	vertLineSum[j][iPtrN[j]] += gPtrN[j]*w1 + gPtrN2[j]*w2;    
	vertLineSum[j][iPtrP[j]] -= gPtrP[j]*w1 + gPtrP2[j]*w2;    
      }
    }

    // Update the cell response for the 1st x-location of the next scan line
    if(i < gradMag->height-hcw-1) {
      dstPtr = ((float*)(imgNew->imageData+imgNew->widthStep*(i+1-hcw)));
      memcpy(dstPtr, imgNew->imageData+imgNew->widthStep*(i-hcw), sizeof(float)*numBins);
      for(j = hcw; j < hcw+(ihw); j++) {
	dstPtr[iPtrN[j]] += gPtrN[j]*w1+gPtrN2[j]*w2;
	dstPtr[iPtrP[j]] -= gPtrP[j]*w1+gPtrP2[j]*w2;
      }
      dstPtr[iPtrN[j]] += (gPtrN[j]*w1+gPtrN2[j]*w2)*w1;
      dstPtr[iPtrP[j]] -= (gPtrP[j]*w1+gPtrP2[j]*w2)*w1;
    }
  }
  free(vertLineSum);
  
  return imgNew;
}




void SiftComputeSquaredMagnitude(IplImage *img, IplImage *sqrMagn) {
  unsigned char *ptr2 = (unsigned char*)img->imageData;
  unsigned char *sqrMagnPtr2 = (unsigned char*)sqrMagn->imageData;
  float *ptr, *sqrMagnPtr;
  int i, j, k;

  // For each pixel, compute the sum squared magnitude over each orientation bin
  for(i = 0; i < img->height; i++, ptr2 += img->widthStep, sqrMagnPtr2 += sqrMagn->widthStep) {
    ptr = (float*)ptr2;
    sqrMagnPtr = (float*)sqrMagnPtr2;
    for(j = 0; j < img->width; j++, ptr += img->nChannels) {
      sqrMagnPtr[j] = 0;
      for(k = 0; k < img->nChannels; k++) 
	sqrMagnPtr[j] += ptr[k]*ptr[k];
    }
  }
}

void  SiftL2Normalize(float *desc, int d, float siftNormEps) {
  float sumSqr = PointMagnitude_sqr<float>(desc, d);
  float m = (sqrt(sumSqr+siftNormEps));  //(sqrt(sumSqr)+siftNormEps);
  for(int i = 0; i < d; i++)
    desc[i] /= m;
}

void SiftNormalize(float *desc, int d, float siftNormMax, float siftNormEps) {
  SiftL2Normalize(desc, d, siftNormEps);
  for(int i = 0; i < d; i++)
    desc[i] = my_min(desc[i], siftNormMax);
  SiftL2Normalize(desc, d, siftNormEps);
}


/* Locally normalize a sift descriptor image, similar to the method used in Dalal/Triggs (and also
 * to Felzenzwalb/Ramanan)
 * Parameters:
 *   img -- the image to be normalized, as returned by SiftComputeGradientHistograms().  This image is 
 *          modified by this function.
 *   cellWidth -- the width in pixels of each histogram cell
 *   siftNormEps, siftNormMax -- constants used for normalization, taken from Felzenzwalb/Ramanan's code
 */
void HogNormalize(IplImage *img, int cellWidth, float siftNormEps, float siftNormMax) {
  // Compute per block normalization constants
  IplImage *norm = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_32F, 1);
  IplImage *norm2 = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_32F, 1);
  cvZero(norm);
  cvZero(norm2);
  unsigned char *ptr2 = (unsigned char*)img->imageData;
  unsigned char *normPtr2 = (unsigned char*)norm->imageData;
  float *ptr, *normPtr;
  int i, j, k;

  // For each pixel, compute the sum squared magnitude over each orientation bin
  for(i = 0; i < img->height; i++, ptr2 += img->widthStep, normPtr2 += norm->widthStep) {
    ptr = (float*)ptr2;
    normPtr = (float*)normPtr2;
    for(j = 0; j < img->width; j++, ptr += img->nChannels) {
      normPtr[j] = 0;
      for(k = 0; k < img->nChannels; k++) 
	normPtr[j] += ptr[k]*ptr[k];
    }
  }


  // Normalize each bin in the descriptor as the average normalization over each 2X2 block 
  normPtr2 = (unsigned char*)norm->imageData;
  unsigned char *normPtr2_2 = (unsigned char*)norm2->imageData;
  float *normPtr_2, *normPtrPrev, *normPtrNext;
  int jPrev, jNext, hw=cellWidth/2, maxX = img->width-1, maxY = img->height-1;
  for(i = 0; i < img->height; i++, normPtr2 += norm->widthStep, normPtr2_2 += norm2->widthStep) {
    normPtr_2 = (float*)normPtr2_2;
    normPtrPrev = (float*)(normPtr2 - my_min(i,cellWidth)*norm->widthStep);
    normPtrNext = (float*)(normPtr2 + my_min(maxY-i,cellWidth)*norm->widthStep);
    for(j = 0; j < img->width; j++) {
      jPrev = j-my_min(j, cellWidth);
      jNext = j+my_min(maxX-j, cellWidth);
      normPtr_2[j] = 1.0f / sqrt(normPtrPrev[jPrev] + normPtrPrev[jNext] + normPtrNext[jPrev] + normPtrNext[jNext] + siftNormEps);
    }
  }
  
  // Normalize each bin in the descriptor as the average normalization over each 2X2 block (as done by Felzenzwalb/Ramanan)
  normPtr2 = (unsigned char*)norm2->imageData;
  ptr2 = (unsigned char*)img->imageData;
  for(i = 0; i < img->height; i++, ptr2 += img->widthStep, normPtr2 += norm2->widthStep) {
    ptr = (float*)ptr2;
    normPtr = (float*)normPtr2;
    normPtrPrev = (float*)(normPtr2 - my_min(i,hw)*norm2->widthStep);
    normPtrNext = (float*)(normPtr2 + my_min(maxY-i,hw)*norm->widthStep);
    for(j = 0; j < img->width; j++, ptr += img->nChannels) {
      jPrev = j-my_min(j, hw);
      jNext = j+my_min(maxX-j, hw);
      for(k = 0; k < img->nChannels; k++) {
	ptr[k] =  my_min(normPtrPrev[jPrev]*ptr[k], siftNormMax) + my_min(normPtrPrev[jNext]*ptr[k], siftNormMax) +
	  my_min(normPtrNext[jPrev]*ptr[k], siftNormMax) + my_min(normPtrNext[jNext]*ptr[k], siftNormMax);
      }
    }
  }
  cvReleaseImage(&norm);
  cvReleaseImage(&norm2);
}


/*
 * Visualize a HOG descriptor (similar to as in Felzenzwalb/Ramanan)
 */
IplImage *VisualizeHOGDescriptor(float *descriptor, int numX, int numY, int cellWidth, int numBins, bool combineOpposite, float mi, float ma) {
  IplImage *img = cvCreateImage(cvSize(cellWidth*numX,cellWidth*numY), IPL_DEPTH_32F, mi < 0 ? 3 : 1);
  float *ptr = descriptor;
  float dx[100], dy[100], ang, v;
  int i, j, k;
  int x = cellWidth/2, y = cellWidth/2;
  cvZero(img);
  if(!mi && !ma) {
    mi=10000000;
    ma=-10000000;
    for(i = 0; i < numX*numY*numBins; i++) {
      mi = my_min(mi, descriptor[i]);
      ma = my_max(ma, descriptor[i]);
    }
  } 

  float s = (float)(combineOpposite ? M_PI : (2*M_PI));
  for(i = 0; i < numBins; i++) {
    ang = s*(i+(numBins%2)/2.0f)/numBins;
    dx[i] = -cellWidth*sin(ang)/2;
    dy[i] = cellWidth*cos(ang)/2;
  }
  
  for(i = 0; i < numY; i++, y += cellWidth) {
    for(j = 0, x = cellWidth/2; j < numX; j++, ptr += numBins, x += cellWidth) {
      for(k = 0; k < numBins; k++) {
	v = img->nChannels > 1 ? ptr[k]/my_max(my_abs(ma),my_abs(mi)) : (ptr[k]-mi)/(ma-mi);
	cvTransparentLine(img, cvPoint2D32f(x-dx[k],y-dy[k]), cvPoint2D32f(x+dx[k],y+dy[k]), v);
      }
    }
  }

  IplImage *retval = img->nChannels == 3 ? MinMaxNormImage(img, 0, NULL, NULL, my_max(my_abs(ma),my_abs(mi))) : MinMaxNormImage(img);
  cvReleaseImage(&img);
  return retval;
}

IplImage *VisualizeHOGDescriptor(float *descriptor, SiftParams *params) {
  return VisualizeHOGDescriptor(descriptor, params->numBlocksX, params->numBlocksY, params->cellWidth, params->numBins, params->combineOpposite);
}

#ifdef TEST_HOG
int main(int argc, char **argv) {
  if(argc < 2) { fprintf(stderr, "USAGE: ./testHOG <imageNameIn>\n"); return -1; }
  IplImage *img = cvLoadImage(argv[1]);
  if(!img) { fprintf(stderr, "Could not open image %s\n", argv[1]); return -1; }
  
  float subsamplePower;
  int i, n;
  SiftParams params;  // Initialize default HOG parameters
  InitHOGParams(&params);
  for(i = 2; i < argc; i++) {
    if(strstr(argv[i], "sift")) InitSiftParams(&params);
    if(sscanf(argv[i], "maxScales=%d", &n)) params.maxScales=n;
    if(sscanf(argv[i], "normalize=%d", &n)) params.normalize=n;
    if(sscanf(argv[i], "combineOpposite=%d", &n)) params.combineOpposite=n;
    if(sscanf(argv[i], "cellWidth=%d", &n)) params.cellWidth=n;
    if(sscanf(argv[i], "numBins=%d", &n)) params.numBins=n;
    if(sscanf(argv[i], "smoothWidth=%d", &n)) params.smoothWidth=n;
    if(sscanf(argv[i], "subsamplePower=%f", &f)) params.subsamplePower=f;
  }
  params.maxScales=4;

  char str[400];
  for(i = 0; i < params.maxScales; i++) {
    params.subsample=i;
    IplImage *hogImg = ComputeSiftImage(img, &params); // Compute an image caching all HOG cell histograms
    int x = params.cellWidth/2, y = params.cellWidth/2, numX = hogImg->width/params.cellWidth, numY = hogImg->height/params.cellWidth;

    // Compute a descriptor that samples cell histograms on a regular grid
    float *descriptor = (float*)malloc(numX*numY*params.numBins*sizeof(float));
    ComputeHOGDescriptor(hogImg, descriptor, x, y, numX, numY, &params);

    // Draw a visualization of the HOG descriptor, and save it to the specified output image
    IplImage *vis = VisualizeHOGDescriptor(descriptor, numX, numY, params.cellWidth, params.numBins, params.combineOpposite);
    sprintf(str, "hog%d.png", i);
    cvSaveImage(str, vis);
  
    cvReleaseImage(&vis);
    cvReleaseImage(&hogImg);
    free(descriptor);
  }
  cvReleaseImage(&img);

  return 0;
}
#endif







/* 
 * Helper function to draw a line into an image with transparency
 */
void cvTransparentLine(IplImage *img, CvPoint2D32f pt1, CvPoint2D32f pt2, float alpha) {
  float x, y, m;
  int ix, iy;
  CvPoint2D32f tmp;
  float *ptr;

  if(my_abs(pt1.x-pt2.x) > my_abs(pt1.y-pt2.y)) {
    if(pt1.x > pt2.x) { tmp = pt1; pt1 = pt2; pt2 = tmp; }
    x = pt1.x; y = pt1.y;
    m = (pt2.y-pt1.y)/(pt2.x-pt1.x);
    while(x < pt2.x) {
      ix = (int)(x+.5f);
      iy = (int)(y+.5f);
      if(ix >= 0 && ix < img->width && iy >= 0 && iy < img->height) {
	if(img->nChannels == 3) {
	  int c = alpha < 0 ? 0 : 2;
	  ptr = ((float*)(img->imageData+iy*img->widthStep)) + ix*3;
	  ptr[c] = my_max(ptr[c], my_abs(alpha)*255);
	} else {
	  ptr = ((float*)(img->imageData+iy*img->widthStep)) + ix;
	  ptr[0] = my_max(ptr[0], alpha*255);
	}
      }
      x++;
      y += m;
    }
  } else {
    if(pt1.y > pt2.y) { tmp = pt1; pt1 = pt2; pt2 = tmp; }
    x = pt1.x; y = pt1.y;
    m = (pt2.x-pt1.x)/(pt2.y-pt1.y);
    while(y < pt2.y) {
      ix = (int)(x+.5f);
      iy = (int)(y+.5f);
      if(ix >= 0 && ix < img->width && iy >= 0 && iy < img->height) {
	if(img->nChannels == 3) {
	  int c = alpha < 0 ? 0 : 2;
	  ptr = ((float*)(img->imageData+iy*img->widthStep)) + ix*3;
	  ptr[c] = my_max(ptr[c], my_abs(alpha)*255);
	} else {
	  ptr = ((float*)(img->imageData+iy*img->widthStep)) + ix;
	  ptr[0] = my_max(ptr[0], alpha*255);
	}
      }
      y++;
      x += m;
    }
  }
}

HOGTemplateFeature::HOGTemplateFeature(FeatureOptions *fo, SiftParams p) : TemplateFeature(fo, p) {
  name = StringCopy(params.name);
  denseFeatures = NULL;
}
    
void HOGTemplateFeature::Clear(bool full) {
  TemplateFeature::Clear(full);
  if(denseFeatures) {
    for(int i = 0; i < params.maxScales; i++) 
      cvReleaseImage(&denseFeatures[i]);
    free(denseFeatures);
    denseFeatures = NULL;
  }
}

IplImage *****HOGTemplateFeature::PrecomputeFeatures(bool flip) {
  if((!flip && !featureImages) || (flip && !featureImages_flip)) {
    if(g_debug > 1) fprintf(stderr, "    precompute %s %s...\n", fo->Name(), Name());
    //featureImages = PrecomputeHOGFastBegin(fo->GetImage(), &params, fo->SpatialGranularity(), fo->NumOrientations(), fo->CellWidth()/fo->SpatialGranularity());
    IplImage *img = fo->GetImage();
    params.maxScales = fo->NumScales();
    if(!denseFeatures) 
      denseFeatures = PrecomputeHOGFastBegin2(img, &params);
    IplImage *****tmp = PrecomputeHOGFastFinish2(denseFeatures, fo->GetImage(), &params, /*fo->SpatialGranularity()*/params.cellWidth, fo->NumOrientations(), NumScales(), flip);
    if(!flip) featureImages = tmp;
    else featureImages_flip = tmp;
  }
  
  //if(fo->NumOrientations() > 1 && !featureImages[1][0][0][0])
  //PrecomputeHOGFastFinish(featureImages, fo->GetImage(), &params, fo->SpatialGranularity(), fo->NumOrientations(), fo->NumScales());

  return !flip ? featureImages : featureImages_flip;  
}

IplImage *HOGTemplateFeature::Visualize(float *f, int w, int h, float mi, float ma) {
  return VisualizeHOGDescriptor(f, w, h, params.cellWidth, params.numBins, params.combineOpposite, mi, ma);
}

// Visualize an assignment to part locations as a HOG image
IplImage *HOGTemplateFeature::Visualize(Classes *classes, PartLocation *locs, bool visualizeWeights, AttributeInstance *attribute) {
  if(attribute) {
    Attribute *am = attribute->Model();
    float *tmp = new float[am->NumWeights()];
    // Visualize a particular attribute detector
    if(visualizeWeights) {
      if(!am->GetWeights(tmp, name)) {
	delete [] tmp;
	return NULL;
      }
    } else
      GetFeaturesAtLocation(tmp, am->Width(), am->Height(), 0, &locs[am->Part()->Id()], attribute->IsFlipped());
    IplImage *retval = VisualizeHOGDescriptor(tmp, am->Width(), am->Height(), params.cellWidth, params.numBins, params.combineOpposite);
    delete [] tmp;
    return retval;
  }


  // Visualize a full part-based detector
  int wi = fo->GetImage()->width, he = fo->GetImage()->height, n;
  IplImage *img = NULL;
  CvRect *rects = new CvRect[classes->NumParts()];
  float *rotations = new float[classes->NumParts()];
  int numRects = 0;
  for(int i = 0; i < classes->NumParts(); i++) {
    int x, y, scale, rot, pose, width, height;
    locs[i].GetDetectionLocation(&x, &y, &scale, &rot, &pose);

    if(pose >= 0) {
      ObjectPose *p = classes->GetPart(i)->GetPose(pose);
      Attribute *attr = p->Appearance();
      if(attr) {
	float *tmp = new float[attr->NumWeights()];
	PartLocation l(locs[i]);  
	  int s=scale, j = 0;
	  while(s < fo->ScaleOffset() && j < attr->NumFeatureTypes()) {
	    s = scale + attr->Feature(j)->scale;
	    j++;
	  }
	  float x2, y2;
	  fo->ConvertDetectionCoordinates(x, y, scale, rot, s, rot, &x2, &y2);
	  l.SetDetectionLocation(x2, y2, s, rot, pose, LATENT, LATENT);
	  if(j < attr->NumFeatureTypes() && attr->Width(j)) {
	    width = attr->Width(j);
	    height = attr->Height(j);
	    if(visualizeWeights) 
	      n = attr->GetWeights(tmp, name);
	    else
	      n = GetFeaturesAtLocation(tmp, width, height, 0, &l, false/*p->IsFlipped()*/);
	  } else n = 0;
	

	if(n) {
	  float xx, yy, scale, rot;
	  l.GetImageLocation(&xx, &yy, &scale, &rot, NULL);
	  //locs[i].GetImageLocation(&xx, &yy, &scale, &rot, NULL, &width, &height);

	  if(!img) {
	    img = cvCreateImage(cvSize(wi,he), IPL_DEPTH_8U, 1);
	    cvZero(img);
	  }
	  
	  IplImage *part = VisualizeHOGDescriptor(tmp, width, height, params.cellWidth, params.numBins, params.combineOpposite);
	  //assert(fo->NumScales() == 1 && fo->NumOrientations() == 1); // Not implemented yet for multiple scales/orientations
	  
	  RotationInfo r = ::GetRotationInfo(width*params.cellWidth, height*params.cellWidth, -rot, scale);
	  IplImage *rotImage = cvCreateImage(cvSize(r.maxX-r.minX, r.maxY-r.minY), IPL_DEPTH_8U, img->nChannels);
	  cvZero(rotImage);
	  CvMat affineMat = cvMat(2, 3, CV_32FC1, r.mat);
	  cvWarpAffineMultiChannel(part, rotImage, &affineMat);
	  
	  rotations[numRects] = rot;
	  rects[numRects] = cvRect((int)(xx), (int)(yy), width*params.cellWidth*scale, height*params.cellWidth*scale);
	  DrawImageIntoImageMax(rotImage, cvRect(0,0,r.maxX-r.minX,r.maxY-r.minY), img,
			        (int)(xx-(r.maxX-r.minX)/2), (int)(yy-(r.maxY-r.minY)/2));
	  numRects++;
	  //p->Draw(img, &locs[i], CV_RGB(0,0,200));
	  cvReleaseImage(&part);
	  cvReleaseImage(&rotImage);
	}
	delete [] tmp;
      }
    }
  }
  if(!img) return NULL;

  IplImage *img2 = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_8U, 3);
  cvConvertImage(img, img2);
  cvReleaseImage(&img);
  for(int i = 0; i < numRects; i++) {
    cvRotatedRect(img2, cvPoint(rects[i].x,rects[i].y), rects[i].width, rects[i].height, -rotations[i], CV_RGB(0,0,255));
  }
  delete [] rects;
  delete [] rotations;

  return img2;
}



