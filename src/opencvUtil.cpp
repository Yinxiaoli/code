/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#include "opencvUtil.h"


/*#ifdef _MSC_VER
#pragma comment(lib, "cv210d.lib")
#pragma comment(lib, "cxcore210d.lib")
#pragma comment(lib, "cvaux210d.lib")
#pragma comment(lib, "highgui210d.lib")
#pragma comment(lib, "ml210d.lib")
#endif*/




/*
 * Normalize an image to be on the range [0,255], and convert to 8-bit
 */
IplImage *MinMaxNormImage(IplImage *img, double *mi_p, double *ma_p, double mi, double ma) {
  CvPoint mil, mal;
  if(img->nChannels == 3) cvSetImageCOI(img, 3);
  IplImage *retval = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_8U, img->nChannels);
  if(!mi && !ma) cvMinMaxLoc(img, &mi, &ma, &mil, &mal);
  if(img->nChannels == 3) cvSetImageCOI(img, 0);
  if(mi_p) *mi_p = mi;
  if(ma_p) *ma_p = ma;
  cvCvtScale(img, retval, 255/(ma-mi), -mi*255.0/(ma-mi));
  return retval;
}


RotationInfo GetRotationInfo(int width, int height, float rot, float scale) {
  RotationInfo r;
  CvMat map_matrix = cvMat(2, 3, CV_32FC1, r.mat);
  CvMat map_matrix_inv = cvMat(2, 3, CV_32FC1, r.invMat);
  float x, y;
  int ix, iy;

  r.center = cvPoint2D32f(width/2.0f,height/2.0f);
  r.rot = rot;
  r.width = width;
  r.height = height;
  cv2DRotationMatrix(r.center, rot*180/M_PI, scale, &map_matrix);
  AffineTransformPoint(r.mat, 0, 0, &x, &y); ix = (int)(x+.5f); iy = (int)(y+.5f);  
  r.minX = r.maxX = ix; r.minY = r.maxY = iy;
  AffineTransformPoint(r.mat, (float)width, 0, &x, &y);ix = (int)(x+.5f); iy = (int)(y+.5f);  
  r.minX = my_min(r.minX, ix); r.maxX = my_max(r.maxX, ix); r.minY = my_min(r.minY, iy); r.maxY = my_max(r.maxY, iy);
  AffineTransformPoint(r.mat, 0, (float)height, &x, &y); ix = (int)(x+.5f); iy = (int)(y+.5f);  
  r.minX = my_min(r.minX, ix); r.maxX = my_max(r.maxX, ix); r.minY = my_min(r.minY, iy); r.maxY = my_max(r.maxY, iy);
  AffineTransformPoint(r.mat, (float)width, (float)height, &x, &y); ix = (int)(x+.5f); iy = (int)(y+.5f);  
  r.minX = my_min(r.minX, ix); r.maxX = my_max(r.maxX, ix); r.minY = my_min(r.minY, iy); r.maxY = my_max(r.maxY, iy);
  cvSet2D(&map_matrix, 0, 2, cvScalar(cvGet2D(&map_matrix, 0, 2).val[0]-r.minX));
  cvSet2D(&map_matrix, 1, 2, cvScalar(cvGet2D(&map_matrix, 1, 2).val[0]-r.minY));
 
  cv2DRotationMatrix(cvPoint2D32f(0,0), -rot*180/M_PI, 1/scale, &map_matrix_inv);
  AffineTransformPoint(&map_matrix_inv, (float)cvGet2D(&map_matrix, 0, 2).val[0], (float)cvGet2D(&map_matrix, 1, 2).val[0], &x, &y);
  cvSet2D(&map_matrix_inv, 0, 2, cvScalar(-x));
  cvSet2D(&map_matrix_inv, 1, 2, cvScalar(-y));

  AffineTransformPoint(&map_matrix, 100, 121, &x, &y);
  AffineTransformPoint(&map_matrix_inv, x, y, &x, &y);
  assert(my_abs(x-100)<.1 && my_abs(y-121)<.1);

  return r;
}



template <class T> 
void FlipImageSub(IplImage *img) {
  T tmp;
  int i, j, k;
  unsigned char *ptr2 = (unsigned char*)img->imageData;
  for(i = 0; i < img->height; i++, ptr2 += img->widthStep) {
    T *ptrBeg = (T*)ptr2;
    T *ptrEnd = ptrBeg + (img->width-1)*img->nChannels;
    for(j = 0; j < img->width/2; j++, ptrBeg += img->nChannels, ptrEnd -= img->nChannels) {
      for(k = 0; k < img->nChannels; k++) {
        tmp = ptrBeg[k];
        ptrBeg[k] = ptrEnd[k];
        ptrEnd[k] = tmp;
      }
    }
  }
}

void FlipImage(IplImage *src) {
  switch(src->depth) {
	case IPL_DEPTH_8U:
	  FlipImageSub<unsigned char>(src); break;
	case IPL_DEPTH_8S:
		FlipImageSub<char>(src); break;
	case IPL_DEPTH_16U:
		FlipImageSub<unsigned short>(src); break;
	case IPL_DEPTH_16S:
		FlipImageSub<short>(src); break;
	case IPL_DEPTH_32S:
		FlipImageSub<int>(src); break;
	case IPL_DEPTH_32F:
		FlipImageSub<float>(src); break;
	case IPL_DEPTH_64F:
		FlipImageSub<double>(src); break;
  }
}

template <class T> 
void SetChannelSub(IplImage *dst, IplImage *src, int cDst, int cSrc = 0, int num = 1) {
  int i, j1, j2;
  T *srcPtr, *dstPtr;
  char *srcPtr2 = src->imageData, *dstPtr2 = dst->imageData;

  assert(num > 0 && num <= 4);
  if(num == 1) {
    for(i = 0; i < src->height; i++, srcPtr2 += src->widthStep, dstPtr2 += dst->widthStep) 
      for(j1 = cSrc, j2 = cDst, srcPtr=(T*)srcPtr2, dstPtr=(T*)dstPtr2; 
	  j1 < src->width*src->nChannels; j1 += src->nChannels, j2 += dst->nChannels) 
	dstPtr[j2] = srcPtr[j1];	
  } else if(num == 2) {
    for(i = 0; i < src->height; i++, srcPtr2 += src->widthStep, dstPtr2 += dst->widthStep) {
      for(j1 = cSrc, j2 = cDst, srcPtr=(T*)srcPtr2, dstPtr=(T*)dstPtr2; 
	  j1 < src->width*src->nChannels; j1 += src->nChannels, j2 += dst->nChannels) {
	dstPtr[j2] = srcPtr[j1]; dstPtr[j2+1] = srcPtr[j1+1];
      }
    }
  } else if(num == 3) {
    for(i = 0; i < src->height; i++, srcPtr2 += src->widthStep, dstPtr2 += dst->widthStep) {
      for(j1 = cSrc, j2 = cDst, srcPtr=(T*)srcPtr2, dstPtr=(T*)dstPtr2; 
	  j1 < src->width*src->nChannels; j1 += src->nChannels, j2 += dst->nChannels) {
	dstPtr[j2] = srcPtr[j1]; dstPtr[j2+1] = srcPtr[j1+1]; dstPtr[j2+2] = srcPtr[j1+2];
      }
    }
  } else if(num == 4) {
    for(i = 0; i < src->height; i++, srcPtr2 += src->widthStep, dstPtr2 += dst->widthStep) {
      for(j1 = cSrc, j2 = cDst, srcPtr=(T*)srcPtr2, dstPtr=(T*)dstPtr2; 
	  j1 < src->width*src->nChannels; j1 += src->nChannels, j2 += dst->nChannels) {
	dstPtr[j2] = srcPtr[j1]; dstPtr[j2+1] = srcPtr[j1+1]; dstPtr[j2+2] = srcPtr[j1+2]; dstPtr[j2+3] = srcPtr[j1+3];
      }
    }
  }
}

template <class T> 
void ExtractChannelSub(IplImage *src, IplImage *dst, int c, int num) {
  SetChannelSub<T>(dst, src, 0, c, num);
}

void cvExtractChannel(IplImage *src, IplImage *dst, int c, int num) {
  switch(src->depth) {
	case IPL_DEPTH_8U:
	  ExtractChannelSub<unsigned char>(src, dst, c, num); break;
	case IPL_DEPTH_8S:
		ExtractChannelSub<char>(src, dst, c, num); break;
	case IPL_DEPTH_16U:
		ExtractChannelSub<unsigned short>(src, dst, c, num); break;
	case IPL_DEPTH_16S:
		ExtractChannelSub<short>(src, dst, c, num); break;
	case IPL_DEPTH_32S:
		ExtractChannelSub<int>(src, dst, c, num); break;
	case IPL_DEPTH_32F:
		ExtractChannelSub<float>(src, dst, c, num); break;
	case IPL_DEPTH_64F:
		ExtractChannelSub<double>(src, dst, c, num); break;
  }
}


void cvSetChannel(IplImage *dst, IplImage *src, int c, int cSrc, int num) {
	switch(src->depth) {
	case IPL_DEPTH_8U:
	  SetChannelSub<unsigned char>(dst, src, c, cSrc, num); break;
	case IPL_DEPTH_8S:
		SetChannelSub<char>(dst, src, c, cSrc, num); break;
	case IPL_DEPTH_16U:
	        SetChannelSub<unsigned short>(dst, src, c, cSrc, num); break;
	case IPL_DEPTH_16S:
		SetChannelSub<short>(dst, src, c, cSrc, num); break;
	case IPL_DEPTH_32S:
		SetChannelSub<int>(dst, src, c, cSrc, num); break;
	case IPL_DEPTH_32F:
		SetChannelSub<float>(dst, src, c, cSrc, num); break;
	case IPL_DEPTH_64F:
		SetChannelSub<double>(dst, src, c, cSrc, num); break;
	}
}


void cvCopyMakeBorderMultiChannel(IplImage* src, IplImage* dst, CvPoint offset,
				  int bordertype, CvScalar value) {
  if(src->nChannels <= 4)
    cvCopyMakeBorder(src, dst,offset, bordertype, value);
  else {
    IplImage *srcC = cvCreateImage(cvSize(src->width,src->height), src->depth, 4);
    IplImage *tmp = cvCreateImage(cvSize(dst->width,dst->height), dst->depth, 4);
    for(int i = 0; i < src->nChannels; i+=4) {
      int n = my_min(4,src->nChannels-i);
      if(n < 4) {
	cvReleaseImage(&srcC);  cvReleaseImage(&tmp);
	srcC = cvCreateImage(cvSize(src->width,src->height), src->depth, n);
	tmp = cvCreateImage(cvSize(dst->width,dst->height), dst->depth, n);
      }
      cvExtractChannel(src, srcC, i, n);
      cvCopyMakeBorder(srcC, tmp, offset, bordertype, value);
      cvSetChannel(dst, tmp, i, 0, n);
    }
    cvReleaseImage(&srcC);
    cvReleaseImage(&tmp);
  }
}

// My own version of an affine warp, because OpenCV version was giving weird boundary effects and wouldn't do nearest neighbor
template <class T> 
void cvWarpAffineMultiChannelCustomSub(IplImage *src, IplImage *dst, float *mat, T fillVal) {
  float dx = mat[0], dy = mat[3], x, y;
  int i, j, n, ix, iy, w = src->width, h= src->height, nc = src->nChannels;
  T *sPtr, *ptr;
  char *ptr2 = dst->imageData;

  assert(src->nChannels == dst->nChannels);
  
  if(nc == 1) {
    // slightly faster version for 1-channel images
    for(i = 0; i < dst->height; i++, ptr2+=dst->widthStep) {
      x = mat[2] + i*mat[1] +.5f;
      y = mat[5] + i*mat[4] +.5f;
      for(j = 0, ptr = (T*)ptr2; j < dst->width; j++) {
	ix = (int)x; iy = (int)y;
	if(ix < 0 || ix >= w || iy < 0 || iy >= h) {
	  // (x,y) is out of bounds
	  ptr[j] = fillVal;
	} else {
	  ptr[j] = ((T*)(src->imageData+iy*src->widthStep))[ix];
	}
	x += dx;  y += dy;  
      }
    }
  } else {
    for(i = 0; i < dst->height; i++, ptr2+=dst->widthStep) {
      x = mat[2] + i*mat[1] +.5f;
      y = mat[5] + i*mat[4] +.5f;
      for(j = 0, ptr = (T*)ptr2; j < dst->width; j++) {
	ix = (int)x; iy = (int)y;
	if(ix < 0 || ix >= w || iy < 0 || iy >= h) {
	  // (x,y) is out of bounds
	  ix = iy = -1;  
	  for(n = 0; n < nc; n++) {
	    ptr[n] = fillVal;
	  }
	} else {
	  sPtr = ((T*)(src->imageData+iy*src->widthStep))+ix*nc;
	  for(n = 0; n < nc; n++) {
	    ptr[n] = sPtr[n];
	  }
	}
	x += dx;  y += dy;  
	ptr += nc;
      }
    }
  }
}

// My own version of an affine warp, because OpenCV version was giving weird boundary effects and wouldn't do nearest neighbor
template <class T> 
void cvWarpAffineMultiChannelCustomGetIndicesSub(IplImage *src, IplImage *dst, IplImage *inds, float *mat, T fillVal) {
  float dx = mat[0], dy = mat[3], x, y;
  int i, j, n, ix, iy, w = src->width, h= src->height, nc = src->nChannels;
  T *sPtr, *ptr;
  int *iPtr;
  char *ptr2=dst->imageData, *iPtr2=inds->imageData;

  assert(inds->depth == (int)IPL_DEPTH_32S && src->nChannels == dst->nChannels && inds->width == dst->width && inds->height == dst->height && 
	 (inds->nChannels == 2 || inds->nChannels == 3));
  
  if(nc == 1) {
    // slightly faster version for 1-channel images
    for(i = 0; i < dst->height; i++, ptr2+=dst->widthStep, iPtr2 += inds->widthStep) {
      x = mat[2] + i*mat[1] +.5f;
      y = mat[5] + i*mat[4] +.5f;
      for(j = 0, ptr = (T*)ptr2, iPtr = (int*)iPtr2; j < dst->width*inds->nChannels; j+=inds->nChannels) {
	ix = (int)x; iy = (int)y;
	if(ix < 0 || ix >= w || iy < 0 || iy >= h) {
	  // (x,y) is out of bounds
	  ix = iy = -1;  
	  *ptr = fillVal;
	  iPtr[j] = iPtr[j+1] = -1;
	} else {
	  *ptr = ((T*)(src->imageData+iy*src->widthStep))[ix];
	  iPtr[j] = ix;  iPtr[j+1] = iy;
	}
	x += dx;  y += dy;  
	ptr++;
      }
    }
  } else {
    for(i = 0; i < dst->height; i++, ptr2+=dst->widthStep, iPtr2 += inds->widthStep) {
      x = mat[2] + i*mat[1] +.5f;
      y = mat[5] + i*mat[4] +.5f;
      for(j = 0, ptr = (T*)ptr2, iPtr = (int*)iPtr2; j < dst->width*inds->nChannels; j+=inds->nChannels) {
	ix = (int)x; iy = (int)y;
	if(ix < 0 || ix >= w || iy < 0 || iy >= h) {
	  // (x,y) is out of bounds
	  ix = iy = -1;  
	  for(n = 0; n < nc; n++) {
	    ptr[n] = fillVal;
	  }
	  iPtr[j] = iPtr[j+1] = -1;
	} else {
	  sPtr = ((T*)(src->imageData+iy*src->widthStep))+ix*nc;
	  for(n = 0; n < nc; n++) {
	    ptr[n] = sPtr[n];
	  }
	  iPtr[j] = ix;  iPtr[j+1] = iy;
	}
	x += dx;  y += dy;  
	ptr += nc;
      }
    }
  }
}

void cvWarpAffineMultiChannelCustomGetIndices(IplImage *src, IplImage *dst, IplImage *inds, float *mat, double fillVal) {
  switch(src->depth) {
  case IPL_DEPTH_8U: cvWarpAffineMultiChannelCustomGetIndicesSub<unsigned char>(src, dst, inds, mat, (unsigned char)fillVal); break;
  case IPL_DEPTH_8S: cvWarpAffineMultiChannelCustomGetIndicesSub<char>(src, dst, inds, mat, (char)fillVal); break;
  case IPL_DEPTH_16U: cvWarpAffineMultiChannelCustomGetIndicesSub<unsigned short>(src, dst, inds, mat, (unsigned short)fillVal); break;
  case IPL_DEPTH_16S: cvWarpAffineMultiChannelCustomGetIndicesSub<short>(src, dst, inds, mat, (short)fillVal); break;
  case IPL_DEPTH_32S: cvWarpAffineMultiChannelCustomGetIndicesSub<int>(src, dst, inds, mat, (int)fillVal); break;
  case IPL_DEPTH_32F: cvWarpAffineMultiChannelCustomGetIndicesSub<float>(src, dst, inds, mat, (float)fillVal); break;
  case IPL_DEPTH_64F: cvWarpAffineMultiChannelCustomGetIndicesSub<double>(src, dst, inds, mat, (double)fillVal); break;
  }
}


void cvWarpAffineMultiChannelCustom(IplImage *src, IplImage *dst, float *mat, double fillVal) {
  switch(src->depth) {
  case IPL_DEPTH_8U: cvWarpAffineMultiChannelCustomSub<unsigned char>(src, dst, mat, (unsigned char)fillVal); break;
  case IPL_DEPTH_8S: cvWarpAffineMultiChannelCustomSub<char>(src, dst, mat, (char)fillVal); break;
  case IPL_DEPTH_16U: cvWarpAffineMultiChannelCustomSub<unsigned short>(src, dst, mat, (unsigned short)fillVal); break;
  case IPL_DEPTH_16S: cvWarpAffineMultiChannelCustomSub<short>(src, dst, mat, (short)fillVal); break;
  case IPL_DEPTH_32S: cvWarpAffineMultiChannelCustomSub<int>(src, dst, mat, (int)fillVal); break;
  case IPL_DEPTH_32F: cvWarpAffineMultiChannelCustomSub<float>(src, dst, mat, (float)fillVal); break;
  case IPL_DEPTH_64F: cvWarpAffineMultiChannelCustomSub<double>(src, dst, mat, (double)fillVal); break;
  }
}
 

void cvWarpAffineMultiChannel(IplImage *src, IplImage *dst, CvMat *mat, int inter, CvScalar fillval) {
  if(src->nChannels <= 4)
    cvWarpAffine(src, dst, mat, inter, fillval);
  else {
    IplImage *srcC = cvCreateImage(cvSize(src->width,src->height), src->depth, 4);
    IplImage *tmp = cvCreateImage(cvSize(dst->width,dst->height), dst->depth, 4);
    for(int i = 0; i < src->nChannels; i+=4) {
      int n = my_min(4,src->nChannels-i);
      if(n < 4) {
	cvReleaseImage(&srcC);  cvReleaseImage(&tmp);
	srcC = cvCreateImage(cvSize(src->width,src->height), src->depth, n);
	tmp = cvCreateImage(cvSize(dst->width,dst->height), dst->depth, n);
      }
      cvExtractChannel(src, srcC, i, n);
      cvWarpAffine(srcC, tmp, mat, inter, fillval);
      cvSetChannel(dst, tmp, i, 0, n);
    }
    cvReleaseImage(&srcC);
    cvReleaseImage(&tmp);
  }
}

void cvMatchTemplateMultiChannel(IplImage *img, IplImage *templ, IplImage *score, int method) {
  if(templ->nChannels == 1) {
    cvMatchTemplate(img, templ, score, method);
  } else {
    IplImage *imgC = cvCreateImage(cvSize(img->width,img->height), img->depth, 1);
    IplImage *templC = cvCreateImage(cvSize(templ->width,templ->height), templ->depth, 1);
    IplImage *tmp = cvCreateImage(cvSize(score->width,score->height), score->depth, 1);
    cvZero(score);
    for(int i = 0; i < img->nChannels; i++) {
      cvExtractChannel(img, imgC, i);
      cvExtractChannel(templ, templC, i);
      cvMatchTemplate(imgC, templC, tmp, method);
      cvAdd(score, tmp, score);			
    }
    //cvSetImageCOI(img, 0);
    //cvSetImageCOI(templ, 0);
    cvReleaseImage(&imgC);
    cvReleaseImage(&templC);
    cvReleaseImage(&tmp);
  }
}


void cvFilter2DMultiChannel(IplImage *img, IplImage *score, IplImage *templ) {
  if(templ->nChannels == 1) {
    CvMat mat = cvMat(templ->height, templ->width, CV_32FC1, templ->imageData);
    if(cvCountNonZero(templ))
      cvFilter2D(img, score, &mat);
    else
      cvZero(score);
  } else {
    IplImage *imgC = cvCreateImage(cvSize(img->width,img->height), img->depth, 1);
    IplImage *templC = cvCreateImage(cvSize(templ->width,templ->height), img->depth, 1);
    IplImage *tmp = cvCreateImage(cvSize(score->width,score->height), score->depth, 1);
    cvZero(score);
    for(int i = 0; i < img->nChannels; i++) {
      cvExtractChannel(templ, templC, i);
      if(cvCountNonZero(templC)) {
	cvExtractChannel(img, imgC, i); 
	CvMat mat = cvMat(templ->height, templ->width, CV_32FC1, templC->imageData);
	cvFilter2D(imgC, tmp, &mat);
	cvAdd(score, tmp, score);	
      }	
    }

    cvReleaseImage(&imgC);
    cvReleaseImage(&templC);
    cvReleaseImage(&tmp);
  }
}

template <class T>
IplImage *cvCumulativeSumSub(IplImage *img, T *sumO, IplImage *sumImg, T gamma, T delta, T mi) {
  assert(img->nChannels==1);
  if(!sumImg) sumImg = cvCloneImage(img);
  T sum = 0, sumLine;
  int i, j;

  if(mi)   // Let mi be the minimum likelihood value for any pixel location
    cvMaxS(img, mi, sumImg);

  // Convert per pixel likelihoods to per pixel probability maps: p(x,y)=exp{gamma*ll(x,y)}/Z, where Z=exp{-delta} 
  if(gamma) {
    cvConvertScale(sumImg, sumImg, gamma, delta);
    cvExp(sumImg, sumImg);
  }

  // Compute the cumulative sum of all probabilities
  char *ptr2 = sumImg->imageData;
  T *ptr;
  for(i = 0; i < sumImg->height; i++, ptr2 += sumImg->widthStep) {
    sumLine = 0;
    for(j = 0, ptr =(T*)ptr2; j < sumImg->width; j++) {
      sumLine += ptr[j];
      ptr[j] = sum + sumLine;
    }
    sum += sumLine;
  }
  *sumO = sum;

  return sumImg;
}
IplImage *cvCumulativeSum(IplImage *img, float *sum, IplImage *sumImg, float gamma, float delta, float mi) { return cvCumulativeSumSub<float>(img, sum, sumImg, gamma, delta, mi); }
IplImage *cvCumulativeSum(IplImage *img, double *sum, IplImage *sumImg, double gamma, double delta, double mi) { return cvCumulativeSumSub<double>(img, sum, sumImg, gamma, delta, mi); }
IplImage *cvCumulativeSum(IplImage *img, int *sum, IplImage *sumImg, int gamma, int delta, int mi) { return cvCumulativeSumSub<int>(img, sum, sumImg, gamma, delta, mi); }

template <class T>
bool cvFindCumulativeSumLocationSub(IplImage *subImg, T sum, CvPoint *pt) {
  assert(subImg->nChannels==1 && subImg->widthStep == (int)(subImg->width*sizeof(T)));

  int s = 0, e = subImg->width*subImg->height, m;
  T *cumSums = (T*)subImg->imageData;
  while(s != e) {
    m = (s+e)>>1;
    if(sum > cumSums[m])
      s = m+1;
    else
      e = m;
  }
  pt->y = s/subImg->width;  pt->x = s%subImg->width;  
  return s < subImg->width*subImg->height && cumSums[s] >= sum;
}
bool cvFindCumulativeSumLocation(IplImage *subImg, float sum, CvPoint *pt) { 
  return cvFindCumulativeSumLocationSub<float>(subImg, sum, pt);
}
bool cvFindCumulativeSumLocation(IplImage *subImg, double sum, CvPoint *pt) { 
  return cvFindCumulativeSumLocationSub<double>(subImg, sum, pt);
}
bool cvFindCumulativeSumLocation(IplImage *subImg, int sum, CvPoint *pt) { 
  return cvFindCumulativeSumLocationSub<int>(subImg, sum, pt);
}

template <class T> 
void DrawImageIntoImageMaxSub(IplImage *src, CvRect from, IplImage *to, int toX, int toY) {
  if(from.x < 0) { from.width += from.x; from.x = 0; }
  if(from.y < 0) { from.height += from.y; from.y = 0; }
  if(from.x+from.width > src->width) { from.width = src->width-from.x; }
  if(from.y+from.height > src->height) { from.height = src->height-from.y; }
  if(toX < 0) { from.width += toX; from.x -= toX; toX = 0; }
  if(toY < 0) { from.height += toY; from.y -= toY; toY = 0; }
  if(toX+from.width > to->width) { from.width = to->width-toX; }
  if(toY+from.height > to->height) { from.height = to->height-toY; }

  unsigned char *srcPtr = (unsigned char*)src->imageData + from.y*src->widthStep + from.x*src->nChannels*sizeof(T);
  unsigned char *dstPtr = (unsigned char*)to->imageData + toY*to->widthStep + toX*to->nChannels*sizeof(T);
  for(int i = 0; i < from.height && from.width > 0; i++, srcPtr += src->widthStep, dstPtr += to->widthStep) {
    for(int j = 0; j < from.width*src->nChannels; j++)
      ((T*)dstPtr)[j] = ((T*)srcPtr)[j] > ((T*)dstPtr)[j] ? ((T*)srcPtr)[j] : ((T*)dstPtr)[j];
  }
}

void DrawImageIntoImageMax(IplImage *src, CvRect from, IplImage *to, int toX, int toY) {
  switch(src->depth) {
  case IPL_DEPTH_8U: DrawImageIntoImageMaxSub<unsigned char>(src, from, to, toX, toY); break;
  case IPL_DEPTH_8S: DrawImageIntoImageMaxSub<char>(src, from, to, toX, toY); break;
  case IPL_DEPTH_16U: DrawImageIntoImageMaxSub<unsigned short>(src, from, to, toX, toY); break;
  case IPL_DEPTH_16S: DrawImageIntoImageMaxSub<short>(src, from, to, toX, toY); break;
  case IPL_DEPTH_32S: DrawImageIntoImageMaxSub<int>(src, from, to, toX, toY); break;
  case IPL_DEPTH_32F: DrawImageIntoImageMaxSub<float>(src, from, to, toX, toY); break;
  case IPL_DEPTH_64F: DrawImageIntoImageMaxSub<double>(src, from, to, toX, toY); break;
  }
}

template <class T> 
void DrawImageIntoImageSub(IplImage *src, CvRect from, IplImage *to, int toX, int toY) {
  if(from.x < 0) { from.width += from.x; from.x = 0; }
  if(from.y < 0) { from.height += from.y; from.y = 0; }
  if(from.x+from.width > src->width) { from.width = src->width-from.x; }
  if(from.y+from.height > src->height) { from.height = src->height-from.y; }
  if(toX < 0) { from.width += toX; from.x -= toX; toX = 0; }
  if(toY < 0) { from.height += toY; from.y -= toY; toY = 0; }
  if(toX+from.width > to->width) { from.width = to->width-toX; }
  if(toY+from.height > to->height) { from.height = to->height-toY; }

  unsigned char *srcPtr = (unsigned char*)src->imageData + from.y*src->widthStep + from.x*src->nChannels*sizeof(T);
  unsigned char *dstPtr = (unsigned char*)to->imageData + toY*to->widthStep + toX*to->nChannels*sizeof(T);
  for(int i = 0; i < from.height && from.width > 0; i++, srcPtr += src->widthStep, dstPtr += to->widthStep) {
    memcpy(dstPtr, srcPtr, from.width*src->nChannels*sizeof(T));
  }
}

void DrawImageIntoImage(IplImage *src, CvRect from, IplImage *to, int toX, int toY) {
  switch(src->depth) {
  case IPL_DEPTH_8U: DrawImageIntoImageSub<unsigned char>(src, from, to, toX, toY); break;
  case IPL_DEPTH_8S: DrawImageIntoImageSub<char>(src, from, to, toX, toY); break;
  case IPL_DEPTH_16U: DrawImageIntoImageSub<unsigned short>(src, from, to, toX, toY); break;
  case IPL_DEPTH_16S: DrawImageIntoImageSub<short>(src, from, to, toX, toY); break;
  case IPL_DEPTH_32S: DrawImageIntoImageSub<int>(src, from, to, toX, toY); break;
  case IPL_DEPTH_32F: DrawImageIntoImageSub<float>(src, from, to, toX, toY); break;
  case IPL_DEPTH_64F: DrawImageIntoImageSub<double>(src, from, to, toX, toY); break;
  }
}


void cvRotatedRect(IplImage *img, CvPoint centerPt, int width, int height, float rot, CvScalar color, int scale) {
	float matf[6], x[4], y[4];
	CvMat mat = cvMat(2, 3, CV_32FC1, matf);
	cv2DRotationMatrix(cvPoint2D32f(0,0), rot*180/M_PI, 1, &mat);
	AffineTransformPoint(matf, (float)-width/2, (float)-height/2, &x[0], &y[0]);
	AffineTransformPoint(matf, (float)width/2, (float)-height/2, &x[1], &y[1]);
	AffineTransformPoint(matf, (float)width/2, (float)height/2, &x[2], &y[2]);
	AffineTransformPoint(matf, (float)-width/2, (float)height/2, &x[3], &y[3]);
	cvLine(img, cvPoint(centerPt.x+(int)x[0],centerPt.y+(int)y[0]), cvPoint(centerPt.x+(int)x[1],centerPt.y+(int)y[1]), color, scale, CV_AA);
	cvLine(img, cvPoint(centerPt.x+(int)x[1],centerPt.y+(int)y[1]), cvPoint(centerPt.x+(int)x[2],centerPt.y+(int)y[2]), color, scale, CV_AA);
	cvLine(img, cvPoint(centerPt.x+(int)x[2],centerPt.y+(int)y[2]), cvPoint(centerPt.x+(int)x[3],centerPt.y+(int)y[3]), color, scale, CV_AA);
	cvLine(img, cvPoint(centerPt.x+(int)x[3],centerPt.y+(int)y[3]), cvPoint(centerPt.x+(int)x[0],centerPt.y+(int)y[0]), color, scale, CV_AA);
}

int NonMaximalSuppression(CvRectScore *boxes_orig, int num, float overlap, int w, int h) {
  int i;
  CvRectScore *boxes = boxes_orig;
  int numNMS = 0;

  if(overlap >= 0) {
    // Greedy "non-maximal suppression"
    while(num) {
      // Greedily select the highest scoring bounding box
      int best = -1;
      float bestS = -10000000;
      for(i = 0; i < num; i++) {
	if(boxes[i].score > bestS) { 
	  bestS = boxes[i].score; 
	  best = i; 
	}
      }
      CvRect b = boxes[best].rect;
      CvRectScore tmp = boxes[0];
      boxes[0] = boxes[best];
      boxes[best] = tmp;
      boxes++;
      numNMS++;
      float A1 = b.width*b.height, A2, inter, inter_over_union;

      // Remove all bounding boxes where the percent area of overlap is greater than overlap
      int numGood = 0, x1, x2, y1, y2, a;
      for(i = 0; i < num-1; i++) {
	x1 = my_max(b.x, boxes[i].rect.x);
	y1 = my_max(b.y, boxes[i].rect.y);
	x2 = my_min(b.x+b.width,  boxes[i].rect.x+boxes[i].rect.width);
	y2 = my_min(b.y+b.height, boxes[i].rect.y+boxes[i].rect.height);
	A2 = boxes[i].rect.width*boxes[i].rect.height;
	inter = (float)((x2-x1)*(y2-y1));
	inter_over_union = inter / (A1+A2-inter);
	if(inter_over_union <= overlap) {
	  tmp = boxes[numGood];
	  boxes[numGood++] = boxes[i];
	  boxes[i] = tmp;
	}
      }
      num = numGood;
    }
  } else {
    // Regular non-maximal suppression
    assert(num == w*h);
    int i = 0;
    CvRectScore *ptr = boxes;
    for(int y = 0; y < h; y++, ptr += w) {
      for(int x = 0; x < w; x++, i++) {
	if(!((x && ptr[x-1].score >= ptr[x].score) || (y && ptr[x-w].score >= ptr[x].score) ||
	     (x<w-1 && ptr[x+1].score >= ptr[x].score) || (y<h-1 && ptr[x+w].score >= ptr[x].score))) {
	  CvRectScore tmp = boxes[numNMS];
	  boxes[numNMS++] = boxes[i];
	  boxes[i] = tmp;
	}
      }
    }
  }

  return numNMS;
}

void MakeThumbnail(IplImage *img, int thumb_width, int thumb_height, const char *outFileName) {
  IplImage *thumb = cvCreateImage(cvSize(thumb_width, thumb_height), img->depth, img->nChannels);
  cvResize(img, thumb);
  cvSaveImage(outFileName, thumb);
  cvReleaseImage(&thumb);
}
