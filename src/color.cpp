/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#include "color.h"
#include "attribute.h"
#include "part.h"
#include "pose.h"
#include "classes.h"

IplImage *SiftSubsampleImage(IplImage *src, float p, int subsample, bool *freeSrc);
IplImage *****AllocatePrecomputeHOGBuffers(int rotations, int maxScales, int num);
void ExtractHOGOffsetImages(IplImage *siftImage, IplImage ***retval, IplImage *img, SiftParams *params, 
			    RotationInfo *r, int rotations, int spatialGranularity);


IplImage *****PrecomputeColor(IplImage *img, SiftParams *params2, int spatialGranularity, int numOrientations, int numScales) {
  int num = my_max(1,params2->cellWidth/spatialGranularity);
  IplImage *****retval = AllocatePrecomputeHOGBuffers(numOrientations, params2->maxScales, num);


  // compute descriptors, ordered by scale and orientation
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for(int j = 0; j < numOrientations; j++) {
    SiftParams tmpP = *params2;
    SiftParams *params = &tmpP;
    int n = params->cellWidth&1 ? params->cellWidth : params->cellWidth+1;
    int i;
    RotationInfo r;
 
    params->rot = 0;
    float rot = (float)(j*2*M_PI/numOrientations);
    r = GetRotationInfo(img->width, img->height, rot);
    IplImage *rotImage = cvCreateImage(cvSize(r.maxX-r.minX, r.maxY-r.minY), IPL_DEPTH_32F, img->nChannels);
    cvZero(rotImage);
    CvMat affineMat = cvMat(2, 3, CV_32FC1, r.mat);
    cvWarpAffineMultiChannel(img, rotImage, &affineMat);

    for(i = 0; i < numScales; i++) {
      params->subsample = i;
      bool freeSrc = false;
      IplImage *siftImage = SiftSubsampleImage(rotImage, params->subsamplePower, params->subsample, &freeSrc);
      IplImage *siftImage2 = cvCloneImage(siftImage);
      cvSmooth(siftImage, siftImage2, CV_BLUR, n, n);
      RotationInfo r = GetRotationInfo(img->width, img->height, (float)(j*2*M_PI/numOrientations), 1.0f/pow(params->subsamplePower, i)/spatialGranularity);
      ExtractHOGOffsetImages(siftImage2, retval[j][i], img, params, &r, numOrientations, spatialGranularity);

      if(freeSrc) cvReleaseImage(&siftImage);
      cvReleaseImage(&siftImage2);
    }
    cvReleaseImage(&rotImage);
  }

  return retval;
}




IplImage **PrecomputeColorFastBegin2(IplImage *colorImg, SiftParams *params2) {
  IplImage **retval = (IplImage**)malloc(sizeof(IplImage*)*params2->maxScales);
  
  for(int i = 0; i < params2->maxScales; i++) {
    SiftParams params = *params2;
    int n = params2->cellWidth&1 ? params2->cellWidth : params2->cellWidth+1;
    params.subsample = i;
    bool freeSrc = false;
    IplImage *siftImage = SiftSubsampleImage(colorImg, params.subsamplePower, params.subsample, &freeSrc);
    retval[i] = cvCloneImage(siftImage);
    cvSmooth(siftImage, retval[i], CV_BLUR, n, n);
    if(freeSrc) cvReleaseImage(&siftImage);
  }
  return retval;
}

  
IplImage *****PrecomputeColorFastFinish2(IplImage **hogImages, IplImage *img, SiftParams *params2, int spatialGranularity, int rotations, int scales, bool flip) {
  int num = my_max(1,params2->cellWidth/spatialGranularity);
  IplImage *****retval = AllocatePrecomputeHOGBuffers(rotations, params2->maxScales, num);
  for(int j = 0; j < rotations; j++) {
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
	  // Extract a rotated version of the color images, by sampling pixels on a grid from the dense HOG image, 
	  // while applying an affine warp (for rotation and scale)
	  int w = numBlocksX + (x >= rx ? 0 : 1);
	  int h = numBlocksY + (y >= ry ? 0 : 1);
	  int k, l, nc = hogImages[i]->nChannels, ix, iy, sx, sy;
	  float xx, yy, *ptr, *srcPtr;
	  char *ptr2;
	  RotationInfo r2 = GetRotationInfo(hogImages[i]->width, hogImages[i]->height, (float)(j*2*M_PI/rotations), 1.0f);
	  float *mat = r2.invMat;
	  float dx = mat[0]*spatialGranularity, dy = mat[3]*spatialGranularity;
	  retval[j][i][x][y] = cvCreateImage(cvSize(w, h), hogImages[i]->depth, hogImages[i]->nChannels);
	  for(k = 0, sy = y, ptr2=retval[j][i][x][y]->imageData; k < retval[j][i][x][y]->height; 
	      k++, ptr2 += retval[j][i][x][y]->widthStep, sy += spatialGranularity) {
	    AffineTransformPoint(mat, x, sy, &xx, &yy);  xx += .5f; yy += .5f;
	    for(l = 0, sx = x, ptr=(float*)ptr2; l < retval[j][i][x][y]->width; l++, ptr += nc, xx += dx, yy += dy, sx += spatialGranularity) {
	      iy = ((int)yy);  ix = ((int)xx);
	      if(ix >= 0 && iy >= 0 && ix < hogImages[i]->width && iy < hogImages[i]->height) {
		srcPtr = ((float*)(hogImages[i]->imageData+hogImages[i]->widthStep*iy))+(ix*nc);
		ptr[0] = srcPtr[0];  ptr[1] = srcPtr[1];  ptr[2] = srcPtr[2];  
	      } else {
		ptr[0] = ptr[1] = ptr[2] = 0;
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





IplImage *VisualizeColorDescriptor(float *descriptor, int numX, int numY, int cellWidth) {
  int i, j, k, l, numBins=3, x, y, xx;
  IplImage *img = cvCreateImage(cvSize(cellWidth*numX,cellWidth*numY), IPL_DEPTH_8U, numBins);
  float *ptr = descriptor;
  unsigned char *dstPtr, *dstPtr2;
  char *dstPtr3=img->imageData, *dstPtr1;
  unsigned char r, g, b;
  cvZero(img);  
  
  for(i = 0, y = 0; i < numY; i++, dstPtr3 += img->widthStep*cellWidth, y += cellWidth) {
    for(j = 0, x = 0, dstPtr2=(unsigned char*)dstPtr3; j < numX; j++, ptr += numBins, dstPtr2 += numBins*cellWidth, x+= cellWidth) {
      r = (unsigned char)(ptr[0]*255);  g = (unsigned char)(ptr[1]*255);  b = (unsigned char)(ptr[2]*255);  
      for(k = y, dstPtr1=(char*)dstPtr2; k < y+cellWidth && k < img->height; k++, dstPtr1 += img->widthStep) {
	for(l = 0, xx=x, dstPtr=(unsigned char*)dstPtr1; xx < img->width && l < cellWidth*numBins; l+=numBins, xx++) {
	  dstPtr[l] = r;  dstPtr[l+1] = g;  dstPtr[l+2] = b;
	}
      }
    }
  }

  return img;
}







ColorTemplateFeature::ColorTemplateFeature(FeatureOptions *fo, SiftParams p) : TemplateFeature(fo, p) {
  colorImg = NULL;
}

ColorTemplateFeature::~ColorTemplateFeature() { 
  if(colorImg) cvReleaseImage(&colorImg);
}

void ColorTemplateFeature::Clear(bool full) {
  TemplateFeature::Clear(full);
  if(colorImg && colorImg != fo->GetImage()) 
    cvReleaseImage(&colorImg); 
}

IplImage *****ColorTemplateFeature::PrecomputeFeatures(bool flip) {
  if((!flip && !featureImages) || (flip && !featureImages_flip)) {
    if(!colorImg) ComputeColorImage();
    params.maxScales = fo->NumScales();
    if(g_debug > 1) fprintf(stderr, "    precompute %s...\n", Name());
    IplImage **denseFeatures = PrecomputeColorFastBegin2(colorImg, &params);
    IplImage *****tmp = PrecomputeColorFastFinish2(denseFeatures, colorImg, &params, 
						   params.cellWidth, fo->NumOrientations(), NumScales(), flip);
    if(!flip) featureImages = tmp;
    else featureImages_flip = tmp;
    for(int i = 0; i < NumScales(); i++) 
      cvReleaseImage(&denseFeatures[i]);
    free(denseFeatures);
  }
  return !flip ? featureImages : featureImages_flip;  
}

IplImage *ColorTemplateFeature::Visualize(float *f, int w, int h, float mi, float ma) {
  float *tmp = new float[NumFeatures(w,h)];
  memcpy(tmp, f, sizeof(float)*w*h*3);
#if NORMALIZE_TEMPLATES
  for(int i = 0; i < w*h*3; i++) tmp[i] *= w*h*3;
#endif
  ConvertToRGB(tmp, tmp, w, h);
  IplImage *retval = VisualizeColorDescriptor(tmp, w, h, params.cellWidth);
  delete [] tmp;
  return retval;
}

// Visualize an assignment to part locations as a HOG image
IplImage *ColorTemplateFeature::Visualize(Classes *classes, PartLocation *locs, bool visualizeWeights, AttributeInstance *attribute) {
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
    ConvertToRGB(tmp, tmp, am->Width(), am->Height());
    IplImage *retval = VisualizeColorDescriptor(tmp, am->Width(), am->Height(), params.cellWidth);
    delete [] tmp;
    return retval;
  }


  // Visualize a full part-based detection model
  int wi = fo->GetImage()->width, he = fo->GetImage()->height, n;
  IplImage *img = NULL;
  for(int i = 0; i < classes->NumParts(); i++) {
    int x, y, scale, rot, pose;
    locs[i].GetDetectionLocation(&x, &y, &scale, &rot, &pose);
    if(pose >= 0) {
      ObjectPose *p = classes->GetPart(i)->GetPose(pose);
      Attribute *attr = p->Appearance();
      if(attr) {
	float *tmp = new float[attr->NumWeights()];
	if(visualizeWeights) {
	  n = attr->GetWeights(tmp, name);
	} else {
	  int s = scale, j = 0;
	  PartLocation l(locs[i]);  
	  while(s < fo->ScaleOffset() && j < attr->NumFeatureTypes()) {
	    s = scale + attr->Feature(j)->scale;
	    j++;
	  }
	  l.SetDetectionLocation(x, y, s, rot, pose, LATENT, LATENT);
	  if(j < attr->NumFeatureTypes() && attr->Width(j)) n = GetFeaturesAtLocation(tmp, attr->Width(j), attr->Height(j), 0, &l, false/*p->IsFlipped()*/);
	  else n = 0;
#if NORMALIZE_TEMPLATES
	  for(int i = 0; i < n; i++) tmp[i] *= n;
#endif
	}
	ConvertToRGB(tmp, tmp, attr->Width(), attr->Height());
	
	if(n) {
	  float xx, yy, scale, rot, width, height;
	  locs[i].GetImageLocation(&xx, &yy, &scale, &rot, NULL, &width, &height);

	  if(!img) {
	    img = cvCreateImage(cvSize(wi,he), IPL_DEPTH_8U, 3);
	    cvZero(img);
	  }
	  IplImage *part = VisualizeColorDescriptor(tmp, attr->Width(), attr->Height(), params.cellWidth);
	  
	  RotationInfo r = ::GetRotationInfo(part->width, part->height, -rot, scale);
	  IplImage *rotImage = cvCreateImage(cvSize(r.maxX-r.minX, r.maxY-r.minY), IPL_DEPTH_8U, img->nChannels);
	  cvZero(rotImage);
	  CvMat affineMat = cvMat(2, 3, CV_32FC1, r.mat);
	  cvWarpAffineMultiChannel(part, rotImage, &affineMat);
	  
	  DrawImageIntoImageMax(rotImage, cvRect(0,0,(r.maxX-r.minX),(r.maxY-r.minY)), img,
				(int)(xx-(r.maxX-r.minX)/2), (int)(yy-(r.maxY-r.minY)/2));
	  cvRotatedRect(img, cvPoint((int)xx, (int)yy), width, height, -rot, CV_RGB(0,0,255));
	  cvReleaseImage(&part);
	  cvReleaseImage(&rotImage);
	}
	delete [] tmp;
      }
    }
  }
  return img;
}


RGBTemplateFeature::RGBTemplateFeature(FeatureOptions *fo, SiftParams p) : ColorTemplateFeature(fo, p) {
  name = StringCopy(params.name);
}

void RGBTemplateFeature::ComputeColorImage() {
  IplImage *img = fo->GetImage();
  colorImg = cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_32F,3);
  cvConvertScale(img, colorImg, 1/255.0f);
}


CIETemplateFeature::CIETemplateFeature(FeatureOptions *fo, SiftParams p) : ColorTemplateFeature(fo, p) {
  name = StringCopy(params.name);
}
void CIETemplateFeature::ComputeColorImage() {
  IplImage *img = fo->GetImage();
  colorImg = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_32F, 3);
  cvConvertScale(img, colorImg);
  cvConvertScale(colorImg, colorImg, 1/255.0f);
  cvCvtColor(colorImg, colorImg, CV_BGR2Lab);
}

void CIETemplateFeature::ConvertToRGB(float *tmp, float *dst, int w, int h) {
  IplImage *img = cvCreateImage(cvSize(w,h), IPL_DEPTH_32F, 3);
  assert(img->widthStep == (int)(w*3*sizeof(float)));
  memcpy(img->imageData, tmp, w*h*3*sizeof(float));
  cvCvtColor(img, img, CV_Lab2BGR);
  //cvConvertScale(img, img, 255.0f);
  memcpy(dst, img->imageData, w*h*3*sizeof(float));
}







// For each pixel location, denseley compute the likelihood that the segmentation around that pixel 
// agrees with the mask, assuming the background and foreground pixels are gaussian distributed
void ComputeColorMaskLikelihood(IplImage *colorImg, CvMat *mask, IplImage *likelihoods=NULL, IplImage *w=NULL, 
				IplImage *t=NULL, IplImage *sigmas=NULL, IplImage *maskAvgs=NULL, IplImage *maskInvAvgs=NULL) {
  IplImage *chanImg = cvCreateImage(cvGetSize(colorImg), colorImg->depth, 1);
  IplImage *maskAvg = cvCreateImage(cvGetSize(colorImg), colorImg->depth, 1);
  IplImage *maskInvAvg = cvCreateImage(cvGetSize(colorImg), colorImg->depth, 1);
  IplImage *chanImgSqr = cvCreateImage(cvGetSize(colorImg), colorImg->depth, 1);
  IplImage *maskAvgSqr = cvCreateImage(cvGetSize(colorImg), colorImg->depth, 1);
  IplImage *maskInvAvgSqr = cvCreateImage(cvGetSize(colorImg), colorImg->depth, 1);
  IplImage *maskAvgMaskInvAvg = cvCreateImage(cvGetSize(colorImg), colorImg->depth, 1);
  IplImage *meanDiffSqr = cvCreateImage(cvGetSize(colorImg), colorImg->depth, 1);
  IplImage *sigma = cvCreateImage(cvGetSize(colorImg), colorImg->depth, 1);
  IplImage *likelihood = cvCreateImage(cvGetSize(colorImg), colorImg->depth, 1);
  CvMat *maskOn = cvCloneMat(mask);
  CvMat *maskOff = cvCloneMat(mask);
  CvMat *aveFilter = cvCloneMat(mask);
  double n = cvSum(maskOn).val[0], N = mask->rows*mask->cols;
  cvConvertScale(maskOn, maskOff, -1/(N-n), 1/(N-n));
  cvConvertScale(maskOn, maskOn, 1/n);
  cvSet(aveFilter, cvScalar(1.0/N));
  if(likelihoods) cvZero(likelihoods);

  for(int i = 0; i < colorImg->nChannels; i++) {
    // Compute the mean color values inside and outside the mask
    cvExtractChannel(colorImg, chanImg, i);
    cvFilter2D(chanImg, maskAvg, maskOn);  // mean pixel value inside the mask
    cvFilter2D(chanImg, maskInvAvg, maskOff);  // mean pixel value outside the mask
    
    // Compute squares of the color image and mean color images
    cvMul(maskAvg, maskAvg, maskAvgSqr);  // the square of the mean pixel value inside the mask
    cvMul(maskInvAvg, maskInvAvg, maskInvAvgSqr);  // the square of the mean pixel value outside the mask
    cvMul(chanImg, chanImg, chanImgSqr);  // the square of each pixel

    if(likelihoods || sigmas) {// Compute (maskAvg-maskInvAvg).^2
      cvAdd(maskAvgSqr, maskInvAvgSqr, meanDiffSqr);
      cvMul(maskAvg, maskInvAvg, maskAvgMaskInvAvg, -2);
      cvAdd(meanDiffSqr, maskAvgMaskInvAvg, meanDiffSqr);

      // Compute the standard deviation sigma, where we have assumed foreground pixels within the 
      // mask and background pixels outside the mask share the same standard deviation
      cvConvertScale(maskAvgSqr, maskAvgSqr, -n/N);
      cvConvertScale(maskInvAvgSqr, maskInvAvgSqr, -(N-n)/N);
      cvFilter2D(chanImgSqr, chanImgSqr, aveFilter);
      cvAdd(chanImgSqr, maskAvgSqr, sigma);
      cvAdd(sigma, maskInvAvgSqr, sigma);
      cvPow(sigma, sigma, .5);

      // For each pixel location, compute the likelihood that the segmentation around that pixel 
      // agrees with the mask, assuming the background and foreground pixels are gaussian distributed
      if(likelihoods) 
	cvDiv(meanDiffSqr, sigma, likelihood);
    }

    // For a particular pixel x, the likelihood of foreground will be wx+t
    if(w) {
      cvSub(maskAvg, maskInvAvg, w);
      cvDiv(t, sigma, w);
    }
    if(t) {
      cvSub(maskInvAvgSqr, maskAvgSqr, t);
      cvDiv(t, sigma, t, .5);
    }

    // Store results to return back to the caller of this function
    if(likelihoods) cvAdd(likelihoods, likelihood, likelihoods);
    if(maskAvgs) cvSetChannel(maskAvgs, maskAvg, i);
    if(maskInvAvgs) cvSetChannel(maskInvAvgs, maskInvAvg, i);
    if(sigmas) cvSetChannel(sigmas, sigma, i);
  }

  // Cleanup
  cvReleaseImage(&chanImg);
  cvReleaseImage(&maskAvg);
  cvReleaseImage(&maskInvAvg);
  cvReleaseImage(&chanImgSqr);
  cvReleaseImage(&maskAvgSqr);
  cvReleaseImage(&maskInvAvgSqr);
  cvReleaseImage(&maskAvgMaskInvAvg);
  cvReleaseImage(&meanDiffSqr);
  cvReleaseImage(&sigma);
  cvReleaseImage(&likelihood);
  cvReleaseMat(&maskOff);
  cvReleaseMat(&maskOn);
  cvReleaseMat(&aveFilter);
}

ColorMaskFeature::ColorMaskFeature(FeatureOptions *fo, SiftParams p) : SlidingWindowFeature(fo) {
  params = p;
  numPoses = 0;
  features = features_flip = NULL;
  colorImages = colorImages_flip = NULL;
  name = StringCopy(p.name);
}

IplImage ***ColorMaskFeature::SlidingWindowDetect(float *weights, int numX, int numY, bool flip, ObjectPose *pose) {
  assert(numX == 1 && numY == 1); 

  IplImage ***images = PrecomputeFeatures(pose, flip);
  IplImage ***retval = (IplImage***)malloc((NumScales())*(sizeof(IplImage**)+fo->NumOrientations()*sizeof(IplImage*)));
  memset(retval, 0, (NumScales())*(sizeof(IplImage**)+fo->NumOrientations()*sizeof(IplImage*)));

  int w, h;
  for(int i = 0; i < NumScales(); i++) {
    retval[i] = ((IplImage**)(retval+NumScales())) + i*fo->NumOrientations();
    for(int j = 0; j < fo->NumOrientations(); j++) {
      if(i >= fo->ScaleOffset()) {
	retval[i][j] = cvCreateImage(cvSize(images[i][j]->width,images[i][j]->height), IPL_DEPTH_32F, 1);
	cvConvertScale(images[i][j], retval[i][j], weights[0]);
      }
    }
  }

  char fname[1000];
  double mi, ma;
  sprintf(fname, "%s.png", pose->Name());
  IplImage *img = MinMaxNormImage(retval[0][0], &mi, &ma);
  if(flip) FlipImage(img);
  cvSaveImage(fname, img);

  return retval;
}

void ColorMaskFeature::Clear(bool full) {
  SlidingWindowFeature::Clear(full);
  if(features) {
    for(int i = 0; i < numPoses; i++) {
      if(features[i]) {
	for(int j = 0; j < fo->NumScales(); j++) 
	  for(int k = 0; k < fo->NumOrientations(); k++) 
	    cvReleaseImage(&features[i][j][k]);
	free(features[i]);
      }
    }
    free(features);
    features = NULL;
  }
  if(features_flip) {
    for(int i = 0; i < numPoses; i++) {
      if(features[i]) {
	for(int j = 0; j < fo->NumScales(); j++) 
	  for(int k = 0; k < fo->NumOrientations(); k++) 
	    cvReleaseImage(&features_flip[i][j][k]);
	free(features_flip[i]);
      }
    }
    free(features_flip);
    features_flip = NULL;
  }
  
  if(colorImages) {
    for(int j = 0; j < fo->NumOrientations(); j++) {
      for(int i = 0; i < NumScales(); i++) {
	cvReleaseImage(&colorImages[j][i][0][0]);
	free(colorImages[j][i][0]);
	free(colorImages[j][i]);
      }
      free(colorImages[j]);
    }
    free(colorImages);
  }
  if(colorImages_flip) {
    for(int j = 0; j < fo->NumOrientations(); j++) {
      for(int i = 0; i < NumScales(); i++) {
	cvReleaseImage(&colorImages_flip[j][i][0][0]);
	free(colorImages_flip[j][i][0]);
	free(colorImages_flip[j][i]);
      }
      free(colorImages_flip[j]);
    }
    free(colorImages_flip);
  }
}

IplImage ***ColorMaskFeature::PrecomputeFeatures(ObjectPose *pose, bool flip) {
  if(!flip && !features) {
    numPoses = pose->GetClasses()->NumPoses();
    features = (IplImage****)malloc(sizeof(IplImage***)*numPoses);
    memset(features, 0, sizeof(IplImage***)*numPoses);
  }
  if(flip && !features_flip) {
    numPoses = pose->GetClasses()->NumPoses();
    features_flip = (IplImage****)malloc(sizeof(IplImage***)*numPoses);
    memset(features_flip, 0, sizeof(IplImage***)*numPoses);
  }
  if((!flip && !features[pose->Id()]) || (flip && !features_flip[pose->Id()])) {
    assert(params.cellWidth == 1);
    params.maxScales = fo->NumScales();
    
    if((!flip && !colorImages) || (flip && !colorImages_flip)) {
      IplImage *colorImg = cvCreateImage(cvSize(fo->GetImage()->width,fo->GetImage()->height), IPL_DEPTH_32F,fo->GetImage()->nChannels);
      cvConvertScale(fo->GetImage(), colorImg);
      IplImage **denseFeatures = PrecomputeColorFastBegin2(colorImg, &params);
      IplImage *****tmp = PrecomputeColorFastFinish2(denseFeatures, colorImg, &params, fo->SpatialGranularity(), fo->NumOrientations(), fo->NumScales(), flip);
      cvReleaseImage(&colorImg);
      for(int i = 0; i < NumScales(); i++) 
	cvReleaseImage(&denseFeatures[i]);
      free(denseFeatures);
      if(flip) colorImages_flip = tmp;
      else colorImages = tmp;
    }
    IplImage *****tmp = flip ? colorImages_flip : colorImages;

    IplImage ***retval = (IplImage***)malloc((fo->NumScales())*(sizeof(IplImage**)+sizeof(IplImage*)*fo->NumOrientations()));
    IplImage **ptr = (IplImage**)(retval+(fo->NumScales()));
    for(int i = 0; i < fo->NumScales(); i++, ptr += fo->NumOrientations()) {
      retval[i] = ptr;
      for(int j = 0; i >= fo->ScaleOffset() && j < fo->NumOrientations(); j++) {
	IplImage *mask = !flip ? pose->GetSegmentationMask() : pose->GetFlipped()->GetSegmentationMask();
	assert(mask);
	IplImage *mask2 = cvCreateImage(cvSize(mask->width/fo->SpatialGranularity(), mask->height/fo->SpatialGranularity()), mask->depth, 1);
	cvResize(mask, mask2);
	CvMat *mat = cvCreateMat(mask2->height, mask2->width, CV_32FC1);
	cvConvertScale(mask2, mat, 1/255.0);
	IplImage *src = tmp[j][i-fo->ScaleOffset()][0][0];
	int w, h;
	fo->GetDetectionImageSize(&w, &h, i-fo->ScaleOffset(), j);
	retval[i][j] = cvCreateImage(cvSize(w,h), IPL_DEPTH_32F, 1);
	ComputeColorMaskLikelihood(src, mat, retval[i][j]);
	cvReleaseMat(&mat);
	cvReleaseImage(&mask2);
      }
    }
    if(flip)
      features_flip[pose->Id()] = retval;
    else
      features[pose->Id()] = retval;
  }
  return flip ? features_flip[pose->Id()] : features[pose->Id()];
}

int ColorMaskFeature::GetFeaturesAtLocation(float *f, int w, int h, int feat_scale, PartLocation *loc, bool flip) {
  int x, y, scale, rot, pose;
  loc->GetDetectionLocation(&x, &y, &scale, &rot, &pose);
  if(scale < fo->ScaleOffset() || scale >= NumScales()) {
    f[0] = 0;
    return 1;
  }

  IplImage ***images = PrecomputeFeatures(loc->GetClasses()->GetPart(loc->GetPartID())->GetPose(pose), flip);
  f[0] = cvGet2D(images[scale][rot], x, y).val[0];

  return 1;
}

