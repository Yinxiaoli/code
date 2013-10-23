/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#include "feature.h"
#include "hog.h"
#include "color.h"
#include "part.h"
#include "pose.h"
#include "attribute.h"
#include "kmeans.h"
#include "classes.h"
#include "dataset.h"
#include "imageProcess.h"
#include "histogram.h"
#include "fisher.h"
#include <stdlib.h>

#ifndef WIN32
#include <unistd.h>
#endif

int g_debug = 1;

bool LoadFeatureWindow(FeatureWindow *f, Json::Value fe) {
  f->name = StringCopy(fe.get("name","").asString().c_str());
  f->w = fe.get("width", 0).asInt();
  f->h = fe.get("height", 0).asInt();
  f->dim = fe.get("dim", 0).asInt();
  f->scale = fe.get("scale", 0).asInt();
  f->orientation = fe.get("orientation", 0).asInt();
  f->dx = fe.get("dx", 0).asInt();
  f->dy = fe.get("dy", 0).asInt();
  f->poseInd = fe.get("poseInd", -1).asInt();
  return true;
} 

Json::Value SaveFeatureWindow(FeatureWindow *f) {
  Json::Value fw;
  fw["name"] = f->name;
  fw["width"] = f->w;
  fw["height"] = f->h;
  fw["dim"] = f->dim;
  fw["scale"] = f->scale;
  fw["orientation"] = f->orientation;
  fw["dx"] = f->dx;
  fw["dy"] = f->dy;
  fw["poseInd"] = f->poseInd;
  return fw;  
}

IplImage *FeatureOptions::GetSegmentation(int scale, int rot) {
  if(!segmentations) {
    assert(segmentationName);
    IplImage *img = GetImage();
    IplImage *seg = cvLoadImage(segmentationName);
    assert(seg && seg->width == img->width && seg->height == img->height && seg->depth == IPL_DEPTH_8U);
    RotationInfo rSrc = ::GetRotationInfo(img->width, img->height, Rotation(0), 1.0f/Scale(my_max(0,params.scaleOffset)));
    segmentations = (IplImage***)malloc(params.numScales*(sizeof(IplImage**)+sizeof(IplImage*)*params.numOrientations));
    IplImage **ptr = (IplImage**)(segmentations+params.numScales);

    /*if(seg->nChannels == 3) {
      IplImage *tmp = cvCreateImage(cvSize(seg->width,seg->height), seg->depth, 1);
      cvCvtColor(seg, tmp, CV_BGR2GRAY);
      cvReleaseImage(&seg);
      seg = tmp;
    }
    */

    for(int i = 0; i < params.numScales; i++, ptr += params.numOrientations) {
      segmentations[i] = ptr;
      for(int j = 0; j < params.numOrientations; j++) {
	RotationInfo r = ::GetRotationInfo(img->width, img->height, Rotation(j), 1.0f/Scale(my_max(i,params.scaleOffset)));
	int w = (int)(r.maxX - r.minX),  h = (int)(r.maxY - r.minY);
	segmentations[i][j] = cvCreateImage(cvSize(w,h), seg->depth, seg->nChannels);
	float mat[6];
	MultiplyAffineMatrices(r.invMat, rSrc.mat, mat);
	cvWarpAffineMultiChannelCustom(seg, segmentations[i][j], mat, CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS);
      }
    }
    cvReleaseImage(&seg);
  }
  return segmentations[scale][rot];
}

void FeatureOptions::GetSegmentationMask(PartLocation *loc, float *f, int w, int h, bool flip, unsigned int color) {
  int x, y, scale, rot, i = 0;
  loc->GetDetectionLocation(&x, &y, &scale, &rot);
  x *= SpatialGranularity();
  y *= SpatialGranularity();
  IplImage *segImg = GetSegmentation(scale, rot);
  IplImage *seg = cvCreateImage(cvSize(w,h),IPL_DEPTH_8U,segImg->nChannels);
  DrawImageIntoImage(segImg, cvRect(x-w/2,y-h/2,w,h), seg, 0, 0);
  if(flip) FlipImage(seg);
  unsigned char *ptr = (unsigned char*)seg->imageData;
  if(color && segImg->nChannels == 3) {
    unsigned char r = (color&0xff0000)>>16, g = (color&0xff00)>>8, b = color&0xff;
    for(y = 0; y < h; y++, ptr += seg->widthStep)
      for(x = 0; x < w; x++, i++)
	f[i] = ptr[x*3]==b && ptr[x*3+1]==g && ptr[x*3+2]==r ? 1 : 0; 
  } else if(segImg->nChannels == 3) {
    for(y = 0; y < h; y++, ptr += seg->widthStep)
      for(x = 0; x < w; x++, i++)
	f[i] = ptr[x*3] || ptr[x*3+1] || ptr[x*3+2] ? 1 : 0; 
  } else {
    for(y = 0; y < h; y++, ptr += seg->widthStep)
      for(x = 0; x < w; x++, i++)
	f[i] = ptr[x]/255.0f;
  }
  cvReleaseImage(&seg);
}
 
FeatureOptions::FeatureOptions(const char *imgName, FeatureParams *p, const char *n, Classes *classes) {
  this->imgName = imgName ? StringCopy(imgName) : NULL;
  img = NULL;
  params = *p;

  name = n ? StringCopy(n) : NULL; 
  features = NULL;
  numFeatureTypes = 0;
  rotations = rotations_base = NULL;
  imageScale = 1;
  segmentationName = NULL;
  segmentations = NULL;

  nthreads = 1;
  
  int i;
  for(i = 0; i < classes->NumFeatureTypes(); i++) {
    SiftParams *feat = classes->GetFeatureType(i);
    if(!strcmp(feat->type, "HOG")) RegisterFeature(new HOGTemplateFeature(this, *feat));
    else if(!strcmp(feat->type, "LBP")) RegisterFeature(new LBPTemplateFeature(this, *feat));
    else if(!strcmp(feat->type, "RGB")) RegisterFeature(new RGBTemplateFeature(this, *feat));
    else if(!strcmp(feat->type, "CIE")) RegisterFeature(new CIETemplateFeature(this, *feat));
    else if(!strcmp(feat->type, "MASK")) RegisterFeature(new ColorMaskFeature(this, *feat));
    else
      assert(0);
  }
  for(int i = 0; i < classes->NumCodebooks(); i++) 
    RegisterFeature(new HistogramFeature(this, classes->GetCodebook(i)));
  for(int i = 0; i < classes->NumFisherCodebooks(); i++) 
    RegisterFeature(new FisherFeature(this, classes->GetFisherCodebook(i)));
}

FeatureOptions::~FeatureOptions() {
  if(name)
    free(name);
  if(imgName)
    free(imgName);

  for(int i = 0; i < numFeatureTypes; i++)
    delete features[i];
  if(features)
    free(features);

  if(img)
    cvReleaseImage(&img);

  if(rotations_base) free(rotations_base);

  if(segmentations) {
    for(int i = 0; i < params.numScales; i++)
      for(int j = 0; j < params.numOrientations; j++)
	cvReleaseImage(&segmentations[i][j]);
    free(segmentations);
  }
}


void FeatureOptions::BuildRotationInfo() {
  int numNegativeScales = 50;
  rotations_base = (RotationInfo**)malloc((params.numScales+numNegativeScales)*(sizeof(RotationInfo*)+params.numOrientations*sizeof(RotationInfo)));
  rotations = rotations_base+numNegativeScales;
  RotationInfo *ptr = (RotationInfo*)(rotations+params.numScales); 
  float sub = SpatialGranularity();
  for(int i = -numNegativeScales; i < params.numScales; i++) {
    rotations[i] = ptr; ptr += params.numOrientations;
    for(int j = 0; j < params.numOrientations; j++) {
      rotations[i][j] = ::GetRotationInfo(img->width, img->height, Rotation(j), 1.0f/sub/Scale(my_max(i,params.scaleOffset)));
    }
  }
}


void FeatureOptions::Clear(bool full) {
  for(int i = 0; i < numFeatureTypes; i++)
    features[i]->Clear(full);
  if(full && img) 
    cvReleaseImage(&img);
  if(segmentations) {
    for(int i = 0; i < params.numScales; i++)
      for(int j = 0; j < params.numOrientations; j++)
	cvReleaseImage(&segmentations[i][j]);
    free(segmentations);
    segmentations = NULL;
  }
}

void FeatureOptions::RegisterFeature(SlidingWindowFeature *f) {
  features = (SlidingWindowFeature**)realloc(features, sizeof(SlidingWindowFeature*)*(numFeatureTypes+1));
  features[numFeatureTypes++] = f;
}

SlidingWindowFeature *FeatureOptions::Feature(const char *n) {
  for(int i = 0; i < numFeatureTypes; i++)
    if(!strcmp(n, features[i]->Name()))
      return features[i];
  return NULL;
}

IplImage ***FeatureOptions::SlidingWindowDetect(float *weights, FeatureWindow *feats, int num, bool flip, ObjectPose *pose) {
  IplImage ***all=NULL, ***curr;
  SlidingWindowFeature *f, *lastF = NULL;
  for(int i = 0; i < num; i++) {
    f = Feature(feats[i].name);
    assert(f);

    curr = f->SlidingWindowDetect(weights, feats[i].w, feats[i].h, flip, pose);
    if(flip) {
      for(int scale = 0; scale < NumScales(); scale++) 
        for(int rot = 0; rot < NumOrientations(); rot++)
	  FlipImage(curr[scale][rot]);
    }

    if(all) {
      for(int scale = 0; scale < NumScales(); scale++) {
        for(int rot = 0; rot < NumOrientations(); rot++) {
	  int s = scale + feats[i].scale, r = (rot+feats[i].orientation+NumOrientations())%NumOrientations();
	  if(s >= params.scaleOffset && s < f->NumScales()) {
	    IplImage *c = curr[s][r];
	    if(feats[i].scale) {
	      c = ConvertDetectionImageCoordinates(curr[s][r], s, r, scale, rot, 0); // scale response image
	    }
	    if(feats[i].dx || feats[i].dy) {  // translate response image
	      IplImage *c2 = cvCreateImage(cvSize(curr[scale][rot]->width,curr[scale][rot]->height),IPL_DEPTH_32F,1);
	      cvZero(c2);
	      DrawImageIntoImage(c, cvRect(0,0,c->width,c->height), c2, feats[i].dx, feats[i].dy);
	      if(c != curr[s][r]) cvReleaseImage(&c);
	      c = c2;
	    }
	    if(all[scale][rot])
	      cvAdd(all[scale][rot], c, all[scale][rot]);
	    else
	      all[scale][rot] = c != curr[s][r] ? c : cvCloneImage(c);
	    if(c != curr[s][r] && c != all[scale][rot]) cvReleaseImage(&c);
	  }
	}
      }
      ReleaseResponses(&curr, f->NumScales());
    } else {
      all = curr;
      assert(!feats[i].scale && !feats[i].orientation && !feats[i].dx && !feats[i].dy && f->NumScales() >= NumScales());
      all = (IplImage***)malloc(NumScales()*(sizeof(IplImage**) + sizeof(IplImage*)*NumOrientations()));
      for(int scale = 0; scale < NumScales(); scale++) {
	all[scale] = ((IplImage**)(all+NumScales())) + scale*NumOrientations();
        for(int rot = 0; rot < NumOrientations(); rot++) {
	  all[scale][rot] = curr[scale][rot];
	}
      }
      for(int i = NumScales(); i < f->NumScales(); i++) 
        for(int rot = 0; rot < NumOrientations(); rot++) 
	  cvReleaseImage(&curr[i][rot]);
      free(curr);
    }
    assert(feats[i].dim == f->NumFeatures(feats[i].w,feats[i].h));
    weights += feats[i].dim;
    lastF = f;
  }
  int w, h;
  for(int scale = 0; scale < params.scaleOffset; scale++) {
    for(int rot = 0; rot < NumOrientations(); rot++) {
      if(!all[scale][rot]) {
	GetDetectionImageSize(&w, &h, scale, rot);
	all[scale][rot] = cvCreateImage(cvSize(w,h),IPL_DEPTH_32F,1);
	cvZero(all[scale][rot]);
      }
    }
  }
  return all;
}


void FeatureOptions::ReleaseResponses(IplImage ****buff, int numScales) {
  if(numScales < 0) numScales = params.numScales;
  for(int i = 0; i < numScales; i++) 
    for(int j = 0; j < params.numOrientations; j++) 
      cvReleaseImage(&(*buff)[i][j]);
  free(*buff);
  *buff = NULL;
}


IplImage *FeatureOptions::ConvertDetectionImageCoordinates(IplImage *srcImg, int srcScale, int srcRot, int dstScale, int dstRot, float d, IplImage *inds) {
  int w,h;
  GetDetectionImageSize(&w, &h, dstScale, dstRot);
  IplImage *retval = cvCreateImage(cvSize(w,h), srcImg->depth, srcImg->nChannels);
  float mat[6];
  RotationInfo rSrc = GetRotationInfo(srcRot, srcScale);
  RotationInfo rDst = GetRotationInfo(dstRot, dstScale);
  MultiplyAffineMatrices(rDst.invMat, rSrc.mat, mat);

  if(inds)
    cvWarpAffineMultiChannelCustomGetIndices(srcImg, retval, inds, mat, d);
  else
    cvWarpAffineMultiChannelCustom(srcImg, retval, mat, d);
  
  return retval;
}


void FeatureOptions::ConvertDetectionCoordinates(float x, float y, int srcScale, int srcRot, int dstScale, int dstRot, float *xx, float *yy) {
  RotationInfo rSrc = GetRotationInfo(srcRot, srcScale);
  RotationInfo rDst = GetRotationInfo(dstRot, dstScale);
  float mat[6];
  MultiplyAffineMatrices(rSrc.invMat, rDst.mat, mat);
  AffineTransformPoint(mat, x, y, xx, yy);  
}


void FeatureOptions::ImageLocationToDetectionLocation(float x, float y, int scale, int rot, int *xx, int *yy) {
  float xxx, yyy;
  int w, h;
 
  
  RotationInfo r = GetRotationInfo(rot, scale);
  AffineTransformPoint(r.mat, x, y, &xxx, &yyy);
  *xx = (int)(xxx+.5f); *yy = (int)(yyy+.5f);

  GetDetectionImageSize(&w, &h, scale, rot);
  if(*xx >= w) *xx = w-1;
  if(*yy >= h) *yy = h-1;
  if(*xx < 0) *xx = 0;
  if(*yy < 0) *yy = 0;
}

void FeatureOptions::DetectionLocationToImageLocation(int x, int y, int scale, int rot, float *xx, float *yy) {
  RotationInfo r = GetRotationInfo(rot, scale);
  AffineTransformPoint(r.invMat, x, y, xx, yy);
}

void FeatureOptions::GetDetectionImageSize(int *w, int *h, int scale, int rot) {
  RotationInfo r = GetRotationInfo(rot,scale);
  *w = (int)(r.maxX - r.minX);
  *h = (int)(r.maxY - r.minY);
}


const char *FeatureOptions::Visualize(Classes *classes, PartLocation *locs, const char *fname_prefix, char *html, bool visualizeWeights, AttributeInstance *attribute) {
  if(html) strcpy(html, "");
    
  if(locs) {
    for(int i = 0; i < numFeatureTypes; i++) {
      IplImage *fimg = features[i]->Visualize(classes, locs, visualizeWeights, attribute);
      if(fimg) {
	char fname[1000], base[1000], desc[1000];
	ExtractFilename(fname_prefix, base);
	sprintf(fname, "%s_%s.png", fname_prefix, features[i]->Name());
	cvSaveImage(fname, fimg);
	cvReleaseImage(&fimg);
	if(html) 
	  sprintf(html+strlen(html), "<br><br><table><center><tr><td><img src=\"%s_%s.png\"></td></tr><tr><td><font size=\"+2\">%s</font> - %s</td></tr></center></table>", base, features[i]->Name(), features[i]->Name(), features[i]->Description(desc));
      }
    }
  } else {
    FeatureWindow *featureWindows = classes->FeatureWindows();
    int numWindows = classes->NumFeatureWindows();
    float *f = new float[classes->NumWindowFeatures()];
    float *fptr = f;
    int width = GetImage()->width, height = GetImage()->height;
    PartLocation l;
    l.Init(classes, width, height, this);
    for(int i = 0; i < numWindows; i++) {
      FeatureWindow *fw = &featureWindows[i];
      SlidingWindowFeature *fe = Feature(fw->name);
      int fscale = fw->scale, w = fw->w, h = fw->h;
      double scale = Scale(fscale);
      if(fw->w > 0 && fw->h > 0) {
	// For features like HOG, compute the closest scale such that a wXh template fills the image
	double x_scale = width/(double)fw->w/classes->SpatialGranularity()/scale;
	double y_scale = height/(double)fw->h/classes->SpatialGranularity()/scale;
	fscale = classes->ScaleInd(my_min(x_scale, y_scale));
	scale = Scale(fscale);
      } else {
	GetDetectionImageSize(&w, &h, fscale, 0);
      if(fw->w < 0) w = w*2/my_abs(fw->w);
      if(fw->h < 0) h = h*2/my_abs(fw->h);
      }
      double x = width/scale*(fw->dx/(float)my_abs(fw->w)+.5);
      double y = height/scale*(fw->dy/(float)my_abs(fw->h)+.5);
      l.SetImageLocation(x, y, scale, 0, NULL);
      float *f_prev = fptr;
      fptr += fe->GetFeaturesAtLocation(fptr, w, h, fscale, &l, false);
      IplImage *fimg = fe->Visualize(f_prev, w, h);
      if(fimg) {
	char fname[1000], base[1000], desc[1000];
	ExtractFilename(fname_prefix, base);
	sprintf(fname, "%s_%s_%d.png", fname_prefix, fe->Name(), i);
	cvSaveImage(fname, fimg);
	cvReleaseImage(&fimg);
	if(html) 
	  sprintf(html+strlen(html), "<br><br><table><center><tr><td><img src=\"%s_%s_%d.png\"></td></tr><tr><td><font size=\"+2\">%s</font> - %s</td></tr></center></table>", base, fe->Name(), i, fe->Name(), fe->Description(desc));
      }
    }
    delete [] f;
  }
  return html;
}


SlidingWindowFeature::SlidingWindowFeature(FeatureOptions *fo) { 
  name = NULL; 
  this->fo = fo; 
  featureImages = NULL;
  featureImages_flip = NULL;
  params = fo->GetParams()->hogParams;
} 


SlidingWindowFeature::~SlidingWindowFeature() { 
  if(name) free(name); 
  Clear(true);
}

void SlidingWindowFeature::Clear(bool full) {
  if(featureImages) {
    int num = my_max(1,params.cellWidth/fo->SpatialGranularity());
    for(int j = 0; j < fo->NumOrientations(); j++) {
      for(int i = 0; i < NumScales()/*+fo->ScaleOffset()*/; i++) {
	for(int x = 0; x < num; x++) {
	  for(int y = 0; y < num; y++) 
	    cvReleaseImage(&featureImages[j][i][x][y]);
	  free(featureImages[j][i][x]);
	}
	free(featureImages[j][i]);
      }
      free(featureImages[j]);
    }
    free(featureImages); featureImages = NULL;
  }
  if(featureImages_flip) {
    int num = my_max(1,params.cellWidth/fo->SpatialGranularity());
    for(int j = 0; j < fo->NumOrientations(); j++) {
      for(int i = 0; i < NumScales()/*+fo->ScaleOffset()*/; i++) {
	for(int x = 0; x < num; x++) {
	  for(int y = 0; y < num; y++) 
	    cvReleaseImage(&featureImages_flip[j][i][x][y]);
	  free(featureImages_flip[j][i][x]);
	}
	free(featureImages_flip[j][i]);
      }
      free(featureImages_flip[j]);
    }
    free(featureImages_flip); featureImages_flip = NULL;
  }
}



TemplateFeature::TemplateFeature(FeatureOptions *fo, SiftParams p) : SlidingWindowFeature(fo) {
  params = p;
}

TemplateFeature::~TemplateFeature() {
}

void TemplateFeature::Clear(bool full) {
  SlidingWindowFeature::Clear(full);
}

IplImage ***TemplateFeature::SlidingWindowDetect(float *weights, int numX, int numY, bool flip, ObjectPose *pose) {
  IplImage *****images = PrecomputeFeatures(flip);
  IplImage *wImg = cvCreateImage(cvSize(numX,numY), IPL_DEPTH_32F, params.numBins);
  assert(wImg->widthStep == (int)(numX*params.numBins*sizeof(float)));

  assert(params.cellWidth >= fo->SpatialGranularity());

#if NORMALIZE_TEMPLATES
  unsigned char *ww = (unsigned char*)wImg->imageData;
  float k = 1.0f / (numX*numY*params.numBins), *wptr;
  int i, j;
  for(i = 0; i < numY; i++, ww += wImg->widthStep) 
    for(j = 0, wptr=(float*)ww; j < numX*params.numBins; j++) 
      wptr[j] = weights[i*numX*params.numBins+j]*k;
#else
  memcpy(wImg->imageData, weights, numX*numY*params.numBins*sizeof(float));
#endif

  if(g_debug > 1) fprintf(stderr, "    score %s...\n", Name());
  

  int nthreads = fo->NumThreads();
  IplImage ***s = ScoreHOG(images, fo->GetImage(), wImg, &params, fo->SpatialGranularity(), fo->NumOrientations(), NumScales(), nthreads);
  IplImage ***retval = (IplImage***)malloc((NumScales())*(sizeof(IplImage**)+fo->NumOrientations()*sizeof(IplImage*)));
  memset(retval, 0, (NumScales())*(sizeof(IplImage**)+fo->NumOrientations()*sizeof(IplImage*)));

  int w, h;
  for(int i = 0; i < NumScales(); i++) {
    retval[i] = ((IplImage**)(retval+NumScales())) + i*fo->NumOrientations();
    for(int j = 0; j < fo->NumOrientations(); j++) {
      retval[i][j] = i >= fo->ScaleOffset() ? s[i-fo->ScaleOffset()][j] : NULL;
      fo->GetDetectionImageSize(&w, &h, i, j);
      assert(i < fo->ScaleOffset() || (w == retval[i][j]->width && h == retval[i][j]->height));
    }
  }
  free(s);


  cvReleaseImage(&wImg);
  return retval;
}

int ExtractFeatures(float *f, int w, int h, PartLocation *loc, IplImage *****images, FeatureOptions *fo, 
		    int cellWidth, int spatialGranularity, bool flip, int ox, int oy) {
  int i, j;
  int ixx, iyy, scale, rot;
  loc->GetDetectionLocation(&ixx, &iyy, &scale, &rot);
  int ix, iy, dx=0, dy=0;
  if(cellWidth >= spatialGranularity) {
    ix = ixx/(cellWidth/spatialGranularity);
    iy = iyy/(cellWidth/spatialGranularity);
    dx = ixx%(cellWidth/spatialGranularity);
    dy = iyy%(cellWidth/spatialGranularity);
  } else {
    ix = ixx*spatialGranularity/cellWidth+ox;
    iy = iyy*spatialGranularity/cellWidth+oy;
  }
  int so = fo->ScaleOffset();
  IplImage *img = images[rot][scale-so][dx][dy];
  if(flip) ix = img->width-1 - ix;
  ix -= w / 2; iy -= h / 2;  
  
  unsigned char *ptr2 = (unsigned char*)img->imageData + iy*img->widthStep + ix*img->nChannels*sizeof(float);
  float *ptr;
  int sz=img->nChannels*sizeof(float);
  float zeros[400] = { 0 };
  for(i = iy; i < iy+h; i++, ptr2 += img->widthStep) {
    for(j = ix, ptr = (float*)ptr2; j < ix+w; j++, ptr += img->nChannels, f += img->nChannels) {
      if(j >= 0 && j < img->width && i >= 0 && i < img->height) 
	memcpy(f, ptr, sz);
      else
	memcpy(f, zeros, sz);
    }
  }
  return w*h*img->nChannels;
}


int TemplateFeature::GetFeaturesAtLocation(float *f, int w, int h, int feat_scale, PartLocation *loc, bool flip, int ox, int oy) {
  int scale;
  loc->GetDetectionLocation(NULL, NULL, &scale);
  params.maxScales = fo->NumScales();
  if(scale < fo->ScaleOffset() || scale >= NumScales()) {
    for(int i = 0; i < w*h*params.numBins; i++)
      f[i] = 0;
    return w*h*params.numBins;
  }

  IplImage *****images = PrecomputeFeatures(flip);
  int n = ExtractFeatures(f, w, h, loc, images, fo, params.cellWidth, fo->SpatialGranularity(), flip, ox, oy);
  
#if NORMALIZE_TEMPLATES
  float k = 1.0f/n;
  for(int i = 0; i < n; i++)
    f[i] *= k;
#endif

  if(strstr(name, "SIFT") && params.normMax) SiftNormalize(f, n, params.normMax, params.normEps);
  
  return n;
}


FeatureWindow *SpatialPyramidFeature(const char *featName, int *numWindows, int numLevels, int dim, 
				     int gridWidthX, int gridWidthY, int startAtLevel, 
				     int w, int h, int scale, int rot, int pose) {
  *numWindows = 0;
  for(int i = 0, px = 1, py = 1; i < numLevels; i++, px *= gridWidthX, py *= gridWidthY)
    if(i >= startAtLevel)
      *numWindows += px*py;

  FeatureWindow *retval = (FeatureWindow*)malloc(*numWindows * sizeof(FeatureWindow));
  if(my_abs(w) % ((int)(pow((double)gridWidthX,numLevels-1))) != 0 || my_abs(h) % ((int)pow((double)gridWidthY,numLevels-1))) {
    fprintf(stderr, "ERROR: width and height for spatial pyramid ROI should be divisible by gridWidth^(numLevels-1)\n");
    return NULL;
  }

  for(int ii = 0, i = 0, px = 1, py = 1; ii < numLevels; ii++, px *= gridWidthX, py *= gridWidthY) {
    if(ii < startAtLevel)
      continue;
    for(int j = 0; j < px; j++) {
      for(int k = 0; k < py; k++, i++) {
	retval[i].name = StringCopy(featName);
	if(w > 0 && h > 0) {
	  // Spatial pyramid localized around a part or bounding box location
	  retval[i].w = w/px;
	  retval[i].h = h/py;
	  retval[i].dx = (int)((j+.5)*w/px-w/2.0);
	  retval[i].dy = (int)((k+.5)*h/py-h/2.0);
	} else {
	  // Image-level spatial pyramid 
	  retval[i].w = -px*2;
	  retval[i].h = -py*2;
	  retval[i].dx = (int)((j+.5)*2-px);
	  retval[i].dy = (int)((k+.5)*2-py);
	}
	retval[i].dim = dim;
	retval[i].scale = scale;
	retval[i].orientation = rot;
	retval[i].poseInd = pose;
      }
    }
  }
  return retval;
}





