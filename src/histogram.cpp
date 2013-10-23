/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#include "histogram.h"
#include "part.h"
#include "pose.h"
#include "attribute.h"
#include "kmeans.h"
#include "classes.h"
#include "dataset.h"
#include "hog.h"
#include "imageProcess.h"
#include <stdlib.h>

#ifndef WIN32
#include <unistd.h>
#endif

HistogramFeature::HistogramFeature(FeatureOptions *fo, FeatureDictionary *d)
  : SlidingWindowFeature(fo) { 
  dictionary = d; 
  params = *fo->Feature(d->baseFeature)->Params();

  char str[1000];
  sprintf(str, "HIST_%s", dictionary->baseFeature);
  name = StringCopy(str);
}

// Compute an image map of assignments to nearest codebook word
IplImage *****HistogramFeature::PrecomputeFeatures(bool flip) {
  if((!flip && !featureImages) || (flip && !featureImages_flip)) {
    params.maxScales = fo->NumScales();
    IplImage *****images = ((TemplateFeature*)fo->Feature(dictionary->baseFeature))->PrecomputeFeatures(flip);
    IplImage *****tmp = (IplImage*****)malloc(sizeof(IplImage****)*fo->NumOrientations());
    int w = dictionary->w, h = dictionary->h;
    int num = fo->CellWidth()/fo->SpatialGranularity();
    int nthreads = fo->NumThreads();

    if(g_debug > 1) fprintf(stderr, "    precompute %s...\n", Name());

    // Build the list of orientations/scales at which to precompute features
    int l_scales[10000], l_rotations[10000], l_x[10000], l_y[10000];
    int num_r = 0;
    for(int j = 0; j < fo->NumOrientations(); j++) {
      tmp[j] = (IplImage****)malloc(sizeof(IplImage***)*NumScales());
      for(int i = 0; i < NumScales(); i++) {
        tmp[j][i] = (IplImage***)malloc(sizeof(IplImage**)*num);
	for(int x = 0; x < num; x++) {
          tmp[j][i][x] = (IplImage**)malloc(sizeof(IplImage*)*num);
          for(int y = 0; y < num; y++) {
	    l_scales[num_r] = i; 
	    l_rotations[num_r] = j;
	    l_x[num_r] = x;
	    l_y[num_r++] = y;
	  }
	}
      }
    }

#ifdef USE_OPENMP
#pragma omp parallel for num_threads(nthreads)
#endif
    for(int m = 0; m < num_r; m++) {
      int k, l, xx, yy, yyy, best;
      float bestD, d, dd;
      int i = l_scales[m], j = l_rotations[m], x = l_x[m], y = l_y[m];

      IplImage *img_o = images[j][i][x][y], *dst;
      tmp[j][i][x][y] = dst = cvCreateImage(cvSize(img_o->width,img_o->height), IPL_DEPTH_32S, 1); 
      IplImage *img = cvCreateImage(cvSize(img_o->width+w,img_o->height+h), IPL_DEPTH_32F, img_o->nChannels);
      int fsz = w*img->nChannels;
      int fsz2 = fsz*sizeof(float);
      int sz = fsz*h;
      float *f = new float[sz];
      float *fptr, *ptr, *wptr;
      int *iPtr;
      char *ptr2 = img->imageData, *iPtr2 = dst->imageData, *ptr3;
      cvCopyMakeBorderMultiChannel(img_o, img, cvPoint(w/2,h/2), IPL_BORDER_CONSTANT, cvScalarAll(0));
      for(yy = 0; yy < img_o->height; yy++, ptr2 += img->widthStep, iPtr2 += dst->widthStep) {
	for(xx = 0, ptr=(float*)ptr2, iPtr=(int*)iPtr2; xx < img_o->width; xx++, ptr += img->nChannels) {
	  // Extract a wXhXc descriptor around xx,yy
	  for(yyy = 0, fptr=f, ptr3=(char*)ptr; yyy < h; yyy++, fptr+=fsz, ptr3 += img->widthStep)
	    memcpy(fptr, ptr3, fsz2);
      if(strstr(name, "SIFT") && params.normMax) SiftNormalize(fptr, sz, params.normMax, params.normEps);
	  
	  if(dictionary->tree_depth) {
	    best = HierarchicalKMeansAssignment(f, dictionary->decision_planes, dictionary->tree_depth, sz);
	  } else {
	    // Find the nearest neighbor in the dictionary to the descriptor
	    bestD = 100000000000000000.0f;
	    for(k = 0, wptr = dictionary->words; k < dictionary->numWords; k++, wptr += sz) {
	      d = 0;
	      for(l = 0; l < sz; l++) {
		dd = wptr[l]-f[l];
		d += dd*dd;
	      }
	      if(d < bestD) { bestD = d; best = k; } 
	    }
	  }
	  iPtr[xx] = best;
	}
      }
      cvReleaseImage(&img);
      delete [] f;
    }

    if(!flip) featureImages = tmp;
    else featureImages_flip = tmp;
  }

  return flip ? featureImages_flip : featureImages;
}

IplImage ***HistogramFeature::SlidingWindowDetect(float *weights, int w, int h, bool flip, ObjectPose *pose) {
  int nthreads = fo->NumThreads();
  int num = fo->CellWidth()/fo->SpatialGranularity();
  int nf = NumFeatures(w,h);
  float a = 1.0f/(w*h), *weights_n = new float[nf+1], *weights_normalized;
  int dx = fo->CellWidth()/fo->SpatialGranularity();

  IplImage *****images = PrecomputeFeatures(flip);
  IplImage ***responses = (IplImage***)malloc((NumScales())*(sizeof(IplImage**)+sizeof(IplImage*)*fo->NumOrientations()));
  memset(responses, 0, (NumScales())*(sizeof(IplImage**)+sizeof(IplImage*)*fo->NumOrientations()));
  for(int i = 0; i < NumScales(); i++)
    responses[i] = ((IplImage**)(responses+NumScales()))+fo->NumOrientations()*i;

  // Since we are computing the dot product <w,f>, and we want to normalize word counts in f by the sliding window area, we
  // can divide w by this area and multiply it times the unnormalized features
  weights_normalized = weights_n+1;
  weights_n[0] = 0;
  for(int j = 0; j < nf; j++) 
    weights_normalized[j] = weights[j]*a;


  // Build the list of orientations/scales at which to precompute features
  int l_scales[10000], l_rotations[10000], l_x[10000], l_y[10000];
  int num_r = 0;
  for(int j = 0; j < fo->NumOrientations(); j++) {
    for(int i = 0; i < NumScales(); i++) {
      for(int x = 0; x < num; x++) {
	for(int y = 0; y < num; y++) {
	  l_scales[num_r] = i; 
	  l_rotations[num_r] = j;
	  l_x[num_r] = x;
	  l_y[num_r++] = y;
	}
      }
    }
  }

#ifdef USE_OPENMP
#pragma omp parallel for num_threads(nthreads)
#endif
  for(int m = 0; m < num_r; m++) {
    int so = fo->ScaleOffset();
    int i = l_scales[m], j = l_rotations[m], x = l_x[m], y = l_y[m];
    int xx, yy;
    char *ptr2, *iPtr2;
    int *ptr;
    float *iPtr, *dstPtr, *iPtr_1, *iPtr_2, *iPtr_3, *iPtr_4;
    int rw, rh; fo->GetDetectionImageSize(&rw, &rh, i, j);
    IplImage *dst = responses[i+so][j] = cvCreateImage(cvSize(rw,rh), IPL_DEPTH_32F, 1);

    IplImage *img_o = images[j][i][x][y]; 
    assert(x+img_o->width*dx <= rw && y+img_o->height*dx <= rh);
    IplImage *img = cvCreateImage(cvSize(img_o->width+w,img_o->height+h), IPL_DEPTH_32S, 1);
    IplImage *integralImg = cvCreateImage(cvSize(img_o->width+w+1,img_o->height+h+1), IPL_DEPTH_32F, 1);
    cvCopyMakeBorder(img_o, img, cvPoint(w/2,h/2), IPL_BORDER_CONSTANT, cvScalarAll(0));

    // Compute the integral image for the histogram features dot product'd with the weight vector
    for(xx = 0, iPtr = (float*)integralImg->imageData; xx <= img->width; xx++) iPtr[xx] = 0;
    for(yy = 0, ptr2=img->imageData, iPtr2=integralImg->imageData+integralImg->widthStep; 
	yy < img->height; yy++, ptr2 += img->widthStep, iPtr2 += integralImg->widthStep) {
      ptr=(int*)ptr2; iPtr=(float*)iPtr2;
      iPtr[0] = *(float*)(iPtr2-integralImg->widthStep);
      for(xx = 0; xx < img->width; xx++) 
	iPtr[xx+1] = iPtr[xx] + weights_normalized[ptr[xx]];
    }

    // Use the integral image to compute the response in a wXh window around each pixel
    for(yy = 0, ptr2=dst->imageData+y*dst->widthStep, iPtr2=integralImg->imageData; yy < img_o->height;  yy++, ptr2 += dst->widthStep*dx, iPtr2 += integralImg->widthStep) {
      iPtr_1=(float*)iPtr2; iPtr_2=iPtr_1+w; iPtr_3=(float*)(iPtr2+h*integralImg->widthStep); iPtr_4=iPtr_3+w;
      for(xx = 0, dstPtr=((float*)ptr2)+x; xx < dst->width; xx++, dstPtr += dx) 
	*dstPtr = iPtr_1[xx]+iPtr_4[xx] - (iPtr_2[xx]+iPtr_3[xx]);
    }

    cvReleaseImage(&img);
    cvReleaseImage(&integralImg);
  }
  
  int ww, hh;
  for(int ii = fo->ScaleOffset(); ii < NumScales(); ii++) {
    for(int jj = 0; jj < fo->NumOrientations(); jj++) {
      fo->GetDetectionImageSize(&ww, &hh, ii, jj);
      assert(ww == responses[ii][jj]->width && hh == responses[ii][jj]->height);
    }
  }

  delete [] weights_n;

  return responses;
}

void HistogramFeature::Clear(bool full) {
  //SlidingWindowFeature *base = fo->Feature(dictionary->baseFeature);
  //if(base) base->Clear(full);
  SlidingWindowFeature::Clear(full);
}

typedef struct {
  int ind;
  float val;
} FeatureValInd;

int FeatureValIndCompare(const void *a1, const void *a2) {
  FeatureValInd *v1 = (FeatureValInd*)a1, *v2 = (FeatureValInd*)a2; 
  float d = (((FeatureValInd*)v2)->val-(((FeatureValInd*)v1)->val));
  return d < 0 ? -1 : (d > 0 ? 1 : (v1->ind-v2->ind)); 
} 
IplImage *HistogramFeature::Visualize(float *f, int w, int h, float mi, float ma) {
  // Sort the histogram bins by the weight of each word
  FeatureValInd *ff = (FeatureValInd*)malloc(sizeof(FeatureValInd)*dictionary->numWords);
  int i;
  for(i = 0; i < dictionary->numWords; i++) {
    ff[i].val = f[i];
    ff[i].ind = i;
  }
  qsort(ff, dictionary->numWords, sizeof(FeatureValInd), FeatureValIndCompare);

  if(mi == 0 && ma == 0) {
    ma = ff[0].val;
    mi = ff[dictionary->numWords-1].val;
  }
  int num = 0;
  for(i = 0; i < dictionary->numWords; i++) 
    if(ff[i].val) 
      num++;

  int sz = dictionary->w*dictionary->h*dictionary->nChannels;
  int hspace = 4;
  int bar_height = 150;
  int c = fo->CellWidth();
  IplImage *img = cvCreateImage(cvSize((dictionary->w*c+hspace)*num+hspace, bar_height+hspace*3+dictionary->h*c), IPL_DEPTH_8U, 3);
  cvSet(img, cvScalar(255,255,255));
  SlidingWindowFeature *base = fo->Feature(dictionary->baseFeature);
  int x = hspace, y = hspace+bar_height;
  for(i = 0; i < dictionary->numWords; i++) {
    if(ff[i].val) {
      IplImage *key = base->Visualize(dictionary->words+ff[i].ind*sz, dictionary->w, dictionary->h);
      if(key->nChannels == 1) {
	IplImage *img2 = cvCreateImage(cvSize(key->width,key->height), IPL_DEPTH_8U, 3);
	cvConvertImage(key, img2);
	cvReleaseImage(&key);
	key = img2;
      }
      
      DrawImageIntoImage(key, cvRect(0,0,key->width,key->height), img, x, hspace+y);
      cvRectangle(img, cvPoint(x,y-bar_height*(ff[i].val-mi)/(ma-mi)), cvPoint(x+key->width,y-bar_height*(-mi)/(ma-mi)), CV_RGB(0,0,255), -1);
      x += hspace + key->width;
      cvReleaseImage(&key);
    }
  }

  free(ff);

  return img;
}

// Visualize an assignment to part locations as a HOG image
IplImage *HistogramFeature::Visualize(Classes *classes, PartLocation *locs, bool visualizeWeights, AttributeInstance *attribute) {
  if(attribute) { // Visualize a particular attribute detector
    // Extract histogram for attribute detector weights
    Attribute *am = attribute->Model();
    float *tmp = new float[am->NumWeights()];
    if(visualizeWeights) {
      if(!am->GetWeights(tmp, name)) {
	delete [] tmp;
	return NULL;
      }
    } else
      GetFeaturesAtLocation(tmp, am->Width(), am->Height(), 0, &locs[am->Part()->Id()], attribute->IsFlipped());
    
    // Extract the max and min of any type of histogram entry
    float mi = 1000000, ma = -10000000;
    for(int i = 0; i < fo->NumFeatureTypes(); i++) {
	float tmp2[100000];
	int n = am->GetWeights(tmp2, fo->GetFeatureType(i)->Name());
	for(int k = 0; k < n; k++) {
	  if(tmp2[k] < mi) mi = tmp2[k];
	  if(tmp2[k] > ma) ma = tmp2[k];
	}
    }

    IplImage *retval = Visualize(tmp, am->Width(), am->Height(), mi, ma);
    delete [] tmp;
    return retval;
  }

  IplImage **images = (IplImage**)malloc(sizeof(IplImage*)*classes->NumParts());
  memset(images, 0, sizeof(IplImage*)*classes->NumParts());
  int max_width = 0;
  int yy[1000], yyy = 0;
  int pad = 10;
  CvFont font;
  double hScale=1.0;
  double vScale=1.0;
  int    lineWidth=1; 
  int baseline, n; 
  CvSize sz; 
  cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale,vScale,0,lineWidth);
  cvGetTextSize("hello", &font, &sz, &baseline);
  
  // Visualize each part separately
  for(int i = 0; i < classes->NumParts(); i++) {
    int x, y, scale, rot, pose;
    locs[i].GetDetectionLocation(&x, &y, &scale, &rot, &pose);

    if(pose >= 0) {
      ObjectPose *p = classes->GetPart(i)->GetPose(pose);
      Attribute *attr = p->Appearance();
      if(attr) {
	float *tmp = new float[attr->NumWeights()];
	if(visualizeWeights) 
	  n = attr->GetWeights(tmp, name);
	else {
	  int s=scale, j = 0;
	  PartLocation l(locs[i]);  
	  while(s < fo->ScaleOffset() && j < attr->NumFeatureTypes()) {
	    s = scale + attr->Feature(j)->scale;
	    j++;
	  }
	  l.SetDetectionLocation(x, y, s, rot, pose, LATENT, LATENT);
	  if(j < attr->NumFeatureTypes() && attr->Width(j)) n = GetFeaturesAtLocation(tmp, attr->Width(j), attr->Height(j), 0, &l, p->IsFlipped());
	  else n = 0;
	}
	yy[i] = yyy;
	if(n) {
	  images[i] = Visualize(tmp, attr->Width(), attr->Height());
	  if(images[i]->width > max_width) 
	    max_width = images[i]->width;
	  yyy += images[i]->height+pad+sz.height;
	}
	delete [] tmp;
      }
    } 
  }
  if(!yyy) {
    free(images);
    return NULL;
  }

  // Stack the visualizations of each part into the output image
  IplImage *img = cvCreateImage(cvSize(max_width,yyy), IPL_DEPTH_8U, 3);
  cvSet(img, cvScalar(255,255,255));
  for(int i = 0; i < classes->NumParts(); i++) {
    if(images[i]) {
      DrawImageIntoImage(images[i], cvRect(0,0,images[i]->width,images[i]->height), img, (max_width-images[i]->width)/2, yy[i]);
      cvGetTextSize(classes->GetPart(i)->Name(), &font, &sz, &baseline);
      cvPutText(img,classes->GetPart(i)->Name(),cvPoint((max_width-sz.width)/2,yy[i]+images[i]->height+sz.height), &font, CV_RGB(0,0,0));
      cvReleaseImage(&images[i]);
    }
  }
  free(images);

  return img;
}


int ExtractHistogramFeatures(float *f, int w, int h, PartLocation *loc, IplImage *****images, 
			     FeatureOptions *fo, int cellWidth, int numWords, bool flip) {
  int i, j;
  float a = 1.0f/(w*h);

  for(i = 0; i < numWords; i++) 
    f[i] = 0;
  
  int ixx, iyy, scale, rot, width, height;
  loc->GetDetectionLocation(&ixx, &iyy, &scale, &rot);
  fo->GetDetectionImageSize(&width, &height, scale, rot);
  if(flip) ixx = width-1 - ixx;
  int ix = ixx/(cellWidth/fo->SpatialGranularity());
  int iy = iyy/(cellWidth/fo->SpatialGranularity());
  int dx = ixx%(cellWidth/fo->SpatialGranularity());
  int dy = iyy%(cellWidth/fo->SpatialGranularity());
  IplImage *img = images[rot][scale-fo->ScaleOffset()][dx][dy];
  ix -= w / 2; iy -= h / 2;  
  unsigned char *ptr2 = (unsigned char*)(img->imageData + iy*img->widthStep);
  int *ptr;
  for(i = iy; i < iy+h; i++, ptr2 += img->widthStep) {
    for(j = ix, ptr = (int*)ptr2; j < ix+w; j++) {
      if(j >= 0 && j < img->width && i >= 0 && i < img->height) {
        f[ptr[j]] += a;
      }
    }
  }
  return numWords;
}
int HistogramFeature::GetFeaturesAtLocation(float *f, int w, int h, int feat_scale, PartLocation *loc, bool flip) {
  int ixx, iyy, scale, rot;
  loc->GetDetectionLocation(&ixx, &iyy, &scale, &rot);

  params.maxScales = fo->NumScales();
  if(scale < fo->ScaleOffset() || scale >= NumScales()) {
    for(int i = 0; i < dictionary->numWords; i++)
      f[i] = 0;
    return dictionary->numWords;
  }

  IplImage *****images = PrecomputeFeatures(flip);
  return ExtractHistogramFeatures(f, w, h, loc, images, fo, fo->CellWidth(), dictionary->numWords, flip);
}



FeatureDictionary::FeatureDictionary(const char *featName, int w, int h, int nChannels, int numWords, int depth) {
  baseFeature = featName ? StringCopy(featName) : NULL;
  this->w = w; this->h = h; this->nChannels = nChannels; this->numWords = numWords; this->words = NULL;
  fileName = NULL;
  tree_depth = depth;
  decision_planes = NULL;
  if(tree_depth > 0) 
    assert(numWords == 1<<tree_depth);
}

FeatureDictionary::~FeatureDictionary() {
  if(baseFeature) free(baseFeature);
  if(words) free(words);
  if(fileName) free(fileName);
  if(decision_planes) free(decision_planes);
}

bool FeatureDictionary::Load(const char *fname) {
  if(fileName) free(fileName);
  fileName = StringCopy(fname);

  FILE *fin = fopen(fname, "rb");
  if(!fin) return false;
  int len;
  if(!fread(&len, sizeof(int), 1, fin) || len <= 0 || len > 1000) {
    fclose(fin);
    return false;
  }
  baseFeature = (char*)malloc(sizeof(char)*(len+1));
  if((int)fread(baseFeature, sizeof(char), len, fin) < len) {
    fclose(fin);
    return false;
  }
  baseFeature[len] = '\0';
  if(!fread(&w, sizeof(int), 1, fin) || !fread(&h, sizeof(int), 1, fin) || !fread(&nChannels, sizeof(int), 1, fin) ||
     !fread(&numWords, sizeof(int), 1, fin)) {
    fclose(fin);
    return false;
  }
  if(!LoadData(fin)){
    fclose(fin);
    return false;
  }

  fclose(fin);
  return true;
}

bool FeatureDictionary::LoadData(FILE *fin) {
  words = (float*)malloc(sizeof(float)*numWords*nChannels*w*h);
  if((int)fread(words, sizeof(float), numWords*nChannels*w*h, fin) != numWords*nChannels*w*h) return false;

  if(fread(&tree_depth, sizeof(int), 1, fin) && tree_depth) {
    assert(numWords == 1<<tree_depth);
    decision_planes = (float**)malloc(sizeof(float*)*(numWords-1) + sizeof(float)*(numWords-1)*(nChannels*w*h+1));
    float *ptr = (float*)(decision_planes+(numWords-1));
    if((int)fread(ptr, sizeof(float), (numWords-1)*(nChannels*w*h+1), fin) != (numWords-1)*(nChannels*w*h+1)) return false;
    for(int i = 0; i < numWords-1; i++, ptr += (nChannels*w*h+1)) 
      decision_planes[i] = ptr;
  }
  return true;
}

bool FeatureDictionary::Save(const char *fname) {
  if(fileName) free(fileName);
  fileName = StringCopy(fname);

  FILE *fout = fopen(fname, "wb");
  if(!fout) return false;
  int len = strlen(baseFeature);
  if(!fwrite(&len, sizeof(int), 1, fout) || !fwrite(baseFeature, sizeof(char), len, fout) || !fwrite(&w, sizeof(int), 1, fout) || !fwrite(&h, sizeof(int), 1, fout) || !fwrite(&nChannels, sizeof(int), 1, fout) ||
     !fwrite(&numWords, sizeof(int), 1, fout)) {
    fclose(fout);
    return false;
  }
  if(!SaveData(fout)){
    fclose(fout);
    return false;
  }
  fclose(fout);
  return true;
}

bool FeatureDictionary::SaveData(FILE *fout) {
  if((int)fwrite(words, sizeof(float), numWords*nChannels*w*h, fout) != numWords*nChannels*w*h) return false;
  if(!fwrite(&tree_depth, sizeof(int), 1, fout))
    return false;
  if(tree_depth > 0 && (int)fwrite(decision_planes+numWords-1, sizeof(float), (numWords-1)*(nChannels*w*h+1), fout) != (numWords-1)*(nChannels*w*h+1)) 
    return false;
  return true;
}




void FeatureDictionary::LearnDictionary(Dataset *dataset, int maxImages, int ptsPerImage, int resize_image_width) {
  int ptSz = w*h*nChannels, i;
  float *ptr;
  int numImages = my_min(maxImages, dataset->NumExamples());
  int numPts = numImages*ptsPerImage;

  // Allocate memory for cluster centers and points
  words = (float*)malloc(sizeof(float)*numWords*ptSz);
  float **pts = (float**)malloc((sizeof(float*)+sizeof(float)*ptSz)*numPts);
  for(i = 0, ptr = (float*)(pts+numPts); i < numPts; i++, ptr += ptSz)
    pts[i] = ptr;

  int *perm = RandPerm(dataset->NumExamples());

  // Extract interest points
#ifdef USE_OPENMP
  #pragma omp parallel for
#endif
  for(int k = 0; k < numImages; k++) {
    int ii = perm[k];
    ImageProcess *process = dataset->GetExampleData(ii)->GetProcess(dataset->GetClasses());
    FeatureOptions *fo = process->Features();
    IplImage *im = fo->GetImage(resize_image_width);
    TemplateFeature *f = (TemplateFeature*)fo->Feature(baseFeature);
    if(g_debug) fprintf(stderr, "Compute %s for %s...\n", f->Name(), fo->Name());
    f->PrecomputeFeatures(false);
    
    FeatureParams *params = fo->GetParams();
    PartLocation loc;
    loc.Init(dataset->GetClasses(), process->Image()->width, process->Image()->height, NULL);
    int nn = k*ptsPerImage;

    //IplImage *img = cvCloneImage(im);
    for(int j = 0; j < ptsPerImage; j++) {
      int ww, hh, scale = 0, rot = 0, x, y;
      fo->GetDetectionImageSize(&ww, &hh, scale, rot);
      x = rand()%ww;
      y = rand()%hh;

      loc.SetDetectionLocation(x, y, scale, rot, 0, LATENT, LATENT);
      f->GetFeaturesAtLocation(pts[nn+j], w, h, 0, &loc, false);

      
      //IplImage *tmp = f->Visualize(pts[nn+j], w, h);
      //if(tmp->nChannels == 1) {
      //IplImage *img2 = cvCreateImage(cvSize(tmp->width,tmp->height), IPL_DEPTH_8U, 3);
      //cvConvertImage(tmp, img2);
      //cvReleaseImage(&tmp);
      //tmp = img2;
      //}
      //float xx, yy;
      //loc.GetImageLocation(&xx, &yy);
      //DrawImageIntoImage(tmp, cvRect(0,0,tmp->width,tmp->height), img, (int)xx-32, (int)yy-32);
      //cvReleaseImage(&tmp);
    }
    //char fname[1000];  sprintf(fname, "tmp%d.png", k);
    //cvSaveImage(fname, img);
    //cvReleaseImage(&img);
    
    //IplImage *****images = f->PrecomputeFeatures(false);
    //sprintf(fname, "Tmp%d.png", k); 
    //IplImage *tmp = f->Visualize((float*)images[0][0][0][0]->imageData, images[0][0][0][0]->width, images[0][0][0][0]->height);
    //cvSaveImage(fname, tmp);
    //cvReleaseImage(&tmp);
    

    dataset->GetExampleData(ii)->GetProcess(dataset->GetClasses())->Clear(); 
  }

  free(perm);

  LearnDictionaryFromPoints(pts, numPts);

  free(pts);
}


void FeatureDictionary::LearnDictionaryFromPoints(float **pts, int numPts) {
  int ptSz = w*h*nChannels, i;

  // Allocate memory for cluster centers
  float **centers = (float**)malloc(sizeof(float*)*numWords);
  float *ptr = words;
  for(i = 0; i < numWords; i++, ptr += ptSz) 
    centers[i] = ptr;

  // If using hierarchical k-means, allocate memory for tree decision planes
  if(tree_depth > 0) {
    assert(numWords == 1<<tree_depth);
    decision_planes = (float**)malloc(sizeof(float*)*(numWords-1) + sizeof(float)*(numWords-1)*(ptSz+1));
    for(i = 0, ptr = (float*)(decision_planes+(numWords-1)); i < numWords-1; i++, ptr += ptSz+1) 
      decision_planes[i] = ptr;
  }

  // Learn vocabulary
  int *assignments = NULL;
  if(tree_depth > 0) assignments = HierarchicalKMeans<float>(pts, numPts, centers, decision_planes, tree_depth, ptSz, 2);
  else assignments = KMeans<float>(pts, numPts, centers, numWords, ptSz, 2);

  delete [] assignments;
  free(centers);
}




