#include "fisher.h"
#include "part.h"
#include "classes.h"
#include "pose.h"
#include "sparse_vector.h"
#include "hog.h"
#include "attribute.h"


float **ProjectPCA_sse(float **pts, int numPts, float **eigvects_i, float *pca_mu, float **ptsProj, int ptSz, int pcaDims);
void ComputeFisherVector_sse(float **pts, float **mu, float **sigma, float *mixtureWeights, 
			     float *fisherVector, int numPts, int K, int ptSz, float *weights, float *wordCounts=NULL);
void NormalizeVector_sse(float *fisherVector, int dim);


FisherFeatureDictionary::FisherFeatureDictionary(const char *featName, int w, int h, int nChannels, int numWords, int pcaDims) : 
  FeatureDictionary(featName, w, h, nChannels, numWords) {
  mus = NULL;
  sigmas = NULL;
  mixtureWeights = NULL;
  eigvects = eigvects_t = NULL;
  eigvals = NULL;
  pca_mu = NULL;
  this->pcaDims = my_min(pcaDims, DescriptorDims());
}

FisherFeatureDictionary::~FisherFeatureDictionary() {
  if(mus) FreeArray(mus, true);
  if(sigmas) FreeArray(sigmas, true);
  if(mixtureWeights) FreeArray(mixtureWeights, true);
  if(eigvects) FreeArray(eigvects, true);
  if(eigvects_t) FreeArray(eigvects_t, true);
  if(eigvals) FreeArray(eigvals, true);
  if(pca_mu) FreeArray(pca_mu, true);
}

bool FisherFeatureDictionary::LoadData(FILE *fin) {
  if(!fread(&pcaDims, sizeof(int), 1, fin)) return false;
  
  mus = Create2DArray<float>(numWords, pcaDims, true);
  sigmas = Create2DArray<float>(numWords, pcaDims, true);
  mixtureWeights = Create1DArray<float>(numWords, true);

  int i;
  for(i = 0; i < numWords; i++)
    if((int)fread(mus[i], sizeof(float), pcaDims, fin) != pcaDims) 
      return false;
  for(i = 0; i < numWords; i++)
    if((int)fread(sigmas[i], sizeof(float), pcaDims, fin) != pcaDims) 
      return false;
  if((int)fread(mixtureWeights, sizeof(float), numWords, fin) != numWords) 
    return false;

  if(pcaDims != DescriptorDims()) {
    int ptSz = DescriptorDims();
    eigvects = Create2DArray<float>(pcaDims, ptSz, true);
    eigvals = Create1DArray<float>(pcaDims, true);
    pca_mu = Create1DArray<float>(ptSz, true);
    if((int)fread(pca_mu, sizeof(float), ptSz, fin) != ptSz) 
      return false;
    for(i = 0; i < pcaDims; i++)
      if((int)fread(eigvects[i], sizeof(float), ptSz, fin) != ptSz) 
	return false;
    if((int)fread(eigvals, sizeof(float), pcaDims, fin) != pcaDims) 
      return false;
    
#if HAVE_SSE   
    eigvects_t = Create2DArray<float>(ptSz,pcaDims,true);
    for(int j = 0; j < ptSz; j++)
      for(int k = 0; k < pcaDims; k++) 
        eigvects_t[j][k] = eigvects[k][j];
#endif
  }

  return true;
}

bool FisherFeatureDictionary::SaveData(FILE *fout) {
  int i;

  if(!fwrite(&pcaDims, sizeof(int), 1, fout)) return false;
  for(i = 0; i < numWords; i++)
    if((int)fwrite(mus[i], sizeof(float), pcaDims, fout) != pcaDims) 
      return false;
  for(i = 0; i < numWords; i++)
    if((int)fwrite(sigmas[i], sizeof(float), pcaDims, fout) != pcaDims) 
      return false;
  if((int)fwrite(mixtureWeights, sizeof(float), numWords, fout) != numWords) 
    return false;

  if(pcaDims != DescriptorDims()) {
    int ptSz = DescriptorDims();
    if((int)fwrite(pca_mu, sizeof(float), ptSz, fout) != ptSz) 
      return false;
    for(i = 0; i < pcaDims; i++)
      if((int)fwrite(eigvects[i], sizeof(float), ptSz, fout) != ptSz) 
	return false;
    if((int)fwrite(eigvals, sizeof(float), pcaDims, fout) != pcaDims) 
      return false;
  }

  return true;
}


void FisherFeatureDictionary::TrainPCA(float **pts, int numPts) {
  fprintf(stderr, "Train PCA on %d points...", numPts);

  // Allocate memory
  int ptSz = DescriptorDims();
  int dims = ptSz, i, j, k;
  float **pts_mean_subtracted = Create2DArray<float>(numPts, ptSz, false);
  float **covar = Create2DArray<float>(ptSz, ptSz, false);
  float **U = Create2DArray<float>(ptSz, ptSz, false);
  float **eigvectsTmp = Create2DArray<float>(dims, ptSz, false);
  eigvects = Create2DArray<float>(pcaDims,ptSz, true);
  eigvals = Create1DArray<float>(dims, true);
  pca_mu = Create1DArray<float>(ptSz, true);

  // Mean subtract the data
  for(j = 0; j < ptSz; j++)
    pca_mu[j] = 0;
  for(i = 0; i < numPts; i++)
    for(j = 0; j < ptSz; j++)
      pca_mu[j] += pts[i][j];
  for(j = 0; j < ptSz; j++)
    pca_mu[j] /= numPts;
  for(i = 0; i < numPts; i++)
    for(j = 0; j < ptSz; j++)
      pts_mean_subtracted[i][j] = pts[i][j]-pca_mu[j];
  
  // Compute covariance matrix
  for(j = 0; j < ptSz; j++)
    for(k = 0; k < ptSz; k++)
      covar[j][k] = 0;
  for(i = 0; i < numPts; i++)
    for(j = 0; j < ptSz; j++)
      for(k = 0; k < ptSz; k++)
	covar[j][k] += pts_mean_subtracted[i][j]*pts_mean_subtracted[i][k];
  for(j = 0; j < ptSz; j++)
    for(k = 0; k < ptSz; k++)
      covar[j][k] /= (numPts-1);

  CvMat UMat = cvMat(ptSz, ptSz, CV_32FC1, U+ptSz);
  CvMat eigvectsMat = cvMat(dims, ptSz, CV_32FC1, eigvectsTmp+dims);
  CvMat eigvalsMat = cvMat(ptSz, 1, CV_32FC1, eigvals);
  CvMat covarMat = cvMat(ptSz, ptSz, CV_32FC1, covar+ptSz);

  // Train pca
  cvSVD(&covarMat, &eigvalsMat, &UMat, &eigvectsMat, CV_SVD_MODIFY_A|CV_SVD_U_T|CV_SVD_V_T);
  for(i = 0; i < pcaDims; i++)
    for(j = 0; j < ptSz; j++)
      eigvects[i][j] = eigvectsTmp[i][j];//sqrt(eigvals[i]);
 
#if HAVE_SSE   
  eigvects_t = Create2DArray<float>(ptSz,pcaDims,true);
  for(int j = 0; j < ptSz; j++)
    for(int k = 0; k < pcaDims; k++) 
      eigvects_t[j][k] = eigvects[k][j];
#endif

  // Cleanup
  FreeArray(pts_mean_subtracted, false);
  FreeArray(U, false);
  FreeArray(eigvectsTmp, false);
  FreeArray(covar, false);

  fprintf(stderr, "done\n");
}

float **FisherFeatureDictionary::ProjectPCA(float **pts, int numPts, float **ptsProj) {
  int ptSz = DescriptorDims(), i, j, k;
  float d;
  if(!ptsProj) 
    ptsProj = Create2DArray<float>(numPts, pcaDims, true);
 
  for(i = 0; i < numPts; i++)
    for(k = 0; k < pcaDims; k++) 
      ptsProj[i][k] = 0;
  for(i = 0; i < numPts; i++) {
    for(j = 0; j < ptSz; j++) {
      d = pts[i][j]-pca_mu[j];
      for(k = 0; k < pcaDims; k++) 
	ptsProj[i][k] += d*eigvects[k][j];
    }
  }

  return ptsProj;
}

float **FisherFeatureDictionary::ReprojectPCA(float **ptsProj, int numPts, float **pts) {
  int ptSz = DescriptorDims(), i, j, k;
  float d;
  if(!pts) 
    pts = Create2DArray<float>(numPts, ptSz, true);
 
  for(i = 0; i < numPts; i++)
    for(k = 0; k < ptSz; k++) 
      pts[i][k] = pca_mu[k];
  for(i = 0; i < numPts; i++) {
    for(j = 0; j < pcaDims; j++) {
      d = ptsProj[i][j];
      for(k = 0; k < ptSz; k++) 
        pts[i][k] += d*eigvects[j][k];
    }
  }

  return pts;
}


void FisherFeatureDictionary::LearnDictionaryFromPoints(float **pts, int numPts) {
  // Allocate memory
  if(mus) FreeArray(mus, true);
  if(sigmas) FreeArray(sigmas, true);
  if(mixtureWeights) FreeArray(mixtureWeights, true);
  mus = Create2DArray<float>(numWords, pcaDims, true);
  sigmas = Create2DArray<float>(numWords, pcaDims, true);
  mixtureWeights = Create1DArray<float>(numWords, true);

  // Use PCA to reduce the dimensionality of the feature descriptor space
  float **ptsProj = pts;
  if(pcaDims != DescriptorDims()) {
    TrainPCA(pts, numPts);
    ptsProj = ProjectPCA(pts, numPts);
  }

  // Train a Gaussian mixture model
  GMMLearn<float>(ptsProj, mus, sigmas, mixtureWeights, numPts, numWords, pcaDims, 2, 0.0001f);

  if(ptsProj != pts)
    FreeArray(ptsProj, true);
}

void FisherFeatureDictionary::ComputeFisherVector(float **pts, int numPts, float *fisherVector, bool normalize, float *weights, float **ptsProjSpace, float *wordCounts) {
  float **ptsProj = pts;

#if HAVE_SSE
  if(pcaDims != DescriptorDims()) {
    ptsProj = ProjectPCA_sse(pts, numPts, eigvects_t, pca_mu, ptsProjSpace, DescriptorDims(), pcaDims);
  }
  
  ::ComputeFisherVector_sse(ptsProj, mus, sigmas, mixtureWeights, fisherVector, numPts, numWords, pcaDims, weights, wordCounts);
  if(normalize)
    NormalizeVector_sse(fisherVector, FisherFeatureDims());

#else
  if(pcaDims != DescriptorDims()) 
    ptsProj = ProjectPCA(pts, numPts, ptsProjSpace);
  
  ::ComputeFisherVector<float>(ptsProj, mus, sigmas, mixtureWeights, fisherVector, numPts, numWords, pcaDims, weights, wordCounts);
  assert(my_abs(pts[0][0] < 10000));
  if(normalize)
    NormalizeVector<float>(fisherVector, FisherFeatureDims());
#endif

  if(ptsProj != pts && ptsProj != ptsProjSpace)
    FreeArray(ptsProj, true);
}



FisherFeature::FisherFeature(FeatureOptions *fo, FisherFeatureDictionary *d)
  : SlidingWindowFeature(fo) { 
  dictionary = d; 
  params = *fo->Feature(d->baseFeature)->Params();

  char str[1000];
  sprintf(str, "FISHER_%s", dictionary->baseFeature);
  name = StringCopy(str);
  visW = NULL;
}

// Not used currently, but in the future could precompute integral images of the power normalized 
// fisher encodings for each dimension, as well as an integral image of the squared L2 norm
IplImage *****FisherFeature::PrecomputeFeatures(bool flip) {
  return NULL; 
}

IplImage ***FisherFeature::SlidingWindowDetect(float *weights, int ww, int hh, bool flip, ObjectPose *pose) {
  int nthreads = fo->NumThreads();
  int num = fo->CellWidth()/fo->SpatialGranularity();
  int dx = fo->CellWidth()/fo->SpatialGranularity();
  IplImage *****images = ((TemplateFeature*)fo->Feature(dictionary->baseFeature))->PrecomputeFeatures(flip);
  IplImage *mask = pose ? pose->GetSegmentationMask() : NULL;

  IplImage ***responses = (IplImage***)malloc((NumScales())*(sizeof(IplImage**)+sizeof(IplImage*)*fo->NumOrientations()));
  memset(responses, 0, (NumScales())*(sizeof(IplImage**)+sizeof(IplImage*)*fo->NumOrientations()));
  for(int i = 0; i < NumScales(); i++)
    responses[i] = ((IplImage**)(responses+NumScales()))+fo->NumOrientations()*i;



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
    int w, h;
    int so = fo->ScaleOffset();
    int i = l_scales[m], j = l_rotations[m], x = l_x[m], y = l_y[m], d;
    float fscale = fo->Scale(i);
    if(mask) {
      int factor = 1;//my_max(1,fo->SpatialGranularity() / cw);
      w = (int)(mask->width*factor/fo->SpatialGranularity()/fscale);
      h = (int)(mask->height*factor/fo->SpatialGranularity()/fscale);
    } else {
      w = ww;
      h = hh;
    }
    int xx, yy, yyy;
    char *ptr2, *ptr3, *iPtr2, *niPtr2;
    float *ptr, *dstPtr;
    float *iPtr, *iPtr_1, *iPtr_2, *iPtr_3, *iPtr_4;
    float *niPtr, *niPtr_1, *niPtr_2, *niPtr_3, *niPtr_4;
    int rw, rh; 
    fo->GetDetectionImageSize(&rw, &rh, i, j);
    IplImage *dst = responses[i+so][j] = cvCreateImage(cvSize(rw,rh), IPL_DEPTH_32F, 1);
    int sz = dictionary->DescriptorDims();
    int fisherDims = dictionary->FisherFeatureDims();
    float *f = new float[sz*2], *fptr, v;
    float *f2 = f+sz, *fisher=Create1DArray<float>(fisherDims,true);
    IplImage *img_o = images[j][i][x][y]; 
    assert(x+img_o->width*dx <= rw && y+img_o->height*dx <= rh);

    // img is zero padded by size (dictionary->w X dictionary->h) so that each codeword can be extracted without bounds checking
    // scoreImg and normImg are zero padded by size (w X h) so that each sliding window can be extracted without bounds checking
    IplImage *img = cvCreateImage(cvSize(img_o->width+dictionary->w,img_o->height+dictionary->h), img_o->depth, img_o->nChannels);
    IplImage *scoreImg = cvCreateImage(cvSize(img_o->width+w,img_o->height+h), img_o->depth, img_o->nChannels);
    IplImage *normImg = cvCreateImage(cvSize(img_o->width+w,img_o->height+h), img_o->depth, img_o->nChannels);
    cvCopyMakeBorderMultiChannel(img_o, img, cvPoint(dictionary->w/2,dictionary->h/2), IPL_BORDER_CONSTANT, cvScalarAll(0));
    int fsz = dictionary->w*img->nChannels;
    int fsz2 = fsz*sizeof(float);
    cvZero(scoreImg);
    cvZero(normImg);

    // Compute the integral image for the fisher vector features dot product'd with the weight vector and the
    // integral image of the squared L2 norm of the fisher vector features
    for(yy = 0, ptr2=img->imageData; yy < img_o->height; yy++, ptr2 += img->widthStep) {
      ptr=(float*)ptr2; 
      float *scorePtr=(float*)(scoreImg->imageData+(yy+h/2)*scoreImg->widthStep)+w/2;
      float *normPtr=(float*)(normImg->imageData+(yy+h/2)*normImg->widthStep)+w/2;
      for(xx = 0; xx < img_o->width; xx++) {
	// Extract a wXhXc descriptor around xx,yy
	for(yyy = 0, fptr=f, ptr3=(char*)ptr; yyy < dictionary->h; yyy++, fptr+=fsz, ptr3 += img->widthStep)
	  memcpy(fptr, ptr3, fsz2);
	dictionary->ComputeFisherVector(&f, 1, fisher, false, NULL, &f2);
	float score = 0, norm = 0;
	for(d = 0; d < fisherDims; d++) {
	  v = my_abs(fisher[d]);
	  score += weights[d]*sqrt(v)*(fisher[d]<0 ? -1 : 1);  // assume fisher vector will be power normalized with alpha=.5
	  norm += v;            // keep track of the squared sum of the power normalized fisher vector
	}
	scorePtr[xx] = score;
        normPtr[xx] = norm;
      }
    }

    
    if(!mask) {
      // Compute fisher vector in w X h region
      IplImage *integralScoreImg = cvCreateImage(cvSize(scoreImg->width+1,scoreImg->height+1), IPL_DEPTH_32F, 1);
      IplImage *integralNormImg = cvCreateImage(cvSize(scoreImg->width+1,scoreImg->height+1), IPL_DEPTH_32F, 1);
      cvIntegral(scoreImg, integralScoreImg);
      cvIntegral(normImg, integralNormImg);
      
      // Use the integral image to compute the response in a wXh window around each pixel
      float score, norm;
      for(yy = 0, ptr2=dst->imageData+y*dst->widthStep, iPtr2=integralScoreImg->imageData, niPtr2=integralNormImg->imageData; 
	  yy < img_o->height;  yy++, ptr2 += dst->widthStep*dx, iPtr2 += integralScoreImg->widthStep, niPtr2 += integralNormImg->widthStep) {
	iPtr_1=(float*)iPtr2; iPtr_2=iPtr_1+w; iPtr_3=(float*)(iPtr2+h*integralScoreImg->widthStep); iPtr_4=iPtr_3+w;
	niPtr_1=(float*)niPtr2; niPtr_2=niPtr_1+w; niPtr_3=(float*)(niPtr2+h*integralNormImg->widthStep); niPtr_4=niPtr_3+w;
	for(xx = 0, dstPtr=((float*)ptr2)+x; xx < dst->width; xx++, dstPtr += dx) {
	  score = (iPtr_1[xx]+iPtr_4[xx] - (iPtr_2[xx]+iPtr_3[xx]));  // the dot product of weights and the fisher vector (before L2 normalization)
	  norm = sqrt(niPtr_1[xx]+niPtr_4[xx] - (niPtr_2[xx]+niPtr_3[xx]));  // denominator for L2 normalization
	  *dstPtr = score / norm;
	}
      }
      cvReleaseImage(&integralScoreImg);
      cvReleaseImage(&integralNormImg);
    } else {
      // Compute fisher vector in w X h region, with each codeword weighted according to "mask"
      IplImage *mask2 = cvCreateImage(cvSize(w, h), mask->depth, 1);
      IplImage *mask3 = cvCreateImage(cvSize(w, h), IPL_DEPTH_32F, 1);
      IplImage *tmpScore = cvCreateImage(cvSize(scoreImg->width, scoreImg->height), IPL_DEPTH_32F, 1);
      IplImage *tmpNorm = cvCreateImage(cvSize(scoreImg->width, scoreImg->height), IPL_DEPTH_32F, 1);
      cvResize(mask, mask2);
      cvConvertScale(mask2, mask3, 1/255.0);

      cvFilter2DMultiChannel(scoreImg, tmpScore, mask3);
      cvFilter2DMultiChannel(normImg, tmpNorm, mask3);
      cvPow(tmpNorm, tmpNorm, -.5);
      cvMul(tmpScore, tmpNorm, tmpScore);
      DrawImageIntoImage(tmpScore, cvRect(w/2,h/2,dst->width,dst->height), dst, 0, 0);

      cvReleaseImage(&mask2);
      cvReleaseImage(&mask3);
    }

    cvReleaseImage(&img);
    cvReleaseImage(&scoreImg);
    cvReleaseImage(&normImg);
    delete [] f;
    FreeArray(fisher, true);
  }

  return responses;
}

void FisherFeature::Clear(bool full) {
  SlidingWindowFeature::Clear(full);
}

int FisherFeature::GetFeaturesAtLocation(float *f, int w, int h, int feat_scale, PartLocation *loc, bool flip) {
  int ixx, iyy, scale, rot, pose;
  int fisherDims = dictionary->FisherFeatureDims();
  SlidingWindowFeature *fe = fo->Feature(dictionary->baseFeature);
  params.maxScales = fo->NumScales();

  if(!visW)
    visW = new float[dictionary->numWords];

  loc->GetDetectionLocation(&ixx, &iyy, &scale, &rot, &pose);

  if(scale < fo->ScaleOffset() || scale >= NumScales() || !loc->IsVisible()) {
    // Invalid scale
    for(int i = 0; i < fisherDims; i++)
      f[i] = 0;
    return fisherDims;//dictionary->numWords;
  }

  // Extract features only from a pose segmentation mask, if it is available
  float *weights = NULL;
  IplImage *mask = loc->GetPartID() >= 0 ? loc->GetClasses()->GetPart(loc->GetPartID())->GetPose(pose)->GetSegmentationMask() : NULL;
  int cw = fe->Params()->cellWidth;
  assert(fo->SpatialGranularity() % cw == 0);
  int factor = 1;//my_max(1,fo->SpatialGranularity() / cw);
  int nx = dictionary->w;
  float fscale = pow(fe->Params()->subsamplePower, feat_scale);

  if(mask) {
    w = (int)(mask->width*factor/fo->SpatialGranularity()/fscale);
    h = (int)(mask->height*factor/fo->SpatialGranularity()/fscale);
    weights = new float[w*h];
    IplImage *mask2 = cvCreateImage(cvSize(w, h), mask->depth, 1);
    IplImage *mask3 = cvCreateImage(cvSize(w, h), IPL_DEPTH_32F, 1);
    cvResize(mask, mask2);
    cvConvertScale(mask2, mask3, 1/255.0);
    //if(flip) cvFlip(mask3, NULL, 1);
    for(int i = 0; i < h; i++) 
      memcpy(weights+i*w, mask3->imageData+mask3->widthStep*i, w*sizeof(float));
    cvReleaseImage(&mask2);
    cvReleaseImage(&mask3);
  } else {
    weights = new float[w*h];
    for(int i = 0; i < w*h; i++) 
      weights[i] = 1;
  }


  // Extract descriptors at patches on a dense grid
  int numPts = 0;
  float **pts = Create2DArray<float>(w*h,dictionary->DescriptorDims(), true);
  PartLocation loc2(*loc);
 
  for(int y = 0; y < h; y++) {
    for(int x = 0; x < w; x++) {
      int xx = (ixx*factor-w/2+x)/factor, yy = (iyy*factor-h/2+y)/factor;
      int ox = (ixx*factor-w/2+x)%factor, oy = (iyy*factor-h/2+y)%factor;
      loc2.SetDetectionLocation(xx, yy, scale, rot, pose, LATENT, LATENT);
      ((TemplateFeature*)fe)->GetFeaturesAtLocation(pts[numPts], dictionary->w, dictionary->h, feat_scale, &loc2, flip, ox, oy); 
      numPts++;
    }
  }

  // Compute Fisher encoding of the descriptors
  //assert(my_abs(pts[0][0] < 10000));
  
  float *f_aligned = Create1DArray<float>(fisherDims,true);
  dictionary->ComputeFisherVector(pts, numPts, f_aligned, true, weights, NULL, visW);
  memcpy(f, f_aligned, (fisherDims)*sizeof(float));
  FreeArray(pts, true);
  FreeArray(f_aligned, true);
  

  if(weights) delete [] weights;
  return fisherDims;
}


typedef struct {
  int ind;
  float val, m1, m2;
} FeatureValInd2;

int FeatureValInd2Compare(const void *a1, const void *a2) {
  FeatureValInd2 *v1 = (FeatureValInd2*)a1, *v2 = (FeatureValInd2*)a2; 
  float d = (((FeatureValInd2*)v2)->val-(((FeatureValInd2*)v1)->val));
  return d < 0 ? -1 : (d > 0 ? 1 : (v1->ind-v2->ind)); 
} 
IplImage *FisherFeature::Visualize(float *f, int w, int h, float mi, float ma) {
  // Sort the histogram bins by the weight of each word
  int fisherDims = dictionary->FisherFeatureDims();
  FeatureValInd2 *ff = (FeatureValInd2*)malloc(sizeof(FeatureValInd2)*dictionary->numWords);
  int i, j;
  int ptSz = dictionary->DescriptorDims();
  float mi_w = INFINITY, ma_w = -INFINITY, mi_v1 = INFINITY, ma_v1 = -INFINITY, mi_v2 = INFINITY, ma_v2 = -INFINITY;
  int pcaDims = dictionary->pcaDims;

  for(int i = 0; i < fisherDims; i++)
    assert(f[i] >= -1 && f[i] <= 1);

  float maxCount = 0;
  float *buff = new float[ptSz*3];
  for(i = 0; i < dictionary->numWords; i++) {
    float m11 = INFINITY, m12 = -INFINITY, m21 = INFINITY, m22 = -INFINITY, mu1 = INFINITY, mu2 = -INFINITY;
    float *f_mu = &f[i*pcaDims], *f_sig = &f[(dictionary->numWords+i)*pcaDims], *mu = dictionary->mus[i]; 
    float *f_mu_out = f_mu, *f_sig_out = f_sig, *mu_out = mu;
    if(pcaDims < ptSz) {
      f_mu_out = buff; f_sig_out = buff+ptSz;  mu_out = buff+ptSz*2;  
      dictionary->ReprojectPCA(&f_mu, 1, &f_mu_out); 
      dictionary->ReprojectPCA(&f_sig, 1, &f_sig_out);
      dictionary->ReprojectPCA(&mu, 1, &mu_out);
    }
    for(j = 0; j < ptSz; j++) {
      m21 = my_min(m21, f_sig_out[j]);
      m22 = my_max(m22, f_sig_out[j]);
      m11 = my_min(m11, f_mu_out[j]);
      m12 = my_max(m12, f_mu_out[j]);
      mu1 = my_min(mu1, mu_out[j]);
      mu2 = my_max(mu2, mu_out[j]);
    }
    mi_w = my_min(mu1, mi_w);
    ma_w = my_max(mu2, ma_w);
    mi_v1 = my_min(m11, mi_v1);
    ma_v1 = my_max(m12, ma_v1);
    mi_v2 = my_min(m21, mi_v2);
    ma_v2 = my_max(m22, ma_v2);
    ff[i].val = visW[i];
    maxCount = my_max(maxCount, ff[i].val);
    ff[i].m1 = m21;
    ff[i].m2 = m22;
    ff[i].ind = i;
  }
  qsort(ff, dictionary->numWords, sizeof(FeatureValInd2), FeatureValInd2Compare);

  int num = dictionary->numWords;

  int sz = dictionary->w*dictionary->h*dictionary->nChannels;
  int hspace = 4;
  int c = fo->CellWidth();
#if VISUALIZE_SOFT_ASSIGNMENTS
  int bar_height = 150;
#else
  int bar_height = 0;
#endif
  IplImage *img = cvCreateImage(cvSize((dictionary->w*c+hspace)*num+hspace, bar_height+3*hspace+(dictionary->h*c+hspace)*3), IPL_DEPTH_8U, 3);
  cvSet(img, cvScalar(255,255,255));
  SlidingWindowFeature *base = fo->Feature(dictionary->baseFeature);
  int x = hspace, y = hspace+bar_height;
  for(i = 0; i < dictionary->numWords; i++) {
    float *f_mu = &f[(ff[i].ind)*pcaDims], *f_sig = &f[(dictionary->numWords+ff[i].ind)*pcaDims], *mu = dictionary->mus[ff[i].ind]; 
    float *f_mu_out = f_mu, *f_sig_out = f_sig, *mu_out = mu;
    if(pcaDims < ptSz) {
      f_mu_out = buff; f_sig_out = buff+ptSz;  mu_out = buff+ptSz*2;  
      dictionary->ReprojectPCA(&f_mu, 1, &f_mu_out); 
      dictionary->ReprojectPCA(&f_sig, 1, &f_sig_out);
      dictionary->ReprojectPCA(&mu, 1, &mu_out);
    }

    if(ff[i].val) {
      IplImage *key = base->Visualize(mu_out, dictionary->w, dictionary->h, mi_w, ma_w);
      if(key->nChannels == 1) {
	IplImage *img2 = cvCreateImage(cvSize(key->width,key->height), IPL_DEPTH_8U, 3);
	cvConvertImage(key, img2);
	cvReleaseImage(&key);
	key = img2;
      }
	  
      DrawImageIntoImage(key, cvRect(0,0,key->width,key->height), img, x, y+3*hspace+2*key->height);
      cvReleaseImage(&key);
      
      key = base->Visualize(f_mu_out, dictionary->w, dictionary->h, my_min(mi_v1,mi_v2), my_max(ma_v1,ma_v2));
      if(key->nChannels == 1) {
	IplImage *img2 = cvCreateImage(cvSize(key->width,key->height), IPL_DEPTH_8U, 3);
	cvConvertImage(key, img2);
	cvReleaseImage(&key);
	key = img2;
      }
      DrawImageIntoImage(key, cvRect(0,0,key->width,key->height), img, x, y+2*hspace+key->height);
      cvReleaseImage(&key);

      key = base->Visualize(f_sig_out, dictionary->w, dictionary->h, my_min(mi_v1,mi_v2), my_max(ma_v1,ma_v2));
      if(key->nChannels == 1) {
	IplImage *img2 = cvCreateImage(cvSize(key->width,key->height), IPL_DEPTH_8U, 3);
	cvConvertImage(key, img2);
	cvReleaseImage(&key);
	key = img2;
      }
      DrawImageIntoImage(key, cvRect(0,0,key->width,key->height), img, x, y+hspace);

#if VISUALIZE_SOFT_ASSIGNMENTS
      cvRectangle(img, cvPoint(x,y-bar_height*(ff[i].val)/(maxCount)), cvPoint(x+key->width,y), CV_RGB(0,0,255), -1);
#endif

      x += hspace + key->width;
      cvReleaseImage(&key);
    }
  }

  free(ff);

  return img;
}

// Visualize an assignment to part locations as a HOG image
IplImage *FisherFeature::Visualize(Classes *classes, PartLocation *locs, bool visualizeWeights, AttributeInstance *attribute) {
  if(attribute) { // Visualize a particular attribute detector
    // Extract histogram for attribute detector weights
    Attribute *am = attribute->Model();
    float *tmp = new float[dictionary->FisherFeatureDims()];;
    if(visualizeWeights) {
      if(!am->GetWeights(tmp, name)) {
	delete [] tmp;
	return NULL;
      }
    } else {
      GetFeaturesAtLocation(tmp, am->Width(), am->Height(), 0, &locs[am->Part()->Id()], attribute->IsFlipped());
    }

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
	float *tmp = new float[dictionary->FisherFeatureDims()];;
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



#if HAVE_SSE

double GMMComputeSoftAssignments_sse(float **pts_i, float **mu, float **sigma, float *mixtureWeights, float **softAssignments_i, 
				     int numPts, int K, int ptSz, float ***exampleDevs, float **log_likelihoods_i=NULL) {
  int i, j, k;
  float *invSigma = Create1DArray<float>(ptSz, true);
  float *ll = Create1DArray<float>(K, true);
  float log_likelihood = 0, d;
  if(!log_likelihoods_i) 
    log_likelihoods_i = softAssignments_i;

  for(k = 0; k < K; k++) {
    // Compute the normalization terms of this gaussian
    float gaussNormTerm = LN(mixtureWeights[k])-K/2*LN(2*M_PI);
    for(j = 0; j < ptSz; j++) {
      gaussNormTerm -= sigma[k][j] ? .5*LN(sigma[k][j]) : 0;
      invSigma[j] = sigma[k][j] ? 1.0 / sigma[k][j] : .00000000001;
    }

    // Compute log probabilities for each point-gaussian pair
    for(i = 0; i < numPts; i++) 
      log_likelihoods_i[k][i] = 0;

    for(j = 0; j < ptSz; j++) {
      // For 4 points at a time, subtract the mean and divide by the standard deviation
      assert(IS_ALIGNED(pts_i[j]) && IS_ALIGNED(log_likelihoods_i[k]) && IS_ALIGNED(exampleDevs[k][j]));
      __m128 *pts_m = (__m128*)pts_i[j], *log_likelihoods_m = (__m128*)log_likelihoods_i[k];
      __m128 mu_m = _mm_load_ps1(&mu[k][j]), invSigma_m = _mm_load_ps1(&invSigma[j]);
      __m128 *exampleDevs_m = (__m128*)exampleDevs[k][j];
      for(i = 0; i < numPts/4; i++) {
	exampleDevs_m[i] = _mm_mul_ps(_mm_sub_ps(pts_m[i], mu_m),invSigma_m);
	log_likelihoods_m[i]  = _mm_add_ps(log_likelihoods_m[i], _mm_mul_ps(exampleDevs_m[i],exampleDevs_m[i]));
      }

      // If the number of points isn't divisible by 4, process the remaining points
      for(i = (numPts/4)*4; i < numPts; i++) {
	exampleDevs[k][j][i] = (pts_i[j][i]-mu[k][j])*invSigma[j];
	log_likelihoods_i[k][i] += SQR(exampleDevs[k][j][i]);
      } 
    }
    for(i = 0; i < numPts; i++)
      log_likelihoods_i[k][i] = gaussNormTerm-.5*log_likelihoods_i[k][i];
  } 
  

  // Compute normalized probabilities for each point
  for(i = 0; i < numPts; i++) {
    float ma = -INFINITY;
    for(k = 0; k < K; k++) 
      if(log_likelihoods_i[k][i] > ma)
	ma = log_likelihoods_i[k][i];
    float Z = 0;
    for(k = 0; k < K; k++) {
      ll[k] = log_likelihoods_i[k][i];
      softAssignments_i[k][i] = exp(log_likelihoods_i[k][i]-ma);
      Z += softAssignments_i[k][i];
    }
    float expected_log_likelihood = 0;
    for(k = 0; k < K; k++) {
      softAssignments_i[k][i] /= Z;
      if(softAssignments_i[k][i] > 0) 
	expected_log_likelihood += softAssignments_i[k][i]*ll[k];
      assert(!isnan(expected_log_likelihood));
    }
    log_likelihood += expected_log_likelihood;
    assert(!isnan(log_likelihood));
  }

  FreeArray(invSigma, true);
  FreeArray(ll, true);
  return -log_likelihood;
}




void ComputeFisherVector_sse(float **pts, float **mu, float **sigma, float *mixtureWeights, float *fisherVector, 
			     int numPts, int K, int ptSz, float *weights, float *wordCounts) {
  int i, j, k, ind = 0;
  ALGNW float ftmp[4] ALGNL = { 0.0f, 0.0f, 0.0f, 0.0f };
  float ***exampleDevs = Create3DArray<float>(K,ptSz,numPts,true);
  float **softAssignments_i = Create2DArray<float>(K,numPts,true);
  float **pts_i = Create2DArray<float>(ptSz,numPts,true);
  for(j = 0; j < ptSz; j++)
    for(i = 0; i < numPts; i++) 
      pts_i[j][i] = pts[i][j];
  GMMComputeSoftAssignments_sse(pts_i, mu, sigma, mixtureWeights, softAssignments_i, numPts, K, ptSz, exampleDevs);
  float sumWeight = numPts;
  if(weights) {
    sumWeight = 0;
    for(i = 0; i < numPts; i++)
      sumWeight += weights[i];
    for(k = 0; k < K; k++) 
      for(i = 0; i < numPts; i++)
        softAssignments_i[k][i] *= weights[i];
  }

  if(wordCounts) {
    for(k = 0; k < K; k++) {
      wordCounts[k] = 0;
      for(i = 0; i < numPts; i++) 
        wordCounts[k] += softAssignments_i[k][i];
    }
  }

  // Fisher vector elements with respect to mu
  for(k = 0; k < K; k++) {
    for(j = 0; j < ptSz; j++) {
      fisherVector[ind] = 0;

      // Compute fisher vector entries with respect to mu_kj for 4 pts at a time 
      assert(IS_ALIGNED(exampleDevs[k][j]) && IS_ALIGNED(softAssignments_i[k]));
      __m128 *exampleDevs_m = (__m128*)exampleDevs[k][j], *softAssignments_m = (__m128*)softAssignments_i[k];
      __m128 fisherVector_m = _mm_load_ps1(&fisherVector[ind]);
      for(i = 0; i < numPts/4; i++) 
        fisherVector_m = _mm_add_ps(fisherVector_m, _mm_mul_ps(softAssignments_m[i],exampleDevs_m[i]));
      _mm_store_ps(ftmp, fisherVector_m);   
      fisherVector[ind] = ftmp[0] + ftmp[1] + ftmp[2] + ftmp[3];
      
      // If the number of points isn't divisible by 4, compute the remaining entries
      for(i = (numPts/4)*4; i < numPts; i++) 
        fisherVector[ind] += softAssignments_i[k][i]*exampleDevs[k][j][i];

      fisherVector[ind++] /= sumWeight*sqrt(mixtureWeights[k]);
    }
  }

  // Fisher vector elements with respect to sigma
  for(k = 0; k < K; k++) {
    for(j = 0; j < ptSz; j++) {
      fisherVector[ind] = 0;

      // Compute fisher vector entries with respect to sigma_kj for 4 pts at a time 
      float one = 1;
      __m128 *exampleDevs_m = (__m128*)exampleDevs[k][j], *softAssignments_m = (__m128*)softAssignments_i[k];
      __m128 fisherVector_m = _mm_load_ps1(&fisherVector[ind]), one_m = _mm_load_ps1(&one), dev_m;
      for(i = 0; i < numPts/4; i++)  {
        dev_m = _mm_sub_ps(_mm_mul_ps(exampleDevs_m[i],exampleDevs_m[i]),one_m);
        fisherVector_m = _mm_add_ps(fisherVector_m, _mm_mul_ps(softAssignments_m[i],dev_m));
      }
      _mm_store_ps(ftmp, fisherVector_m);   
      fisherVector[ind] = ftmp[0] + ftmp[1] + ftmp[2] + ftmp[3];

      // If the number of points isn't divisible by 4, compute the remaining entries
      for(i = (numPts/4)*4; i < numPts; i++) 
        fisherVector[ind] += softAssignments_i[k][i]*(SQR(exampleDevs[k][j][i])-1);

      fisherVector[ind++] /= sumWeight*sqrt(2*mixtureWeights[k]);
    }
  }

  // Cleanup
  FreeArray(exampleDevs, true);
  FreeArray(softAssignments_i, true);
  FreeArray(pts_i, true);
}

void NormalizeVector_sse(float *fisherVector, int dim) {
  int i;

  // For 4 entries at a time, compute the sum of the absolute value of all entries
  assert(IS_ALIGNED(fisherVector));
  float sum = 0;
  __m128 sum_m = _mm_load_ps1(&sum);
  __m128 *fisherVector_m = (__m128*)fisherVector;
  ALGNW float ftmp[4] ALGNL = { 0.0f, 0.0f, 0.0f, 0.0f };
  static const __m128 SIGNMASK = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));  // sign bit of float32
  for(i = 0; i < dim/4; i++) 
    sum_m = _mm_add_ps(sum_m, _mm_andnot_ps(SIGNMASK, fisherVector_m[i]));
  _mm_store_ps(ftmp, sum_m);   
  sum = ftmp[0] + ftmp[1] + ftmp[2] + ftmp[3];

  // If the number of points isn't divisible by 4, compute the remaining entries
  for(i = (dim/4)*4; i < dim; i++) 
    sum += my_abs(fisherVector[i]);


  // For 4 entries at a time, power normalize each entry (take the sqrt while preserving sign) and then L2 normalize 
  // (divide by the norm of the power normalized vector)
  float iSum = 1.0/sqrt(sum);
  __m128 iSum_m = _mm_load_ps1(&iSum);
  for(i = 0; i < dim/4; i++) 
    fisherVector_m[i] =_mm_or_ps(_mm_and_ps(SIGNMASK, fisherVector_m[i]),
	                         _mm_mul_ps(_mm_sqrt_ps(_mm_andnot_ps(SIGNMASK, fisherVector_m[i])), iSum_m));
  
  // If the number of points isn't divisible by 4, compute the remaining entries
  for(i = (dim/4)*4; i < dim; i++) 
    fisherVector[i] = fisherVector[i] < 0 ? -sqrt(-fisherVector[i])*iSum : sqrt(fisherVector[i])*iSum;
}



float **ProjectPCA_sse(float **pts, int numPts, float **eigvects_i, float *pca_mu, float **ptsProj, int ptSz, int pcaDims) {
  int i, j, k;
  float d;
  if(!ptsProj) 
    ptsProj = Create2DArray<float>(numPts, pcaDims, true);
 
  for(i = 0; i < numPts; i++)
    for(k = 0; k < pcaDims; k++) 
      ptsProj[i][k] = 0;

  for(i = 0; i < numPts; i++) {
    for(j = 0; j < ptSz; j++) {
      d = pts[i][j]-pca_mu[j];

      // For 4 eigenvectors at a time, update the dot product of the current point vector and each eigenvector 
      // (update dot product with respect to feature element j)
      assert(IS_ALIGNED(eigvects_i[j]) && IS_ALIGNED(ptsProj[i]));
      __m128 *eigvects_m = (__m128*)eigvects_i[j], *ptsProj_m = (__m128*)ptsProj[i], d_m = _mm_load_ps1(&d);
      for(k = 0; k < pcaDims/4; k++) 
        ptsProj_m[k] = _mm_add_ps(ptsProj_m[k], _mm_mul_ps(d_m,eigvects_m[k]));
      
      // If the number of eigenvectors divisible by 4, compute the remaining entries
      for(k = (pcaDims/4)*4; k < pcaDims; k++) 
	ptsProj[i][k] += d*eigvects_i[j][k];
    }
  }

  return ptsProj;
}

#endif
