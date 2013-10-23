#ifndef __FISHER_H
#define __FISHER_H

/**
 * @file fisher.h
 * @brief Implementation of Fisher vector features (Perronnin et al. ECCV 2010)
 */

#include "gmm.h"
#include "feature.h"
#include "histogram.h"

#define FISHER_PCA_DIMS 128

class FisherFeature;

class FisherFeatureDictionary : public FeatureDictionary {
  float **mus;  /**< A numWordsXpcaDims vector of standard deviations for each cluster center */
  float **sigmas;  /**< A numWordsXpcaDims vector of standard deviations for each cluster center */
  float *mixtureWeights;  /**< A numWords vector of weights for each mixture component */
  int pcaDims;  /**< PCA reduced dimensionality (which is applied before the fisher encoding) */
  float **eigvects, *eigvals, *pca_mu,  **eigvects_t;

 public:
  FisherFeatureDictionary(const char *featName=NULL, int w=0, int h=0, int nChannels=0, int numWords=0, int pcaDims=FISHER_PCA_DIMS);
  ~FisherFeatureDictionary();

  bool LoadData(FILE *fin);
  bool SaveData(FILE *fout);

  void TrainPCA(float **pts, int numPts);
  float **ProjectPCA(float **pts, int numPts, float **ptsProj=NULL);
  float **ReprojectPCA(float **ptsProj, int numPts, float **pts=NULL);
  void LearnDictionaryFromPoints(float **pts, int numPts);
  void ComputeFisherVector(float **pts, int numPts, float *fisherVector, bool normalize=true, float *weights=NULL, float **ptsProjSpace=NULL, float *wordCounts=NULL);
  int DescriptorDims() { return w*h*nChannels; }
  int FisherFeatureDims() { return 2*numWords*pcaDims; }

  friend class FisherFeature;
};


/**
 * @class FisherFeature
 *
 * @brief A sliding window detector that applies a set of weights to a fisher encoding of patches extracted from a window of arbitrary size and a customizable decriptor
 */
class FisherFeature : public SlidingWindowFeature {
  FisherFeatureDictionary *dictionary;
  float *visW;

public:
  /**
   * @brief Constructor
   * @param fo A pointer to the FeatureOptions object that contains the image and all feature definitions 
   * @param d A dictionary defining a set of codewords for computing this histogram
   */
  FisherFeature(FeatureOptions *fo, FisherFeatureDictionary *d);
  IplImage *****PrecomputeFeatures(bool flip);
  IplImage ***SlidingWindowDetect(float *weights, int w, int h, bool flip, ObjectPose *pose);
  int GetFeaturesAtLocation(float *f, int w, int h, int feat_scale, PartLocation *loc, bool flip);
  int NumFeatures(int w, int h) { return dictionary->FisherFeatureDims(); }
  float MaxFeatureSumSqr(int w, int h) { return 1; }  

  void Clear(bool full=true);

  const char *Description(char *str) { sprintf(str, "Fisher encoded %s features (%d words) in a sliding window", dictionary->baseFeature, dictionary->numWords); return str; }
  IplImage *Visualize(float *f, int w, int h, float mi=0, float ma=0);
  IplImage *Visualize(Classes *classes, PartLocation *locs, bool visualizeWeights=false, AttributeInstance *attr=NULL);
};

template <typename T>
void ComputeFisherVector(T **pts, T **mu, T **sigma, T *mixtureWeights, T *fisherVector, 
			int numPts, int K, int ptSz, float *weights, float *wordCounts=NULL) {
  int i, j, k, ind = 0;
  T ***exampleDevs = Create3DArray<T>(K,ptSz,numPts, true);
  T **softAssignments = Create2DArray<T>(numPts,K, true);
  GMMComputeSoftAssignments(pts, mu, sigma, mixtureWeights, softAssignments, numPts, K, ptSz, exampleDevs);
  float sumWeight = numPts;
  if(weights) { 
    sumWeight = 0;
    for(i = 0; i < numPts; i++) {
      for(k = 0; k < K; k++) 
        softAssignments[i][k] *= weights[i];
      sumWeight += weights[i];
    }
  }

  if(wordCounts) {
    for(k = 0; k < K; k++)
      wordCounts[k] = 0;
    for(i = 0; i < numPts; i++) 
      for(k = 0; k < K; k++)
        wordCounts[k] += softAssignments[i][k];
  }

  // Fisher vector elements with respect to mu
  for(k = 0; k < K; k++) {
    for(j = 0; j < ptSz; j++) {
      fisherVector[ind] = 0;
      for(i = 0; i < numPts; i++) 
        fisherVector[ind] += softAssignments[i][k]*exampleDevs[k][j][i];
      fisherVector[ind++] /= sumWeight*sqrt(mixtureWeights[k]);
    }
  }

  // Fisher vector elements with respect to sigma
  for(k = 0; k < K; k++) {
    for(j = 0; j < ptSz; j++) {
      fisherVector[ind] = 0;
      for(i = 0; i < numPts; i++) 
        fisherVector[ind] += softAssignments[i][k]*(SQR(exampleDevs[k][j][i])-1);
      fisherVector[ind++] /= sumWeight*sqrt(2*mixtureWeights[k]);
    }
  }

  // Cleanup
  FreeArray(exampleDevs, true);
  FreeArray(softAssignments, true);
}


template <typename T>
void NormalizeVector(T *fisherVector, int dim, double powerNormalization = .5, bool l2Normalization = true) {
  int i;

  if(powerNormalization==.5 && l2Normalization) {
    // Special case for power normalization of .5 and L2 normalization
    T sum = 0;
    for(i = 0; i < dim; i++) 
      sum += my_abs(fisherVector[i]);
    T iSum = 1.0/sqrt(sum);
    for(i = 0; i < dim; i++) 
      fisherVector[i] = fisherVector[i] < 0 ? -sqrt(-fisherVector[i])*iSum : sqrt(fisherVector[i])*iSum;
  } else {
    // Normalize each element z as sign(z)*|z|^powerNormalization
    if(powerNormalization) {
      for(i = 0; i < dim; i++) 
	fisherVector[i] = (fisherVector[i] < 0 ? -1 : 1)*pow((double)my_abs(fisherVector[i]),powerNormalization);
    }

    // Apply L2 normalization
    if(l2Normalization) {
      T sum = 0;
      for(i = 0; i < dim; i++) 
	sum += SQR(fisherVector[i]);
      T iSum = 1.0/sqrt(sum);
      for(i = 0; i < dim; i++) 
	fisherVector[i] *= iSum;
    }
  }
}


#endif
