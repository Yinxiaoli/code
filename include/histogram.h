/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#ifndef __HISTOGRAM_H
#define __HISTOGRAM_H

#include "feature.h"
#include "util.h"

/**
 * @file histogram.h
 * @brief Structures and methods for extraction of histogrammed features (e.g. bag of words SIFT or color histograms)
 * that can be evaluated in a sliding window fashion
 */




/**
 * @class FeatureDictionary
 *
 * @brief A dictionary of codewords that can be used for vector quantization for bag of words or histogram features
 */
class FeatureDictionary {
protected:
  char *baseFeature;  /**< The name of the base feature that makes up the descriptor for this codeword */
  int w;  /**< The width of the descriptor (e.g. for SIFT it's 4 HOG bins) */
  int h;  /**< The height of the descriptor (e.g. for SIFT it's 4 HOG bins) */
  int nChannels;  /**< The number of channels in the feature descriptor, should be the same as baseFeature->Params().numBins */

  int numWords;  /**< The number of codewords in the codebook */
  float *words;  /**< A wXhXnChannelsXnumWords vector of codewords cluster centers */

  char *fileName;  /**< The name of the file where the codebook was loaded from */

  int tree_depth; /**< If the clustering method is hierarchical k-means, this is the depth of the tree */
  float **decision_planes;   /**< If the clustering method is hierarchical k-means, this is used to send a novel point down the tree */

public:

  /**
   * @brief Constructor
   * @param featName  The name of the base feature that makes up the descriptor for this codeword
   * @param w The width of the descriptor (e.g. for SIFT it's 4 HOG bins)
   * @param h The height of the descriptor (e.g. for SIFT it's 4 HOG bins)
   * @param nChannels The number of channels in the feature descriptor, should be the same as baseFeature->Params().numBins
   * @param numWords The number of codewords in the codebook
   * @param tree_depth If non-zero, learn a dictionary using hierarchical kmeans with the specified tree depth
   */
  FeatureDictionary(const char *featName=NULL, int w=0, int h=0, int nChannels=0, int numWords=0, int tree_depth=0);

  /**
   * @brief Destructor
   */
  virtual ~FeatureDictionary();

  virtual bool LoadData(FILE *fin);
  virtual bool SaveData(FILE *fout);
  virtual void LearnDictionaryFromPoints(float **pts, int numPts);

  /**
   * @brief Load a dictionary from file
   * @param fname The file name where to load the codebook from
   * @return true on success
   */
  bool Load(const char *fname);

  /**
   * @brief Save a dictionary to file
   * @param fname The file name where to save the codebook to
   * @return true on success
   */
  bool Save(const char *fname);

  /**
   * @brief Learn a new codebook by clustering interest points
   * @param d A dataset of images
   * @param maxImages The maximum number of images used to construct the codebook
   * @param ptsPerImage The number of interest points to extract per image
   */
  void LearnDictionary(Dataset *d, int maxImages, int ptsPerImage, int resize_image_width=0);

  /**
   * @brief Get the file name where this dictionary is stored on disk
   * @return the file name
   */
  const char *FileName() { return fileName; }

  const char *BaseFeatureName() { return baseFeature; }
  int NumWords() { return numWords; }

  friend class HistogramFeature;
};

/**
 * @class HistogramFeature
 *
 * @brief A sliding window detector that applies a set of weights to a bag of words or color histogram extracted from a window of arbitrary shape
 */
class HistogramFeature : public SlidingWindowFeature {
  FeatureDictionary *dictionary;

public:
  /**
   * @brief Constructor
   * @param fo A pointer to the FeatureOptions object that contains the image and all feature definitions 
   * @param d A dictionary defining a set of codewords for computing this histogram
   */
  HistogramFeature(FeatureOptions *fo, FeatureDictionary *d);
  IplImage *****PrecomputeFeatures(bool flip);
  IplImage ***SlidingWindowDetect(float *weights, int w, int h, bool flip, ObjectPose *pose);
  int GetFeaturesAtLocation(float *f, int w, int h, int feat_scale, PartLocation *loc, bool flip);
  int NumFeatures(int w, int h) { return dictionary->numWords; }
  float MaxFeatureSumSqr(int w, int h) { return 1; }  


  /**
   * @brief Create a visualization image of the feature space
   * @param f An array of features
   * @param w The width of the sliding window, at which f was extracted
   * @param h The height of the sliding window, at which f was extracted
   * @param mi If mi or ma are non-zero, the features are visualized with respect to a minimum and maximum feature response of mi and ma
   * @param ma If mi or ma are non-zero, the features are visualized with respect to a minimum and maximum feature response of mi and ma
   * @return An allocated image of the visualization image of the feature space
   */
  IplImage *Visualize(float *f, int w, int h, float mi=0, float ma=0);
  IplImage *Visualize(Classes *classes, PartLocation *locs, bool visualizeWeights=false, AttributeInstance *attr=NULL);
  
  void Clear(bool full=true);

  const char *Description(char *str) { sprintf(str, "Vector quantized %s features (%d words), histogrammed in a sliding window", dictionary->baseFeature, dictionary->numWords); return str; }
};


#endif

