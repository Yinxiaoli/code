/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#ifndef __ENTROPY_H
#define __ENTROPY_H

#include "util.h"

/**
 * @file entropy.h
 * @brief Helper routines for computing entropy and information gain
 */

/**
 * @class EntropyBuffers
 * 
 * @brief Simple helper class used for computing entropy and information gain
 *
 */
class EntropyBuffers {
 public:
  int numClasses; /**< The total number of classes */
  int maxAnswers; /**< The maximum number of possible answers of any question processed so far (defines the size of pre-allocated memory buffers) */
  int numAnswers; /**< The number of possible answers to the question we are currently processing */

  double *answerProbs; /**< An array of size numAnswers defining the estimated probability of each possible user response */
  double **answerUnnormalizedClassProbs; /**< A numAnswersXnumClasses array of induced unnormalized class probabilities for every possible answer */
  double **answerClassProbs; /**< A numAnswersXnumClasses array of induced unnormalized class probabilities for every possible answer */

  /**
   * @brief Constructor
   * @param numClasses The number of different classes
   */
  EntropyBuffers(int numClasses);

  /**
   * @brief Destructor
   */
  ~EntropyBuffers();

  /**
   * @brief Initialize memory buffers for a question with numAnswers possible answers
   * @param numAnswers The number of possible answers to the question we are currently processing
   */
  void InitializeAndClear(int numAnswers);

  /**
   * @brief Normalizes probabilities (Computes answerClassProbs and answerProbs from answerUnnormalizedClassProbs)
   */
  void NormalizeAnswerProbabilities(bool isLog=false);

  /**
   * @brief Computes expected entropy, as a weighted average of class entropies for each possible question answer
   * @return The expected entropy
   */
  double ComputeExpectedEntropy();
};

/**
 * @brief Used for sorting a list of class probabilities
 */ 
typedef struct {
  double prob; /**< The probability of a particular class */
  int classID; /**< The class id (index in Classes->classes) */
} ClassProb;

/**
 * @brief Normalize an array of probabilities
 * @param unnormalizedProbs An array of numClasses unnormalized probabilities
 * @param classProbs An array of numClasses probabilities, which is set by this function
 * @param numClasses The number of classes
 * @return The sum of all entries in unnormalizedProbs
 */
double NormalizeProbabilities(double *unnormalizedProbs, double *classProbs, int numClasses, bool isLog=false);

/**
 * @brief Computes entropy
 * @param classProbs An array of numClasses normalized probabilities
 * @param numClasses The size of the array
 * @return The computed entropy
 */
double ComputeEntropy(double *classProbs, int numClasses);

/**
 * @brief Compute a ranked list of classes, from highest to lowest probability
 * @param classProbs An array of numClasses probabilities
 * @param numClasses The size of the array
 * @return A dynamically allocated sorted list of classes (must be deallocated using free())
 */
ClassProb *BestClasses(double *classProbs, int numClasses);

#endif
