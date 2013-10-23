/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#include "entropy.h"


EntropyBuffers::EntropyBuffers(int numClasses) {
  this->numClasses = numClasses;
  maxAnswers = 0;
  answerProbs = NULL;
  answerUnnormalizedClassProbs = answerClassProbs = NULL;
}

EntropyBuffers::~EntropyBuffers() {
  if(answerProbs) free(answerProbs);
  if(answerUnnormalizedClassProbs) {
    for(int i = 0; i < maxAnswers; i++) {
      free(answerUnnormalizedClassProbs[i]);
      free(answerClassProbs[i]);
    }
    free(answerUnnormalizedClassProbs);
    free(answerClassProbs);
  }
}

// Setup memory buffers that can be used to compute expected entropy
void EntropyBuffers::InitializeAndClear(int numAnswers) {
  int i, j;

  this->numAnswers = numAnswers;
  if(numAnswers > maxAnswers) {
    // Grow the memory buffers if necessary
    answerProbs = (double*)realloc(answerProbs, sizeof(double)*numAnswers);
    answerUnnormalizedClassProbs = (double**)realloc(answerUnnormalizedClassProbs, sizeof(double*)*numAnswers);
    answerClassProbs = (double**)realloc(answerClassProbs, sizeof(double*)*numAnswers);
    for(i = maxAnswers; i < numAnswers; i++) {
      answerUnnormalizedClassProbs[i] = (double*)malloc(sizeof(double)*numClasses);
      answerClassProbs[i] = (double*)malloc(sizeof(double)*numClasses);
    }
    maxAnswers = numAnswers;
  }

  // Clear all memory buffers
  for(i = 0; i < numAnswers; i++) {
    answerProbs[i] = 0;
    for(j = 0; j < numClasses; j++) {
      answerUnnormalizedClassProbs[i][j] = answerClassProbs[i][j] = 0;
    }
  }
}



// For each possible question answer, assume we have already computed unnormalized class probabilties (weights)
// answerUnnormalizedClassProbs.  The probability of each answer is proportional to the sum over the unnormalized
// weights.  We also compute the normalized class probabilities for each answer
void EntropyBuffers::NormalizeAnswerProbabilities(bool isLog) {
  double sumWeights = 0;
  for(int i = 0; i < numAnswers; i++) {
    answerProbs[i] = NormalizeProbabilities(answerUnnormalizedClassProbs[i], answerClassProbs[i], numClasses, isLog);
    sumWeights += answerProbs[i];
  }
  if(sumWeights) {
    for(int i = 0; i < numAnswers; i++)
      answerProbs[i] /= sumWeights;
  }
}


// Compute expected entropy.  Assumes normalized answer and class probabilities have already been computed
// for each possible answer.  Expected entropy is simply a weighted average over entropies for each possible answer
double EntropyBuffers::ComputeExpectedEntropy() {
  double entropy = 0, answerEntropy;
  for(int i = 0; i < numAnswers; i++) {
    answerEntropy = ComputeEntropy(answerClassProbs[i], numClasses);
    entropy += answerEntropy*answerProbs[i];
    //fprintf(stderr, "  answer_prob=%f entropy=%f\n", answerProbs[i], answerEntropy);
  }

  return entropy;
}


// Input an array of numClasses unnormalized probabilties and output an array of numClasses 
// normalized probabilties.  Returns the sum (inverse of the normalization factor)
double NormalizeProbabilities(double *unnormalizedProbs, double *classProbs, int numClasses, bool isLog) {
  int j;

  double sum = 0;
  if(isLog) {
    double ma = -INFINITY;
	for(j = 0; j < numClasses; j++) 
      ma = my_max(unnormalizedProbs[j],ma);
    for(j = 0; j < numClasses; j++) {
      classProbs[j] = exp(unnormalizedProbs[j]-ma);
      sum += classProbs[j];
    }
    if(sum) {
      for(j = 0; j < numClasses; j++) 
        classProbs[j] /= sum;
    }
  } else {
    for(j = 0; j < numClasses; j++) 
      sum += unnormalizedProbs[j];

    if(sum) {
      for(j = 0; j < numClasses; j++) 
        classProbs[j] = unnormalizedProbs[j] / sum;
    }
  }
  return sum;
}

// Computes the class entropy of a normalized array of class probabilities
double ComputeEntropy(double *classProbs, int numClasses) {
  double entropy = 0;
  for(int j = 0; j < numClasses; j++) {
    if(classProbs[j] > 0)
      entropy -= classProbs[j]*log(classProbs[j]);
  }

  assert(!isnan(entropy));
  return entropy;
}



int ClassProbCompare(const void *a1, const void *a2) {
  ClassProb *v1 = (ClassProb*)a1, *v2 = (ClassProb*)a2; 
  double d = (((ClassProb*)v2)->prob-(((ClassProb*)v1)->prob));
  return d < 0 ? -1 : (d > 0 ? 1 : (v1->classID-v2->classID)); 
} 
ClassProb *BestClasses(double *classProbs, int numClasses) {
  ClassProb *probs = (ClassProb*)malloc(sizeof(ClassProb)*numClasses);
  
  for(int i = 0; i < numClasses; i++) {
    probs[i].prob = classProbs[i];
    probs[i].classID = i;
  }
  qsort(probs, numClasses, sizeof(ClassProb), ClassProbCompare);

  return probs;
}
