#ifndef __GMM_H
#define __GMM_H

#include "kmeans.h"

#define LEARN_MIXTURE_WEIGHTS


/**
 * @file gmm.h
 * @brief Gaussian mixture model learning
 */

template <typename T>
inline void GMMUpdateModelComponent(int k, T **pts, T **mu, T **sigma, T *mixtureWeights, T **softAssignments, int numPts, int K, int ptSz, double minSigma);
template <typename T>
inline double GMMComputeSoftAssignments(T **pts, T **mu, T **sigma, T *mixtureWeights, T **softAssignments, 
					int numPts, int K, int ptSz, T ***exampleDevs=NULL, T **log_likelihoods=NULL);
template <typename T>
inline double GMMLearn(T **pts, T **mu, T **sigma, T *mixtureWeights, int numPts, int K, int ptSz, int debug=1, T minSigma=.0001);
template <typename T>
inline void GMMUpdateModel(T **pts, T **mu, T **sigma, T *mixtureWeights, T **softAssignments, int numPts, int K, int ptSz, double minSigma);



template <typename T>
inline double GMMComputeSoftAssignments(T **pts, T **mu, T **sigma, T *mixtureWeights, T **softAssignments, 
					int numPts, int K, int ptSz, T ***exampleDevs, T **log_likelihoods) {
  int i, j, k;
  T *invSigma = Create1DArray<T>(ptSz);
  T *ll = Create1DArray<T>(K);
  T log_likelihood = 0, d;
  if(!log_likelihoods) 
    log_likelihoods = softAssignments;

  for(k = 0; k < K; k++) {
    // Compute the normalization terms of this gaussian
    T gaussNormTerm = LN(mixtureWeights[k])-K/2*LN(2*M_PI);
    for(j = 0; j < ptSz; j++) {
      gaussNormTerm -= sigma[k][j] ? .5*LN(sigma[k][j]) : 0;
      invSigma[j] = sigma[k][j] ? 1.0 / sigma[k][j] : .00000000001;
    }

    // Compute log probabilities for each point-gaussian pair
    for(i = 0; i < numPts; i++) {
      log_likelihoods[i][k] = 0;
      if(exampleDevs) {
	for(j = 0; j < ptSz; j++) {
	  exampleDevs[k][j][i] = (pts[i][j]-mu[k][j])/sigma[k][j];
	  log_likelihoods[i][k] += SQR(exampleDevs[k][j][i]);
	}
      } else {
	for(j = 0; j < ptSz; j++) {
	  d = (pts[i][j]-mu[k][j])/sigma[k][j];
	  log_likelihoods[i][k] += SQR(d);
	  assert(!isnan(log_likelihoods[i][k]));
	}
      }	
      log_likelihoods[i][k] = gaussNormTerm-.5*log_likelihoods[i][k];
      assert(!isnan(log_likelihoods[i][k]) && log_likelihoods[i][k] <= 10000000);
    }
  }

  // Compute normalized probabilities for each point
  for(i = 0; i < numPts; i++) {
    T ma = -INFINITY;
    for(k = 0; k < K; k++) 
      if(log_likelihoods[i][k] > ma)
	ma = log_likelihoods[i][k];
    T Z = 0;
    for(k = 0; k < K; k++) {
      ll[k] = log_likelihoods[i][k];
      softAssignments[i][k] = exp(log_likelihoods[i][k]-ma);
      Z += softAssignments[i][k];
    }
    T expected_log_likelihood = 0;
    for(k = 0; k < K; k++) {
      softAssignments[i][k] /= Z;
      if(softAssignments[i][k] > 0) 
	expected_log_likelihood += softAssignments[i][k]*ll[k];
      assert(!isnan(expected_log_likelihood));
    }
    log_likelihood += expected_log_likelihood;
    assert(!isnan(log_likelihood));
  }

  free(invSigma);
  free(ll);
  return -log_likelihood;
}


template <typename T>
inline T GMMSplitRandomCenter(int k_replace, T **pts, T **mu, T **sigma, T *mixtureWeights, T **softAssignments, int numPts, int K, int ptSz, double minSigma) {
  // Randomly choose a center to split
  int k_split = 0, i;
  T Z = 0;
  for(i = 0; i < K; i++) 
    Z += mixtureWeights[i];
  double r = RAND_DOUBLE*Z;
  double s = 0;
  while(k_split < K-1 && s+mixtureWeights[k_split] < r)
    s += mixtureWeights[k_split++];

  // Randomly split the assignment of each point between k_replace and k_split
  s = 0;
  for(i = 0; i < numPts; i++) {
    r = RAND_DOUBLE;
    softAssignments[i][k_replace] = r*softAssignments[i][k_split];
    softAssignments[i][k_split] = (1-r)*softAssignments[i][k_split];
    s += r;
  }

#ifdef LEARN_MIXTURE_WEIGHTS
  mixtureWeights[k_replace] = s/numPts*mixtureWeights[k_split];
  mixtureWeights[k_split] = mixtureWeights[k_split]-mixtureWeights[k_replace];
  GMMUpdateModelComponent(k_split, pts, mu, sigma, mixtureWeights, softAssignments, numPts, K, ptSz, minSigma);
#endif

  GMMUpdateModelComponent(k_replace, pts, mu, sigma, mixtureWeights, softAssignments, numPts, K, ptSz, minSigma);

  return mixtureWeights[k_replace];
}

template <typename T>
inline void GMMUpdateModelComponent(int k, T **pts, T **mu, T **sigma, T *mixtureWeights, T **softAssignments, int numPts, int K, int ptSz, double minSigma) {
  int i, j;

  // Update mixtureWeights as the number of points assigned to each mixture component, weighted by softAssignments
  T Z = 0;
  for(i = 0; i < numPts; i++) 
    Z += softAssignments[i][k];
#ifdef LEARN_MIXTURE_WEIGHTS
  mixtureWeights[k] = Z/numPts;
#endif
  //T iZ = 1.0/Z;

  // Update mu as the average of pts assigned to each mixture component, weighted by softAssignments
  if(Z) {
    for(j = 0; j < ptSz; j++) 
      mu[k][j] = sigma[k][j] = 0; 
    for(i = 0; i < numPts; i++) 
      for(j = 0; j < ptSz; j++) 
	mu[k][j] += (softAssignments[i][k])*pts[i][j];	
    for(j = 0; j < ptSz; j++) 
      mu[k][j] /= Z;

    // Update sigma as the std deviation of pts assigned to each mixture component, weighted by softAssignments
    for(i = 0; i < numPts; i++) 
      for(j = 0; j < ptSz; j++) 
	sigma[k][j] += (softAssignments[i][k])*SQR(pts[i][j]-mu[k][j]);	
    for(j = 0; j < ptSz; j++) 
      sigma[k][j] /= Z;
    
    for(j = 0; j < ptSz; j++) {
      sigma[k][j] = sqrt(sigma[k][j]);
      sigma[k][j] = my_max(sigma[k][j], minSigma);
    }
  } else {
    printf("  component %d has no assignments\n", k);
    GMMSplitRandomCenter(k, pts, mu, sigma, mixtureWeights, softAssignments, numPts, K, ptSz, minSigma);
  }
}

template <typename T>
inline void GMMUpdateModel(T **pts, T **mu, T **sigma, T *mixtureWeights, T **softAssignments, int numPts, int K, int ptSz, double minSigma) {
  for(int k = 0; k < K; k++) 
    GMMUpdateModelComponent(k, pts, mu, sigma, mixtureWeights, softAssignments, numPts, K, ptSz, minSigma);
}

template <typename T>
inline double GMMLearn(T **pts, T **mu, T **sigma, T *mixtureWeights, int numPts, int K, int ptSz, int debug, T minSigma) {
  int i, j, k;
  int numIter = 0;
  T lastError, error = INFINITY;
  T **softAssignments = Create2DArray<T>(numPts,K);
  T **log_likelihoods = Create2DArray<T>(numPts,K);
  int *num = Create1DArray<int>(K);

  for(k = 0; k < K; k++) 
    for(j = 0; j < ptSz; j++)
      sigma[k][j] = mu[k][j] = 0;

  // Initialize using kmeans
  int *assigned = KMeans<T>(pts, numPts, mu, K, ptSz, debug);

  /*for(k = 0; k < K; k++) {
    for(j = 0; j < ptSz; j++)
		sigma[k][j] = 1;
    mixtureWeights[k] = 1.0/K;
  }
  delete [] assigned;
  return 0;*/

  for(i = 0; i < numPts; i++) {
    k = assigned[i];
    num[k]++;
    for(j = 0; j < ptSz; j++) 
      sigma[k][j] += SQR(pts[i][j]-mu[k][j]);
  }
  for(k = 0; k < K; k++) {
    if(num[k])
      for(j = 0; j < ptSz; j++) 
        sigma[k][j] /= num[k];
    for(j = 0; j < ptSz; j++) 
      sigma[k][j] = my_max(sigma[k][j],minSigma);
      
#ifdef LEARN_MIXTURE_WEIGHTS
    mixtureWeights[k] = ((T)num[k])/numPts;
#else
    mixtureWeights[k] = 1.0/K;
#endif
  }

  // Iteratively update centers and center assignments until convergence
  do {
    lastError = error;
    error = GMMComputeSoftAssignments(pts, mu, sigma, mixtureWeights, softAssignments, numPts, K, ptSz, (T***)NULL, log_likelihoods);
    GMMUpdateModel(pts, mu, sigma, mixtureWeights, softAssignments, numPts, K, ptSz, minSigma);
    if(debug > 1) fprintf(stderr, "  gmm iter=%d ave_cost=%f\n", numIter, (float)(error/numPts));
  } while(numIter++ < MAX_ITER_KMEANS && error < lastError);

  // Cleanup
  free(softAssignments);
  free(log_likelihoods);
  free(num);
  delete [] assigned;

  if(debug > 0) fprintf(stderr, " done iter=%d ave_cost=%f\n", numIter, (float)(error/numPts));

  return error;
}







#endif

