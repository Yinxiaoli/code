/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#ifndef __KMEANS_H
#define __KMEANS_H

/**
 * @file kmeans.h
 * @brief Routines for k-means clustering
 */

#include "util.h"
#include "distance.h"

#define MAX_ITER_KMEANS 15 /**< maximum number of iterations to run k-means for */
#define NUM_RANDOM_INITIALIZATIONS 10 /**< Number of times to try random initialization for k-means++ */


//template <typename T> int *KMeans(T **pts, int numPts, T **centers, int k, int ptSz, int debug=1, T (*distFunc)(T *d1, T *d2, int ptSz)=PointDistanceL2_sqr<T>);
template <typename T> T KMeansOneIteration(T **pts, int numPts, T **centers, int k, int ptSz, int *assignedClusters, 
					   T **meanSum, int *num, int debug=1, T (*distFunc)(T *d1, T *d2, int ptSz)=PointDistanceL2_sqr<T>, bool mirroredPoints=false);
template <typename T> double KMeansInitialize(T **pts, int numPts, T **centers, int k, int ptSz, int debug=1, T (*distFunc)(T *d1, T *d2, int ptSz)=PointDistanceL2_sqr<T>, bool mirroredPoints=false);
template <typename T> int NearestCluster(T *x, T **centers, int k, int ptSz, T *dist=NULL, T (*distFunc)(T *d1, T *d2, int ptSz)=PointDistanceL2_sqr<T>);



/**
 * @brief Cluster using k-means
 *
 * Initializes cluster centers using k-means++, then runs regular k-means until convergence
 *
 * @param pts a numPtsXptSz array of dataPts to be clustered
 * @param numPts number of data points
 * @param centers a kXptSz array used to store the centers computed by k-means
 * @param k the number of centers to be found by k-means
 * @param ptSz The dimensionality of the feature space
 * @param debug used to control how much debug info is printed
 * @param distFunc The distance function to use (should be PointDistanceL2_sqr()) for k-means
 * @return an array of length numPts encoding the index of the nearest center for each data point
 *
 */
template <typename T> 
inline int *KMeans(T **pts, int numPts, T **centers, int k, int ptSz, int debug=1, T (*distFunc)(T *d1, T *d2, int ptSz)=PointDistanceL2_sqr<T>, bool mirroredPoints=false) {
  if(debug > 0) fprintf(stderr, "Running k-means on %d points...", numPts);


  int i;
  int numIter = 0;
  T lastError, error = INFINITY;
  T **meanSum = new T*[k];
  for(i = 0; i < k; i++)
    meanSum[i] = new T[ptSz];
  int *num = new int[k];
  int *assignedClusters = new int[numPts];
  double bestCost = INFINITY;

  if(debug > 1) fprintf(stderr, "initializing...");
  for(int i = 0; i < NUM_RANDOM_INITIALIZATIONS; i++) {
    T **centers_curr = Create2DArray<T>(k, ptSz);
    double cost = KMeansInitialize(pts, numPts, centers_curr, k, ptSz, debug, distFunc, mirroredPoints);
    cost = KMeansOneIteration(pts, numPts, centers_curr, k, ptSz, assignedClusters, meanSum, num, debug, distFunc, mirroredPoints);
    cost = KMeansOneIteration(pts, numPts, centers_curr, k, ptSz, assignedClusters, meanSum, num, debug, distFunc, mirroredPoints);
    if(cost < bestCost) {
      bestCost = cost;
      for(int j = 0; j < k; j++)
	memcpy(centers[j], centers_curr[j], sizeof(T)*ptSz);
    }
    free(centers_curr);
    if(debug > 1) fprintf(stderr, "%d.", i);
  }


  // Iteratively update centers and center assignments until convergence
  do {
    lastError = error;
    error = KMeansOneIteration(pts, numPts, centers, k, ptSz, assignedClusters, meanSum, num, debug, distFunc, mirroredPoints);
    if(debug > 1) fprintf(stderr, "  kmeans iter=%d ave_cost=%f\n", numIter, (float)(error/numPts));
  } while(numIter++ < MAX_ITER_KMEANS && error < lastError);

  /*
  fprintf(stderr, "\n");
  for(i = 0; i < k; i++) {
    fprintf(stderr, "k=%d:", i);
    for(int j = 0; j < ptSz; j++) 
      fprintf(stderr, " %d:%.4lf", j, centers[i][j]);
    fprintf(stderr, "\n");
  }
  */

  // Cleanup
  for(i = 0; i < k; i++)
    delete [] meanSum[i];
  delete [] meanSum;
  delete [] num;

  if(debug > 0) fprintf(stderr, " done iter=%d ave_cost=%f\n", numIter, (float)(error/numPts));

  return assignedClusters;
}


/**
 * @brief Find the nearest center to point 'x'
 * @param x an array of size ptSize
 * @param centers pts a kXptSz array of k cluster centers
 * @param k the number of cluster centers 
 * @param ptSz the feature space dimensionality
 * @param dist a pointer into which the distance to the nearest cluster center is stored
 * @param distFunc The distance function to use (should be PointDistanceL2_sqr()) for k-means
 * @return the index into centers of the nearest cluster center
 *
 */
template <typename T> 
inline int NearestCluster(T *x, T **centers, int k, int ptSz, T *dist, T (*distFunc)(T *d1, T *d2, int ptSz)) {
    int n = 0;
    T minD = INFINITY, d;
    for(int i = 0; i < k; i++) {
      d = (*distFunc)(centers[i], x, ptSz);
      if(d < minD) {
	minD = d;
	n = i;
      }
    }
    if(dist) *dist = minD;
    return n;
}

template <typename T> 
inline int NearestClusterMirrored(T *x1, T *x2, T **centers, int k, int ptSz, T *dist, T (*distFunc)(T *d1, T *d2, int ptSz)) {
    int n = 0;
    T minD = INFINITY, d;
    for(int i = 0; i < k; i++) {
      d = (*distFunc)(centers[i], x1, ptSz) + (*distFunc)(centers[(i/2)*2 + (i%2 ? 0 : 1)], x2, ptSz);
      if(d < minD) {
	minD = d;
	n = i;
      }
    }
    if(dist) *dist = minD;
    return n;
}

/**
 * @brief Do one iteration of k-means (assign points to nearest center, and update each center mean)
 *
 * @param pts a numPtsXptSz array of dataPts to be clustered
 * @param numPts number of data points
 * @param centers a kXptSz array used to store the centers computed by k-means
 * @param k the number of centers to be found by k-means
 * @param ptSz The dimensionality of the feature space
 * @param assignedClusters a numPts array into which the assigned cluster (an index into centers) of each point in pts is stored 
 * @param meanSum An array of size k storing the sum of all points assigned to each cluster
 * @param num An array of size k storing how many points were assigned to each cluster
 * @param debug used to control how much debug info is printed
 * @param distFunc The distance function to use (should be PointDistanceL2_sqr()) for k-means
 * @return the total k-means cost
 */
template <typename T> 
inline T KMeansOneIteration(T **pts, int numPts, T **centers, int k, int ptSz, int *assignedClusters, 
			    T **meanSum, int *num, int debug, T (*distFunc)(T *d1, T *d2, int ptSz), bool mirroredPoints) {
  int i, j;
  T error = 0;
  
  for(i = 0; i < k; i++) {
    for(j = 0; j < ptSz; j++)
      meanSum[i][j] = 0;
    num[i] = 0;
  }

  // Special case where it is assumed that examples come in pairs, such that each example must
  // be in the opposite cluster than its paired example
  if(mirroredPoints) 
    assert(k%2==0 && numPts%2==0);

  // for each data point, find the nearest center
#ifdef USE_OPENMP
  //#pragma omp parallel for
#endif
  for(int ii = 0; ii < numPts; ii += (mirroredPoints ? 2 : 1)) {
    T d;
    if(!mirroredPoints) 
      assignedClusters[ii] = NearestCluster(pts[ii], centers, k, ptSz, &d, distFunc);
    else {
      assignedClusters[ii] = NearestClusterMirrored(pts[ii], pts[ii+1], centers, k, ptSz, &d, distFunc);
      assignedClusters[ii+1] = (assignedClusters[ii]/2)*2 + (assignedClusters[ii]%2 ? 0 : 1);
    }
  }
  
  for(i = 0; i < numPts; i++) {
    error += (*distFunc)(centers[assignedClusters[i]], pts[i], ptSz);
    
    // Update the appropriate mean based on the assigned center
    for(j = 0; j < ptSz; j++)
      meanSum[assignedClusters[i]][j] += pts[i][j];
    num[assignedClusters[i]]++;
  }
  
  // Finish computing the mean for each center
  for(i = 0; i < k; i++)
    for(j = 0; j < ptSz; j++)
      centers[i][j] = meanSum[i][j] / num[i]; 
  
  return error;
}



/**
 * @brief Initialize k-means using k-means++, an initializer that produces centers that are within expected error of O(log(k))
 *
 * k-means++ works by iteratively adding a new center, at each iteration randomly selecting a point as a new center, where the
 * probability of picking a point is proportional to its current distance to the nearest center
 *
 * @param pts a numPtsXptSz array of dataPts to be clustered
 * @param numPts number of data points
 * @param centers a kXptSz array used to store the centers computed by k-means++
 * @param k the number of centers to be found by k-means++
 * @param ptSz The dimensionality of the feature space
 * @param debug used to control how much debug info is printed
 * @param distFunc The distance function to use (should be PointDistanceL2_sqr()) for k-means
 */
// 
template <typename T> 
inline double KMeansInitialize(T **pts, int numPts, T **centers, int k, int ptSz, int debug, T (*distFunc)(T *d1, T *d2, int ptSz), bool mirroredPoints) {
  double totalDist = 0;
  int i, j;
  double d;
  int pt_ind;

  double *dists = new double[numPts];
  for(i = 0; i < numPts; i++)
    dists[i] = INFINITY;

  for(i = 0; i < k; i += (mirroredPoints ? 2 : 1)) {
    if(i == 0) {
      // Initialize the first mean to a random data point
      if(!mirroredPoints) pt_ind = rand()%numPts;
      else pt_ind = (rand()%(numPts/2))*2;
    } else {
      // Initialize the next mean to a random data point, chosen with probability
      // proportional to its distance to its nearest center
      double d = ((double)rand())/RAND_MAX * totalDist;
      double sum = 0;
      for(j = 0; j < numPts-1; j++) {
	sum += dists[j];
	if(d <= sum) break;
      }
      pt_ind = mirroredPoints ? (j/2)*2 : j;
      if(debug > 2) fprintf(stderr, "  chose center %d at %f away\n", i, (float)dists[j]);
    }
    memcpy(centers[i], pts[pt_ind], sizeof(T)*ptSz);
    if(mirroredPoints) memcpy(centers[i+1], pts[pt_ind+1], sizeof(T)*ptSz);

    // Update each point if its distance to the new center is closer than any previous center 
    for(j = 0; j < numPts; j++) {
      d = (double)(*distFunc)(centers[i], pts[j], ptSz);
      if(d < dists[j])
	dists[j] = d;
      if(mirroredPoints) {
	d = (double)(*distFunc)(centers[i+1], pts[j], ptSz);
	if(d < dists[j])
	  dists[j] = d;
      }
    }

    totalDist = 0;
    for(j = 0; j < numPts; j++) {
      totalDist += dists[j];
    }
  }

  delete [] dists;

  return totalDist;
}


/**
 * @brief Cluster using hierarchical k-means, producing a set of 2^depth cluster centers that can be reached by traversing a binary tree
 *
 * Recursively calls k-means using KMeans() with k=2.  
 *
 * @param pts a numPtsXptSz array of dataPts to be clustered
 * @param numPts number of data points
 * @param centers a kXptSz array used to store the centers computed by k-means
 * @param decision_planes A (2^depth-1)X(numPts+1) array, whose values are set by this function.  The user should pass this to HierarchicalKMeansAssignment()
 *  to determine which cluster center to assign a novel point
 * @param depth The depth of the tree.  The number of centers will be k=2^depth
 * @param ptSz The dimensionality of the feature space
 * @param debug used to control how much debug info is printed
 * @return an array of length numPts encoding the index of the assigned center for each data point
 *
 */
template <typename T> 
inline int *HierarchicalKMeans(T **pts, int numPts, T **centers, T **decision_planes, int depth, int ptSz, int debug=1, bool mirroredPoints=false) {
  assert(depth >= 1);
  int i;

  // Run regular kmeans with k=2, for creating the next tree branch
  int *assigned = KMeans(pts, numPts, centers, 2, ptSz, debug, PointDistanceL2_sqr<T>, mirroredPoints);

  // For efficiency, create a vector w (decision_plane), such that the decision to assign a new point x to the right branch
  // occurs iff the dot product is greater than 0, e.g. <w,[x,1]> > 0 
  decision_planes[0][ptSz] = 0;
  for(i = 0; i < ptSz; i++) {  // compute the equation of the plane where the left and right points are equidistant
    decision_planes[0][i] = centers[1][i]-centers[0][i];
    decision_planes[0][ptSz] -= decision_planes[0][i]*(centers[1][i]+centers[0][i])/2;
  }
  decision_planes++;

  if(depth > 1) {
    // Assign each point to the left or right child
    int num_left = 0, num_right = 0;
    T **pts_left = new T*[numPts], **pts_right = new T*[numPts];
    for(i = 0; i < numPts; i++) {
      if(assigned[i] == 0) pts_left[num_left++] = pts[i];
      else pts_right[num_right++] = pts[i];
    }

    assert(num_left);
    if(num_right == 0) {  // If the right child has 0 pts, then never send a point to that branch 
      for(i = 0; i < ptSz; i++) decision_planes[-1][i] = 0;
      decision_planes[-1][ptSz] = -10000;
    }

    // Recurse on the left and right branches
    int *assigned_left = HierarchicalKMeans(pts_left, num_left, centers, decision_planes, depth-1, ptSz, debug, mirroredPoints); 
    int *assigned_right = num_right ? HierarchicalKMeans(pts_right, num_right, centers + (1<<(depth-1)), decision_planes + (1<<(depth-2)), depth-1, ptSz, debug, mirroredPoints) : NULL; 
    
    // Correct the assignment indices to be the correct indices
    num_left = num_right = 0;
    for(i = 0; i < numPts; i++) {
      if(assigned[i] == 0) assigned[i] = assigned_left[num_left++];
      else assigned[i] = assigned_right[num_right++] + (1<<(depth-1));
    }

    delete [] pts_left; 
    delete [] pts_right; 
    delete [] assigned_left; 
    if(assigned_right) delete [] assigned_right; 
  }

  return assigned;
}

/**
 * @brief Assign a novel point to a cluster center, where the set of cluster centers was created using HierarchicalKMeans()
 *
 * The returned value will be a number from 0...(2^depth-1).  
 *
 * @param pt A vector of size ptSz.  The point we want to assign
 * @param decision_planes A (2^depth-1)X(numPts+1) array, whose values are set by HierarchicalKMeans()
 * @param depth The depth of the tree.  The number of centers will be k=2^depth
 * @param ptSz The dimensionality of the feature space
 * @return An integer identifier from 0...(2^depth-1).  Note that if the id is encoded in binary, one can read off the internal branches in the tree assignment
 *
 */
template <typename T> 
inline int HierarchicalKMeansAssignment(T *pt, T **decision_planes, int depth, int ptSz) {
  int i;
  T sum = 0;
  T *w;
  int ind = 0;
  int w_ind = 0;

  while(depth) {
    w = decision_planes[w_ind++];  // dot product with the decision plane to choose assignment to the left or right branch
    sum = w[ptSz];
    for(i = 0; i < ptSz; i++)
      sum += w[i]*pt[i];

    if(sum > 0) { // right branch
      ind += 1<<(depth-1);
      if(depth > 1) w_ind += 1<<(depth-2);
    }
    depth--;
  }

  return ind;
}

#endif 
