/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#ifndef __DISTANCE_H
#define __DISTANCE_H

#include <math.h>

/**
 * @file distance.h
 * @brief Routines for computing different types of distance functions
 */


/**
 * @brief Compute the squared magnitude of a vector d: d'd
 * @param d An array of size ptSz
 * @param ptSz The number of elements in the array
 * @return The squared magnitude of d
 */
template <typename T> 
inline T PointMagnitude_sqr(T *d, int ptSz) {
  int i = 0;
  T sum = 0;

  for(i = 0; i < ptSz; i++) 
    sum += d[i]*d[i];

  return sum;
}


/**
 * @brief Computes the squared L2 distance between two vectors of length ptSz 
 * @param d1 An array of size ptSz storing the first vector
 * @param d2 An array of size ptSz storing the second vector
 * @param ptSz The number of elements in the array
 * @return (d1-d2)'(d1-d2)
 */
template <typename T> 
inline T PointDistanceL2_sqr(T *d1, T *d2, int ptSz) {
  int i = 0;
  T sum = 0;
  T d;

  for(i = 0; i < ptSz; i++) {
    d = d1[i]-d2[i];
    sum += d*d;
  }

  return sum;
}


/**
 * @brief Computes the L2 distance between two vectors of length ptSz 
 * @param d1 An array of size ptSz storing the first vector
 * @param d2 An array of size ptSz storing the second vector
 * @param ptSz The number of elements in the array
 * @return sqrt((d1-d2)'(d1-d2))
 */
template <typename T> 
inline T PointDistanceL2(T *d1, T *d2, int ptSz) {
  return sqrt(PointDistanceL2_sqr<T>(d1, d2, ptSz));
}


/**
 * @brief Computes the L1-norm between two vectors of length ptSz 
 * @param d1 An array of size ptSz storing the first vector
 * @param d2 An array of size ptSz storing the second vector
 * @param ptSz The number of elements in the array
 * @return sum_{i=1}^ptSz |d1[i]-d2[i]|
 */
template <typename T> 
inline T PointDistanceL1(T *d1, T *d2, int ptSz) {
  int i = 0;
  T sum = 0;
  T d;

  for(i = 0; i < ptSz; i++) {
    d = d1[i]-d2[i];
    sum += d < 0 ? -d : d;
  }

  return sum;
}


/**
 * @brief Computes the chi histogram intersection of two vectors of length ptSz
 * @param d1 An array of size ptSz storing the first vector
 * @param d2 An array of size ptSz storing the second vector
 * @param ptSz The number of elements in the array
 * @return 1/2 sum_{i=1}^ptSz (d1[i]-d2[i])^2/(d1[i]+d2[i])
 */
template <typename T> 
inline T PointDistanceChi(T *d1, T *d2, int ptSz) {
  int i = 0;
  T sum = 0;
  T dn, dd;

  for(i = 0; i < ptSz; i++) {
    dn = d1[i]-d2[i];
    dd = d1[i]+d2[i];
    if(dd) sum += dn*dn/dd;
  }

  return sum/2;
}

#endif
