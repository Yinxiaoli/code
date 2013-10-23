/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/


// Taken from Pedro Felzenszwalb's distance transform source code and then slightly modified
// The modifications include:
//   1) The distance transform routines in this file maximize a score rather than minimize a cost
//   2) The ideal offsets mu_x,mu_y are passed explicitly to this function, instead of assuming
//      the caller will shift the images
//   3) The spatial costs are precomputed as a lookup table
//   4) The inputs to the distance transform functions are OpenCV images
//   5) Another routine to do distance transforms over scales and orientations was added
#include "opencvUtil.h"

#define DT_PAD 0


/*
 * Generalized distance transforms.
 * We use a simple nlog(n) divide and conquer algorithm instead of the
 * theoretically faster linear method, for no particular reason except
 * that this is a bit simpler and I wanted to test it out.
 */


// O(n log n) 1D distance transform by divide and conquer 
void dt1d_log(float *unary_score, float *combined_score, int *best_loc, 
	       int s1, int s2, int d1, int d2, float *spatial_score) {
 if (d2 >= d1) {
   int d = (d1+d2) >> 1;
   int s = s1;
   float cs = unary_score[s] + spatial_score[s-d], cp;
   for (int p = s1+1; p <= s2; p++) {
     cp = unary_score[p] + spatial_score[p-d];
     if (cs < cp) {
       s = p;
       cs = cp;
     }
   }
   combined_score[d] = cs;
   best_loc[d] = s;
   dt1d_log(unary_score, combined_score, best_loc, s1, s, d1, d-1, spatial_score);
   dt1d_log(unary_score, combined_score, best_loc, s, s2, d+1, d2, spatial_score);
 }
}

// O(n) 1D distance transform by computing an upper envelope
void dt1d_linear(float *unary_score, float *combined_score, int *best_loc, int n, int mu, float wx, float wxx, int *v, float *z, float *spatial_scores, float *cache) {
  int k = 0, q; 
  float s, dx, i2wxx = wxx ? 1.0f/(2*wxx) : 1;
  float o = -mu + wx / (2*wxx);
  v[0] = 0;
  z[0] = -INFINITY;
  cache[0] = 0;
  
  for(q = 1; q < n; q++) {
    // s is the position in which the parabola centered at q could become part of the upper envelope,
    // and is found by intersecting the parabola centered at q with the previous parabola in the upper envelope
    cache[q] = unary_score[q]*i2wxx + SQR(q)/2;
    s = o + cache[q]-cache[q-1];  // s = o + (unary_score[v[k]]-unary_score[q] + wxx*(SQR(v[k])-SQR(q))) / (2*wxx*(v[k]-q));
    while(s <= z[k]) {
      k--;  // remove a parabola that is strictly dominated by the parabolas at q and v[k-1]
      s = o + (cache[v[k]] - cache[q]) / (v[k]-q);  // s = o + (unary_score[v[k]]-unary_score[q] + wxx*(SQR(v[k])-SQR(q))) / (2*wxx*(v[k]-q));
    }
    k++;
    v[k] = q;
    z[k] = s;
  }
  z[k+1] = INFINITY;
  
  for(q = 0, k = 0; q < n; q++) {
    // The parabola at v[k] encodes the optimal solution for all points z[k] <= q < z[k+1]
    while(z[k+1] < q)
      k++;
    best_loc[q] = v[k];
    //dx = q+mu-v[k];  combined_score[q] = unary_score[v[k]] + wxx*SQR(dx) + wx*dx;
    combined_score[q] = unary_score[v[k]] + spatial_scores[v[k]-q];
  }
}



// dt of 1d array
void dt1d(float *src, float *dst, int *ptr, int n, float *spatial_scores, int mu, float wx, float wxx) {
  int b = (mu < 0 ? -mu : mu);
  float *z = (spatial_scores+n+2*b+2);
  float *cache = (z+n+1);
  int *v = (int*)(cache+n);

  if(wxx >= 0) {
    // Distance transform is invalid if wxx >= 0.  In a special case to handle this,
    // simply choose the child position with maximum score, ignoring the spatial component
    // (equivalent to setting wxx=wx=0)
    float b = -INFINITY;
    int i, ii, dx;
    for(i = 0; i < n; i++) {
      if(src[i] > b) {
	b = src[i];
	ii = i;
      }
    }
    for(i = 0; i < n; i++) {
      ptr[i] = ii;
      dx = i+mu-ii;
      dst[i] = src[ii] + (dx*wx+SQR(dx)*wxx);
    }
  } else {
    dt1d_linear(src, dst, ptr, n, mu, wx, wxx, v, z, spatial_scores, cache);
    //dt1d_log(src, dst, ptr, DT_PAD, n-1-DT_PAD, 0, n-1, spatial_scores);
  }

#ifdef EXTRA_DEBUG
  for(int i = 0; i < n; i++) {
    int x = ptr[i];
    int dx = i+mu-x;
    float d =  dst[i] - (src[x]+dx*wx+SQR(dx)*wxx);
    assert(d > -.001 && d < .001);
  }
#endif
}

float *dt_spatial_cost_cache_table(int n, float wxx, float wx, int mu, float **ptr2, int max_change) {
  int b = (mu < 0 ? -mu : mu);
  float *retval = (float*)malloc(sizeof(float)*(2*n+1+4*b+1)+(n+1)*(sizeof(float)*2+sizeof(int)));
  float *ptr = *ptr2 = retval+n+2*b;

  for(int i = -n-b; i <= n+b; i++) {
    if(i < -max_change || i > max_change) ptr[i+mu] = -INFINITY;
    else ptr[i+mu] = wxx*(i*i) + wx*(i);
  }

  return retval;
}

void DistanceTransform(IplImage *img, float wxx, float wyy, float wx, float wy, int mu_x, int mu_y, IplImage **best_child_locs, IplImage **score, int max_change_x, int max_change_y) {
  int x, y;
  IplImage *imgT = cvCreateImage(cvSize(img->height,img->width), IPL_DEPTH_32F, 1);
  IplImage *My = cvCreateImage(cvSize(img->height,img->width), IPL_DEPTH_32F, 1);
  IplImage *Ix = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_32S, 1);
  IplImage *Iy = cvCreateImage(cvSize(img->height,img->width), IPL_DEPTH_32S, 1);
  IplImage *MyT = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_32F, 1);
  IplImage *Mx = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_32F, 1);
  IplImage *IyT = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_32S, 1);
  cvTranspose(img, imgT);
  
  // First compute the 1D distance transform along each vertical line
  float *cache_table;
  float *cache_table_buff = dt_spatial_cost_cache_table(imgT->width, wyy, wy, mu_y, &cache_table, max_change_y);
  unsigned char *ptr = (unsigned char*)imgT->imageData;
  unsigned char *dstPtr = (unsigned char*)My->imageData;
  unsigned char *iPtr = (unsigned char*)Iy->imageData;
  for (y = 0; y < imgT->height; y++, ptr += imgT->widthStep, dstPtr += My->widthStep, iPtr += Iy->widthStep)
    dt1d((float*)ptr, (float*)dstPtr, (int*)iPtr, imgT->width, cache_table, mu_y, wy, wyy);
  free(cache_table_buff);
  cvTranspose(My, MyT);
  cvTranspose(Iy, IyT);
  
  // Next compute the 1D distance transform along each horizontal line, taking as input
  // the vertical line scores
  cache_table_buff = dt_spatial_cost_cache_table(MyT->width, wxx, wx, mu_x, &cache_table, max_change_x);
  ptr = (unsigned char*)MyT->imageData;
  dstPtr = (unsigned char*)Mx->imageData;
  iPtr = (unsigned char*)Ix->imageData;
  for (y = 0; y < MyT->height; y++, ptr += MyT->widthStep, dstPtr += Mx->widthStep, iPtr += Ix->widthStep)
    dt1d((float*)ptr, (float*)dstPtr, (int*)iPtr, img->width, cache_table, mu_x, wx, wxx);
  free(cache_table_buff);


  // Package the x,y child locations into a 2-channel image of locations
  *best_child_locs = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_16S, 2);
  dstPtr = (unsigned char*)(*best_child_locs)->imageData;
  char *xPtr = Ix->imageData;
  char *yPtr = IyT->imageData;
  for(y = 0; y < img->height; y++, dstPtr += (*best_child_locs)->widthStep, xPtr += Ix->widthStep, yPtr += IyT->widthStep) {
    for(x = 0; x < img->width; x++) {
      ((int16_t*)dstPtr)[x<<1] = (int16_t)((int*)xPtr)[x];
      ((int16_t*)dstPtr)[(x<<1)+1] = (int16_t)((int*)yPtr)[((int*)xPtr)[x]];
    }
  }
  *score = Mx;

  // Cleanup
  cvReleaseImage(&imgT);
  cvReleaseImage(&My);
  cvReleaseImage(&Ix);
  cvReleaseImage(&Iy);
  cvReleaseImage(&IyT);
  cvReleaseImage(&MyT);
}

/**
 *
 * @param scores A num_scalesXnum_orientations array of scores, where x and y have already been optimized over using distance transforms
 * @param scale_offsets_p A pointer to a num_scalesXnum_orientations array into which optimal scale indices for the child part is stored
 * @param orientation_offsets_p A pointer to a num_scalesXnum_orientations array into which optimal orientation indices for the child part is stored
 */
IplImage ***DistanceTransformScaleOrientation(IplImage ***scores,  IplImage ****best_child_scales_p, IplImage ****best_child_orientations_p,
                       int num_scales, int num_orientations, int mu_s, int mu_o, float wss, float woo, float ws, float wo, 
                       int max_change_scale, int max_change_ori) {
  int w = scores[0][0]->width, h = scores[0][0]->height;
  int sz = w*h;
  int o_pad1 = (num_orientations)/2;
  int o_pad = o_pad1+(mu_o < 0 ? -mu_o : mu_o);
  int buff_sz = (num_orientations > num_scales ? num_orientations : num_scales)+o_pad*2;
  float *child_scores_p = (float*)malloc(buff_sz*sizeof(float));
  float *best_child_scores_p = (float*)malloc(buff_sz*sizeof(float));
  int *best_child_inds_p = (int*)malloc(buff_sz*sizeof(int));
  float *child_scores = child_scores_p+o_pad, *best_child_scores = best_child_scores_p+o_pad;
  int *best_child_inds = best_child_inds_p+o_pad;
  int i, j, k, k2, l, l2;

  // Allocate temporary buffers and return values
  IplImage ***best_scores = (IplImage***)malloc(num_scales*(sizeof(IplImage**)+num_orientations*sizeof(IplImage*)));
  IplImage ***best_orientation_scores = (IplImage***)malloc(num_scales*(sizeof(IplImage**)+num_orientations*sizeof(IplImage*)));
  IplImage ***best_child_orientations = (IplImage***)malloc(num_scales*(sizeof(IplImage**)+num_orientations*sizeof(IplImage*)));
  IplImage ***best_child_orientations2 = *best_child_orientations_p = (IplImage***)malloc(num_scales*(sizeof(IplImage**)+num_orientations*sizeof(IplImage*)));
  IplImage ***best_child_scales = *best_child_scales_p = (IplImage***)malloc(num_scales*(sizeof(IplImage**)+num_orientations*sizeof(IplImage*)));
  for(j = 0; j < num_scales; j++) {
    best_scores[j] = ((IplImage**)(best_scores+num_scales)) + j*num_orientations;
    best_orientation_scores[j] = ((IplImage**)(best_orientation_scores+num_scales)) + j*num_orientations;
    best_child_orientations[j] = ((IplImage**)(best_child_orientations+num_scales)) + j*num_orientations;
    best_child_orientations2[j] = ((IplImage**)(best_child_orientations2+num_scales)) + j*num_orientations;
    best_child_scales[j] = ((IplImage**)(best_child_scales+num_scales)) + j*num_orientations;
    for(k = 0; k < num_orientations; k++) {
      assert(w == scores[0][0]->width && h == scores[0][0]->height);
      best_scores[j][k] = cvCreateImage(cvSize(w,h), IPL_DEPTH_32F, 1);  
      assert((int)(best_scores[j][k]->widthStep*best_scores[j][k]->height/sizeof(float))==sz);
      best_orientation_scores[j][k] = cvCreateImage(cvSize(w,h), IPL_DEPTH_32F, 1);
      assert((int)(best_orientation_scores[j][k]->widthStep*best_orientation_scores[j][k]->height/sizeof(float))==sz);
      best_child_orientations[j][k] = cvCreateImage(cvSize(w,h), IPL_DEPTH_32S, 1);
      assert((int)(best_child_orientations[j][k]->widthStep*best_child_orientations[j][k]->height/sizeof(int))==sz);
      best_child_orientations2[j][k] = cvCreateImage(cvSize(w,h), IPL_DEPTH_32S, 1);
      assert((int)(best_child_orientations2[j][k]->widthStep*best_child_orientations2[j][k]->height/sizeof(int))==sz);
      best_child_scales[j][k] = cvCreateImage(cvSize(w,h), IPL_DEPTH_32S, 1);
      assert((int)(best_child_scales[j][k]->widthStep*best_child_scales[j][k]->height/sizeof(int))==sz);
    }
  }



  // First compute the optimal orientation in the child for every pixel/scale/orientation in the parent
  float *cache_table;
  float *cache_table_buff = dt_spatial_cost_cache_table(num_orientations+2*o_pad1, woo, wo, mu_o, &cache_table, max_change_ori);
  for(j = 0; j < num_scales; j++) {
    for(i = 0; i < sz; i++) {
      // Pack the child detection scores for all orientations for the ith pixel and jth scale into an array child_scores
      for(k = 0; k < num_orientations; k++) 
	child_scores[k] = ((float*)scores[j][k]->imageData)[i];
      
      // add an extra num_orientations/2 entries to each side of the array, since the score can wrap around in a circle
      for(l = 0, k = num_orientations; k < num_orientations+o_pad1+mu_o; l++, k++) 
        child_scores[k] = ((float*)scores[j][l]->imageData)[i];
      for(k2 = -1, l2 = num_orientations-1; k2 >= -o_pad1+mu_o; k2--, l2--)
	child_scores[k2] = ((float*)scores[j][l2]->imageData)[i];

      //dt1d_log(child_scores, best_child_scores, best_child_inds, -o_pad1+mu_o, num_orientations+o_pad1+mu_o-1, 0, num_orientations-1, cache_table);
      
      int pad = -(-o_pad1+mu_o);
      dt1d(child_scores-pad, best_child_scores-pad, best_child_inds-pad, num_orientations+2*o_pad1, cache_table, mu_o, wo, woo);
      for(k = 0; k < num_orientations; k++) 
        best_child_inds[k] -= pad;

      for(k = 0; k < num_orientations; k++) {
        if(best_child_inds[k] < 0) {
	  best_child_inds[k] += num_orientations;
	  if(best_child_inds[k] < 0) best_child_inds[k] += num_orientations;
	}
	if(best_child_inds[k] >= num_orientations) {
	  best_child_inds[k] -= num_orientations;
	  if(best_child_inds[k] >= num_orientations) best_child_inds[k] -= num_orientations;
	}
        ((float*)best_orientation_scores[j][k]->imageData)[i] = best_child_scores[k];
        ((int*)best_child_orientations[j][k]->imageData)[i] = best_child_inds[k];

	if(woo >= 0) {
int r = ((int*)best_child_orientations[j][k]->imageData)[i];
int dr = k+mu_o-r;
if(dr < -num_orientations/2) dr += num_orientations;
if(dr > num_orientations/2) dr -= num_orientations;
 ((float*)best_orientation_scores[j][k]->imageData)[i] = (((float*)scores[j][r]->imageData)[i]+dr*wo+SQR(dr)*woo);
	}
#ifdef EXTRA_DEBUG
#include <assert.h>
int r = ((int*)best_child_orientations[j][k]->imageData)[i];
int dr = k+mu_o-r;
if(dr < -num_orientations/2) dr += num_orientations;
if(dr > num_orientations/2) dr -= num_orientations;
 float d =  ((float*)best_orientation_scores[j][k]->imageData)[i] - (((float*)scores[j][r]->imageData)[i]+dr*wo+SQR(dr)*woo);
assert(d > -.001 && d < .001);
#endif

      }
    }
  }
  free(cache_table_buff);
  
  

  // Next compute the optimal scale in the child for every pixel/scale/orientation in the parent
  cache_table_buff = dt_spatial_cost_cache_table(num_scales, wss, ws, mu_s, &cache_table, max_change_scale);
  for(k = 0; k < num_orientations; k++) { 
    for(i = 0; i < sz; i++) {
      // Pack the child detection scores for all scales for the ith pixel and kth orientation into an array child_scores
      for(j = 0; j < num_scales; j++) 
        child_scores[j] = ((float*)best_orientation_scores[j][k]->imageData)[i];
      //dt1d_log(child_scores, best_child_scores, best_child_inds, 0, num_scales-1, 0, num_scales-1, cache_table);
      dt1d(child_scores, best_child_scores, best_child_inds, num_scales, cache_table, mu_s, ws, wss);

      for(j = 0; j < num_scales; j++) {
        ((float*)best_scores[j][k]->imageData)[i] = best_child_scores[j];
	((int*)best_child_scales[j][k]->imageData)[i] = best_child_inds[j];
	((int*)best_child_orientations2[j][k]->imageData)[i] = ((int*)best_child_orientations[best_child_inds[j]][k]->imageData)[i];


#ifdef EXTRA_DEBUG
#include <assert.h>
int s = ((int*)best_child_scales[j][k]->imageData)[i];
int r = ((int*)best_child_orientations2[j][k]->imageData)[i];
int ds = j+mu_s-s;
int dr = k+mu_o-r;
if(dr < -num_orientations/2) dr += num_orientations;
if(dr > num_orientations/2) dr -= num_orientations;
 float d = ((float*)best_scores[j][k]->imageData)[i] - (((float*)scores[s][r]->imageData)[i]+ds*ws+SQR(ds)*wss+ dr*wo+SQR(dr)*woo);
assert(d > -.001 && d < .001);
#endif
      }
    }
  }
  free(cache_table_buff);


  // Cleanup
  for(j = 0; j < num_scales; j++) {
    for(k = 0; k < num_orientations; k++) {
      cvReleaseImage(&best_orientation_scores[j][k]); 
      cvReleaseImage(&best_child_orientations[j][k]); 
    }
  }
  free(child_scores_p);  
  free(best_child_inds_p);  
  free(best_child_scores_p);
  free(best_orientation_scores);
  free(best_child_orientations);

  return best_scores;
}
