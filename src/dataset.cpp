/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#include "dataset.h"
#include "classes.h"
#include "class.h"
#include "question.h"
#include "imageProcess.h"
#include "pose.h"
#include "part.h"
#include "attribute.h"
#include "spatialModel.h"
#include "interactiveLabel.h"
#include "structured_svm_partmodel.h"
#include "kmeans.h"
#include "svgPlotter.h"
#include "visualizationTools.h"
#include "histogram.h"
#include "fisher.h"

#include <time.h>
#include <stdio.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

class PartLocalizedStructuredSVMWithUserResponses : public PartModelStructuredSVM {
 public:
  PartLocalizedStructuredSVMWithUserResponses(Classes *c) : PartModelStructuredSVM() { classes = c; }
  StructuredLabel *NewStructuredLabel(StructuredData *x) { return new PartLocalizedStructuredLabelWithUserResponses(x); }
};

class MultiObjectStructuredSVMWithUserResponses : public MultiObjectStructuredSVM {
 public:
  MultiObjectStructuredSVMWithUserResponses(Classes *c) : MultiObjectStructuredSVM() { classes = c; }
  StructuredLabel *NewStructuredLabel(StructuredData *x) { return new MultiObjectLabelWithUserResponses(x); }
};

bool Dataset::Load(const char *fname) {
  if(examples) delete examples;
  MultiObjectStructuredSVMWithUserResponses loader(classes);
  return((examples = loader.LoadDataset(fname)) != NULL);
}

bool Dataset::Save(const char *fname) {
  MultiObjectStructuredSVMWithUserResponses saver(classes);
  return saver.SaveDataset(examples, fname);
}

Dataset *Dataset::ExtractSubset(int *ids, int id, bool inPlace) {
  if(inPlace) {
    int num = 0;
    for(int i = 0; i < NumExamples(); i++) {
      if(ids[i] == id)
	examples->examples[num++] = examples->examples[i];
      else
	delete examples->examples[i];
    }
    examples->num_examples = num;
    return this;
  } else {
    MultiObjectStructuredSVMWithUserResponses loader(classes);
    Dataset *retval = new Dataset(classes);
    for(int i = 0; i < NumExamples(); i++) {
      if(!ids || ids[i] == id)
	retval->AddExample(loader.CopyExample(GetExampleData(i),GetExampleLabel(i),GetExampleLatentLabel(i)));
    }
    return retval;
  }
}

void Dataset::AssignZeroOrientations() {
  for(int i = 0; i < NumExamples(); i++) {
    for(int o = 0; o < GetExampleLabel(i)->NumObjects(); o++) {
      PartLocation *locs = GetExampleLabel(i)->GetObject(o)->GetPartLocations();
      for(int j = 0; j < classes->NumParts(); j++) {
        float x, y, scale, rot;
        const char *pose;
        if(!locs[j].IsLatent()) {
          locs[j].GetImageLocation(&x, &y, &scale, &rot, &pose);
          locs[j].SetImageLocation(x, y, scale, 0, pose);
        }
      }
    }
  }
}

void Dataset::ApplyScaleConversion(float minWidth) {
  float maxScale = 0, minScale = 10000;
  for(int i = 0; i < NumExamples(); i++) {
    for(int o = 0; o < GetExampleLabel(i)->NumObjects(); o++) {
      PartLocation *locs = GetExampleLabel(i)->GetObject(o)->GetPartLocations();
      float sumDists = 0;
      int num = 0;
      for(int j = 0; j < classes->NumParts(); j++) {
        float x, y, child_x, child_y;
        int pose, child_pose;
        if(!locs[j].IsLatent()) {
  	  locs[j].GetImageLocation(&x, &y);
	  locs[j].GetDetectionLocation(NULL, NULL, NULL, NULL, &pose);
	  if(!classes->GetPart(j)->GetPose(pose)->IsNotVisible()) {
	    for(int k = 0; k < classes->GetPart(j)->NumParts(); k++) {
	      locs[classes->GetPart(j)->GetPart(k)->Id()].GetImageLocation(&child_x, &child_y);
	      if(!IS_LATENT(child_x) && !IS_LATENT(child_y)) {
	        locs[classes->GetPart(j)->GetPart(k)->Id()].GetDetectionLocation(NULL, NULL, NULL, NULL, &child_pose);
	        if(!classes->GetPart(j)->GetPart(k)->GetPose(child_pose)->IsNotVisible()) {
	  	  sumDists += sqrt(SQR(x-child_x)+SQR(y-child_y));
		  num++;
	        }
	      }
	    }
	  }
        }
      }
    
      float sz = sumDists / num;
      float scale = sz / (float)minWidth;
      maxScale = my_max(maxScale, scale);
      minScale = my_min(minScale, scale);
      assert(!isnan(scale));
    
      for(int j = 0; j < classes->NumParts(); j++) {
	float x, y, rot;
	const char *pose;
	locs[j].GetImageLocation(&x, &y, NULL, &rot, &pose);
	locs[j].SetImageLocation(x, y, my_max(1,scale), rot, pose);
      }
    }
  }
  fprintf(stderr, "max scale was %f\n", maxScale);
  fprintf(stderr, "min scale was %f\n", minScale);
}


float ***Dataset::ComputeSegmentationMasks(int part_width, int part_height, bool useMirroredPoses, int *num) {
  int numObjects = 0;
  int m = (useMirroredPoses ? 2 : 1);
  for(int i = 0; i < NumExamples(); i++) 
    numObjects += GetExampleLabel(i)->NumObjects(); 

  fprintf(stderr, "Computing part segmentation masks...");

  float ***pts = Create3DArray<float>(classes->NumParts(), numObjects*m, part_width*part_width);
  int *inds = new int[NumExamples()];
  for(int i = 0, ind = 0; i < NumExamples(); i++) {
    inds[i] = ind;
    ind += GetExampleLabel(i)->NumObjects();
  }

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for(int i = 0; i < NumExamples(); i++) {
    int ii = inds[i];
    ImageProcess *process = GetExampleData(i)->GetProcess(classes);
    FeatureOptions *fo = process->Features();
    for(int o = 0; o < GetExampleLabel(i)->NumObjects(); o++, ii++, numObjects++) {
      PartLocation *locs = GetExampleLabel(i)->GetObject(o)->GetPartLocations();

      for(int j = 0; j < classes->NumParts(); j++) {
	int x, y, scale, rot, pose;
	locs[j].GetDetectionLocation(&x, &y, &scale, &rot, &pose);
	if(pose) {
	  int ind = ii*m;
	  if(num) {
	    ind = num[j]; 
	    num[j] += m;
	  }
	  fo->GetSegmentationMask(&locs[j], pts[j][ind], part_width, part_height, false, 
				  classes->GetPart(j)->GetColor());
	  if(useMirroredPoses) 
	    fo->GetSegmentationMask(&locs[j], pts[j][ind+1], part_width, part_height, true, 
				  classes->GetPart(j)->GetColor());
	}
      }
    }
    fo->Clear();
  }
  fprintf(stderr, "\n");
  delete [] inds;
  
  return pts;
}

float ***Dataset::ClusterPosesBySegmentationMasks(int numPoses, int part_width, int part_height, bool useMirroredPoses, const char *maskDir) {
  int num[1000] = {0};
  float ***pts = ComputeSegmentationMasks(part_width, part_height, useMirroredPoses, num);
  float ***centers = Create3DArray<float>(classes->NumParts(), numPoses, part_width*part_height);
  int numObjects = 0;
  for(int i = 0; i < NumExamples(); i++) 
    numObjects += GetExampleLabel(i)->NumObjects(); 
  if(useMirroredPoses) {
    assert(numPoses%2 == 0);
    numObjects *= 2;
  }


#ifdef USE_OPENMP
  #pragma omp parallel for
#endif
  for(int j = 0; j < classes->NumParts(); j++) {
    int *poses = KMeans<float>(pts[j], num[j], centers[j], numPoses, part_width*part_height, 
			       2, PointDistanceL2_sqr, useMirroredPoses);
    delete [] poses;
    
    if(maskDir) {
      char fname[1000];
      IplImage *seg = cvCreateImage(cvSize(part_width,part_height),IPL_DEPTH_8U,1);
      for(int k = 0; k < numPoses; k++) {
	unsigned char *ptr = (unsigned char*)seg->imageData;
	int x, y, i = 0;
	for(y = 0; y < part_height; y++, ptr += seg->widthStep)
	  for(x = 0; x < part_width; x++, i++)
	    ptr[x] = (unsigned char)(centers[j][k][i]*255);
	sprintf(fname, "%s/%s_%d.png", maskDir, classes->GetPart(j)->Name(), k+1);
	cvSaveImage(fname, seg);
	classes->GetPart(j)->GetPose(k+1)->SetSegmentation(fname);
      }
      cvReleaseImage(&seg);
    }
  }
  free(pts);

  if(useMirroredPoses) {
    for(int j = 0; j < classes->NumParts(); j++) {
      ObjectPart *part = classes->GetPart(j);
      if(part->GetFlipped() && part->Id() < part->GetFlipped()->Id()) {
	for(int k = 0; k < numPoses; k+=2) {
	  float *tmp = centers[j][k+1];
	  centers[j][k+1] = centers[part->GetFlipped()->Id()][k+1];
	  centers[part->GetFlipped()->Id()][k+1] = tmp;
	}
      }
    }
  }

  return centers;
}

void Dataset::AssignPosesBySegmentationMasks(float ***centers, int numPoses, int part_width, int part_height, bool assignParentParts) {
  float ***pts = ComputeSegmentationMasks(part_width, part_height, false, NULL);
  int i, j, ii;

  // Assumes the 1st pose for each part is the non-visible pose
  for(i = 0, ii = 0; i < NumExamples(); i++) {
    for(int o = 0; o < GetExampleLabel(i)->NumObjects(); o++, ii++) {
      PartLocation *locs = GetExampleLabel(i)->GetObject(o)->GetPartLocations();

      for(j = 0; j < classes->NumParts(); j++) {
	if(!assignParentParts && classes->GetPart(j)->NumParts())
	  continue;
	int x, y, scale, rot, pose;
	locs[j].GetDetectionLocation(&x, &y, &scale, &rot, &pose);
	if(pose) {
	  int p = NearestCluster(pts[j][ii], centers[j], numPoses, part_width*part_height);
	  locs[j].SetDetectionLocation(x, y, scale, rot, p+1, LATENT, LATENT);
	}
      }
    }
  }
  free(pts);
}

float **Dataset::ComputeGroupOffsets(int par, float nonVisibleCost, bool useMirroredPoses) {
  int i, ii;
  int numObjects = 0;
  for(i = 0; i < NumExamples(); i++) 
    numObjects += GetExampleLabel(i)->NumObjects(); 


  // A "group" is a set of parts (e.g., head or body).  There will be one group for every non-leaf node in the tree
  // The feature vector for each group will be of size numParts*3, where for each child part the first 2 entries are
  // x,y offsets and the 3rd is 0 if both parent and child are visible and nonVisibleCost otherwise.  In this
  // case, nonVisibleCost is a constant used to tradeoff the relative importance of visibility compared to Euclidean distance.
  int m = (useMirroredPoses ? 2 : 1);
  ObjectPart *part = classes->GetPart(par);
  float **group_pts = Create2DArray<float>(numObjects*m, part->NumParts()*3);
  for(i = 0, ii = 0; i < NumExamples(); i++) {
    for(int o = 0; o < GetExampleLabel(i)->NumObjects(); o++, ii++) {
      PartLocation *locs = GetExampleLabel(i)->GetObject(o)->GetPartLocations();
      ComputeExampleGroupOffsets(locs, par, group_pts[ii*m], useMirroredPoses ? group_pts[ii*m+1] : NULL, nonVisibleCost, 
				 GetExampleData(i)->Width(), GetExampleData(i)->Height());
    }
  }

  return group_pts;
}



void Dataset::ComputeExampleGroupOffsets(PartLocation *locs, int par, float *group_pts, float *group_pts_mirrored, 
					 float nonVisibleCost, int w, int h) {
  ObjectPart *part = classes->GetPart(par);
  int curr = 0, inds[10000];
  for(int c = 0; c < part->NumParts(); c++) {
    int x, y, scale, rot, pose;
    int j = part->GetPart(c)->Id();
    locs[j].GetDetectionLocation(&x, &y, &scale, &rot, &pose);
    bool isVisible = pose >= 0 && !part->GetPart(c)->GetPose(pose)->IsNotVisible();
    if(!isVisible) {
      group_pts[curr*3] = group_pts[curr*3+1] = 0;
      group_pts[curr*3+2] = nonVisibleCost;
    } else {
      int par_x, par_y, par_scale, par_rot;
      float x0, y0;
      locs[par].GetDetectionLocation(&par_x, &par_y, &par_scale, &par_rot);
      classes->ConvertDetectionCoordinates(x, y, scale, rot, par_scale, par_rot, w, h, &x0, &y0);
      group_pts[curr*3]   = x0 - par_x;  // x offset from the parent
      group_pts[curr*3+1] = y0 - par_y;  // y offset from the parent
      group_pts[curr*3+2] = 0;
    }
    inds[j] = curr;
    curr++;
  }

  if(group_pts_mirrored) {
    for(int c = 0; c < part->NumParts(); c++) {
      // handle parts like the left and right eye, which get swapped when the model is mirrored
      int j = part->GetPart(c)->Id();
      int src = part->GetPart(c)->GetFlipped() ? part->GetPart(c)->GetFlipped()->Id() : j;
      
      group_pts_mirrored[inds[j]*3] = -group_pts[inds[src]*3];
      group_pts_mirrored[inds[j]*3+1] = group_pts[inds[src]*3+1];
      group_pts_mirrored[inds[j]*3+2] = group_pts[inds[src]*3+2];
    }
  }
}

float **Dataset::ClusterPosesByOffsets(int par, int numPoses, float nonVisibleCost, bool useMirroredPoses) {
  ObjectPart *part = classes->GetPart(par);
  float **group_pts = ComputeGroupOffsets(par, nonVisibleCost, useMirroredPoses);
  float **centers = Create2DArray<float>(numPoses, part->NumParts()*3);
  int numObjects = 0;
  for(int i = 0; i < NumExamples(); i++) 
    numObjects += GetExampleLabel(i)->NumObjects(); 
  if(useMirroredPoses) {
    assert(numPoses%2 == 0);
    numObjects *= 2;
  }

  int *poses = KMeans<float>(group_pts, numObjects, centers, numPoses, 3*part->NumParts(), 
			     1, PointDistanceL2_sqr, useMirroredPoses);
  delete [] poses;
  free(group_pts);

  return centers;
}

void Dataset::AssignPosesByOffsets(int par, float **centers, int numPoses, float nonVisibleCost, bool assignChildren) {
  float **group_pts = ComputeGroupOffsets(par, nonVisibleCost, false);
  int x, y, scale, rot, pose, i, k, ii;
  ObjectPart *part = classes->GetPart(par);

  // Assumes the 1st pose for each part is the non-visible pose
  for(i = 0, ii = 0; i < NumExamples(); i++) {
    for(int o = 0; o < GetExampleLabel(i)->NumObjects(); o++, ii++) {
      PartLocation *locs = GetExampleLabel(i)->GetObject(o)->GetPartLocations();
      int p = NearestCluster(group_pts[ii], centers, numPoses, 3*part->NumParts());
      if(assignChildren) {
	for(k = 0; k < part->NumParts(); k++) {
	  if(!locs[part->GetPart(k)->Id()].IsLatent() && !part->GetPart(k)->NumParts()) {
	    locs[part->GetPart(k)->Id()].GetDetectionLocation(&x, &y, &scale, &rot, &pose);
	    if(pose) locs[part->GetPart(k)->Id()].SetDetectionLocation(x, y, scale, rot, p+1, LATENT, LATENT);
	  }
	}
      }
      locs[par].GetDetectionLocation(&x, &y, &scale, &rot, &pose);
      if(pose) locs[par].SetDetectionLocation(x, y, scale, rot, p+1, LATENT, LATENT);
    }
  }
  free(group_pts);
}


void Dataset::BuildCroppedImages(const char *srcString, const char *dstString) {
  CvRect from;
  for(int i = 0; i < NumExamples(); i++) {
    IplImage *img = GetExampleData(i)->GetProcess(classes)->Image();
    const char *srcName = GetExampleData(i)->GetProcess(classes)->ImageName();
    for(int o = 0; o < GetExampleLabel(i)->NumObjects() && o < 1; o++) {
      GetExampleLabel(i)->GetObject(o)->GetBoundingBox(&from.x, &from.y, &from.width, &from.height);
      IplImage *cropped = cvCreateImage(cvSize(from.width,from.height), img->depth, img->nChannels);
      DrawImageIntoImage(img, from, cropped, 0, 0);
      char *fname = StringReplace(srcName, srcString, dstString);
      assert(strcmp(fname, srcName));
      cvSaveImage(fname, cropped);
      free(fname);
      cvReleaseImage(&cropped);
    }
  }
}


void Dataset::LearnPartMinimumSpanningTree(float nonVisbleCost) {
  int partPts[200][3]; 
  float partPairwiseDistances[200][200]; 
  int num_visible[200];
  assert(classes->NumParts() < 200);

  for(int j = 0; j < classes->NumParts(); j++) {
    for(int k = 0; k < classes->NumParts(); k++) 
      partPairwiseDistances[j][k] = 0;
    num_visible[j] = 0;
  }

  // Compute part pairwise distances
  int pose;
  for(int i = 0; i < NumExamples(); i++) {
    for(int o = 0; o < GetExampleLabel(i)->NumObjects(); o++) {
      PartLocation *locs = GetExampleLabel(i)->GetObject(o)->GetPartLocations();
      for(int j = 0; j < classes->NumParts(); j++) {
	locs[j].GetDetectionLocation(&partPts[j][0], &partPts[j][1], NULL, NULL, &pose); 
	if(classes->GetPart(j)->GetPose(pose)->IsNotVisible()) {
	  partPts[j][0] = partPts[j][1] = 0;
	  partPts[j][2] = 1;
	} else {
	  partPts[j][2] = 0;
	  num_visible[j]++;
	}
      }
      for(int j = 0; j < classes->NumParts(); j++) {
	for(int k = 0; k < classes->NumParts(); k++) {
	  partPairwiseDistances[j][k] += partPts[j][2] == partPts[k][2] ? 
	    SQR(partPts[j][0]-partPts[k][0])+SQR(partPts[j][1]-partPts[k][1]) : nonVisbleCost;
	}
      }
    }
  }

  // Create a part tree as a minimum spanning tree between part distances.  Assume parts[0] is the root part
  int root = 0;
  for(int j = 1; j < classes->NumParts(); j++)  // use the part that is visible in the most examples as the root of the tree
    if(num_visible[j] > num_visible[root])
      root = j;
  bool partUsed[100];
  float minDist[100], bestDist;
  int closestPart[100], best;
  partUsed[root] = true;
  for(int j = 0; j < classes->NumParts(); j++) {
    partUsed[j] = j != root;
    minDist[j] = partPairwiseDistances[root][j];
    closestPart[j] = root;
  }
  for(int j = 1; j < classes->NumParts(); j++) {
    bestDist = INFINITY;
    best = -1;
    for(int k = 0; k < classes->NumParts(); k++) {
      if(!partUsed[k] && minDist[k] < bestDist) {
	best = k;
	bestDist = minDist[k];
      }
    }
    classes->GetPart(closestPart[best])->AddPart(classes->GetPart(best));
    partUsed[best] = true;
    for(int k = 0; k < classes->NumParts(); k++) {
      if(!partUsed[k] && partPairwiseDistances[best][k] < minDist[k]) {
	minDist[k] = partPairwiseDistances[best][j];
	closestPart[k] = best;
      }
    }
  }

  // Topollogically sort part tree array
  int inds[200];
  classes->TopologicallySortParts(inds);
  for(int i = 0; i < NumExamples(); i++) {
    for(int o = 0; o < GetExampleLabel(i)->NumObjects(); o++) {
      PartLocation *locs = GetExampleLabel(i)->GetObject(o)->GetPartLocations();
      PartLocation *locs_old = PartLocation::CopyPartLocations(locs);
      for(int j = 0; j < classes->NumParts(); j++) {
	locs[j] = locs_old[inds[j]];
	locs[j].SetPart(classes->GetPart(j));
      }
      delete [] locs_old;
    }
  }
}


typedef struct {
  ObjectPart *part;
  Dataset *dataset;
  float *w;
  int numExamples;
} PartDataset;


float example_neg_log_likelihood_normalized(ObjectPartInstance *inst, ObjectPartInstance *clickInst, ImageProcess *process, PartLocation *locs, PartLocation *click_loc) {
  float ex_ll;
  Classes *classes = process->GetClasses();
  int nw = classes->NumWeights(true, false);
  float *f = (float*)malloc(sizeof(float)*nw*2);
  float ll = 0, ll_click = 0;

  inst->ComputeCumulativeSums(true);  // computes the delta parameter that normalizes the probabilities
    
  // Solve for the component of the log likelihood w.r.t. this image
  ll = inst->GetScoreAt(&locs[inst->Id()]);
  assert(ll <= inst->GetMaxScore());

  ex_ll = inst->Model()->GetGamma()*my_max(ll+ll_click, inst->GetLowerBound())+inst->GetDelta();
  //fprintf(stderr, "%s: ex_ll=%g ll=%g ll_click=%g, gamma=%g delta=%g, lower=%f, max=%f\n", process->ImageName(), 
  //	  ex_ll, ll, ll_click, inst->Model()->GetGamma(), inst->GetDelta(), inst->GetLowerBound(), inst->GetBestLoc().GetScore());
  if(ex_ll > 0) {
    fprintf(stderr, "error ll=%f, gamma=%f\n", ex_ll, inst->Model()->GetGamma()); assert(ex_ll <= 0);
    inst->ComputeCumulativeSums(true);
    ex_ll = 0;
  }
  
  free(f);
  inst->ClearCumulativeSums();

  return ex_ll;
}

// Computes the log likelihood of the dataset for a specified value of gamma and gamma_click
float gamma_neg_log_likelihood_normalized(float gamma, const void *p) {
  ObjectPart *part = ((PartDataset*)p)->part;
  Dataset *dataset = ((PartDataset*)p)->dataset;
  float sum = 0;
  int id = part->Id();
  int numExamples = ((PartDataset*)p)->numExamples;
  Classes *classes = part->GetClasses();
  part->SetGamma(gamma);

#ifdef USE_OPENMP
  omp_lock_t my_lock;
  omp_init_lock(&my_lock);
  #pragma omp parallel for
#endif
  for(int i = 0; i < numExamples; i++) {
    for(int o = 0; o < dataset->GetExampleLabel(i)->NumObjects(); o++) {
      float ex_ll = example_neg_log_likelihood_normalized(dataset->GetExampleData(i)->GetProcess(classes)->GetPartInst(id), NULL, 
							dataset->GetExampleData(i)->GetProcess(classes),
							dataset->GetExampleLabel(i)->GetObject(o)->GetPartLocations(), NULL);
    
#ifdef USE_OPENMP         
      omp_set_lock(&my_lock);
#endif
      
      sum += ex_ll;
  
#ifdef USE_OPENMP         
      omp_unset_lock(&my_lock);
#endif
    }
  }

fprintf(stderr, "%f: %f\n", gamma, sum);
  
  return -sum;
}

// For this particular value of gamma_click, solves for the value of gamma that maximizes the log likelihood 
// of the dataset
float click_gamma_neg_log_likelihood_normalized(float gamma_click, const void *p) {
  Dataset *dataset = ((PartDataset*)p)->dataset;
  float *w = ((PartDataset*)p)->w;
  int i;
  int numExamples = ((PartDataset*)p)->numExamples;
  PartDataset p2 = *((PartDataset*)p); p2.part = NULL;
  float sum = 0;
  Classes *classes = dataset->GetClasses();
  float *W = (float*)malloc(sizeof(float)*classes->NumWeights(true,false));

  // Scale the part click Gaussian parameters by gamma_click
  float *ptr = w;
  for(int j = 0; j < classes->NumSpatialTransitions(); j++) {
    int num = classes->GetSpatialTransition(j)->NumWeights();
    if(classes->GetSpatialTransition(j)->IsClick()) {
      for(i = 0; i < num; i++)
	W[i] = ptr[i]*gamma_click;
      classes->GetSpatialTransition(j)->SetWeights(W);
    }
    ptr += num;
  }
  free(W);
  
  // Recompute the probability map for each training example with the new gamma_click
#ifdef USE_OPENMP
  omp_lock_t my_lock;
  omp_init_lock(&my_lock);
  #pragma omp parallel for
#endif
  for(int ii = 0; ii < numExamples; ii++) {
    ImageProcess *process = dataset->GetExampleData(ii)->GetProcess(classes);
    for(int o = 0; o < dataset->GetExampleLabel(i)->NumObjects(); o++) {
      float ex_ll = 0;
      for(int jj = 0; jj < classes->NumClickParts(); jj++) {
        ObjectPartInstance *inst = process->GetClickPartInst(jj);	
        ObjectPartInstance *part = inst->GetParent();
        for(int k = 0; k < dataset->GetExampleLabel(ii)->GetObject(o)->NumUsers() && k < 1; k++) {
  	  PartLocation *clickResponses = dataset->GetExampleLabel(ii)->GetObject(o)->GetPartClickLocations(k);
	  if(clickResponses) {
	    inst->SetClickPoint(clickResponses+jj);
	    ex_ll += example_neg_log_likelihood_normalized(process->GetPartInst(jj), inst, process,
							 dataset->GetExampleLabel(ii)->GetObject(o)->GetPartLocations(), 
							 clickResponses+jj);
	    for(int j = 0; j < part->Model()->NumPoses(); j++) 
	      part->GetPose(j)->FreeUnaryExtra();
	  }
        }
        for(int j = 0; j < part->Model()->NumPoses(); j++)
	  part->GetPose(j)->Clear(true, true, false);
      }

#ifdef USE_OPENMP         
      omp_set_lock(&my_lock);
#endif

      sum += ex_ll;

#ifdef USE_OPENMP         
      omp_unset_lock(&my_lock);
#endif
    }
  }

  fprintf(stderr, "%f: %f\n", gamma_click, sum);

  return -sum;
}


typedef struct {
  Dataset *dataset;
  int attribute_ind;
  int num_examples;
} DatasetAttribute;

float gamma_class_neg_log_likelihood_normalized(float gamma_class, const void *p) {
  double ll = 0;
  Dataset *dataset = ((DatasetAttribute*)p)->dataset;
  int num_examples= ((DatasetAttribute*)p)->num_examples;
  Classes *classes = dataset->GetClasses();
  classes->SetClassGamma(gamma_class);

#ifdef USE_OPENMP
  omp_lock_t my_lock;
  omp_init_lock(&my_lock);
  #pragma omp parallel for
#endif
  for(int i = 0; i < num_examples; i++) {
    for(int o = 0; o < dataset->GetExampleLabel(i)->NumObjects(); o++) {
      double classLikelihoods[50000];
      ImageProcess *process = dataset->GetExampleData(i)->GetProcess(classes);
      process->ComputeClassLogLikelihoodsAtLocation(classLikelihoods, dataset->GetExampleLabel(i)->GetObject(o)->GetPartLocations());
      float l = classLikelihoods[dataset->GetExampleLabel(i)->GetObject(o)->GetClassID()];

#ifdef USE_OPENMP         
      omp_set_lock(&my_lock);
#endif
      ll += l;
#ifdef USE_OPENMP         
      omp_unset_lock(&my_lock);
    }
#endif
  }
  return (float)-ll;
}

float gamma_attribute_neg_log_likelihood_normalized(float gamma_attribute, const void *p) {
  double ll = 0;
  Dataset *dataset = ((DatasetAttribute*)p)->dataset;
  int attr_ind     = ((DatasetAttribute*)p)->attribute_ind;
  int num_examples = ((DatasetAttribute*)p)->num_examples;
  Classes *classes = dataset->GetClasses();
  classes->GetAttribute(attr_ind)->SetGamma(gamma_attribute);

#ifdef USE_OPENMP
  omp_lock_t my_lock;
  omp_init_lock(&my_lock);
#pragma omp parallel for
#endif
  for(int i = 0; i < num_examples; i++) {
    ImageProcess *process = dataset->GetExampleData(i)->GetProcess(classes);
    AttributeInstance *a_i = process->GetAttributeInst(classes->GetAttribute(attr_ind)->Id());
    for(int o = 0; o < dataset->GetExampleLabel(i)->NumObjects(); o++) {
      float l = a_i->GetLogLikelihoodAtLocation(dataset->GetExampleLabel(i)->GetObject(o)->GetPartLocations()+a_i->Model()->Part()->Id(), 
  					        dataset->GetExampleLabel(i)->GetObject(o)->GetClass()->GetAttributeWeight(attr_ind));

#ifdef USE_OPENMP         
      omp_set_lock(&my_lock);
#endif
      ll += l;
#ifdef USE_OPENMP         
      omp_unset_lock(&my_lock);
#endif
    }
  }
  return (float)-ll;
}

int f_cmp(const void *f1, const void *f2) { 
  float d = *((float*)f1) - *((float*)f2);
  return d < 0 ? -1 : (d > 0 ? 1 : 0); 
}

// Learn parameters to 1) tradeoff classification score and user click location scores, and 2) convert
// scores into probabilities.  For any given part or part click node, we wish to learn parameters gamma_p
// and gamma_click_p, where argmax_{Theta|theta_p_click} is the maximum likelihood position of all parts
// given a click location theta_p_click
// p(theta_p_click) = gamma_p*[w*psi(x,theta_p,argmax_{Theta|theta_p_click})+gamma_click_p*log(p(theta_p_click|theta_p)]
void Dataset::LearnPartDetectionParameters(int numEx) {
  //numEx=1;
  if(numEx > NumExamples()) numEx = NumExamples();

#ifdef USE_OPENMP
  #pragma omp parallel for
#endif
  for(int ii = 0; ii < numEx; ii++) { 
    // Precompute the part/pose detection scores and feature responses, free everything else
    GetExampleData(ii)->GetProcess(classes)->SetBidirectional(true);
    GetExampleData(ii)->GetProcess(classes)->SetComputeClickParts(true);
    GetExampleData(ii)->GetProcess(classes)->Detect(); 
    GetExampleData(ii)->GetProcess(classes)->Clear(true, false, false, false);
  }

  // Solve for a global gamma parameter to scale the detection scores such that probabilities sum to 1 and
  // the log likelihood is maximized
  float gamma;
  PartDataset d; d.dataset = this; d.part = classes->GetPart(classes->NumParts()-1); d.numExamples = numEx;
  gamma = line_search<float>(0.000001f, 10000, &d, gamma_neg_log_likelihood_normalized, .0001f);
  fprintf(stderr, "Detection probability scaling parameter is %f\n", gamma);

  for(int i = 0; i < classes->NumParts(); i++) {
    classes->GetPart(i)->SetGamma(gamma);
    if(classes->NumParts() == classes->NumClickParts()) classes->GetClickPart(i)->SetGamma(gamma);
  } 

  float *w = (float*)malloc(sizeof(float)*classes->NumWeights(true,false));
  int n = 0;
  for(int j = 0; j < classes->NumSpatialTransitions(); j++) {
    classes->GetSpatialTransition(j)->GetWeights(w+n);
    n += classes->GetSpatialTransition(j)->NumWeights();
  } 
  d.dataset = this; d.part = NULL; d.w = w; d.numExamples = numEx;
  float gamma_click=line_search<float>(0.01f, 10000, &d, click_gamma_neg_log_likelihood_normalized, .01f);
  classes->SetClickGamma(gamma_click);
  fprintf(stderr, "Scaling parameter for part click weights is %f ll=%f\n", gamma_click, 
	    click_gamma_neg_log_likelihood_normalized(gamma_click, &d));
  free(w);

  for(int ii = 0; ii < numEx; ii++) 
    GetExampleData(ii)->GetProcess(classes)->Clear();
}

void Dataset::LearnClassAttributeDetectionParameters(int num_ex) {
  DatasetAttribute d; 
  d.dataset = this; 
  d.num_examples = my_min(NumExamples(), num_ex);

  float gamma_class = line_search<float>(0.0000001f, 10000, &d, gamma_class_neg_log_likelihood_normalized, .01f);
  fprintf(stderr, "Class gamma parameter is %f\n", gamma_class);
  classes->SetClassGamma(gamma_class);

  for(int i = 0; i < classes->NumAttributes(); i++) {
    d.attribute_ind = i;
    if(classes->GetAttribute(i)->NumFeatureTypes()) {
      float gamma_attribute = line_search<float>(0.0000001f, 10000, &d, gamma_attribute_neg_log_likelihood_normalized, .01f);
      fprintf(stderr, "Attribute %s gamma parameter is %f\n", classes->GetAttribute(i)->Name(), gamma_attribute);
      classes->GetAttribute(i)->SetGamma(gamma_attribute);
    }
  }
}



// Learn gaussian parameters defining the mean and variance of a part click location w.r.t. 
// the ground truth part location
void Dataset::LearnUserClickProbabilities() {
  int i, l;
  Question *q;
  float *times = NULL;
  float *w = (float*)malloc(sizeof(float)*classes->NumWeights(true, false));
  PartLocation loc;

  // Compute variance parameters for click parts
  classes->AddSpatialTransitions(this, 1, true, false);
  classes->ComputePartPoseTransitionWeights(this, true, false);

  for(i = 0; i < classes->NumParts(); i++) {
    int numGood = 0;
    PartLocation *locs;
    for(l = 0; l < NumExamples(); l++) {
      for(int o = 0; o < GetExampleLabel(i)->NumObjects(); o++) {
        for(int k = 0; k < GetExampleLabel(l)->GetObject(o)->NumUsers(); k++) {
	  locs = GetExampleLabel(l)->GetObject(o)->GetPartClickLocations(k);
	  if(locs) {
	    loc.Copy(locs[i]);
            times = (float*)realloc(times, sizeof(float)*(numGood+1));
	    times[numGood++] = loc.GetResponseTime();
	  }
        }
      }
    }
 
    // Set expected time to answer as the median time
    qsort(times, numGood, sizeof(float), f_cmp);
    if(numGood) {
      q = classes->GetClickPart(i)->GetQuestion(); assert(q != NULL);
      q->SetExpectedSecondsToAnswer(times[numGood/2]);
    }
  }
  free(w);
  free(times);
}

// Learn class-attribute-certainty probabilities p(a,r|c) for each class c, where a is 
// an attribute value and r is a certainty.  This is done using a training set of user
// responses.  beta_class is a beta prior with mean p(a|c) and beta_cert[k] is a beta
// prior with mean p(a,r)
void Dataset::LearnClassAttributeUserProbabilities(float beta_class, float *beta_cert) {
  int i, j, k, c, user;
  float num_class, num_attr;
  float **pos, **neg;
  Question *q;
  float w[1000], maxW = 0;
  float **times = (float**)malloc(sizeof(float*)*classes->NumAttributes());
  int *numGood = (int*)malloc(sizeof(int)*classes->NumAttributes());
  memset(numGood, 0, sizeof(int)*classes->NumAttributes());
  for(i = 0; i < classes->NumAttributes(); i++)
    times[i] = NULL;

  // Learn global attribute priors p(a,r) and class attribute priors p(a|c), where a is an 
  // attribute, r is a certainty, and c is a class
  float *globalAttributePriors = (float*)malloc(sizeof(float)*classes->NumAttributes());   // p(a)
  float *numAttributeExamples = (float*)malloc(sizeof(float)*classes->NumAttributes());
  float **globalCertaintyAttributePriors = (float**)malloc(sizeof(float*)*classes->NumCertainties());   // p(a,r)
  float **classAttributePriors = (float**)malloc(sizeof(float*)*classes->NumClasses());  // p(a|c)
  float **numClassAttributeExamples = (float**)malloc(sizeof(float*)*classes->NumClasses());  
  float **numCertaintyAttributeExamples = (float**)malloc(sizeof(float*)*classes->NumCertainties());
  for(i = 0; i < classes->NumCertainties(); i++) { // allocate global priors
    w[i] = !strcmp(classes->GetCertainty(i), "not visible") ? 0 : 1/(beta_cert[i]+.5);
    if(w[i] > maxW) maxW = w[i];
    globalCertaintyAttributePriors[i] = (float*)malloc(2*sizeof(float)*classes->NumAttributes());
    numCertaintyAttributeExamples[i] = (float*)malloc(2*sizeof(float)*classes->NumAttributes());
    for(j = 0; j < 2*classes->NumAttributes(); j++) 
      numCertaintyAttributeExamples[i][j] = globalCertaintyAttributePriors[i][j] = 0;
  } 
  for(j = 0; j < classes->NumAttributes(); j++) 
    numAttributeExamples[j] = globalAttributePriors[j] = 0;
  for(i = 0; i < classes->NumCertainties(); i++) 
    w[i] /= maxW;
  classes->SetCertaintyWeights(w);

  for(i = 0; i < classes->NumClasses(); i++) {    // allocate class priors
    classAttributePriors[i] = (float*)malloc(sizeof(float)*classes->NumAttributes());
    numClassAttributeExamples[i] = (float*)malloc(sizeof(float)*classes->NumAttributes());
    for(j = 0; j < classes->NumAttributes(); j++) 
      classAttributePriors[i][j] = numClassAttributeExamples[i][j] = 0;
  }
  int num_ex = 0;
  for(i = 0; i < NumExamples(); i++) {              // compute unnormalized global & class priors
    for(int o = 0; o < GetExampleLabel(i)->NumObjects(); o++) {
      c = GetExampleLabel(i)->GetObject(o)->GetClassID();
      for(j = 0; j < classes->NumAttributes(); j++) {
        for(user = 0; user < GetExampleLabel(i)->GetObject(o)->NumUsers(); user++) {
	  AttributeAnswer *resps = GetExampleLabel(i)->GetObject(o)->GetAttributes(user);
	  if(resps) {
	    globalCertaintyAttributePriors[resps[j].certainty][2*j+resps[j].answer] += 1;
	    if(resps[j].answer) globalAttributePriors[j] += w[resps[j].certainty];
	    numAttributeExamples[j] += w[resps[j].certainty];
	    if(resps[j].answer) classAttributePriors[c][j] += w[resps[j].certainty];
            times[j] = (float*)realloc(times[j], sizeof(float)*(numGood[j]+1));
	    times[j][numGood[j]++] = resps[j].responseTimeSec;
	    numClassAttributeExamples[c][j] += w[resps[j].certainty];
	    numCertaintyAttributeExamples[resps[j].certainty][j] += w[resps[j].certainty];
	    if(j == 0) num_ex++;
          }
	}
      }
    }
  }
  for(j = 0; j < 2*classes->NumAttributes(); j++) {
    for(i = 0; i < classes->NumCertainties(); i++) { // normalize global priors
      globalCertaintyAttributePriors[i][j] /= num_ex;
      if(j%2==1) {
        fprintf(stderr, "p(a,r): p(%s,%s)=%f p(~%s,%s)=%f\n", classes->GetAttribute(j/2)->Name(), classes->GetCertainty(i), 
          globalCertaintyAttributePriors[i][j], classes->GetAttribute(j/2)->Name(), classes->GetCertainty(i), globalCertaintyAttributePriors[i][j-1]);
      }
    }
  }
  for(j = 0; j < classes->NumAttributes(); j++) 
    globalAttributePriors[j] /= numAttributeExamples[j];  
  for(i = 0; i < classes->NumClasses(); i++) { // normalize class priors
    for(j = 0; j < classes->NumAttributes(); j++) {
      classAttributePriors[i][j] /= (numClassAttributeExamples[i][j]+.0000001f);
      fprintf(stderr, "p(a|c): p(%s|%s)=%f\n", classes->GetAttribute(j)->Name(), classes->GetClass(i)->Name(), 
	      classAttributePriors[i][j]);
    }
  }

  // Extract the median time duration as the expected duration of a question, 
  // for the purpose of discarding outliers.  Could instead consider discarding the
  // last 10%
  for(i = 0; i < classes->NumAttributes(); i++) {
    qsort(times[i], numGood[i], sizeof(float), f_cmp);
    q = classes->GetAttribute(i)->GetQuestion(); assert(q != NULL);
    q->SetExpectedSecondsToAnswer(times[i][numGood[i]/2]);
  }
  for(i = 0; i < classes->NumQuestions(); i++) {
    Question *q = classes->GetQuestion(i);
    if(!strcmp(q->Type(), "multiple_choice")) {
      int attr = ((MultipleChoiceAttributeQuestion*)q)->GetChoice(0);
      float tm = classes->GetAttribute(attr)->GetQuestion()->ExpectedSecondsToAnswer();
      q->SetExpectedSecondsToAnswer(tm);
    } else if(!strcmp(q->Type(), "batch")) {
      float tm = ((BatchQuestion*)q)->GetQuestion(0)->ExpectedSecondsToAnswer();
      q->SetExpectedSecondsToAnswer(tm);
    }
  }

  // Now compute the actual class-attribute-certainty probabilities p(a,r|c)
  float ***num = (float***)malloc(sizeof(float**)*classes->NumClasses());
  for(i = 0; i < classes->NumClasses(); i++) {  
    // initialize unnormalized class-attribute-certainty probabilities with prior terms added in
    num[i] = (float**)malloc(sizeof(float*)*classes->NumAttributes());
    neg = classes->GetClass(i)->GetAttributeNegativeUserProbs();
    pos = classes->GetClass(i)->GetAttributePositiveUserProbs();
    for(j = 0; j < classes->NumAttributes(); j++) {
      num_class = my_min(beta_class, numClassAttributeExamples[i][j]);
      num[i][j] = (float*)malloc(sizeof(float)*classes->NumCertainties());
      for(k = 0; k < classes->NumCertainties(); k++) {
        num_attr = my_min(beta_cert[k], NumExamples());
        num[i][j][k] = num_class + num_attr;
        neg[k][j] = (1-classAttributePriors[i][j])*num_class + globalCertaintyAttributePriors[k][2*j]*num_attr;
        pos[k][j] = classAttributePriors[i][j]*num_class + globalCertaintyAttributePriors[k][2*j+1]*num_attr;
      }
    }
  }
  for(i = 0; i < NumExamples(); i++) { // update unnormalized class-attribute-certainty probabilities
    for(int o = 0; o < GetExampleLabel(i)->NumObjects(); o++) {
      c = GetExampleLabel(i)->GetObject(o)->GetClass()->Id();
      neg = classes->GetClass(c)->GetAttributeNegativeUserProbs();
      pos = classes->GetClass(c)->GetAttributePositiveUserProbs();
      for(j = 0; j < classes->NumAttributes(); j++) {
        for(user = 0; user < GetExampleLabel(i)->GetObject(o)->NumUsers(); user++) {
	  AttributeAnswer *resps = GetExampleLabel(i)->GetObject(o)->GetAttributes(user);
	  if(resps) {
	    if(!resps[j].answer) neg[resps[j].certainty][j]++;
	    else pos[resps[j].certainty][j]++;
	    num[GetExampleLabel(i)->GetObject(o)->GetClass()->Id()][j][resps[j].certainty]++;
	  }
        }
      }
    }
  }
  for(i = 0; i < classes->NumClasses(); i++) { // normalize class-attribute-certainty probabilities
    neg = classes->GetClass(i)->GetAttributeNegativeUserProbs();
    pos = classes->GetClass(i)->GetAttributePositiveUserProbs();
    for(j = 0; j < classes->NumAttributes(); j++) {
      for(k = 0; k < classes->NumCertainties(); k++) {
        pos[k][j] /= (num[i][j][k]+1e-7);
	neg[k][j] /= (num[i][j][k]+1e-7);
        fprintf(stderr, "p(a,r|c) %s %s %s: %f %f\n", classes->GetClass(i)->Name(), classes->GetAttribute(j)->Name(),
          classes->GetCertainty(k), pos[k][j], neg[k][j]);
      }
    }
    for(j = 0; j < classes->NumAttributes(); j++) {
      num_attr = my_min(beta_class, numAttributeExamples[j]);
      classAttributePriors[i][j] = (classAttributePriors[i][j]*numClassAttributeExamples[i][j] + num_attr*globalAttributePriors[j])/(numClassAttributeExamples[i][j]+num_attr+1e-7);
      fprintf(stderr, "p(%s|%s)=%f\n", classes->GetAttribute(j)->Name(), classes->GetClass(i)->Name(), classAttributePriors[i][j]);
    }
    classes->GetClass(i)->SetAttributeUserProbabilities(classAttributePriors[i]);
  }

  // Cleanup
  for(i = 0; i < classes->NumClasses(); i++) {
    for(j = 0; j < classes->NumAttributes(); j++) 
      free(num[i][j]);
    free(num[i]);
    free(numClassAttributeExamples[i]);
  }
  for(i = 0; i < classes->NumCertainties(); i++)
    free(globalCertaintyAttributePriors[i]);
  for(i = 0; i < classes->NumAttributes(); i++)
    free(times[i]);
  free(num);
  free(globalCertaintyAttributePriors);
  free(globalAttributePriors);
  free(classAttributePriors);
  free(numClassAttributeExamples);
  free(times);
  free(numGood);
}

//void Dataset::LearnQuestionNormalization(float beta_class, float *beta_cert) {
//  float gamma=line_search<float>(0.000001f, 1, &d, gamma_neg_log_likelihood_normalized, .01f);


void Dataset::BuildCodebook(const char *dictionaryOutFile, const char *featName, 
			    int w, int h, int numWords, int maxImages, int ptsPerImage, int tree_depth, int resize_image_width) {
  SlidingWindowFeature *f = GetExampleData(0)->GetProcess(classes)->Features()->Feature(featName);
  assert(f);
  FeatureDictionary *d = new FeatureDictionary(featName, w, h, f->Params()->numBins, numWords, tree_depth);
  d->LearnDictionary(this, maxImages, ptsPerImage);
  d->Save(dictionaryOutFile);
  classes->AddCodebook(d);
}

void Dataset::BuildFisherCodebook(const char *dictionaryOutFile, const char *featName, 
				  int w, int h, int numWords, int maxImages, int ptsPerImage, int pcaDims, int resize_image_width) {
  SlidingWindowFeature *f = GetExampleData(0)->GetProcess(classes)->Features()->Feature(featName);
  assert(f);
  FisherFeatureDictionary *d = new FisherFeatureDictionary(featName, w, h, f->Params()->numBins, numWords, pcaDims);
  d->LearnDictionary(this, maxImages, ptsPerImage, resize_image_width);
  d->Save(dictionaryOutFile);
  classes->AddFisherCodebook(d);
}

void Dataset::EvaluateTestset20Q(int maxQuestions, double timeInterval, bool isInteractive, bool stopEarly, 
				 int isCorrectWindow, QuestionSelectMethod method, double *accuracy, 
				 double ***perQuestionConfusionMatrix, int **perQuestionPredictions, 
				 double ***perQuestionClassProbabilities, int **questionsAsked, 
				 int **responseTimes, int *numQuestionsAsked, int *gtClasses, 
				 bool disableClick, bool disableBinary, bool disableMultiple, bool disableCV, bool disableCertainty, 
				 const char *debugDir, int debugNumClassPrint, bool debugProbabilityMaps, 
				 bool debugClickProbabilityMaps, int debugNumSamples, bool debugQuestionEntropies, 
				 bool debugMaxLikelihoodSolution, const char *matlabProgressOut) {

  int num_so_far = 0;
  time_t last_time = time(NULL);
  double *accuracy_curr = (double*)malloc(sizeof(double)*(maxQuestions+1));
  int **perQuestionGT = Create2DArray<int>(maxQuestions+1, NumExamples());
  int **perQuestionPred = Create2DArray<int>(maxQuestions+1, NumExamples());
  const char **imageNames = Create1DArray<const char*>(NumExamples());
  char **linkNames = Create1DArray<char*>(NumExamples());
  bool debugSessions = debugNumClassPrint || debugProbabilityMaps || debugClickProbabilityMaps || 
    debugNumSamples || debugQuestionEntropies || debugMaxLikelihoodSolution;
  float *old_certainties = disableCertainty ? new float[classes->NumCertainties()*2] : NULL;
  if(disableCertainty) {
    memcpy(old_certainties, classes->GetCertaintyWeights(), sizeof(float)*classes->NumCertainties());
    for(int i = classes->NumCertainties(); i < classes->NumCertainties()*2; i++) old_certainties[i] = 1;
    classes->SetCertaintyWeights(old_certainties+classes->NumCertainties());
  }

  if(debugDir) 
    CreateDirectoryIfNecessary(debugDir);

  int curr = 0;
#ifdef USE_OPENMP
  omp_lock_t my_lock;
  omp_init_lock(&my_lock);
  //#pragma omp parallel for
#endif
  for(int ii = 0; ii < NumExamples(); ii++) {
#ifdef USE_OPENMP         
    omp_set_lock(&my_lock);
#endif
    int i = curr++;
#ifdef USE_OPENMP           
    omp_unset_lock(&my_lock); 
#endif
    assert(GetExampleLabel(i)->NumObjects() == 1);
    for(int o = 0; o < GetExampleLabel(i)->NumObjects(); o++) {
      int j, k;

      QuestionAskingSession *session = new QuestionAskingSession(GetExampleData(i)->GetProcess(classes), GetExampleLabel(i)->GetObject(o), isInteractive, method, 
							       debugSessions ? debugDir : NULL, debugNumClassPrint, debugProbabilityMaps, 
							       debugClickProbabilityMaps, debugNumSamples, debugQuestionEntropies, 
							       debugMaxLikelihoodSolution);
      session->DisableQuestions(disableClick, disableBinary, disableMultiple, disableCV);
      int class_id = GetExampleLabel(i)->GetObject(o)->GetClassID();
      if(gtClasses)
        gtClasses[i] = class_id;
      session->AskAllQuestions(maxQuestions, stopEarly);
    
#ifdef USE_OPENMP         
      omp_set_lock(&my_lock);
#endif

      if(debugDir) {
	char fname[1000], fname2[1000], thumb[1000], thumb2[1000], link[1000];
        char *str = (char*)malloc(300000);
	sprintf(link, "%s.html", GetExampleData(i)->GetProcess(classes)->Features()->Name());
        if(debugSessions)
          sprintf(str, "<a href=\"%s\">View Detailed Output</a><br>", link);
        else
          strcpy(str, "");
        session->PrintSummary(str+strlen(str), 1);

	sprintf(fname2, "%s_gt.png", GetExampleData(i)->GetProcess(classes)->Features()->Name());
	sprintf(fname, "%s/%s", debugDir, fname2);
	strcpy(thumb, fname2);
	StripFileExtension(thumb);
	sprintf(thumb+strlen(thumb), "_thumb.%s", GetFileExtension(fname2));
	IplImage *img = cvLoadImage(fname);
	sprintf(thumb2, "%s/%s", debugDir, thumb);
	MakeThumbnail(img, 75, 75, thumb2);
	cvReleaseImage(&img);
	GetExample(i)->AddExampleVisualization(fname2, thumb, str, (double)session->GetNumQuestionsAsked());
	imageNames[i] = GetExample(i)->visualization->fname;
	linkNames[i] = StringCopy(link);
        free(str);
      }

      int q = 0;
      double evalTime = 0;
      double sessionTime = 0;
      if(g_debug > 0) fprintf(stderr, "%d/%d:", (num_so_far+1), NumExamples());
      for(j = 0; j < maxQuestions; j++, evalTime += timeInterval) {
        if(!timeInterval) {
	  q = my_min(j, session->GetNumQuestionsAsked());  // plot # of questions asked on x-axis
        } else {
          if(q < session->GetNumQuestionsAsked() && // plot human labor (time) on x-axis
            evalTime > sessionTime+session->GetQuestionAsked(q+1)->GetResponseTime(session->GetQuestionResponse(q+1))) {
            q++;
            sessionTime += session->GetQuestionAsked(q)->GetResponseTime(session->GetQuestionResponse(q));
          }
        }

        ClassProb *probs = session->GetClassProbs(q);
        int pred = probs[0].classID;

        if(perQuestionConfusionMatrix) 
          perQuestionConfusionMatrix[j][class_id][pred]++;
        if(perQuestionPredictions)
          perQuestionPredictions[i][j] = pred;
	perQuestionPred[j][i] = pred;
	perQuestionGT[j][i] = class_id;
        if(perQuestionClassProbabilities)
          for(k = 0; k < classes->NumClasses(); k++)
            perQuestionClassProbabilities[i][j][probs[k].classID] = probs[k].prob;
        if(questionsAsked)
          questionsAsked[i][j] = j>0 && j<=session->GetNumQuestionsAsked() ? session->GetQuestionAsked(j-1)->Id() : -1;
        if(responseTimes) 
          responseTimes[i][j] = j>0 && j<=session->GetNumQuestionsAsked() ? 
        session->GetQuestionAsked(j-1)->GetResponseTime(session->GetQuestionResponse(j-1)) : 0;
        if(accuracy) {
          for(k = 0; k < isCorrectWindow; k++) {
            if(probs[k].classID == class_id)
              accuracy[j]++;
          }
          accuracy_curr[j] = accuracy[j]/(num_so_far+1);
          if(g_debug > 0) fprintf(stderr, " %d:%f", j, accuracy_curr[j]);
        }
      }
      if(numQuestionsAsked)
        numQuestionsAsked[i] = session->GetNumQuestionsAsked();

      if(matlabProgressOut && time(NULL) - last_time > 60) {
        SaveMatlab20Q(matlabProgressOut, maxQuestions, classes->NumClasses(), NumExamples(), 
		    accuracy_curr, perQuestionConfusionMatrix, perQuestionPredictions, 
		    perQuestionClassProbabilities, questionsAsked, responseTimes, numQuestionsAsked, gtClasses);
        last_time = time(NULL);
      }

      num_so_far++;

#ifdef USE_OPENMP           
      omp_unset_lock(&my_lock); 
#endif
    
      delete session;
      GetExampleData(i)->Clear();
      if(g_debug > 0) fprintf(stderr, "\n");
    }
  }   
#ifdef USE_OPENMP
  omp_destroy_lock(&my_lock);
#endif

  char header[10000];
  strcpy(header, "Accuracy by question:");
  for(int j = 0; j < maxQuestions; j++) {
    accuracy[j] /= NumExamples();
    sprintf(header+strlen(header), " %d:%f", j, (float)accuracy[j]);
  }
  sprintf(header+strlen(header), "\n<br><a href=confMats.html>View Confusion Matrices</a>");

  if(debugDir) {
    DatasetSortFunc funcs[4] = { NULL, PartLocationsGTPoseCmp, PartLocationsClassCmp, ExampleLossCmp };
    const char *names[4] = { "Unsorted", "Pose", "Class", "Questions Asked" };
    BuildGalleries(debugDir, "images.html", "Visualization of Questions For All Test Images", header, funcs, names, 4);
    sprintf(header, "<br><a href=images.html>Browse Images</a>");
    BuildConfusionMatrices(debugDir, "index.html", perQuestionPred, perQuestionGT, maxQuestions, 
			   "Class Confusion Matrices By Questions Asked", header, imageNames, (const char**)linkNames);
  }

  for(int k = 0; k < NumExamples(); k++) {
    if(linkNames[k]) free(linkNames[k]);
  }

  free(accuracy_curr);
  free(perQuestionGT);
  free(perQuestionPred);
  free(imageNames);
  free(linkNames);
  if(old_certainties) delete [] old_certainties;
}


void Dataset::EvaluateTestsetInteractive(int maxDrag, float stopThresh, const char *debugDir, bool debugImages, bool debugProbabilityMaps, double *aveLoss, 
					 double **partLoss, int ***perDraggedPredictions, 
					 int **partsDragged, double **dragTimes, const char *matlabProgressOut) {						 
  int num_i = 0;
  time_t last_time = time(NULL);
  double *aveLossCurr = (double*)malloc(maxDrag*sizeof(double));
  int numTotalDragged = 0;

  for(int j = 0; j < maxDrag; j++) {
    aveLossCurr[j] = aveLoss[j] = 0;
  }

  char htmlName[1000]; 
  if(debugDir) {
    CreateDirectoryIfNecessary(debugDir);
    sprintf(htmlName, "%s/index.html", debugDir);
  }

  // Getting strange problems in Visual C++ Release mode.  num_i and numTotalDragged
  // aren't getting shared in the main loop below unless I run this loop first
  #pragma omp parallel for shared(num_i, numTotalDragged, aveLoss, aveLossCurr, last_time, partLoss, perDraggedPredictions, partsDragged, dragTimes)
  for(int i = 0; i < 100; i++) {
    num_i++;
    numTotalDragged++;
  }
  num_i = 0;
  numTotalDragged = 0;

#ifdef USE_OPENMP
  omp_lock_t my_lock;
  omp_init_lock(&my_lock);
  #pragma omp parallel for shared(num_i, numTotalDragged, aveLoss, aveLossCurr, last_time, my_lock, partLoss, perDraggedPredictions, partsDragged, dragTimes)
#endif
  for(int i = 0; i < NumExamples(); i++) {
    assert(GetExampleLabel(i)->NumObjects() == 1);
    int o = 0;
    int j, k;
    ImageProcess *process = GetExampleData(i)->GetProcess(classes);
    InteractiveLabelingSession *session = new InteractiveLabelingSession(process, GetExampleLabel(i)->GetObject(o), false, true, 
						  false, false, true, 1.0f, debugImages ? debugDir : NULL, debugProbabilityMaps);
    PartLocation *locs = session->Label(maxDrag, stopThresh);
    int numDragged = session->NumDragged();
    float losses[1000];

    fprintf(stderr, "%s number dragged=%d\n", process->Features()->Name(), numDragged);

#ifdef USE_OPENMP         
    omp_set_lock(&my_lock);
#endif
      
    numTotalDragged += numDragged;
    if(debugDir) {
      char fname[1000], fname2[1000], thumb[1000], thumb2[1000], description[1000];
      IplImage *img = cvCloneImage(process->Features()->GetImage());
      process->Draw(img, locs, CV_RGB(0,0,255), false, false, true, false, true, -1, true);
      sprintf(fname, "%s/%s_il.png", debugDir, process->Features()->Name());
      cvSaveImage(fname, img);
      
      ExtractFilename(fname, fname2);
      strcpy(thumb, fname2);
      StripFileExtension(thumb);
      sprintf(thumb+strlen(thumb), "_thumb.%s", GetFileExtension(fname2));
      sprintf(thumb2, "%s/%s", debugDir, thumb);
      MakeThumbnail(img, 75, 75, thumb2);
      if(debugImages) 
	sprintf(description, "<a href=\"%s.html\">View Interactive User Annotations</a>", process->Features()->Name());
      GetExample(i)->AddExampleVisualization(fname2, thumb, debugImages ? description : NULL, (double)numDragged);
      cvReleaseImage(&img);
    }
    num_i++;

    // Record information for loss and part predictions for the sequence of part movements
    for(j = 0; j <= numDragged; j++) {
      PartLocation *locs = session->GetPartLocations(j);

      // Part loss is the number of parts outside stopThresh standard deviations of the ground truth part
      partLoss[i][j] = 0;
      session->GetLoss(locs, losses, NULL, stopThresh);
      for(k = 0; k < classes->NumParts(); k++)
	if(losses[k] > stopThresh)
	  partLoss[i][j]++;
      aveLoss[j] += partLoss[i][j];
      aveLossCurr[j] = aveLoss[j]/num_i;

      // Store the predicted location of each part
      for(k = 0; k < classes->NumParts(); k++) {
	float ix, iy;
	locs[k].GetDetectionLocation(NULL, NULL, &perDraggedPredictions[i][k][2], 
				     &perDraggedPredictions[i][k][3], &perDraggedPredictions[i][k][4]); 
	locs[k].GetImageLocation(&ix, &iy); 
	perDraggedPredictions[i][k][0] = ix;
	perDraggedPredictions[i][k][1] = iy;
      }

      // Store which part was dragged
      if(j > 0) {
        PartLocation loc(session->GetMovedPart(j-1));
        partsDragged[i][j] = loc.GetPartID();
        dragTimes[i][j] = loc.GetResponseTime();
      } else {
        partsDragged[i][j] = -1;
        dragTimes[i][j] = 0;
      }
    }

    // If the user stopped the labeling process early, zero out the remainder of the arrays
    for(j = numDragged+1; j < maxDrag; j++) {
      partLoss[i][j] = 0;
      aveLossCurr[j] = aveLoss[j]/num_i;
      memcpy(perDraggedPredictions[i][j], perDraggedPredictions[i][numDragged], sizeof(int)*5);
      partsDragged[i][j] = -1;
    }
    
    

    if(matlabProgressOut && time(NULL) - last_time > 60) {
      SaveMatlabInteractive(matlabProgressOut, maxDrag, NumExamples(), 
			    aveLossCurr, partLoss, perDraggedPredictions, partsDragged, dragTimes);
      last_time = time(NULL);
    }

    if(g_debug > 0) {
      fprintf(stderr, "Loss");
      for(j = 0; j < maxDrag; j++) 
	fprintf(stderr, " %d:%f", j, (float)aveLossCurr[j]);
      fprintf(stderr, "\n");
      fprintf(stderr, "Average number of parts dragged: %f\n", ((float)numTotalDragged)/num_i);
    }

#ifdef USE_OPENMP           
    omp_unset_lock(&my_lock); 
#endif

    delete session;
    GetExampleData(i)->Clear();
  } 

#ifdef USE_OPENMP
  omp_destroy_lock(&my_lock);
#endif  

  char header[10000];
  strcpy(header, "");

  fprintf(stderr, "Final loss:");
  sprintf(header+strlen(header), "Final loss:");
  for(int j = 0; j < maxDrag; j++) {
    aveLoss[j] /= num_i;
    fprintf(stderr, " %d:%f", j, aveLoss[j]);
    sprintf(header+strlen(header), " %d:%f", j, aveLoss[j]);
  }
  fprintf(stderr, "\n Final average number of parts dragged: %f\n", ((float)numTotalDragged)/num_i);
  sprintf(header+strlen(header), "\n<br>Final average number of parts dragged: %f\n", ((float)numTotalDragged)/num_i);

  if(debugDir) {
    DatasetSortFunc funcs[4] = { NULL, PartLocationsGTPoseCmp, PartLocationsClassCmp, ExampleLossCmp };
    const char *names[4] = { "Unsorted", "Ground Truth Pose", "Class", "Parts Dragged" };
    BuildGalleries(debugDir, "index.html", "Visualization of Parts Dragged For Each Test Image", header, funcs, names, 4);
  }

  free(aveLossCurr);
}

      //#define SAVE_TXT_LOCS

int TaxonomicalLoss(Classes *classes, ObjectClass *pred, ObjectClass *gt);

float Dataset::EvaluateTestset(bool evaluatePartDetection, bool evaluateClassification, const char *imagesDirOut, 
			       int *predictedClasses, int *trueClasses, double **classScores, double *localizationLoss, 
			       int ***predictedLocations, int ***trueLocations, double *predictedLocationScores,
			       const char *matlabProgressOut, double **localizationLossComparedToUsers) {
  float sumLoss = 0, sumLossUserByPart[100], sumLossUser = 0;
  int num = 0;
  int *predClasses = Create1DArray<int>(NumExamples()), *gtClasses = Create1DArray<int>(NumExamples());
  char **imageNames = Create1DArray<char*>(NumExamples());
  float sumTaxoLoss = 0;
  bool useTaxoLoss = classes->NumClasses() && strlen(classes->GetClass(0)->GetMeta("Order")) > 0;

  for(int j = 0; j < classes->NumParts(); j++)
    sumLossUserByPart[j] = 0;

  int num_i = 0;
  int numMisclassified = 0;
  int numGTClass = 0;
  time_t last_time = time(NULL);
  PartDetectionLossType lossType = classes->GetDetectionLossMethod();

#ifdef SAVE_TXT_LOCS
  if(imagesDirOut) {
    char dirName[1000];
    sprintf(dirName, "%s/part_locs", imagesDirOut);
    CreateDirectoryIfNecessary(dirName);
    for(int k = 0; k < classes->NumClasses(); k++) {
      sprintf(dirName, "%s/part_locs/%s", imagesDirOut, classes->GetClass(k)->Name());
      CreateDirectoryIfNecessary(dirName);
    }
    sprintf(dirName, "%s/part_locs/parts.txt", imagesDirOut);
    FILE *fout = fopen(dirName, "w");
    for(int j = 0; j < classes->NumParts(); j++) 
      fprintf(fout, "%d %s\n", j+1, classes->GetPart(j)->Name());
    fclose(fout);
  }
#endif

#ifdef USE_OPENMP
  omp_lock_t my_lock;
  omp_init_lock(&my_lock);
  #pragma omp parallel for
#endif
  for(int i = 0; i < NumExamples(); i++) {
    assert(GetExampleLabel(i)->NumObjects() == 1);
    int o = 0;
    float score = 0, loss=0;
    char classL[1000]; strcpy(classL, "");
    char partL[1000];  strcpy(partL, "");
    char poseL[1000];   strcpy(poseL, "");
    char userLoss[10000];
    ImageProcess *process = GetExampleData(i)->GetProcess(classes);
    PartLocation *locs = GetExampleLabel(i)->GetObject(o)->GetPartLocations();
    if(evaluatePartDetection) {
      char loc_str[4000], loc_str2[4000];
      score = process->Detect();
      locs = process->ExtractPartLocations();
      PartLocation *gt_locs = GetExampleLabel(i)->GetObject(o)->GetPartLocations();
      fprintf(stderr, "%s score=%f", process->ImageName(), score);

#ifdef SAVE_TXT_LOCS
      if(imagesDirOut) {
	char txtName[1000], iname[1000];
	ExtractFilename(GetExampleData(i)->GetImageName(), iname);
	StripFileExtension(iname);
	ObjectClass *cl = GetExampleLabel(i)->GetObject(0)->GetClass();
	sprintf(txtName, "%s/part_locs/%s/%s.txt", imagesDirOut, cl ? cl->Name() : "root", iname);
	FILE *txtFile = fopen(txtName, "w");
	for(int j = 0; j < classes->NumParts(); j++) {
	  float ix, iy;
	  locs[j].GetDetectionLocation(NULL, NULL, &predictedLocations[i][j][2], 
				       &predictedLocations[i][j][3], &predictedLocations[i][j][4]); 
	  locs[j].GetImageLocation(&ix, &iy); 
	  fprintf(txtFile, "%d %d %d\n", (int)(ix+.5f), (int)(iy+.5f), predictedLocations[i][j][4] ? 1 : 0);
	}
	fclose(txtFile);
      }
#endif

      if(gt_locs) {
	process->SetLossImages(gt_locs);
	loss = process->ComputeLoss(locs);
	fprintf(stderr, " loss=%f", loss);
	sumLoss += loss;
	num++;

	strcpy(userLoss, "");
	if(localizationLossComparedToUsers) {
	  float sumLossUserTmp = 0;
	  for(int j = 0; j < classes->NumParts(); j++) {
	    localizationLossComparedToUsers[i][j] = process->GetPartInst(j)->GetUserLoss(locs+j);
	    sumLossUserByPart[j] += (float)localizationLossComparedToUsers[i][j];
	    sumLossUserTmp += (float)localizationLossComparedToUsers[i][j];
	    fprintf(stderr, " (%s loss = %f)", classes->GetPart(j)->Name(), (float)localizationLossComparedToUsers[i][j]);
	  }
	  sumLossUser += sumLossUserTmp;
	  sprintf(userLoss, ", average part loss=%f std dev", sumLossUserTmp/classes->NumParts());
	  for(int j = 0; j < classes->NumParts(); j++) 
	    sprintf(userLoss+strlen(userLoss), " (%s = %f std dev)", classes->GetPart(j)->Name(), (float)localizationLossComparedToUsers[i][j]);
	}

	if(trueLocations) { 
	  for(int j = 0; j < classes->NumParts(); j++) {
	    float ix, iy;
	    gt_locs[j].GetDetectionLocation(NULL, NULL, &trueLocations[i][j][2], 
					    &trueLocations[i][j][3], &trueLocations[i][j][4]); 
	    gt_locs[j].GetImageLocation(&ix, &iy); 
	    trueLocations[i][j][0] = (int)(ix+.5f);
	    trueLocations[i][j][1] = (int)(iy+.5f);
	  }
	}
	if(localizationLoss) localizationLoss[i] = loss;
      }
      if(predictedLocations) { 
	for(int j = 0; j < classes->NumParts(); j++) {
	  float ix, iy;
	  locs[j].GetDetectionLocation(NULL, NULL, &predictedLocations[i][j][2], 
				       &predictedLocations[i][j][3], &predictedLocations[i][j][4]); 
	  locs[j].GetImageLocation(&ix, &iy); 
	  predictedLocations[i][j][0] = (int)(ix+.5f);
	  predictedLocations[i][j][1] = (int)(iy+.5f);
	}
      }
      if(predictedLocationScores) predictedLocationScores[i] = score;

      fprintf(stderr, " locs=[%s] gt_locs=[%s]\n", process->PrintPartLocations(locs, loc_str), 
	      gt_locs ? process->PrintPartLocations(gt_locs, loc_str2) : "");
      GetExampleLabel(i)->GetObject(o)->SetPartLocations(locs);

      sprintf(partL, "<br>score=%f, loss=%f%s\n", score, loss, userLoss); 
      for(int j = 0; j < classes->NumParts(); j++) 
	sprintf(partL+strlen(partL), " (%s loss = %f)", classes->GetPart(j)->Name(), (float)process->GetPartInst(j)->GetLoss(locs+j));
    
    }

    double classLikelihoods[50000];
    float bestL = -1000000, classSlack = 0;
    int best = -1;
    strcpy(classL, "");
    if(evaluateClassification) {
      // Predict the class on the current set of part locations
      process->ComputeClassLogLikelihoodsAtLocation(classLikelihoods, GetExampleLabel(i)->GetObject(o)->GetPartLocations(), false);
      for(int j = 0; j < classes->NumClasses(); j++) { 
	if(classLikelihoods[j] > bestL) {
	  bestL = classLikelihoods[j];
	  best = j;
	}
      }
      gtClasses[i] = GetExampleLabel(i)->GetObject(o)->GetClass()->Id();
      predClasses[i] = best;
      classSlack = bestL - classLikelihoods[gtClasses[i]];
      sprintf(classL, "<br>Predicted Class %s, score=%f", classes->GetClass(best)->Name(), bestL);
      if(GetExampleLabel(i)->GetObject(o)->GetClass()) {
	sprintf(classL+strlen(classL), " (true class is %s, score=%f)", GetExampleLabel(i)->GetObject(o)->GetClass()->Name(),
		classLikelihoods[GetExampleLabel(i)->GetObject(o)->GetClassID()]);
      }
      if(useTaxoLoss) fprintf(stderr, "%s ave_err=%f, ave_taxo_loss=%f\n", classL, numMisclassified/(float)numGTClass, sumTaxoLoss/(float)numGTClass);
      else fprintf(stderr, "%s ave_err=%f\n", classL, numMisclassified/(float)numGTClass);

      if(predictedClasses) predictedClasses[i] = best;
      if(trueClasses) trueClasses[i] = GetExampleLabel(i)->GetObject(o)->GetClassID();
      if(classScores) {
	for(int j = 0; j < classes->NumClasses(); j++) { 
	  classScores[i][j] = classLikelihoods[j];
	}
      }
    }

    if(imagesDirOut) {
      IplImage *img = cvCloneImage(process->Image());
      process->Draw(img, locs, CV_RGB(1,1,1), false, false, true, false, false, -1, true);
      char fname[1000], fname2[1000], thumb[1000], thumb2[1000], description[10000];
      sprintf(fname2, "%s.png", process->Features()->Name());
      sprintf(fname, "%s/%s", imagesDirOut, fname2);
      cvSaveImage(fname, img);
      imageNames[i] = StringCopy(fname2);

      strcpy(thumb, fname2);
      StripFileExtension(thumb);
      sprintf(thumb+strlen(thumb), "_thumb.%s", GetFileExtension(fname2));
      sprintf(thumb2, "%s/%s", imagesDirOut, thumb);
      MakeThumbnail(img, 75, 75, thumb2);
      sprintf(description, "%s%s%s", partL, classL, poseL);
      cvReleaseImage(&img);
      
      GetExample(i)->AddExampleVisualization(fname2, thumb, strlen(description) ? description : NULL, (double)(evaluateClassification ? classSlack : loss ));
    }
#ifdef USE_OPENMP         
    omp_set_lock(&my_lock);
#endif
    if(evaluateClassification && GetExampleLabel(i)->GetObject(o)->GetClass()) {
      if(best != GetExampleLabel(i)->GetObject(o)->GetClassID()) {
	numMisclassified++;
	if(useTaxoLoss) sumTaxoLoss += TaxonomicalLoss(classes, classes->GetClass(best), GetExampleLabel(i)->GetObject(o)->GetClass());
      }
      numGTClass++;
    }
    num_i++;

    if(matlabProgressOut && time(NULL) - last_time > 60) {
      SaveMatlabTest(matlabProgressOut, classes->NumClasses(), NumExamples(), classes->NumParts(), 
		     predictedClasses, trueClasses, classScores, localizationLoss,
		     predictedLocations, trueLocations, predictedLocationScores, localizationLossComparedToUsers);
      last_time = time(NULL);
    }

#ifdef USE_OPENMP         
    omp_unset_lock(&my_lock);
#endif
    

    process->Clear();
    GetExampleData(i)->Clear();
  }

  char header[10000];
  strcpy(header, "");
  if(num)  {
    fprintf(stderr, "Average part localization loss is %f\n", sumLoss/num);
    sprintf(header+strlen(header), "Average part localization loss was %f", sumLoss/num);
    if(lossType == LOSS_NUM_INCORRECT) 
      sprintf(header+strlen(header), " (measured as # of parts outside %f standard deviations compared to MTurk users)", NUM_DEVIATIONS_BEFORE_INCORRECT);
    else if(lossType == LOSS_PART_AVERAGE_AREA_UNION_OVER_INTERSECTION) 
      sprintf(header+strlen(header), " (measured as the average area of union the area of intersection over all parts)");
    else if(lossType == LOSS_BOUNDING_BOX_AREA_UNION_OVER_INTERSECTION) 
      sprintf(header+strlen(header), " (measured as the average area of union the area of intersection of the root bounding box)");
    else if(lossType == LOSS_USER_STANDARD_DEVIATIONS) 
      sprintf(header+strlen(header), " (measured as the number of standard deviations from ground truth compared to MTurk users, capped at %f deviations per part)", (float)MAX_DEVIATIONS);
    sprintf(header+strlen(header), "\n<br>");

    if(localizationLossComparedToUsers) {
      fprintf(stderr, "Average part localization standard deviations is %f\n  ", sumLossUser/num/classes->NumParts());
      sprintf(header+strlen(header), "Average part localization standard deviations was %f\n  ", sumLossUser/num/classes->NumParts());
      for(int i = 0; i < classes->NumParts(); i++) {
	fprintf(stderr, " %s:%f", classes->GetPart(i)->Name(), sumLossUserByPart[i]/num);
	sprintf(header+strlen(header), " %s:%f", classes->GetPart(i)->Name(), sumLossUserByPart[i]/num);
      }
      sprintf(header+strlen(header), "\n<br>");
    }
  }
  if(evaluateClassification) {
    fprintf(stderr, "Average classification loss is %f\n", numMisclassified/(float)numGTClass);
    sprintf(header+strlen(header), "Average classification loss was %f\n", numMisclassified/(float)numGTClass);
    if(useTaxoLoss) fprintf(stderr, "Average taxonomical loss is %f\n", sumTaxoLoss/(float)numGTClass);
  }

  if(imagesDirOut) {
    char title[1000];
    if(evaluateClassification) {
      DatasetSortFunc funcs[4] = { ExampleLossCmp, PartLocationsClassCmp, PartLocationsGTPoseCmp, NULL };
      const char *names[4] = { "Classification Error (slack)", "Class", "Ground Truth Pose", "Unsorted" };
      sprintf(title, "Classification Predictions For All Test Images (%d classes)", classes->NumClasses());
      BuildGalleries(imagesDirOut, "index.html", title, header, funcs, names, 4);
    } else {
      DatasetSortFunc funcs[6] = { PartLocationsScoreCmp, ExampleLossCmp, PartLocationsGTPoseCmp, PartLocationsPoseCmp, PartLocationsClassCmp, NULL };
      const char *names[6] = { "Detection Score", "Detection Loss", "Ground Truth Pose", "Predicted Pose", "Class", "Unsorted" };
      sprintf(title, "Part Localization Predictions For All Test Images");
      BuildGalleries(imagesDirOut, "index.html", title, header, funcs, names, 6);
    }

    if(evaluateClassification) {
      const char *classNames[10000];
      char confName[1000];
      for(int i = 0; i < classes->NumClasses(); i++) 
	classNames[i] = classes->GetClass(i)->Name();
      sprintf(confName, "%s/confMat.html", imagesDirOut);
      BuildConfusionMatrix(predClasses, gtClasses, NumExamples(), confName,
			   classNames, "Class Confusion Matrix", NULL, (const char**)imageNames);
    }
  }

  free(predClasses);
  free(gtClasses);
  for(int i = 0; i <  NumExamples(); i++)
    if(imageNames[i])
      free(imageNames[i]);
  free(imageNames);

  return num ? sumLoss/num : 0;
}





#ifdef HAVE_MATLAB
#include <mat.h>
#endif

void SaveMatlab20Q(const char *matfileOut, int maxQuestions, int numClasses, int numExamples, 
		   double *accuracy, double ***perQuestionConfusionMatrix, int **perQuestionPredictions, 
		   double ***perQuestionClassProbabilities, int **questionsAsked, int **responseTimes, int *numQuestionsAsked, int *gtClasses) {
  fprintf(stderr, "Saving %s...", matfileOut);
  
#ifdef HAVE_MATLAB  
  // Save everything into matlab matrices
  MATFile *pmat = matOpen(matfileOut, "wb"); assert(pmat);
  const mwSize accuracyDims[] = {maxQuestions}; 
  mxArray *accuracyM = mxCreateNumericArray(1, accuracyDims, mxDOUBLE_CLASS, mxREAL); 
  memcpy(mxGetData(accuracyM), accuracy, sizeof(double)*maxQuestions);
  matPutVariable(pmat, "accuracy", accuracyM);
  const mwSize perQuestionConfusionMatrixDims[] = {numClasses,numClasses,maxQuestions};
  mxArray *perQuestionConfusionMatrixM = mxCreateNumericArray(3, perQuestionConfusionMatrixDims, mxDOUBLE_CLASS, mxREAL);
  memcpy(mxGetData(perQuestionConfusionMatrixM), &perQuestionConfusionMatrix[0][0][0], sizeof(double)*maxQuestions*numClasses*numClasses);
  matPutVariable(pmat, "perQuestionConfusionMatrix", perQuestionConfusionMatrixM);
  const mwSize perQuestionClassProbabilitiesDims[] = {numClasses, maxQuestions, numExamples};
  mxArray *perQuestionClassProbabilitiesM = mxCreateNumericArray(3, perQuestionClassProbabilitiesDims, mxDOUBLE_CLASS, mxREAL);
  memcpy(mxGetData(perQuestionClassProbabilitiesM), &perQuestionClassProbabilities[0][0][0], sizeof(double)*numExamples*maxQuestions*numClasses);
  matPutVariable(pmat, "perQuestionClassProbabilities", perQuestionClassProbabilitiesM);
  const mwSize perQuestionPredictionsDims[] = {maxQuestions, numExamples};
  mxArray *perQuestionPredictionsM = mxCreateNumericArray(2, perQuestionPredictionsDims, mxINT32_CLASS, mxREAL);
  memcpy(mxGetData(perQuestionPredictionsM), &perQuestionPredictions[0][0], sizeof(int)*numExamples*maxQuestions);
  matPutVariable(pmat, "perQuestionPredictions", perQuestionPredictionsM);
  const mwSize questionsAskedDims[] = {maxQuestions, numExamples};
  mxArray *questionsAskedM = mxCreateNumericArray(2, questionsAskedDims, mxINT32_CLASS, mxREAL);
  memcpy(mxGetData(questionsAskedM), &questionsAsked[0][0], sizeof(int)*numExamples*maxQuestions);
  matPutVariable(pmat, "questionsAsked", questionsAskedM);
  const mwSize responseTimesDims[] = {maxQuestions, numExamples};
  mxArray *responseTimesM = mxCreateNumericArray(2, responseTimesDims, mxINT32_CLASS, mxREAL);
  memcpy(mxGetData(responseTimesM), &responseTimes[0][0], sizeof(int)*numExamples*maxQuestions);
  matPutVariable(pmat, "responseTimes", responseTimesM);
  const mwSize numQuestionsAskedDims[] = {numExamples};
  mxArray *numQuestionsAskedM = mxCreateNumericArray(1, numQuestionsAskedDims, mxINT32_CLASS, mxREAL);
  memcpy(mxGetData(numQuestionsAskedM), &numQuestionsAsked[0], sizeof(int)*numExamples);
  matPutVariable(pmat, "numQuestionsAsked", numQuestionsAskedM);
  const mwSize gtClassesDims[] = {numExamples};
  mxArray *gtClassesM = mxCreateNumericArray(1, gtClassesDims, mxINT32_CLASS, mxREAL);
  memcpy(mxGetData(gtClassesM), &gtClasses[0], sizeof(int)*numExamples);
  matPutVariable(pmat, "gtClasses", gtClassesM);
  matClose(pmat);

  // cleanup
  mxDestroyArray(accuracyM); mxDestroyArray(perQuestionConfusionMatrixM); mxDestroyArray(perQuestionClassProbabilitiesM); mxDestroyArray(gtClassesM);
  mxDestroyArray(perQuestionPredictionsM); mxDestroyArray(questionsAskedM); mxDestroyArray(responseTimesM);  mxDestroyArray(numQuestionsAskedM);
#else
  char fname[1000];
  //fprintf(stderr, "Can't output Matlab file %s.  You must define -DHAVE_MATLABin the Makefile.  Saving to binary file instead...\n", matfileOut);
  strcpy(fname, matfileOut);
  StripFileExtension(fname);
  strcat(fname, ".bin");
  FILE *fout = fopen(fname, "wb");
  int num_mats = ((accuracy ? 1 : 0) + (perQuestionConfusionMatrix ? 1 : 0) + (perQuestionClassProbabilities ? 1 : 0) + 
		  (perQuestionPredictions ? 1 : 0) + (questionsAsked ? 1 : 0) + (responseTimes ? 1 : 0) + (numQuestionsAsked ? 1 : 0) + (gtClasses ? 1 : 0));
  bool b = (fwrite(&num_mats, sizeof(int), 1, fout) &&
	 (!accuracy || Save1DArray<double>(fout, accuracy, maxQuestions, "accuracy", "double")) &&
	 (!perQuestionConfusionMatrix || Save3DArray<double>(fout, perQuestionConfusionMatrix, maxQuestions, numClasses, 
							     numClasses, "perQuestionConfusionMatrix", "double")) &&
	 (!perQuestionClassProbabilities || Save3DArray<double>(fout, perQuestionClassProbabilities, numExamples, maxQuestions, 
								numClasses, "perQuestionClassProbabilities", "double")) &&
	 (!perQuestionPredictions || Save2DArray<int>(fout, perQuestionPredictions, numExamples, maxQuestions, "perQuestionPredictions", "int32")) &&
	 (!questionsAsked || Save2DArray<int>(fout, questionsAsked, numExamples, maxQuestions, "questionsAsked", "int32")) &&
	 (!responseTimes || Save2DArray<int>(fout, responseTimes, numExamples, maxQuestions, "responseTimes", "int32")) &&
	 (!numQuestionsAsked || Save1DArray<int>(fout, numQuestionsAsked, numExamples, "numQuestionsAsked", "int32")) &&
         (!gtClasses || Save1DArray<int>(fout, gtClasses, numExamples, "gtClasses", "int32")));
  assert(b);
  fclose(fout);
  
  StripFileExtension(fname);
  strcat(fname, ".m");
  SaveMatlabImport(fname);
#endif
  fprintf(stderr, "done saving\n");
}

void SaveMatlabTest(const char *matfileOut, int numClasses, int numExamples, int numParts,
		    int *predictedClasses, int *trueClasses, double **classScores, double *localizationLoss,
		    int ***predictedLocations, int ***trueLocations, double *predictedLocationScores,
		    double **localizationLossComparedToUsers) {

#ifdef HAVE_MATLAB 
  // Save everything into matlab matrices
  MATFile *pmat = matOpen(matfileOut, "wb"); assert(pmat);
  if(predictedClasses) {
    const mwSize predictedClassesDims[] = {numExamples}; 
    mxArray *predictedClassesM = mxCreateNumericArray(1, predictedClassesDims, mxINT32_CLASS, mxREAL); 
    memcpy(mxGetData(predictedClassesM), predictedClasses, sizeof(int)*numExamples);
    matPutVariable(pmat, "predictedClasses", predictedClassesM);
    mxDestroyArray(predictedClassesM); 
    free(predictedClasses);
  }
  if(trueClasses) {
    const mwSize trueClassesDims[] = {numExamples}; 
    mxArray *trueClassesM = mxCreateNumericArray(1, trueClassesDims, mxINT32_CLASS, mxREAL); 
    memcpy(mxGetData(trueClassesM), trueClasses, sizeof(int)*numExamples);
    matPutVariable(pmat, "trueClasses", trueClassesM);
    mxDestroyArray(trueClassesM); 
    free(trueClasses);
  }
  if(classScores) {
    const mwSize classScoresDims[] = {numClasses, numExamples};
    mxArray *classScoresM = mxCreateNumericArray(2, classScoresDims, mxDOUBLE_CLASS, mxREAL);
    memcpy(mxGetData(classScoresM), &classScores[0][0], sizeof(double)*numExamples*numClasses);
    matPutVariable(pmat, "classScores", classScoresM);
    mxDestroyArray(classScoresM); 
    free(classScores);
  }
  if(localizationLoss) {
    const mwSize localizationLossDims[] = {numExamples}; 
    mxArray *localizationLossM = mxCreateNumericArray(1, localizationLossDims, mxDOUBLE_CLASS, mxREAL); 
    memcpy(mxGetData(localizationLossM), localizationLoss, sizeof(double)*numExamples);
    matPutVariable(pmat, "localizationLoss", localizationLossM);
    mxDestroyArray(localizationLossM); 
    free(localizationLoss);
  }
  if(predictedLocationScores) {
    const mwSize predictedLocationScoresDims[] = {numExamples}; 
    mxArray *predictedLocationScoresM = mxCreateNumericArray(1, predictedLocationScoresDims, mxDOUBLE_CLASS, mxREAL); 
    memcpy(mxGetData(predictedLocationScoresM), predictedLocationScores, sizeof(double)*numExamples);
    matPutVariable(pmat, "predictedLocationScores", predictedLocationScoresM);
    mxDestroyArray(predictedLocationScoresM); 
    free(predictedLocationScores);
  }
  if(predictedLocations) {
    const mwSize predictedLocationsDims[] = {5, numParts, numExamples}; 
    mxArray *predictedLocationsM = mxCreateNumericArray(3, predictedLocationsDims, mxINT32_CLASS, mxREAL); 
    memcpy(mxGetData(predictedLocationsM), &predictedLocations[0][0][0], sizeof(int)*numExamples*numParts*5);
    matPutVariable(pmat, "predictedLocations", predictedLocationsM);
    mxDestroyArray(predictedLocationsM); 
    free(predictedLocations);
  }
  if(trueLocations) {
    const mwSize trueLocationsDims[] = {5, numParts, numExamples}; 
    mxArray *trueLocationsM = mxCreateNumericArray(3, trueLocationsDims, mxINT32_CLASS, mxREAL); 
    memcpy(mxGetData(trueLocationsM), &trueLocations[0][0][0], sizeof(int)*numExamples*numParts*5);
    matPutVariable(pmat, "trueLocations", trueLocationsM);
    mxDestroyArray(trueLocationsM); 
    free(trueLocations);
  }
  if(localizationLossComparedToUsers) {
    const mwSize localizationLossComparedToUsersDims[] = {numParts, numExamples}; 
    mxArray *localizationLossComparedToUsersM = mxCreateNumericArray(2, localizationLossComparedToUsersDims, mxDOUBLE_CLASS, mxREAL); 
    memcpy(mxGetData(localizationLossComparedToUsersM), localizationLossComparedToUsers, sizeof(double)*numExamples*numParts);
    matPutVariable(pmat, "localizationLossComparedToUsers", localizationLossComparedToUsersM);
    mxDestroyArray(localizationLossComparedToUsersM); 
    free(localizationLossComparedToUsers);
  }
  matClose(pmat);
#else
  char fname[1000];
  //fprintf(stderr, "Can't output Matlab file %s.  You must define -DHAVE_MATLABin the Makefile.  Saving to binary file instead...\n", matfileOut);
  strcpy(fname, matfileOut);
  StripFileExtension(fname);
  strcat(fname, ".bin");
  FILE *fout = fopen(fname, "wb");
  int num_mats = ((predictedClasses ? 1 : 0) + (trueClasses ? 1 : 0) + (classScores ? 1 : 0) + 
		  (localizationLoss ? 1 : 0) + (predictedLocations ? 1 : 0) + (trueLocations ? 1 : 0) + 
		  (predictedLocationScores ? 1 : 0) + (localizationLossComparedToUsers ? 1 : 0));
  bool b = (fwrite(&num_mats, sizeof(int), 1, fout) &&
	 (!predictedClasses || Save1DArray<int>(fout, predictedClasses, numExamples, "predictedClasses", "int32")) &&
	 (!trueClasses || Save1DArray<int>(fout, trueClasses, numExamples, "trueClasses", "int32")) &&
	 (!classScores || Save2DArray<double>(fout, classScores, numExamples, numClasses, "classScores", "double")) &&
	 (!localizationLoss || Save1DArray<double>(fout, localizationLoss, numExamples, "localizationLoss", "double")) &&
	 (!predictedLocations || Save3DArray<int>(fout, predictedLocations, numExamples, numParts, 5, "predictedLocations", "int32")) &&
	 (!trueLocations || Save3DArray<int>(fout, trueLocations, numExamples, numParts, 5, "trueLocations", "int32")) &&
	 (!predictedLocationScores || Save1DArray<double>(fout, predictedLocationScores, numExamples, "predictedLocationScores", "double")) &&
	 (!localizationLossComparedToUsers || Save2DArray<double>(fout, localizationLossComparedToUsers, numExamples, numParts, "localizationLossComparedToUsers", "double")));
  assert(b);
  fclose(fout);
  
  StripFileExtension(fname);
  strcat(fname, ".m");
  SaveMatlabImport(fname);
#endif
}


  
void SaveMatlabInteractive(const char *matfileOut, int maxDrag, int numExamples, double *aveLoss, 
			   double **partLoss, int ***partsDraggedPredictions, int **partsDragged, double **dragTimes) {
  fprintf(stderr, "Saving %s...", matfileOut);
  
  char fname[1000];
#ifdef HAVE_MATLAB  
  // Save everything into matlab matrices
  MATFile *pmat = matOpen(matfileOut, "wb"); assert(pmat);
  const mwSize aveLossDims[] = {maxDrag}; 
  mxArray *aveLossM = mxCreateNumericArray(1, aveLossDims, mxDOUBLE_CLASS, mxREAL); 
  memcpy(mxGetData(aveLossM), aveLoss, sizeof(double)*maxDrag);
  matPutVariable(pmat, "aveLoss", aveLossM);
  const mwSize partLossDims[] = {maxDrag,numExamples}; 
  mxArray *partLossM = mxCreateNumericArray(2, partLossDims, mxDOUBLE_CLASS, mxREAL); 
  memcpy(mxGetData(partLossM), partLoss, sizeof(double)*maxDrag*numExamples);
  matPutVariable(pmat, "partLoss", partLossM);
  const mwSize partsDraggedPredictionsDims[] = {5,maxDrag,numExamples}; 
  mxArray *partsDraggedPredictionsM = mxCreateNumericArray(3, partsDraggedPredictionsDims, mxINT32_CLASS, mxREAL); 
  memcpy(mxGetData(partsDraggedPredictionsM), partsDraggedPredictions, sizeof(int)*5*maxDrag*numExamples);
  matPutVariable(pmat, "partsDraggedPredictions", partsDraggedPredictionsM);
  const mwSize partsDraggedDims[] = {maxDrag,numExamples}; 
  mxArray *partsDraggedM = mxCreateNumericArray(2, partsDraggedDims, mxINT32_CLASS, mxREAL); 
  memcpy(mxGetData(partsDraggedM), partsDragged, sizeof(int)*maxDrag*numExamples);
  matPutVariable(pmat, "partsDragged", partsDraggedM);
  const mwSize dragTimesDims[] = {maxDrag,numExamples};
  mxArray *dragTimesM = mxCreateNumericArray(2, dragTimesDims, mxDOUBLE_CLASS, mxREAL);
  memcpy(mxGetData(dragTimesM), dragTimes, sizeof(double)*maxDrag*numExamples);
  matPutVariable(pmat, "dragTimes", partsDraggedM);

  // cleanup
  mxDestroyArray(aveLossM); mxDestroyArray(partLossM); mxDestroyArray(partsDraggedPredictionsM); mxDestroyArray(partsDraggedM); mxDestroyArray(dragTimesM);
#else
  //fprintf(stderr, "Can't output Matlab file %s.  You must define -DHAVE_MATLABin the Makefile.  Saving to binary file instead...\n", matfileOut);
  strcpy(fname, matfileOut);
  StripFileExtension(fname);
  strcat(fname, ".bin");
  FILE *fout = fopen(fname, "wb");
  int num_mats = ((aveLoss ? 1 : 0) + (partLoss ? 1 : 0) + (partsDraggedPredictions ? 1 : 0) + (partsDragged ? 1 : 0) + (dragTimes ? 1 : 0)) ;
  bool b = (fwrite(&num_mats, sizeof(int), 1, fout) &&
	 (!aveLoss || Save1DArray<double>(fout, aveLoss, maxDrag, "aveLoss", "double")) &&
	 (!partLoss || Save2DArray<double>(fout, partLoss, numExamples, maxDrag, "partLoss", "double")) &&
	 (!partsDraggedPredictions || Save3DArray<int>(fout, partsDraggedPredictions, numExamples, maxDrag, 5, "perDraggedPredictions", "int32")) &&
	 (!partsDragged || Save2DArray<int>(fout, partsDragged, numExamples, maxDrag, "partsDragged", "int32")) &&
        (!dragTimes || Save2DArray<double>(fout, dragTimes, numExamples, maxDrag, "dragTimes", "double")));
  assert(b);
  fclose(fout);
  
  StripFileExtension(fname);
  strcat(fname, ".m");
  SaveMatlabImport(fname);
#endif
  SVGPlotter plotter;
  plotter.AddPlot(NULL, aveLoss, maxDrag, 0, 1, NULL, "class1");
  plotter.SetXLabel("# Parts Labeled");
  plotter.SetYLabel("# Incorrect Parts");
  plotter.SetTitle("Interactive Part Labeling");
  strcpy(fname, matfileOut); StripFileExtension(fname); strcat(fname, "_plot.svg"); plotter.Save(fname);
  strcpy(fname, matfileOut); StripFileExtension(fname); strcat(fname, "_plot.m"); plotter.Save(fname);

  fprintf(stderr, "done saving\n");
}

void Dataset::Sort(int ( * comparator ) ( const void *, const void * )) {
  qsort(examples->examples, examples->num_examples, sizeof(StructuredExample*), comparator);
}

void Dataset::BuildConfusionMatrices(const char *debugDir, const char *fname, int **perQuestionPred, int **perQuestionGT, 
				     int num_conf, const char *title, const char *header, const char **imageNames, const char **linkNames) {
  StructuredExample **tmp = new StructuredExample*[NumExamples()];
  memcpy(tmp, examples->examples, NumExamples()*sizeof(StructuredExample*));

  char htmlName[1000];
  sprintf(htmlName, "%s/%s", debugDir, fname);
  FILE *fout = fopen(htmlName, "w");
  fprintf(fout, "<html>\n<head>\n<script type=\"text/javascript\">\n");
  fprintf(fout, "function loadFrame() {\n");
  fprintf(fout, "  var s=document.getElementById(\"order\").value;\n");
  fprintf(fout, "  document.getElementById(\"loadContainer\").innerHTML=\"\"\n");
  fprintf(fout, "  var iframe = document.createElement(\"iframe\");\n");
  fprintf(fout, "  iframe.src = s;\n");
  fprintf(fout, "  iframe.style.width = \"100%%\";\n");
  fprintf(fout, "  iframe.style.height = \"100%%\";\n");
  fprintf(fout, "  document.getElementById(\"loadContainer\").appendChild(iframe);\n");
  fprintf(fout, "}\n");
  fprintf(fout, "</script>\n</head>\n<body onload=loadFrame()>\n\n");
  fprintf(fout, "<div>\n");
  if(header) fprintf(fout, "<br>%s\n", header);
  fprintf(fout, "<form>After\n");
  fprintf(fout, "<select id='order' onchange=\"loadFrame()\">\n");

  const char *classNames[10000];
  int i;
  for(i = 0; i < classes->NumClasses(); i++) 
    classNames[i] = classes->GetClass(i)->Name();
  for(i = 0; i < num_conf; i++) {			
    char fname[1000];
    sprintf(fname, "%s/confMat%d.html", debugDir, i);
    BuildConfusionMatrix(perQuestionPred[i], perQuestionGT[i], NumExamples(), fname,
			 classNames, "Class Confusion Matrix", NULL, imageNames, linkNames);
    fprintf(fout, "  <option value=\"confMat%d.html\">%d Questions</option>\n", i, i);
  }

  fprintf(fout, "</select>\n</form>\n</div>\n\n");
  fprintf(fout, "<div id=\"loadContainer\"></div>\n");
  fprintf(fout, "</body>\n</html>");
  fclose(fout);

  memcpy(examples->examples, tmp, NumExamples()*sizeof(StructuredExample*));
  delete [] tmp;
}
    

void Dataset::BuildGalleries(const char *dir, const char *fname, const char *title, const char *header, DatasetSortFunc *sort_funcs, const char **labels, int num_sort) {
  char fname2[1000], lname[1000], sys[1000];
  CreateDirectoryIfNecessary(dir);
  for(int i = 0; i < NumExamples(); i++) {
    if(!GetExample(i)->visualization) {
      GetRelativePath(GetExampleData(i)->GetImageName(), dir, lname);
      sprintf(sys, "cd %s; ln -s %s", dir, lname);
      system(sys);
      ExtractFilename(GetExampleData(i)->GetImageName(), fname2);
      GetExample(i)->AddExampleVisualization(fname2, fname2, NULL, 0);
    }
  }

  StructuredExample **tmp = new StructuredExample*[NumExamples()];
  memcpy(tmp, examples->examples, NumExamples()*sizeof(StructuredExample*));

  char htmlName[1000];
  sprintf(htmlName, "%s/%s", dir, fname);
  FILE *fout = fopen(htmlName, "w");
  fprintf(fout, "<html>\n<head>\n<script type=\"text/javascript\">\n");
  fprintf(fout, "function loadFrame() {\n");
  fprintf(fout, "  var s=document.getElementById(\"order\").value;\n");
  fprintf(fout, "  document.getElementById(\"loadContainer\").innerHTML=\"\"\n");
  fprintf(fout, "  var iframe = document.createElement(\"iframe\");\n");
  fprintf(fout, "  iframe.src = s;\n");
  fprintf(fout, "  iframe.style.width = \"100%%\";\n");
  fprintf(fout, "  iframe.style.height = \"100%%\";\n");
  fprintf(fout, "  document.getElementById(\"loadContainer\").appendChild(iframe);\n");
  fprintf(fout, "}\n");
  fprintf(fout, "</script>\n</head>\n<body onload=loadFrame()>\n\n");
  if(title) fprintf(fout, "<h1>%s</h1>\n", title);
  fprintf(fout, "<div>\n");
  fprintf(fout, "%s\n", header);
  fprintf(fout, "<form>Order Images By:\n");
  fprintf(fout, "<select id='order' onchange=\"loadFrame()\">\n");

  for(int i = 0; i < num_sort; i++) {
    if(!sort_funcs[i]) 
      memcpy(examples->examples, tmp, NumExamples()*sizeof(StructuredExample*));
    else 
      Sort(sort_funcs[i]);
    char fname[1000];
    sprintf(fname, "%s/%s.html", dir, labels[i]);
    examples->MakeGallery(fname, NULL, NULL);
    fprintf(fout, "  <option value=\"%s.html\">%s</option>\n", labels[i], labels[i]);
  }

  fprintf(fout, "</select>\n</form>\n</div>\n\n");
  fprintf(fout, "<div id=\"loadContainer\"></div>\n");
  fprintf(fout, "</body>\n</html>");
  fclose(fout);

  memcpy(examples->examples, tmp, NumExamples()*sizeof(StructuredExample*));
  delete [] tmp;
}

// Sort images by pose and then by class
int PartLocationsPoseCmp(const void *v1, const void *v2) {
  MultiObjectLabelWithUserResponses *oy1 = (MultiObjectLabelWithUserResponses*)(*(StructuredExample**)v1)->y;
  MultiObjectLabelWithUserResponses *oy2 = (MultiObjectLabelWithUserResponses*)(*(StructuredExample**)v2)->y;
  if(oy1->NumObjects() == 0 || oy2->NumObjects() == 0) return oy2->NumObjects()-oy1->NumObjects();
  PartLocalizedStructuredLabelWithUserResponses *y1 = oy1->GetObject(0);
  PartLocalizedStructuredLabelWithUserResponses *y2 = oy2->GetObject(0);
  PartLocation *locs1 = y1->GetPartLocations();
  PartLocation *locs2 = y2->GetPartLocations();

  if(!locs1 && locs2) return 1;
  else if(locs1 && !locs2) return -1;
  else if(!locs1 && !locs2) return 0;

  Classes *classes = locs1 ? locs1[0].GetClasses() : locs2[0].GetClasses();
  int pose1, pose2;
  for(int i = classes->NumParts()-1; i >= 0; i--) {
    //if(classes->GetPart(i)->NumParts()) {
      locs1[i].GetDetectionLocation(NULL, NULL, NULL, NULL, &pose1);
      locs2[i].GetDetectionLocation(NULL, NULL, NULL, NULL, &pose2);
      if(pose1 != pose2) return pose1-pose2;
      //}
  }
  return y1->GetClassID() - y2->GetClassID();
}


// Sort images by ground truth pose and then by class
int PartLocationsGTPoseCmp(const void *v1, const void *v2) {
  MultiObjectLabelWithUserResponses *oy1 = (MultiObjectLabelWithUserResponses*)(*(StructuredExample**)v1)->y;
  MultiObjectLabelWithUserResponses *oy2 = (MultiObjectLabelWithUserResponses*)(*(StructuredExample**)v2)->y;
  if(oy1->NumObjects() == 0 || oy2->NumObjects() == 0) return oy2->NumObjects()-oy1->NumObjects();
  PartLocalizedStructuredLabelWithUserResponses *y1 = oy1->GetObject(0);
  PartLocalizedStructuredLabelWithUserResponses *y2 = oy2->GetObject(0);
  PartLocation *locs1 = y1->GetGTPartLocations();
  PartLocation *locs2 = y2->GetGTPartLocations();

  if(!locs1 && locs2) return 1;
  else if(locs1 && !locs2) return -1;
  else if(!locs1 && !locs2) return 0;

  Classes *classes = locs1 ? locs1[0].GetClasses() : locs2[0].GetClasses();
  int pose1, pose2;
  for(int i = classes->NumParts()-1; i >= 0; i--) {
    //if(classes->GetPart(i)->NumParts()) {
      locs1[i].GetDetectionLocation(NULL, NULL, NULL, NULL, &pose1);
      locs2[i].GetDetectionLocation(NULL, NULL, NULL, NULL, &pose2);
      if(pose1 != pose2) return pose1-pose2;
      //}
  }
  return y1->GetClassID() - y2->GetClassID();
}

// Sort images by by class
int PartLocationsClassCmp(const void *v1, const void *v2) {
  MultiObjectLabelWithUserResponses *oy1 = (MultiObjectLabelWithUserResponses*)(*(StructuredExample**)v1)->y;
  MultiObjectLabelWithUserResponses *oy2 = (MultiObjectLabelWithUserResponses*)(*(StructuredExample**)v2)->y;
  if(oy1->NumObjects() == 0 || oy2->NumObjects() == 0) return oy2->NumObjects()-oy1->NumObjects();
  PartLocalizedStructuredLabelWithUserResponses *y1 = oy1->GetObject(0);
  PartLocalizedStructuredLabelWithUserResponses *y2 = oy2->GetObject(0);

  return y1->GetClassID() - y2->GetClassID();
}

// Sort images by detection score
int PartLocationsScoreCmp(const void *v1, const void *v2) {
  MultiObjectLabelWithUserResponses *oy1 = (MultiObjectLabelWithUserResponses*)(*(StructuredExample**)v1)->y;
  MultiObjectLabelWithUserResponses *oy2 = (MultiObjectLabelWithUserResponses*)(*(StructuredExample**)v2)->y;
  if(oy1->NumObjects() == 0 || oy2->NumObjects() == 0) return oy2->NumObjects()-oy1->NumObjects();
  PartLocalizedStructuredLabelWithUserResponses *y1 = oy1->GetObject(0);
  PartLocalizedStructuredLabelWithUserResponses *y2 = oy2->GetObject(0);
  PartLocation *locs1 = y1->GetPartLocations();
  PartLocation *locs2 = y2->GetPartLocations();
  Classes *classes = locs1[0].GetClasses();
  int n = classes->NumParts()-1;
  float a1 = locs1[n].GetScore();
  float a2 = locs2[n].GetScore();
  return a1 < a2 ? 1 : (a2 < a1 ? -1 : 0);
}

// Sort images by image aspect ratio
int PartLocationsAspectRatioCmp(const void *v1, const void *v2) {
  PartLocalizedStructuredData *x1 = (PartLocalizedStructuredData*)(*(StructuredExample**)v1)->x;
  PartLocalizedStructuredData *x2 = (PartLocalizedStructuredData*)(*(StructuredExample**)v2)->x;
  float a1 = (x1->Width()/x1->Height()), a2 = (x2->Width()/x2->Height());

  return a1 < a2 ? -1 : (a2 < a1 ? 1 : 0);
}


// Sort images by by class
int ExampleLossCmp(const void *v1, const void *v2) {
  StructuredExample *e1 = *(StructuredExample**)v1;
  StructuredExample *e2 = *(StructuredExample**)v2;
  if(!e1->visualization)
    return !e2->visualization ? 0 : -1;
  else if(!e2->visualization)
    return 1;
  else {
    double d = e1->visualization->loss - e2->visualization->loss;
    return d < 0 ? -1 : (d > 0 ? 1 : 0);
  }
}
