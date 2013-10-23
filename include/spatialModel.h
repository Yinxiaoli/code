/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#ifndef __SPATIAL_MODEL_H
#define __SPATIAL_MODEL_H

/**
 * @file spatialModel.h
 * @brief Definition of a spatial model between child and parent parts
 */

#include "feature.h"
#include "part.h"

class ObjectPart;
class ObjectPose;
class Attribute;
class ObjectPartInstance;
class ObjectPoseInstance;
class AttributeInstance;
class FeatureOptions;
class ImageProcess;
class Classes;


// Don't allow the quadratic term of a spatial weight to be negative
//#define ENFORCE_NON_NEGATIVE_WEIGHTS

#define POSE_TRANSITION_FEATURE (1/(NORMALIZE_TEMPLATES ? 441.0 : 1)) /**< Constant feature for the transition between a pair of parent and child poses (will be multiplied by a weight)*/


/**
 * @class ObjectPartPoseTransition
 * @brief Defines a spatial model transition between a particular pair of parent/child parts and poses,
 *   which is a function of the position, scale, pose, and orientation of the parent and child parts
 * 
 * This is a helper function of ObjectPart and all routines should go through that class
 */
class ObjectPartPoseTransition {
  ObjectPart *parentPart;  /**< The parent part */
  ObjectPart *childPart;   /**< The child part */
  ObjectPose *parentPose;  /**< A pose associated with the parent part */
  ObjectPose **childPoses;   /**< A pose associated with the child pose */
  int numChildPoses;

  bool isClick;

  CvPoint2D32f offset;     /**< The optimal part offsets (in pixels) from parentPart to childPart */
  float offset_norm;       /**< A constant used to scale the part offsets so that they are typically in the range 0 to ~1 */
  float offset_scale, offset_orientation;  /**< The optimal part offsets (in terms of the scales/orientation indices in the features) from parent to child*/

  char *parentPartName, *childPartName, *parentPoseName, **childPoseNames;
  int id;

  /**
   * An 8+numChildPoses dimensional array of weights defining spatial scores between adjacent parts and poses.  
   *
   * The spatial cost for transitioning between these parent and child poses is: 
   *   dx=((parentPart.x-childPart.x)-offset.x)*offset_norm
   *   dy=((parentPart.y-childPart.y)-offset.y)*offset_norm
   *   dr=parentPart->rotation-childPart->rotation
   *   ds=parentPart->scale-childPart->scale
   * is W[0]*dx+W[1]*dy+W[2]*dr+W[3]*ds + W[4]*dx^2+W[5]*dy^2+W[6]*dr^2+W[7]*ds^2 + W[8]*1
   */
  float *W; 

  int *numTrainExamples;
  int *offset_orientation_count;
  int max_change_x, max_change_y, max_change_scale, max_change_rot;

 public:

  int Id() { return id; }
  void SetId(int i) { id = i; }

  int NumTrainExamples(int childPose) { return numTrainExamples[childPose]; }
  void Reset();
  void ZeroWeights();
  int NumChildPoses() { return numChildPoses; }

  /**
   * @brief Get a vector of weights for this spatial model
   */
  float *GetWeights() { return W; }
  
  /**
   * @brief Get the ideal offsets for this spatial model (e.g., ideal spring offsets from parent to child part locations)
   * @param offset_norm Factor to multiply spatial offsets, used to control the relative magnitude of spatial offsets compared to other features during regularization
   * @param offset_x the x-offset mu_x
   * @param offset_y the y-offset mu_y
   * @param offset_scale the scale-offset mu_scale
   * @param offset_rotation the orientation-offset mu_rot
   */
  void GetOffsets(float *offset_norm, float *offset_x, float *offset_y, float *offset_scale, float *offset_rotation) {
    if(offset_norm) *offset_norm = this->offset_norm;
    if(offset_x) *offset_x = this->offset.x;
    if(offset_y) *offset_y = this->offset.y;
    if(offset_scale) *offset_scale = this->offset_scale;
    if(offset_rotation) *offset_rotation = this->offset_orientation;
  }

  /**
   * @brief Get the weights for this spatial model
   * @param wx the weight to multiply (x_parent+mu_x-x_child)
   * @param wxx the weight to multiply (x_parent+mu_x-x_child)^2
   * @param wy the weight to multiply (y_parent+mu_y-y_child)
   * @param wyy the weight to multiply (y_parent+mu_y-y_child)^2
   * @param ws the weight to multiply (scale_parent+mu_scale-scale_child)
   * @param wss the weight to multiply (scale_parent+mu_scale-scale_child)^2
   * @param wr the weight to multiply (rot_parent+mu_rot-rot_child)
   * @param wrr the weight to multiply (rot_parent+mu_rot-rot_child)^2
   * @param wt the weight (cost) associated with the transition between the parent and child poses
   */
  void GetWeights(float *wx, float *wxx, float *wy, float *wyy, float *ws=NULL, float *wss=NULL, float *wr=NULL, float *wrr=NULL, float *wt=NULL, int childPose=-1, float *WW=NULL) {
    if(WW == NULL) WW = W;
    if(wx) *wx = WW[0];
    if(wy) *wy = WW[1];
    if(wr) *wr = WW[2];
    if(ws) *ws = WW[3];
    if(wxx) *wxx = WW[4];
    if(wyy) *wyy = WW[5];
    if(wrr) *wrr = WW[6];
    if(wss) *wss = WW[7];
    if(wt) { assert(childPose >= 0); *wt = WW[8+childPose]; }
  }

 private:
  /**
   * @brief Constructor
   */  
  ObjectPartPoseTransition();

  ~ObjectPartPoseTransition();
  

  /**
   * @brief Save a definition of this spatial model to a JSON object
   * 
   * @return A JSON encoding of this spatial model
   */
  Json::Value Save();

  /**
   * @brief Load an spatial model object from a JSON encoding
   * @param root A JSON encoding of this spatial model
   * @return True if successful
   */
  bool Load(const Json::Value &root);
 
  /**
   * @brief Get the parent part of this transition
   * @return The parent part
   */
  ObjectPart *GetParentPart() { return parentPart; }

  /**
   * @brief Get the child part of this transition
   * @return The child part
   */
  ObjectPart *GetChildPart() { return childPart; }

  /**
   * @brief Get the pose of the parent part of this transition
   * @return The parent part's pose
   */
  ObjectPose *GetParentPose() { return parentPose; }

  /**
   * @brief Get the pose of the child part of this transition
   * @return The child part's pose
   */
  ObjectPose *GetChildPose(int childPose) { return childPoses[childPose]; }



  /**
   * @brief Get an upper bound on the squared l2 norm of the feature space
   */
  float MaxFeatureSumSqr();

  int GetWeightConstraints(int *wc, bool *learn_weights, bool *regularize);

  /**
   * @brief Resolve pointers to other objects, parts, or attributes
   *
   * Typically, this is called after all classes, parts, and attribute definitions have been loaded
   */
  bool ResolveLinks(Classes *classes);

  int DebugWeights(float *w);

  friend class ObjectPart;
  friend class ObjectPartPoseTransitionInstance;
  friend class ObjectPoseInstance;
  friend class Classes;


  // deprecated
  int parentPartId, childPartId, parentPoseId, childPoseId;
  char *LoadFromString(char *str);
  char *ToString(char *str);
  bool ResolveLinksOld(Classes *classes);


 public:
  bool IsClick() { return isClick; }
  
  int ChildPoseInd(ObjectPose *pose);

  int AddChildPose(ObjectPose *pose);

  /**
   * @brief Get the weights associated with this part/pose transition
   * @param w a 9 dimensional vector of weights that is set by this function
   * @return The number of weights extracted
   */
  int GetWeights(float *w);

  /**
   * @brief Get the weights associated with this part/pose transition
   * @param w a 9 dimensional vector of weights that is set by this function
   */
  void SetWeights(float *w);
 
  /**
   * @brief Get the number of weights associated with this part/pose transition
   * @return The number of weights
   */ 
  int NumWeights();
};

/**
 * @class ObjectPartPoseTransitionInstance
 * @brief Defines a spatial model detection interface for a particular parent/child part and pose in a particular image
 * This class contains code to do inference to integrate out the child part; however, it is a helper function
 * of ObjectPoseInstance and all routines should go through that class
 */
class ObjectPartPoseTransitionInstance {
  ObjectPartPoseTransition *model;  /**< The model for this instance */
  ImageProcess *process; /**< Container for image processing */

  IplImage ***scores, ***scores_no_wt;  /**< A numScalesXnumOrientations array of responses for the parent part/pose that integrates out the child part/pose **/
  IplImage ***scores_scale_rot;  /**< A numScalesXnumOrientations array of responses for the parent part/pose that integrates out only the scale/orientation portion of thethe child part/pose **/
  IplImage ***best_offsets;  /**< A numScalesXnumOrientations array of maps of indices into the pixel location in the child pose that yields the highest score **/
  IplImage ***spatialScores; /**< A numScalesXnumOrientations array of spatial scores for a particular click point */

  float *custom_weights;
  float *custom_weights_buff; /**< Override the model weights */

  IplImage ***offsets;
  bool isFlipped;
  bool isReversed;
  IplImage ***childPoseMaxInds;

 public:
	 
  int NumChildPoses();

  /**
   * @brief Get the model associated with this part
   */
  ObjectPartPoseTransition *Model() { return model; }

  /**
   * @brief Get the ideal offsets for this spatial model (e.g., ideal spring offsets from parent to child part locations)
   * @param offset_norm Factor to multiply spatial offsets, used to control the relative magnitude of spatial offsets compared to other features during regularization
   * @param offset_x the x-offset mu_x
   * @param offset_y the y-offset mu_y
   * @param offset_scale the scale-offset mu_scale
   * @param offset_rotation the orientation-offset mu_rot
   */
  void GetOffsets(float *offset_norm, float *offset_x, float *offset_y, float *offset_scale, float *offset_rotation) {
    model->GetOffsets(offset_norm, offset_x, offset_y, offset_scale, offset_rotation);
    if(isReversed) {
      if(offset_x) *offset_x = -*offset_x;
      if(offset_y) *offset_y = -*offset_y;
      if(offset_scale) *offset_scale = -*offset_scale;
      if(offset_rotation) *offset_rotation = -*offset_rotation;
    }
    if(offset_x && isFlipped) *offset_x = -*offset_x;
  }

  /**
   * @brief Get the weights for this spatial model
   * @param wx the weight to multiply (x_parent+mu_x-x_child)
   * @param wxx the weight to multiply (x_parent+mu_x-x_child)^2
   * @param wy the weight to multiply (y_parent+mu_y-y_child)
   * @param wyy the weight to multiply (y_parent+mu_y-y_child)^2
   * @param ws the weight to multiply (scale_parent+mu_scale-scale_child)
   * @param wss the weight to multiply (scale_parent+mu_scale-scale_child)^2
   * @param wr the weight to multiply (rot_parent+mu_rot-rot_child)
   * @param wrr the weight to multiply (rot_parent+mu_rot-rot_child)^2
   * @param wt the weight (cost) associated with the transition between the parent and child poses
   */
  void GetWeights(float *wx, float *wxx, float *wy, float *wyy, float *ws=NULL, float *wss=NULL, float *wr=NULL, float *wrr=NULL, float *wt=NULL, int childPose=-1) {
    model->GetWeights(wx, wxx, wy, wyy, ws, wss, wr, wrr, wt, childPose, custom_weights ? custom_weights_buff : NULL);
    if(isReversed) {
      if(wx) *wx = -*wx;
      if(wy) *wy = -*wy;
      if(ws) *ws = -*ws;
      if(wr) *wr = -*wr;
    }
    if(wx && isFlipped) *wx = -*wx;
  }

 private:

  /**
   * @brief Constructor
   * @param m The model for this instance
   */
  ObjectPartPoseTransitionInstance(ObjectPartPoseTransition *m, ImageProcess *p, bool isFlip=false, bool isReversed=false);

  /**
   * @brief Destructor 
   */
  ~ObjectPartPoseTransitionInstance();


  /**
   * @brief Set custom detection weights for this spatial model instance, which override those of the spatial models
   * @param w An array of detection weights, with the same ordering as Model()->GetWeights()
   */
  int SetCustomWeights(float *w);

  
  /**
   * @brief Help build a list of all scales/orientations we need to run inference for
   * @param l_parts An array of part indices (to be appended with part_ind) by this function
   * @param l_poses An array of pose indices (to be appended with pose_ind) by this function
   * @param l_scales An array of scale indices (to be appended with scale indices) by this function
   * @param l_rotations An array of rotation indices (to be appended with orientation indices) by this function
   * @param part_ind The index of the child part
   * @param part_ind The index of the child pose
   * @param num A pointer to an index.  This is updated by this function
   */
  void InitDetect(int *l_parts, int *l_poses, int *l_scales, int *l_rotations, int part_ind, int pose_ind, int *num);

  /**
   * @brief For every possible position of the parent part, marginalize out the child part
   * @return A image of scores for every position of the parent part
   */
  IplImage *Detect(ObjectPoseInstance *parPose, int scale, int rot);

  /**
   * @brief Free all memory caches for detection
   */
  void Clear();

  /**
   * @brief Gets the location of the child part with maximum score for a given location of the
   *  parent.  Assumes Detect() has already been called
   * @return The location of the child part
   */
  PartLocation GetChildPartLocation(PartLocation *parentLoc);

  /**
   * @brief Build an image of spatial costs centered at (x,y)
   * @param loc The location of the parent part
   */
  IplImage ***BuildSpatialScores(PartLocation *loc);

  /**
   * @brief Get the spatial scores (assumed to have been computed by a previous call to GetSpatialScores()
   * @return a numScalesXnumOrientations array of spatial scores
   */
  IplImage ***GetSpatialScores() { return spatialScores; }


  float NormalizationConstant() { return 1.0f/sqrt(model->W[4]*model->W[5]); }

  int Debug(ObjectPoseInstance *parPose, float *w, float *f, PartLocation *parent_loc, PartLocation *child_loc, 
	    bool debug_scores, bool print_weights, float *f_gt=NULL);


  void FreeSpatialScores();
  void AllocateCacheTables();
  void AddUnaryWeights(int scale, int rot, float wt);

  void FinishDetect();

  CvPoint2D32f Offset() { CvPoint2D32f o = model->offset; if(isFlipped) { o.x=-o.x; } return o; }

  void SanityCheckLocation(PartLocation *par_loc);

  
  void ConvertDetectionImageCoordinatesUpdateMax(IplImage *srcImg, IplImage *dstImg, IplImage *srcOffset, IplImage *dstOffset, 
						 int srcScale, int srcRot, int dstScale, int dstRot);

 public:

  int ChildPoseInd(ObjectPoseInstance *pose);

  /**
   * @brief Get the parent part of this transition
   * @return The parent part
   */
  ObjectPartInstance *GetParentPart();

  /**
   * @brief Get the child part of this transition
   * @return The child part
   */
  ObjectPartInstance *GetChildPart();

  /**
   * @brief Get the pose of the parent part of this transition
   * @return The parent part's pose
   */
  ObjectPoseInstance *GetParentPose();

  /**
   * @brief Get the pose of the child part of this transition
   * @return The child part's pose
   */
  ObjectPoseInstance *GetChildPose(int childPose);

  /**
   * @brief Get the features associated with this part/pose transition
   * @param f a 9 dimensional vector of features that is set by this function
   * @param parent_loc The location of the parent part
   * @param child_loc The location of the child part
   * @param w If non-null, get the corresponding weights for each feature, storing them into this 9 dimensional vector 
   * @return The number of weights extracted
   */
  int GetFeatures(float *f, PartLocation *parent_loc, PartLocation *child_loc, float *w=NULL);

  friend class ObjectPoseInstance;


};

#endif

