/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#ifndef __INTERACTIVE_LABEL_H
#define __INTERACTIVE_LABEL_H

#include "opencvUtil.h"
#include "part.h"

/**
 * @file interactiveLabel.h
 * @brief  A class that handles interactive labeling of a deformable part model, including a simple OpenCV-based GUI and simulated testing using a testset.  A web-based GUI is available in examples/html/interactive_label.html
 */


class ImageProcess;
class Classes;
class ObjectPart;
class PartLocalizedStructuredLabelWithUserResponses;

/**
 * @class InteractiveLabelingSession
 * 
 * @brief A class that handles interactive labeling of a deformable part model, including a simple OpenCV-based GUI and simulated testing using a testset.  A web-based GUI is available in examples/html/interactive_label.html
 */
class InteractiveLabelingSession  {
  ImageProcess *process; /**< Contains the image we are looking at and all data structures used by computer vision */
  PartLocalizedStructuredLabelWithUserResponses *responses; /**< When evaluating performance on a testset, use this to lookup question answers */
  bool isInteractive; /**< If true, get responses from a real user.  Otherwise, get responses from a test example */
  double elapsedTime; /**< Duration of this session so far (in seconds) */

  char *htmlDebugDir;  /** If non-null, save an html visualization of the questions asked, probability maps, and part location samples */
  bool debugProbabilityMaps;

  int numMoved;                        /**< The number of parts that have been dragged */
  PartLocation **movedPartSequence;     /**< a numMoved array encoding the set of parts dragged (ordered by time) */
  PartLocation **partLocationHistory;  /**< a numMovedXnumParts array encoding the maximum likelihood location of all parts at each timestep */
  PartLocation *partLocations; /** a numParts array encoding the current maximum likelihood location of all parts */

  int overPart, dragPart;

  bool drawRects, drawLabels, drawPoints, drawTree;

  float *totalLosses, *partDists;
  CvPoint dragPoint;

  float zoom;

public:
  /**
   * @brief Constructor
   * 
   * @param p Contains the image we are looking at and all data structures used by computer vision
   * @param responses If non-NULL, when evaluating performance on a testset, use this to lookup question answers
   * @param isInteractive If true, get responses from a real user.  Otherwise, get responses from a test example
   * @param debugDir If non-null save an html visualization of the questions asked, probability maps, and part location samples
   * @param drawPoints draw part locations as circles
   * @param drawLabels draw text next to part locations of the part names
   * @param drawRects draw bounding boxes around part locations 
   * @param drawTree draw the part tree using lines between parent and child parts
   * @param zoom If zoom!=1, scale the image in the interface to make it bigger/smaller
   * @param debugProbabilityMaps during debugging, generate a visualization of the evolution of part probability maps as the user drags parts
   */
  InteractiveLabelingSession(ImageProcess *p, PartLocalizedStructuredLabelWithUserResponses *responses, bool isInteractive, bool drawPoints=true, 
			     bool drawLabels=false, bool drawRects=false, bool drawTree=true, float zoom=1, 
			     const char *debugDir=NULL, bool debugProbabilityMaps=false);

  /**
   * @brief Destructor
   */
  ~InteractiveLabelingSession();

  /**
   * @brief Label part locations interactively
   * 
   * @param maxActions When simulating the interface, the maximum number of parts that can be moved
   * @param stopThresh When simulating the interface, the number of standard deviations (measured using the mean
   * and standard deviation of multiple users clicking on the same part in the same image) that the predicted part
   * location must be within for the simulated user to approve the location
   * @return A numParts array of user-verified part locations
   */
  PartLocation *Label(int maxActions=6000, float stopThresh=0);

  /**
   * @brief Preprocess an image, running part detectors
   * @return the detection score
   */
  double Preprocess();

   /**
   * @brief Get the number of parts that have been dragged in this session
   */
  int NumDragged() { return numMoved; }

  /**
   * @brief Get the ith part moved during interactive labeling
   * @param i the index into the sequence of moved parts
   */
  PartLocation GetMovedPart(int i) { return *movedPartSequence[i]; }

  /**
   * @brief Get the max likelihood location of all part locations
   * @param i if i>=0, an index into the sequence of moved parts, such that we return the max likelihood locations after the ith part was moved.  Otherwise, we return the current max likelihood solution after the entire sequence of moved parts
   * @return a NumParts() array of part locations
   */
  PartLocation *GetPartLocations(int i=-1) { return i < 0 ? partLocations : partLocationHistory[i]; }

  /**
   * @brief Allow the user to submit a final location of a part (e.g., they have released the mouse when dragging a part)
   * @param loc the location of the dragged part
   * @param partDist optionaly, during simulation, store the distance in which the part was moved
   * @param totalLoss optionaly, during simulation, store the loss associated with the moved part before it was moved
   */
  void FinalizePartLocation(PartLocation *loc, float partDist=0, float totalLoss=0);

  /**
   * @brief During simulation, get the loss associated with some predicted part locations 
   * @param locs a NumParts() array defining the predicted location of each part 
   * @param losses a NumParts() array of losses populated by this function
   * @param worstPart If non-null, a pointer to an integer, into which the index of the part with highest loss is stored
   * @param stopThresh A threshold on the loss or distance, such that if a predicted part is within this distance it is considered to be correct
   * @param totalLoss If non-null, a pointer to a float into which the total number of incorrect parts is stored
   */
  float GetLoss(PartLocation *locs, float *losses, int *worstPart=NULL, float stopThresh=0, float *totalLoss=NULL);

  /// @cond
  bool DrawRects() { return drawRects; }
  bool DrawLabels() { return drawLabels; }
  bool DrawPoints() { return drawPoints; }
  bool DrawTree() { return drawTree; }

  void PrintDebugInfo();
  int PointOnPart(int x, int y);
  
  ImageProcess *Process() { return process; }
  int OverPart() { return overPart; }
  int DragPart(CvPoint *pt) { 
    if(pt) *pt = dragPoint;
    return dragPart; 
  }
  void SetDragPart(int d, CvPoint pt) { dragPart = d; dragPoint = pt; }
  void SetOverPart(int o) { overPart = o; }
  float Zoom() { return zoom; }
  /// @endcond
};


#endif
