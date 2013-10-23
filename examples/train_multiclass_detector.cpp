/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#include "dataset.h"
#include "main.h"


/*
 * @file train_multiclass_detector.cpp
 * @brief Train a localized multiclass classifier with a shared part model for each class
 */

/**
 * @example train_multiclass_detector.cpp
 *
 * This is an example of how to train a localized multiclass classifier with a shared part model for each class.  For example, a bird species detector with 
 * a shared bird part model.  That part model is the same model as described in \ref train_detector.cpp.  The multiclass classifier can be done by training
 * localized attribute detectors, which are shared by different classes.  Attribute classifiers are trained to optimize classification accuracy.
 *
 * See BuildCodebooks(), TrainDetectors(), and TrainAttributes() for more options
 *
 * Example usage:
 * - Train a multiclass part-based object detector with a shared part model on training set train.txt, with model definition classes.txt, outputting the learned model to classes.txt.detector
\htmlonly <div style="padding: 0.5em 1em; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; background-color: #eaeafa;">
    $ examples/bin/release_static/train_multiclass_detector.out CUB_200_2011_data/train.txt CUB_200_2011_data/classes.txt CUB_200_2011_data/classes.txt.multi
</div> \endhtmlonly
 *
 */

int main(int argc, const char **argv) {
  // TODO: change these parameters or make them arguments to the program
  int maxImages=1000, ptsPerImage=50;

  char detector_name[1000];

  if(argc < 4) {
    fprintf(stderr, "Train a multiclass part-based object detector with shared part model\n");
    fprintf(stderr, "USAGE: ./train_multiclass_detector.out <training_set_file> <class_definition_file> <class_definition_out_file> <loss_type> <C_detect> <epsilon_detect> <C_multiclass> <epsilon_multiclass>\n");
    fprintf(stderr, "         <training_set_file>: The list of training images and ground truth part/pose locations (see data_pose/part_train.txt)\n");
    fprintf(stderr, "         <class_definition_file>: A configuration file containing the definition of all parts/poses (see data_pose/classes.txt)\n");
    fprintf(stderr, "         <class_definition_file_out>: Output file for the learned model parameters\n");
    fprintf(stderr, "         <loss_type>: Optional, one of intersection_parts, intersection_object, num_click_std_dev, num_parts_incorrect\n");
    fprintf(stderr, "         <C_detect>: Optional regularization constant for part detection training\n");
    fprintf(stderr, "         <epsilon_detect>: Optional training approximation level for part detection\n");
    fprintf(stderr, "         <C_multiclass>: Optional regularization constant for multiclass training\n");
    fprintf(stderr, "         <epsilon_multiclass>: Optional training approximation level for multiclass training\n");
    return -1;
  }

  bool trainDetector = true;
  if(argc > 4 && (!strcmp(argv[4], "0") || !strcmp(argv[4], "false")))
    trainDetector = false;
  PartDetectionLossType lossMethod = argc > 4 && trainDetector ? Classes::DetectionLossMethodFromString(argv[4]) :
    LOSS_PART_AVERAGE_AREA_UNION_OVER_INTERSECTION;
  double C_detect = argc > 5 ? atof(argv[5]) : 0;
  double eps_detect = argc > 6 ? atof(argv[6]) : 0;
  double C_multiclass = argc > 7 ? atof(argv[7]) : 0;
  double eps_multiclass = argc > 8 ? atof(argv[8]) : 0;
  bool attributeBased = argc > 9 ? (atoi(argv[9])!=0 || !strcmp(argv[9],"true")) : false;
  bool trainJointly = argc <= 10 || (atoi(argv[10])!=0 && strcmp(argv[10],"false"));

  if(trainDetector) {
    sprintf(detector_name, "%s.tmp.codebook.detector", argv[3]);
    if(!FileExists(detector_name)) 
      TrainDetectors(argv[1], argv[2], detector_name, lossMethod, NULL, C_detect, eps_detect);
  } else
    strcpy(detector_name, argv[2]);

  if(!FileExists(argv[3])) {
    if(attributeBased) {
      // Train attribute-based classification
      TrainAttributes(argv[1], detector_name, argv[3], trainJointly, NULL, C_multiclass, eps_multiclass, argc > 11 ? argv[11] : NULL, argc > 12 ? argv[12] : NULL);
    } else {
      // Train multiclass classification
      TrainMulticlass(argv[1], detector_name, argv[3], NULL, C_multiclass, eps_multiclass);
    }
  }

  fprintf(stderr, "Finished!\nTo evaluate performance on a testset, use ./test_multiclass_detector.out\n");
  return 0;
}

