/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#include "dataset.h"
#include "main.h"


/*
 * @file train_detector.cpp
 * @brief Train deformable part model
 */

/**
 * @example train_detector.cpp
 *
 * This is an example of how to train a part detector.  The model allows for parts with hierarchical spatial relationships, sliding window
 * appearance score (using HOG templates or bag of words with SIFT or color features), rotatable/scalable parts, mixtures of poses, and
 * parts that can change rotation, orientation, pose, or visibility independently.  It can learn the optimal appearance and spatial parameters within an 
 * arbitrary epsilon for customizable loss functions.
 *
 * See TrainDetectors() for more options
 *
 * Example usage:
* - Train a part-based object detector on training set train.txt, with model definition classes.txt, outputting the learned model to classes.txt.detector
\htmlonly <div style="padding: 0.5em 1em; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; background-color: #eaeafa;">
    $ examples/bin/release_static/train_detector.out CUB_200_2011_data/train.txt CUB_200_2011_data/classes.txt CUB_200_2011_data/classes.txt.detector
</div> \endhtmlonly
 *
 */

int main(int argc, const char **argv) {
  if(argc < 4) {
    fprintf(stderr, "Train a part-based object detector\n");
    fprintf(stderr, "USAGE: ./train_detector.out <training_set_file> <class_definition_file> <class_definition_out_file> <loss_type> <C> <epsilon>\n");
    fprintf(stderr, "         <training_set_file>: The list of training images and ground truth part/pose locations (see data_pose/part_train.txt)\n");
    fprintf(stderr, "         <class_definition_file>: A configuration file containing the definition of all parts/poses (see data_pose/classes.txt)\n");
    fprintf(stderr, "         <class_definition_file_out>: Output file for the learned model parameters\n");
    fprintf(stderr, "         <background_training_set_file>: Optional list of background training images.\n");
    fprintf(stderr, "         <loss_type>: Optional, one of intersection_parts, intersection_object, num_click_std_dev, num_parts_incorrect\n");
    fprintf(stderr, "         <C>: Optional regularization constant\n");
    fprintf(stderr, "         <epsilon>: Optional training approximation level\n");
    return -1;
  }

  PartDetectionLossType lossMethod = argc > 4 ? Classes::DetectionLossMethodFromString(argv[4]) :
    LOSS_PART_AVERAGE_AREA_UNION_OVER_INTERSECTION;

  TrainDetectors(argv[1], argv[2], argv[3], lossMethod, NULL, argc>5 ? atof(argv[5]) : 0, argc>6 ? atof(argv[6]) : 0);
  fprintf(stderr, "Finished!\nTo evaluate performance on a testset, use ./test_detector.out\n");
  return 0;
}

