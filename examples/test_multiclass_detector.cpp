/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#include "dataset.h"
#include "main.h"


/*
 * @file test_multiclass_detector.cpp
 * @brief Evaluate part localization accuracy and/or localized multiclass classification after having 
 * learned a part model and localized multiclass classifiers
 */

/**
 * @example test_multiclass_detector.cpp
 *
 * This is an example of how to evaluate part localization accuracy and/or localized multiclass classification after having 
 * learned a part model and localized attribute detectors (see \ref train_multiclass_detector.cpp or \ref train_localized_v20q.cpp).
 *
 * See EvaluateTestset() for more options
 *
 * Example usage:
 * - Evaluate performance on a testset test.txt, output results to part_preds.txt
\htmlonly <div style="padding: 0.5em 1em; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; background-color: #eaeafa;">
    $ examples/bin/release_static/test_multiclass_detector.out CUB_200_2011_data/test.txt CUB_200_2011_data/classes.txt.multi part_preds.txt true detector_results.mat
</div> \endhtmlonly
 *
 */
int main(int argc, const char **argv) {
  if(argc < 4) {
    fprintf(stderr, "Evaluate a part-based object detector on a testset\n");
    fprintf(stderr, "USAGE: ./test_multiclass_detector.out <test_set_file> <class_definition_file> <predictions_file_out> <use_ground_truth_part_locations> <matlab_file_out> <debug_dir>\n");
    fprintf(stderr, "         <test_set_file>: The list of training images with optional ground truth part/pose locations (see test_pose/part_test.txt)\n");
    fprintf(stderr, "         <class_definition_file>: The output of ./train_multiclass_detectors.out\n");
    fprintf(stderr, "         <predictions_file_out>: Output file of predicted part locations, in the same format as test_set_file\n");
    fprintf(stderr, "         <use_ground_truth_part_locations>: Should be true or false.  If true, assume the ground truth part locations are given\n");
    fprintf(stderr, "         <matlab_file_out>: Optional output file with various statistics to a Matlab .mat file\n");
    fprintf(stderr, "         <matlab_file_out>: Optional output directory with html visualization of results\n");
    return -1;
  }
  bool useGTPartLocations = argc > 4 ? !strcmp(argv[4], "true") : false;
  EvaluateTestset(argv[1], argv[2],  !useGTPartLocations, true, argv[3], argc > 6 ? argv[6] : NULL, argc > 5 ? argv[5] : NULL);
  return 0;
}


