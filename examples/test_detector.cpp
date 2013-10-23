/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#include "dataset.h"
#include "main.h"


/*
 * @file test_detector.cpp
 * @brief Evaluate part localization accuracy after having learned a part model
 */

/**
 * @example test_detector.cpp
 *
 * This is an example of how to evaluate part localization accuracy after having learned a part model (see \ref train_detector.cpp or 
 * \ref train_multiclass_detector.cpp or \ref train_localized_v20q.cpp).
 *
 * See EvaluateTestset() for more options
 *
 * Example usage:
 * - Evaluate performance on a testset test.txt, output results to part_preds.txt
\htmlonly <div style="padding: 0.5em 1em; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; background-color: #eaeafa;">
    $ examples/bin/release_static/test_detector.out CUB_200_2011_data/test.txt CUB_200_2011_data/classes.txt.detector CUB_200_2011_data/part_preds.txt CUB_200_2011_data/detector_html CUB_200_2011_data/detector_results.mat
</div> \endhtmlonly
 *
 */
int main(int argc, const char **argv) {
  if(argc < 4) {
    fprintf(stderr, "Evaluate a part-based object detector on a testset\n");
    fprintf(stderr, "USAGE: ./test_detector.out <test_set_file> <class_definition_file> <predictions_file_out> <html_dir> <matlab_file_out>\n");
    fprintf(stderr, "         <test_set_file>: The list of training images with optional ground truth part/pose locations (see test_pose/part_test.txt)\n");
    fprintf(stderr, "         <class_definition_file>: The output of ./train_detectors.out\n");
    fprintf(stderr, "         <predictions_file_out>: Output file of predicted part locations, in the same format as test_set_file\n");
    fprintf(stderr, "         <html_dir>: Optional output directory where an html visualization of prediction results will be stored\n");
    fprintf(stderr, "         <matlab_file_out>: Optional output file with various statistics to a Matlab .mat file\n");
    return -1;
  }
  EvaluateTestset(argv[1], argv[2],  true, false, argv[3], argc > 4 ? argv[4] : NULL, argc > 5 ? argv[5] : NULL);
  return 0;
}


 
