/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#include "dataset.h"
#include "main.h"
#include "interactiveLabel.h"


/*
 * @file interactive_label.cpp
 * @brief GUI for interactive part labeling on an image of your choice.  An alternate, web-based GUI is available in html/interactive_label.html
 */

//#include <vld.h>

/**
 * @example interactive_label.cpp
 *
 * This is an example of how to invoke a GUI for interactive part labeling on an image of your choice.  An alternate,
 * web-based GUI is available in html/interactive_label.html.  The interface
 * displays in realtime the maximum likelihood location (according to a learned part detector) of all parts as you click 
 * and drag different parts.  It assumes part detectors have already been 
 * trained (see \ref train_detector.cpp, \ref train_multiclass_detector.cpp, or \ref train_localized_v20q.cpp)
 *
 * See InteractiveLabelingSession for more options
 *
 * Example usage:
 * - Interactively label the parts of bird.png:
\htmlonly <div style="padding: 0.5em 1em; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; background-color: #eaeafa;">
 $ examples/bin/release_static/interactive_20q.out CUB_200_2011_data/classes.txt.detector bird.png
</div> \endhtmlonly
 * - Simulate the interactive labeling interface on a testset test.txt, plotting the results
\htmlonly <div style="padding: 0.5em 1em; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; background-color: #eaeafa;">
$ examples/bin/release_static/interactive_label.out -t CUB_200_2011_data/test.txt CUB_200_2011_data/classes.txt.v20q CUB_200_2011_data/interactive_label 1.5 1
</div> \endhtmlonly
 *
 */
int main(int argc, const char **argv) {
  float zoom = 1;
  bool drawPoints=true, drawLabels=false, drawRects=false, drawTree=true;
  bool debugProbabilityMaps=false;
 
  char classes_name[1000], image_name[1000], debug_dir[1000], mat_file[1000]; 
  strcpy(debug_dir, "");
  strcpy(mat_file, "");

  for(int i = 1; i < argc; i++) {
    if(!strcmp(argv[i], "-t")) {
      if(argc < i+3) {
	fprintf(stderr, "A GUI to interactively label an image\n");
	fprintf(stderr, "USAGE: ./interactive_label.out -t <test_set> <class_definition_file> <debug_dir_out> <stop_thresh> <debug_images> <debug_probability_maps>\n");
	fprintf(stderr, "         <test_set>: The test set of images and click responses (e.g. part_train.txt)\n");
	fprintf(stderr, "         <class_definition_file>: The output of ./train_detector, ./train_multiclass_detectors.out, or ./train_localized_v20q\n");
	fprintf(stderr, "         <debug_dir_out>: Optional directory to store HTML visualization of the sequence of GUI operations\n");
	fprintf(stderr, "         <stop_thresh>: The simulated user corrects a label if the current predicted part location is not within <stop_thresh> standard deviations of their true click response, where std dev is measured from MTurkers\n");
	return -1;
      }

      if(argc > 3) { 
	strcpy(debug_dir, argv[i+3]); 
	sprintf(mat_file, "%s/interactive.mat", debug_dir);
      }
      if(argc > i+7) strcpy(mat_file, argv[i+7]);
      EvaluateTestsetInteractive(argv[i+1], argv[i+2], argc > i+4 ? atof(argv[i+4]) : 1.5f, strlen(mat_file) ? mat_file : NULL, 
				 debug_dir, argc > i+5 ? atoi(argv[i+5]) : 0, argc > i+6 ? atoi(argv[i+6]) : 0, NULL); 
      return 0;
    }
  }

  if(argc < 3) {
    fprintf(stderr, "A GUI to interactively label an image\n");
    fprintf(stderr, "USAGE: ./interactive_label.out <class_definition_file> <image_name> <debug_dir_out>\n");
    fprintf(stderr, "         <class_definition_file>: The output of ./train_detector, ./train_multiclass_detectors.out, or ./train_localized_v20q\n");
    fprintf(stderr, "         <image_name>: The name of the image you want to label\n");
    fprintf(stderr, "         <debug_dir_out>: Optional directory to store HTML visualization of the sequence of GUI operations\n");
    return -1;
  }

  strcpy(classes_name, argv[1]);
  strcpy(image_name, argv[2]);
  if(argc > 3) strcpy(debug_dir, argv[3]);
  

  char str[10000];
  Classes *classes = new Classes();
  assert(classes->Load(classes_name));
  ImageProcess *process = new ImageProcess(classes, image_name, IM_MAXIMUM_LIKELIHOOD, true, false, true);
  InteractiveLabelingSession *l_session = new InteractiveLabelingSession(process, NULL, true, drawPoints, drawLabels, 
                                              drawRects, drawTree, zoom, argc > 3 ? debug_dir : NULL, debugProbabilityMaps);
  PartLocation *locs = l_session->Label();
  fprintf(stderr, "image=%s part_locations=[%s]\n", image_name, process->PrintPartLocations(locs, str));

  return 0;
}


