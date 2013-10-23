/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#include "dataset.h"
#include "main.h"
//#include <vld.h>

/**
 * @example visualize.cpp
 *
 * Visualize a dataset, features, or learned model
 *
 * See VisualizeDataset() for more options
 *
 * Example usage:
 * - Visualize a dataset, learned part models, learned attribute models, and localized feature spaces
\htmlonly <div style="padding: 0.5em 1em; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; background-color: #eaeafa;">
    $ examples/bin/release_static/visualize.out CUB_200_2011_data/test.txt CUB_200_2011_data/classes.txt.v20q CUB_200_2011_data/html images,labels,features
</div> \endhtmlonly
 *
 */
int main(int argc, const char **argv) {
  if(argc < 5) {
    fprintf(stderr, "Visualize a dataset, learned part model, learned attribute model, labels, or image features\n");
    fprintf(stderr, "USAGE: ./visualize.out <test_set_file> <class_definition_file> <html_dir> <options>\n");
    fprintf(stderr, "         <test_set_file>: The list of test images with optional ground truth part/pose locations (see test_pose/part_test.txt)\n");
    fprintf(stderr, "         <class_definition_file>: The output of ./train_detectors_multiclass.out\n");
    fprintf(stderr, "         <html_dir>: output directory where an html visualization of prediction results will be stored\n");
    fprintf(stderr, "         <options>: Any combination of options parts,attributes,images,labels,features separated by commas, which respectively stand for visualizing a learned part model, learned attribute model, testset images, testset labels, or image features\n");
    fprintf(stderr, "         <sort_method>: Optional parameter determining the ordering that images are presented in the visualization.  Can be 'pose' (group images by pose), 'class' (group by class), score (sort by detection score), aspect (sort by image aspect ratio) \n");
    return -1;
  }

  char testSet[1000], classNameIn[1000], imagesDirOut[1000], opts[1000];
  int ( * sort ) ( const void *, const void * ) = NULL;
  strcpy(testSet, argv[1]);
  strcpy(classNameIn, argv[2]);
  strcpy(imagesDirOut, argv[3]);
  strcpy(opts, argv[4]);
  if(argc > 5) {
    if(!strcmp(argv[5], "pose"))
      sort = PartLocationsPoseCmp;
    else if(!strcmp(argv[5], "class"))
      sort = PartLocationsClassCmp;
    else if(!strcmp(argv[5], "score"))
      sort = PartLocationsScoreCmp;
    else if(!strcmp(argv[5], "aspect"))
      sort = PartLocationsAspectRatioCmp;
    else {
      fprintf(stderr, "Unsupported sort function %s\n", argv[5]);
      return -1;
    }
  }
  bool visualizePartModel = strstr(opts, "parts") != NULL;
  bool visualizeAttributeModels = strstr(opts, "attributes") != NULL;
  bool showImageGallery = strstr(opts, "images") != NULL;
  bool visualizeLabels = strstr(opts, "labels") != NULL;
  bool visualizeImageFeatures = strstr(opts, "features") != NULL;
  VisualizeDataset(testSet, classNameIn, imagesDirOut, visualizePartModel, visualizeAttributeModels, showImageGallery,
		   visualizeLabels, visualizeImageFeatures, sort);

  return 0;
}


 
