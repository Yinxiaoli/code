/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#include "dataset.h"
#include "main.h"


int main(int argc, const char **argv) {
  if(argc < 3) {
    fprintf(stderr, "USAGE: ./extract_image_features.out <training_set_file> <class_definition_file> <training_set_features.bin> <extract_localized_features> <max_images>\n");
    fprintf(stderr, "  e.g. ./extract_image_features.out CUB_200_2011_data/train.txt CUB_200_2011_data/classes.txt CUB_200_2011_data/train_image_features.bin false\n");
    fprintf(stderr, "         <training_set_file>: The list of training images and ground truth part/pose locations (see data_pose/part_train.txt)\n");
    fprintf(stderr, "         <class_definition_file>: A configuration file containing the definition of all parts/poses (see data_pose/classes.txt)\n");
    fprintf(stderr, "         <training_set_features.bin>: Output file for the extracted features\n");
    fprintf(stderr, "         <extract_localized_features>: Optional. If true, extract features around parts.  Otherwise, extract features from the full image\n");
    fprintf(stderr, "         <max_images>: Optional. Save features for a smaller dataset (randomly chosen) of max_images images\n");
    return -1;
  }
  
  ExtractImageFeatures(argv[1], argv[2], argv[3], argc > 4 ? (!strcmp(argv[4],"true") || !strcmp(argv[4],"1")) : false, argc > 5 ? atoi(argv[5]) : -1, argc > 6 ? atoi(argv[6]) : -1);

  fprintf(stderr, "To train a multiclass classifier, use ./structured_svm_multiclass.out -d %s -o learned_model.txt\n", argv[3]);
  fprintf(stderr, "To test a multiclass classifier,  use ./structured_svm_multiclass.out -t test.bin -i learned_model.txt\n");
  return 0;
}


 
