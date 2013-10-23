#include "dataset.h"
#include "main.h"

/*
 * @file train_localized_v20q.cpp
 * @brief Train an interactive recognition system that combines part localization, pose-normalized multiclass classification, and interactive human feedback
 */

/**
 * @example train_localized_v20q.cpp
 *
 * This is an example of how to train computer vision and user models for the Visual 20 Questions Game With Clicks.  This allows people
 * to classify objects using a combination of computer vision (part detectors and multiclass classification) and user responses (asking
 * a user to click on a part, or asking the user to answer a yes/no question).  This routine first trains computer vision models
 * (same as \ref train_multiclass_detector.cpp), then uses a validation set to convert detection scores to probabilities, and finally
 * uses a training set of human responses to click and yes/no questions to learn a model of user input.
 *
 * See BuildCodebooks(), TrainDetectors(), TrainMulticlass(), TrainAttributes(), LearnAttributeProbabilities(), LearnAttributeUserProbabilities(), 
 * LearnUserClickProbabilities(), LearnDetectionProbabilities() for more options
 *
 * Example usage:
 * - Train an interactive classification system, including a part-based object detector, multiclass detector, and user models for different types of user input, using training set train.txt, model definition classes.txt, outputting the learned model to classes.txt.detector
\htmlonly <div style="padding: 0.5em 1em; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; background-color: #eaeafa;">
    $ examples/bin/release_static/train_localized_v20q.out CUB_200_2011_data/train.txt CUB_200_2011_data/validation.txt CUB_200_2011_data/classes.txt CUB_200_2011_data/classes.txt.v20q
</div> \endhtmlonly
 *
 */

int main(int argc, const char **argv) {
  // TODO: change these parameters or make them arguments to the program
  int maxImagesCodebook=1000, ptsPerImage=50, maxImagesAttributeProb=200, maxImagesDetectionProb=15;
  float gamma_cert[4] = { 32, 16, 8, 4 }, gamma_class=10;
  bool attributeBased = false;

  char detector_name[1000], attribute_name[1000], p1[1000], p2[1000];

  if(argc < 5) {
    fprintf(stderr, "Train a multiclass part-based object detector with shared part model and user response statistics for attribute questions and part click questions\n");
    fprintf(stderr, "USAGE: ./train_localized_v20q.out <training_set_file> <validation_set_file> <class_definition_file> <class_definition_out_file>\n");
    fprintf(stderr, "         <training_set_file>: The list of training images and ground truth part/pose locations (see data_pose/part_train.txt)\n");
    fprintf(stderr, "         <validation_set_file>: The list of validation images and ground truth part/pose locations, which are used to convert part and attribute detection scores to probabilities (see validation_pose/part_train.txt)\n");
    fprintf(stderr, "         <class_definition_file>: A configuration file containing the definition of all parts/poses (see data_pose/classes.txt)\n");
    fprintf(stderr, "         <class_definition_file_out>: Output file for the learned model parameters\n");
    return -1;
  }

  sprintf(p2, "%s.tmp.codebook.prob", argv[4]);
  if(!FileExists(p2))
    LearnAttributeUserProbabilities(argv[1], argv[3], p2, gamma_class, gamma_cert);

  //sprintf(p3, "%s.tmp.codebook.prob2", argv[4]);
  //if(!FileExists(p3))
  //LearnUserClickProbabilities(argv[1], p2, p3);
 
  sprintf(detector_name, "%s.tmp.codebook.detector", argv[4]);
  if(!FileExists(detector_name)) 
    TrainDetectors(argv[1], p2, detector_name, LOSS_NUM_INCORRECT);

  sprintf(attribute_name, "%s.tmp.codebook.detector.attributes", argv[4]);
  if(!FileExists(attribute_name)) {
    if(attributeBased) {
      // Train attribute-based classification
      TrainAttributes(argv[1], detector_name, attribute_name, true);
    } else {
      // Train multiclass classification
      TrainMulticlass(argv[1], detector_name, attribute_name);
    }
  }

  sprintf(p1, "%s.tmp.codebook.detector.attributes.prob", argv[4]);
  if(!FileExists(p1))
    LearnAttributeProbabilities(argv[2], attribute_name, p1, maxImagesAttributeProb);

  if(!FileExists(argv[4]))
    LearnDetectionProbabilities(argv[2], p1, argv[4], maxImagesDetectionProb);
  
  fprintf(stderr, "Finished!\nTo evaluate performance on a testset, use ./test_localized_v20q.out\n");
  return 0;
}

