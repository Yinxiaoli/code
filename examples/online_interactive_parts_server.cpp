/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#include "online_interactive_parts_server.h"


/*
 * @file online_interactive_parts_server.cpp
 * @brief A server for online learning of deformable part models with interactive labeling
 */

/**
 * @example online_interactive_parts_server.cpp
 *
 * This example trains a part detection model in online fashion.  While the model is being trained, a server listens over a network
 * connection for incoming requests.  A client can connect to this server and add new training images, use the current model to detect
 * parts on a test image, interactively label a new image, or evaluate performance on a testset.
 *
 * See TrainDetectors() for more options
 *
 * Example usage:
 * - Start a new server on port 8086, where a client can connect in to add training examples and/or interactively label part locations, where CUB_200_2011_data/train.txt is an initial training set.  See PartModelStructuredLearnerRpc for details on the network protocol 
\htmlonly <div style="padding: 0.5em 1em; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; background-color: #eaeafa;">
 $ examples/bin/release_static/online_interactive_parts_server.out -P 8086 -c CUB_200_2011_data/classes.txt -d CUB_200_2011_data/train.txt
</div> \endhtmlonly
 * - Start a new server on port 8086, where a client can interactively label a test image by connecting in over the network and CUB_200_2011_data/classes.txt.detector is the output of \ref train_detector.cpp, \ref train_multiclass_detector.cpp, or \ref train_localized_v20q.cpp.  See PartModelStructuredLearnerRpc for details on the network protocol
\htmlonly <div style="padding: 0.5em 1em; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; background-color: #eaeafa;">
 $ examples/bin/release_static/online_interactive_parts_server.out -P 8086 -c CUB_200_2011_data/classes.txt.detector 
</div> \endhtmlonly
 */

 /*
 *   - Train using a fixed dataset without running in server mode, outputting the learned model to CUB_200_2011_data/classes.txt.detector
\htmlonly <div style="padding: 0.5em 1em; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; background-color: #eaeafa;">
 examples/bin/release_static/online_interactive_parts_server.out -c CUB_200_2011_data/classes.txt -d CUB_200_2011_data/train.txt -o CUB_200_2011_data/classes.txt.detector
</div> \endhtmlonly
 *   - Evaluate a testset without running in server mode from a previously saved model:
\htmlonly <div style="padding: 0.5em 1em; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; background-color: #eaeafa;">
 $ examples/bin/release_static/online_interactive_parts_server.out -i CUB_200_2011_data/classes.txt.detector -t CUB_200_2011_data/test.txt CUB_200_2011_data/test.txt.predictions
</div> \endhtmlonly
 */

int main(int argc, const char **argv) {
  PartModelStructuredLearnerRpc v;
  return v.main(argc, argv);
}

