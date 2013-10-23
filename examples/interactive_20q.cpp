/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#include "dataset.h"
#include "main.h"
#include "question.h"
#include "class.h"

/*
 * @file interactive_20q.cpp
 * @brief Invoke a debug GUI for the Visual 20 Questions Game on an image of your choice.  A much better GUI is available as a web application in examples/html/20q.html.
 */

/** 
 * @example interactive_20q.cpp
 * @brief Invoke a debug GUI for the Visual 20 Questions Game on an image of your choice.  A much better GUI is available as a web application in examples/html/20q.html.
 *
 * The Visual 20 Questions Game is an interactive method for fine-grained category classification that uses a combination of computer vision 
 * and user responses (mouse clicks and attribute question answers).  It assumes the computer vision routines have already been 
 * trained (see \ref train_localized_v20q.cpp).
 *
 * See QuestionAskingSession for more options
 *
 * Example usage:
 * - Open a GUI to interactively predict the species of bird.png, where classes.txt.v20q is the output of  \ref train_localized_v20q.cpp 
\htmlonly <div style="padding: 0.5em 1em; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; background-color: #eaeafa;">
 $ examples/bin/release_static/interactive_20q.out CUB_200_2011_data/classes.txt.v20q bird.png
</div> \endhtmlonly
 *
 */

int main(int argc, const char **argv) {
  char classes_name[1000], image_name[1000], debug_dir[1000];  
  if(argc < 3) {
    fprintf(stderr, "A GUI to interactively identify the class of an image\n");
    fprintf(stderr, "USAGE: ./interactive_20q.out <class_definition_file> <image_name> <debug_dir_out>\n");
    fprintf(stderr, "         <class_definition_file>: The output of ./train_localized_v20q\n");
    fprintf(stderr, "         <image_name>: The name of the image you want to label\n");
    fprintf(stderr, "         <debug_dir_out>: Optional directory to store HTML visualization of the sequence of GUI operations\n");
    return -1;
  }
  strcpy(classes_name, argv[1]);
  strcpy(image_name, argv[2]);
  if(argc > 3) strcpy(debug_dir, argv[3]);

  Classes *classes = new Classes();
  bool b = classes->Load(classes_name);
  assert(b);
  ImageProcess *process = new ImageProcess(classes, image_name, IM_MAXIMUM_LIKELIHOOD, true, true);
  QuestionAskingSession *session = new QuestionAskingSession(process, NULL, true, QS_INFORMATION_GAIN, argc > 3 ? debug_dir : NULL); 
  int c = session->AskAllQuestions();
  fprintf(stderr, "Predicted class is %s\n", classes->GetClass(c)->Name());

  return 0;
}


