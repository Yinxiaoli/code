/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/
 
#include "dataset.h"
#include "main.h"

/*
 * @file test_localized_v20q.cpp
 * @brief test performance of the Visual 20 Questions Game on a testset after having learned computer vision and user models
 */

//#include <vld.h>   // Uncomment to check for memory leaks in Visual Studio

/**
 * @example test_localized_v20q.cpp
 *
 * This is an example of how to test performance of the Visual 20 Questions Game With Clicks on a testset after having
 * learned computer vision and user models (see \ref train_localized_v20q.cpp).  This routine allows you to measure performance
 * for many different ways of selecting which question to ask, and then to plot all results on the same graph in matlab
 *
 * See EvaluateTestset20Q() for more options
 *
 * Example usage:
 * - Test performance of the Visual 20 Questions Game on testset test.txt, where classes.txt.v20q is the output of \ref train_localized_v20q.cpp, and CUB_200_2011_data/test20q is a directory that will store HTML visualizations:
\htmlonly <div style="padding: 0.5em 1em; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; background-color: #eaeafa;">
 $ examples/bin/release_static/test_localized_v20q.out CUB_200_2011_data/test.txt CUB_200_2011_data/classes.txt.v20q "stop_ig_all|stop_ig_binary|stop_ig_click|stop_ig_multiple|stop_rand_all|nocv_stop_ig_all" CUB_200_2011_data/test20q CUB_200_2011_data/test20q.mat
</div> \endhtmlonly
 * 
 */
#include "svgPlotter.h"

int main(int argc, const char **argv) {
  int isCorrectWindow=1, maxQuestions=60;


  if(argc < 4) {
    fprintf(stderr, "Evaluate a the visual 20 questions game with part clicks on a testset, possibly using several different criteria for choosing questions\n");
    fprintf(stderr, "USAGE: ./test_localized_v20q.out <test_set_file> <class_definition_file> <options> <html_dir> <matlab_file_out> <debug_html_num_class_print> <debug_html_generate_prob_maps>\n");
    fprintf(stderr, "         <test_set_file>: The list of training images with optional ground truth part/pose locations (see test_pose/part_test.txt)\n");
    fprintf(stderr, "         <class_definition_file>: The output of ./train_multiclass_detectors.out\n");
    fprintf(stderr, "         <options>: Options defining what experiment(s) to run.  Can be any subset of options\n");
    fprintf(stderr, "             [ig_all|ig_binary|ig_click|ig_multiple|time_binary|time_click|time_multiple|rand_all|rand_binary|rand_click|rand_multiple\n");
    fprintf(stderr, "              |stop_ig_all|stop_ig_binary|stop_ig_click|stop_ig_multiple|stop_time_all|stop_time_binary|stop_time_click|stop_time_multiple\n");
    fprintf(stderr, "              |stop_rand_all|stop_rand_binary|stop_rand_click|stop_rand_multiple\n");
    fprintf(stderr, "              |nocv_ig_all|nocv_ig_binary|nocv_ig_click|nocv_ig_multiple|nocv_time_all|nocv_time_binary|nocv_time_click|nocv_time_multiple\n");
    fprintf(stderr, "              |nocv_rand_all|nocv_rand_binary|nocv_rand_click|nocv_rand_multiple\n");
    fprintf(stderr, "              |nocv_stop_ig_all|nocv_stop_ig_binary|nocv_stop_ig_click|nocv_stop_ig_multiple|nocv_stop_time_all|nocv_stop_time_binary\n");
    fprintf(stderr, "              |nocv_stop_time_click|nocv_stop_time_multiple|nocv_stop_rand_all|nocv_stop_rand_binary|nocv_stop_rand_click|nocv_stop_rand_multiple]\n");
    fprintf(stderr, "           You can run multiple experiments by putting options between vertical bars.  'ig' selects \n");
    fprintf(stderr, "           questions by minimizing expected information gain, 'time' by minimizing expected time, 'random' randomly.\n");
    fprintf(stderr, "           'all' uses all types of questions, 'click' only uses clicks, 'binary' only uses binary, and 'multiple' only\n"); 
    fprintf(stderr, "           uses multiple choice.  If the prefix 'stop' is added, then it assumes the user will stop the session early (method 2), \n");
    fprintf(stderr, "           whereas by default it just measures accuarcy vs # of questions (method 1).  If the prefix 'nocv' is added, computer\n");
    fprintf(stderr, "           vision is disabled\n");
    fprintf(stderr, "         <html_dir>: Optional output directory where an html visualization of prediction results will be stored\n");
    fprintf(stderr, "         <matlab_file_out>: Optional output file with various statistics to a Matlab .mat file\n");
    return -1;
  }


  char opts[1000];
  strcpy(opts, argv[3]);
  const char *ptr = strtok(opts, "|");
  int num = 0;
  char labels[10000], files[10000];
  char id[1000], mfile[1000], mfile2[1000], mfile_folder[1000], mfile_file[1000], label[1000], dir[1000];
  strcpy(labels, "");
  strcpy(files, "");
  do {
    /*********************** Parse options ********************************************/
    bool disableClick, disableBinary, disableMultiple, stopEarly, disableComputerVision, disableCertainty = false;
    QuestionSelectMethod method;
    strcpy(id, ptr);

    if(!strncmp(ptr, "nocv_", 5)) { ptr+=5; disableComputerVision = true; strcpy(label, "No CV, "); }
    else { disableComputerVision = false; strcpy(label, ""); }

    if(!strncmp(ptr, "stop_", 5)) { ptr+=5; stopEarly = true; }
    else stopEarly = false;

    if(!strncmp(ptr, "ig_", 3)) { ptr+=3; method = QS_INFORMATION_GAIN; strcat(label, "By IG, "); }
    else if(!strncmp(ptr, "rand_", 5)) { ptr+=5; method = QS_RANDOM; strcat(label, "Random, "); }
    else if(!strncmp(ptr, "time_", 5)) { ptr+=5; method = QS_TIME_REDUCTION; strcat(label, "By Time, "); }
    else { fprintf(stderr, "Unrecognized option %s, %s\n", id, ptr); return -1; }

    if(!strncmp(ptr, "all", 3)) { disableClick=false; disableBinary=false; disableMultiple=false; strcat(label, "All"); ptr += 3; }
    else if(!strncmp(ptr, "click", 5)) { disableClick=false; disableBinary=true; disableMultiple=true; strcat(label, "Click"); ptr += 5; }
    else if(!strncmp(ptr, "binary", 6)) { disableClick=true; disableBinary=false; disableMultiple=true; strcat(label, "Binary"); ptr += 6; }
    else if(!strncmp(ptr, "nobinary", 8)) { disableClick=false; disableBinary=true; disableMultiple=false; strcat(label, "No Binary"); ptr += 8; }
    else if(!strncmp(ptr, "multiple", 8)) { disableClick=true; disableBinary=false; disableMultiple=false; strcat(label, "Multiple"); ptr += 8; }
    else { fprintf(stderr, "Unrecognized option %s, %s\n", id, ptr); return -1; }
    
    if(ptr && !strncmp(ptr, "_nocertainty", 12)) disableCertainty=true; 

    if(argc > 5) {
      strcpy(mfile, argv[5]); 
      StripFileExtension(mfile); 
      sprintf(mfile+strlen(mfile), "_%s.mat", id);
      strcpy(mfile2, mfile); StripFileExtension(mfile2); strcat(mfile2, ".bin");
      sprintf(labels+strlen(labels), "%s'%s'", num ? ", " : "", label);
      ExtractFolderAndFileName(mfile, mfile_folder, mfile_file);
	  StripFileExtension(mfile_file);
      sprintf(files+strlen(files), "%s'%s'", num ? ", " : "", mfile_file);
    }
    if(argc > 4 && strcmp(argv[4], "NULL")) 
      sprintf(dir, "%s_%s", argv[4], id);
    num++;



    /*********************** Run 20 Questions Game on Testset for specified options *******************/
    if(argc <= 5 || (!FileExists(mfile) && !FileExists(mfile2)))
      EvaluateTestset20Q(argv[1], argv[2], argc > 5 ? mfile : NULL, maxQuestions, 0, isCorrectWindow, stopEarly,
                       method, disableClick, disableBinary, disableMultiple, disableComputerVision, disableCertainty, argc > 4 && strcmp(argv[4], "NULL") ? dir : NULL, argc > 6 ? atoi(argv[6]) : 10, argc > 7 ? !strcmp(argv[7],"true") : false);
  } while((ptr=strtok(NULL,"|")) != NULL);


  /***************** Auto-Generate file for Matlab plotting *****************************/
  if(argc > 5) {
    char fname[1000];
    strcpy(fname, argv[5]);
    StripFileExtension(fname);
    strcat(fname, "_plot.m");
    FILE *fout = fopen(fname, "w");
    fprintf(fout, "plotStyle = { '--', '-', '-.', '-v', '-p', '-s', '-x', '*', '+' };\n"
	    "plotColor = { 'b', 'r', 'g', 'c', 'm', 'y', 'k' };\n"
	    "matfiles = {%s};\n"
	    "labels = {%s};\n"
	    "timeInterval=1;\n"
	    "figure(1); clf; hold on;\n"
	    "title('Classification w/ Humans in the Loop');\n"
	    "xlabel('Number of Questions Asked');\n"
	    "ylabel('Classification Accuracy');\n"
	    "for i=1:length(matfiles)\n"
	    "  eval(matfiles{i});\n"
	    "  plot(0:timeInterval:(length(accuracy)-1), accuracy, plotStyle{i}, 'LineWidth', 3, 'Color', plotColor{i});\n"
	    "end\n"
	    "legend(labels, 'Location', 'Southeast');\n", files, labels);
    fclose(fout);
  }

  return 0;
}


