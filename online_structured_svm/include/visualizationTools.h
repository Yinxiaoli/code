#ifndef __VISUALIZATION_TOOLS_H
#define __VISUALIZATION_TOOLS_H

#include "util.h"

/**
 * @file visualizationTools.h
 * @brief Helper routines for making image galleries or interactive confusion matrices
 */


typedef struct _ExampleVisualization {
  char *fname;
  char *thumb;
  char *description;
  double loss;
} ExampleVisualization;


ExampleVisualization *AllocateExampleVisualization(const char *fname, const char *thumb, const char *description, double loss);

void BuildGallery(const char **images, int numImages, const char *outFileName, const char *title, 
		  const char *header, const char **thumbs, const char **imageDescriptions, int numThumbs=15);
void BuildConfusionMatrix(int *predLabels, int *gtLabels, int numExamples, const char *outFileName,
			  const char **classNames, const char *title, const char *header, const char **imageNames, 
			  const char **imageLinkNames=NULL, int *classGroups=NULL, int numGroups=0, int confMatWidth=1600);


#endif
