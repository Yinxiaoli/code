#ifndef __DATASET_IO_H
#define __DATASET_IO_H

/**
 * @file datasetIO.h
 * @brief Implements routines to import various public datasets into our dataset format
 */


/**
 * @brief Import the Caltech-UCSD-Birds-200-2011 dataset into the format of this toolbox
 * @param datasetDir The Caltech-UCSD-Birds-200-2011 directory.  The extracted contents of http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
 * @param modelDefinitionOut The outputted model definition file (defines the set of classes, parts, poses, and attributes)
 * @param trainsetOut The outputted training set file
 * @param testsetOut The outputted test set file
 * @param validationSetOut The outputted validation set file
 * @param numValidation The number of images to use in the validation set
 * @param numPoses The number of aspects or mixture components to use in the pose model
 */
bool ImportCUBXBirds200(const char *datasetDir, const char *modelDefinitionOut, const char *trainsetOut, 
			const char *testsetOut, const char *validationSetOut, int numValidation, int numPoses);

///@cond
bool ImportVOC2009PoseletKeypointAnnotations(const char *matfile, const char *voc2009Dir, const char *trainsetOut, const char *testsetOut, const char *classesOut, int numPoses);
///@endcond

#endif

