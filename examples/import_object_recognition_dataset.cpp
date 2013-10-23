/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#include "dataset.h"
#include "main.h"
#include "fisher.h"
#include "structured_svm_multiclass.h"

#define FSIFT_CODEBOOK_SIZE 100
#define FSIFT4_CODEBOOK_SIZE 100
#define FCIE_CODEBOOK_SIZE 100
#define NUM_SPATIAL_PYR_LEVELS 2
#define RESIZE_IMAGE_WIDTH 640

extern int g_start, g_end;

int strcmp2(const void *s1,const void *s2) { 
  return strcmp(*(const char**)s1,*(const char**)s2); 
}

  
/**
 * @example import_object_recognition_dataset.cpp
 *
 * Import an object recognition dataset in the format of Caltech-101, or caltech-256, computing image features.
 *
 * Example usage:
 * - Import from Caltech-101, where http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz is extracted into the directory 101_ObjectCategories, excluding the directories BACKGROUND_Google and Faces_easy, with 30 training images per category
\htmlonly <div style="padding: 0.5em 1em; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; background-color: #eaeafa;">
$ examples/bin/release_static/import_object_recognition_dataset.out 101_ObjectCategories 101_ObjectCategories_data 30 BACKGROUND_Google,Faces_easy
</div> \endhtmlonly
 * - Alternatively, import from Caltech-256, where http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar is extracted into the directory 256_ObjectCategories, excluding the directory 257.clutter
 \htmlonly <div style="padding: 0.5em 1em; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; background-color: #eaeafa;">
 $ examples/bin/release_static/import_object_recognition_dataset.out 256_ObjectCategories 256_ObjectCategories_data 30 257.clutter
</div> \endhtmlonly
 * - Train a multiclass SVM on the computed image features:
\htmlonly <div style="padding: 0.5em 1em; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; background-color: #eaeafa;">
 $ online_structured_svm/examples/bin/release_static/structured_svm_multiclass.out -d 101_ObjectCategories_data/train.feat.bin -o 101_ObjectCategories_data/model.txt -T 1 
</div> \endhtmlonly
 * - Evaluate the learned multiclass SVM on the testset:
\htmlonly <div style="padding: 0.5em 1em; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; background-color: #eaeafa;">
 $ online_structured_svm/examples/bin/release_static/structured_svm_multiclass.out -t 101_ObjectCategories_data/test.feat.bin 101_ObjectCategories_data/test.txt.preds -i 101_ObjectCategories_data/model.txt -T 1 -D no_dataset
</div> \endhtmlonly

*/
int main(int argc, const char **argv) {
  if(argc < 4) {
    fprintf(stderr, "USAGE: ./import_object_recognition_dataset.out <dataset_dir> <output_dir> <num_train_images_per_class> <exclude_dirs>\n");
    fprintf(stderr, "  e.g. ./import_object_recognition_dataset.out 256_ObjectCategories 256_ObjectCategories_data 30 257.clutter\n");
    return -1;
  }
  char datasetDir[1000], outputDir[1000], excludeStr[1000], dir[1000], fname[1000];
  char *excludeClasses[1000];


  strcpy(datasetDir, argv[1]);
  strcpy(outputDir, argv[2]);
  int numTrainPerCategory = atoi(argv[3]);
  strcpy(excludeStr, argc >= 5 ? argv[4] : "");
  int numExcludeClasses = argc >= 5 ? SplitString(excludeStr, excludeClasses, ",") : 0;

  char classFile[1000], trainFile[1000], testFile[1000], trainFeaturesFile[1000], testFeaturesFile[1000], svmParams[1000];
  sprintf(classFile, "%s/classes.txt", outputDir);
  sprintf(trainFile, "%s/train.txt", outputDir);
  sprintf(testFile, "%s/test.txt", outputDir);
  sprintf(trainFeaturesFile, "%s/train.feat.bin", outputDir);
  sprintf(testFeaturesFile, "%s/test.feat.bin", outputDir);
  sprintf(svmParams, "%s/svm_params.txt", outputDir);

  // Add SIFT features (at 2 scales, cellWidth=4,8) and CIE color features, with a Fisher-codebook
  int numSift_2x2, numSift4_2x2, numCIE_2x2, numSift_1x3, numSift4_1x3, numCIE_1x3;
  Classes *classes = new Classes();
  FeatureWindow *fsift_2x2 = SpatialPyramidFeature("FISHER_SIFT", &numSift_2x2, NUM_SPATIAL_PYR_LEVELS, FSIFT_CODEBOOK_SIZE*my_min(128,FISHER_PCA_DIMS)*2, 2, 2);
  FeatureWindow *fsift4_2x2 = SpatialPyramidFeature("FISHER_SIFT4", &numSift4_2x2, NUM_SPATIAL_PYR_LEVELS, FSIFT4_CODEBOOK_SIZE*my_min(128,FISHER_PCA_DIMS)*2, 2, 2);
  FeatureWindow *fcie_2x2 = SpatialPyramidFeature("FISHER_CIE", &numCIE_2x2, NUM_SPATIAL_PYR_LEVELS, FCIE_CODEBOOK_SIZE*48*2, 2, 2);
  FeatureWindow *fsift_1x3 = SpatialPyramidFeature("FISHER_SIFT", &numSift_1x3, NUM_SPATIAL_PYR_LEVELS, FSIFT_CODEBOOK_SIZE*my_min(128,FISHER_PCA_DIMS)*2, 1, 3, 1);
  FeatureWindow *fsift4_1x3 = SpatialPyramidFeature("FISHER_SIFT4", &numSift4_1x3, NUM_SPATIAL_PYR_LEVELS, FSIFT4_CODEBOOK_SIZE*my_min(128,FISHER_PCA_DIMS)*2, 1, 3, 1);
  FeatureWindow *fcie_1x3 = SpatialPyramidFeature("FISHER_CIE", &numCIE_1x3, NUM_SPATIAL_PYR_LEVELS, FCIE_CODEBOOK_SIZE*48*2, 1, 3, 1);
  classes->AddFeatureWindows(fsift_2x2, numSift_2x2);
  classes->AddFeatureWindows(fsift4_2x2, numSift4_2x2);
  /*classes->AddFeatureWindows(fsift_2x2, numSift_2x2);
    classes->AddFeatureWindows(fsift4_2x2, numSift4_2x2);
    classes->AddFeatureWindows(fcie_2x2, numCIE_2x2);
    classes->AddFeatureWindows(fsift_1x3, numSift_1x3);
    classes->AddFeatureWindows(fsift4_1x3, numSift4_1x3);
    classes->AddFeatureWindows(fcie_1x3, numCIE_1x3);
  */
  free(fsift_2x2);
  free(fsift4_2x2);
  free(fcie_2x2);
  free(fsift_1x3);
  free(fsift4_1x3);
  free(fcie_1x3);

  Dataset *trainset = new Dataset(classes);
  Dataset *testset = new Dataset(classes);

  // Assume the dataset directory has a bunch of sub-directories, where each sub-directory contains images for one category
  int numCategories;
  char **dirs = ScanDir(argv[1], "dir", &numCategories);
  qsort(dirs, numCategories, sizeof(char*), strcmp2);
  int *cat_counts = Create1DArray<int>(numCategories), num = 0, numGoodCat = 0;
  for(int i = 0; i < numCategories; i++) {
    bool exclude = !strcmp(dirs[i], ".") || !strcmp(dirs[i], "..");
    for(int j = 0; j < numExcludeClasses; j++) 
      if(!strcmp(excludeClasses[j], dirs[i]))
	exclude = true;
    if(exclude)
      continue;

    ObjectClass *cl = new ObjectClass(classes, dirs[i]);
    classes->AddClass(cl);
    fprintf(stderr, "Processing class %s...\n", dirs[i]);

    // Add the images for this category
    int numImages;
    sprintf(dir, "%s/%s", argv[1], dirs[i]);
    char **images = ScanDir(dir, "jpg|JPG|jpeg|JPEG|png|PNG|gif|GIF", &numImages);
    int *split = RandSplit(numTrainPerCategory, numImages);
    for(int j = 0; j < numImages; j++) {
      sprintf(fname, "%s/%s", dir, images[j]);
      StructuredExample *ex = new StructuredExample;
      PartLocalizedStructuredData *x = new PartLocalizedStructuredData();
      MultiObjectLabelWithUserResponses *y = new MultiObjectLabelWithUserResponses(x);
      PartLocalizedStructuredLabelWithUserResponses *l = new PartLocalizedStructuredLabelWithUserResponses(x);
      l->SetClassID(cl->Id());
      y->AddObject(l);
      ex->x = x;   ex->y = y;   
      x->SetImageName(StringCopy(fname));

      if(split[j] == 0)
	trainset->AddExample(ex);
      else 
	testset->AddExample(ex);
    }

    cat_counts[numGoodCat] = my_max(0, numImages-numTrainPerCategory);
    num += cat_counts[numGoodCat++];
      
    free(split);
    free(images);
  }
    
  free(dirs);
  
  CreateDirectoryIfNecessary(outputDir);
  classes->Save(classFile);
  trainset->Randomize();
  trainset->Save(trainFile);
  testset->Randomize();
  testset->Save(testFile);
  delete trainset;
  delete testset;
  delete classes;

  // Optional cost-sensitive version, where each class is weighted by its inverse frequency, such that
  // the average loss will be the class-average loss
  double **confCosts = Create2DArray<double>(numGoodCat, numGoodCat);
  for(int i = 0; i < numGoodCat; i++) 
    for(int j = 0; j < numGoodCat; j++) 
      confCosts[i][j] = i==j || !cat_counts[i] ? 0 : (((double)num)/numCategories)/cat_counts[i];
  MulticlassStructuredSVM *svm = new MulticlassStructuredSVM;
  svm->SetNumClasses(numGoodCat);
  svm->SetClassConfusionCosts(confCosts);
  ((StructuredSVM*)svm)->Save((const char*)svmParams);
  free(confCosts);
  free(cat_counts);
  delete svm;
  
  ExtractImageFeatures(trainFile, classFile, trainFeaturesFile, false, -1, RESIZE_IMAGE_WIDTH);
  ExtractImageFeatures(testFile, classFile, testFeaturesFile, false, -1, RESIZE_IMAGE_WIDTH);

  fprintf(stderr, "To train a multiclass classifier, use structured_svm_multiclass.out -d %s -o learned_model.txt\n", trainFeaturesFile);
  fprintf(stderr, "    or to train a version where each class is weighted by its inverse frequence use structured_svm_multiclass.out -i %s -d %s -o learned_model.txt\n", svmParams, trainFeaturesFile);
  fprintf(stderr, "To test a multiclass classifier,  use structured_svm_multiclass.out -t %s -i learned_model.txt\n", testFeaturesFile);
  return 0;
}


 
