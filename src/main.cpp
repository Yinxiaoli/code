/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/


#include "feature.h"
#include "imageProcess.h"
#include "part.h"
#include "pose.h"
#include "attribute.h"
#include "question.h"
#include "dataset.h"
#include "class.h"
#include "online_interactive_server.h"
#include "structured_svm_partmodel.h"
#include "structured_svm_class_attribute.h"
#include "structured_svm_multiclass.h"
#include "main.h"
#include "fisher.h"

#ifdef HAVE_MATLAB  
#include "mat.h"
#include "engine.h"
#endif


#define DESCRIPTOR_SPATIAL_WIDTH 4

/*
 * Train part detectors from a set of images and part locations (defined in trainSet).
 * classNameIn defines the set of parts/poses.  The learned detectors are written out
 * to classNameOut, which is the same format file as classNameIn
 */
void TrainDetectors(const char *trainSet, const char *classNameIn, const char *classNameOut, PartDetectionLossType lossType, double *partLoss, double C, double eps) {
  fprintf(stderr, "Training part detectors %s...\n", classNameOut);
  Classes *classes = new Classes(); 
  bool b = classes->Load(classNameIn);  assert(b); 
  if(eps == 0) eps = .05;
  classes->SetDetectionLossMethod(lossType);

  // Add in the set of allowed transitions between parts/poses, and compute the mean offset between parent and child parts
  if(!classes->HasPartPoseTransitions()) {
    Dataset *dv = new Dataset(classes);
    bool b = dv->Load(trainSet); assert(b);
    fprintf(stderr, "Computing initial part-aspect spatial offsets...");
    classes->AddSpatialTransitions(dv, 1, false, true); 
    fprintf(stderr, "...");
    classes->ComputePartPoseTransitionWeights(dv, false, true);
    fprintf(stderr, "done\n");
    delete dv;
    char fname[1000];
    sprintf(fname, "%s.spatial", classNameIn);
    classes->Save(fname);
  }
  if((lossType == LOSS_USER_STANDARD_DEVIATIONS || lossType == LOSS_NUM_INCORRECT) && !classes->HasClickPartPoseTransitions()) {
    Dataset *dv = new Dataset(classes);
    bool b = dv->Load(trainSet);  assert(b);
    fprintf(stderr, "Computing part click statistics...");
    dv->LearnUserClickProbabilities();
    fprintf(stderr, "done\n");
    delete dv;
    char fname[1000];
    sprintf(fname, "%s.click", classNameIn);
    classes->Save(fname);
  }

  if(!partLoss) {
    partLoss = (double*)malloc(sizeof(double)*classes->NumParts());
    for(int i = 0; partLoss && i < classes->NumParts(); i++)
      partLoss[i] = 1.0/classes->NumParts();
  }

  double sum = partLoss ? 0 : classes->NumParts();
  for(int i = 0; partLoss && i < classes->NumParts(); i++)
    sum += partLoss[i];
  
  if(!C)
    C = .1*sum*classes->NumPoses()/classes->NumParts()*(NORMALIZE_TEMPLATES ? SQR(441) : 1);
    //C = .1*sum*classes->NumPoses()/classes->NumParts();

  PartModelStructuredSVM *learner = classes->SupportsMultiObjectDetection() && lossType != LOSS_DETECTION ? 
    new MultiObjectStructuredSVM : new PartModelStructuredSVM;
  StructuredSVMTrainParams *params = learner->GetTrainParams();
  learner->SetClasses(classes);
  learner->SetPartLosses(partLoss);
  params->C = C;
  params->eps = eps*sum;
  //learner->RunMultiThreaded(0);
  if(classes->SupportsMultiObjectDetection() && lossType == LOSS_DETECTION) {
    // Convert a dataset with multiple objects per image to a dataset with at most one object per image,
    // by making each positive occurrence of an object into a training example
    MultiObjectStructuredSVM learner2;
    learner2.SetClasses(classes);
    learner2.LoadTrainset(trainSet);
    StructuredDataset *binary_dataset = learner2.ConvertToSingleObjectDataset(learner2.GetTrainset());
    char binaryTrainSet[1000];
    sprintf(binaryTrainSet, "%s.binary", trainSet);
    learner->SaveDataset(binary_dataset, binaryTrainSet);
    learner->SetTrainset(binary_dataset);
  } else
    learner->LoadTrainset(trainSet);
  learner->GetTrainset()->Randomize();
  learner->Train();
  
  SparseVector *w = learner->GetCurrentWeights();
  float *ww = w->get_non_sparse<float>(learner->GetSizePsi());
  classes->SetWeights(ww, true, false); 
  classes->Save(classNameOut);
  delete learner;
  
  delete w;
  free(ww);
  delete classes;
}
 
/*
 * Train attribute detectors from a set of images and part locations (defined in trainSet).
 * classNameIn defines the set of parts/poses.  The learned detectors are written out
 * to classNameOut, which is the same format file as classNameIn
 */
void TrainAttributes(const char *trainSet, const char *classNameIn, const char *classNameOut, bool trainJointly, double **classConfusionCost, double C, double eps, const char *validationFile, const char *optMethod) {
  fprintf(stderr, "Training attribute detectors %s...\n", classNameOut);
  if(eps == 0) eps = .03;

  Classes *classes = new Classes(); bool b=classes->Load(classNameIn);  assert(b);
  
  if(!classes->NumCodebooks() && !classes->NumFisherCodebooks()) {
    char cname[1000];
    sprintf(cname, "%s.codebook", classNameIn);
    BuildCodebooks(trainSet, classNameIn, classNameIn, cname, 1000, 50);
    delete classes;
    classes = new Classes();
    bool b = classes->Load(classNameIn);  assert(b);
  }

  classes->SetScaleAttributesIndependently(!trainJointly);

  int maxLoss = classConfusionCost ? 0 : 1;
  for(int i = 0; classConfusionCost && i < classes->NumClasses(); i++) 
    for(int j = 0; j < classes->NumClasses(); j++) 
      maxLoss = my_max(classConfusionCost[i][j], maxLoss);

  ClassAttributeStructuredSVM learner;
  StructuredSVMTrainParams *params = learner.GetTrainParams();
  learner.SetTrainJointly(trainJointly);
  if(optMethod) params->method = OptimizationMethodFromString(optMethod);
  learner.SetClasses(classes);

  char fname[400];
  char *modelout = NULL;
  if(validationFile) {
    params->dumpModelStartTime = 10;
    params->dumpModelFactor = 1.5;
    sprintf(fname, "%s.dump", classNameOut);
    modelout = fname;
    learner.SetValidationFile(validationFile);
  }

  //if(classConfusionCost)
  learner.SetClassConfusionCosts(classConfusionCost);
  //else
  //learner.SetClassConfusionCostsByAttribute();
  
  //learner.SetUseClassConfusionLoss(trainJointly);
  //learner.SetUseAttributeDetectionLoss(!trainJointly);
  params->C = C ? C : 5000.0*maxLoss;
  params->eps = eps*maxLoss;
  learner.LoadTrainset(trainSet);
  learner.GetTrainset()->Randomize();
  learner.Train(modelout);
  
  SparseVector *w = learner.GetCurrentWeights();
  float *ww = w->get_non_sparse<float>(learner.GetSizePsi());
  classes->SetWeights(ww, false, true); 
  classes->Save(classNameOut);
  
  delete w;
  free(ww);
  delete classes;
}
  
void TrainMulticlass(const char *trainSet, const char *classNameIn, const char *classNameOut, double **classConfusionCost, double C, double eps) {
  char feat[1000], mod[1000];
  sprintf(feat, "%s.bin", trainSet);
  if(!FileExists(feat))
    ExtractImageFeatures(trainSet, classNameIn, feat, true);
  else 
    fprintf(stderr, "%s already computed\n", feat);

  MulticlassStructuredSVM *learner = new MulticlassStructuredSVM;
  double maxLoss = 1;
  Classes *classes = new Classes(); bool b=classes->Load(classNameIn);  assert(b); 
  
  if(!classes->NumCodebooks() && !classes->NumFisherCodebooks()) {
    char cname[1000];
    sprintf(cname, "%s.codebook", classNameIn);
    BuildCodebooks(trainSet, classNameIn, classNameIn, cname, 1000, 50);
    delete classes;
    classes = new Classes();
    bool b = classes->Load(classNameIn);  assert(b);
  }
  

  int n = classes->NumWindowFeatures()*classes->NumParts();
  if(eps == 0) 
    eps = .01;
  if(classConfusionCost)
    learner->SetClassConfusionCosts(classConfusionCost);
  StructuredSVMTrainParams *params = learner->GetTrainParams();
  params->runMultiThreaded = 1; // multi-threaded, detect number of cores
  params->memoryMode = MEM_KEEP_DUAL_VARIABLES_IN_MEMORY;  // handle case where dataset is larger than memory
  params->C = C ? C : 5000.0*maxLoss;
  params->eps = eps*maxLoss;
  learner->LoadTrainset(feat);
  learner->GetTrainset()->Randomize();
  learner->Train();

  //sprintf(mod, "%s.svm", classNameOut);
  //((StructuredSVM*)&learner)->Save(mod, false);

  SparseVector *w = learner->GetCurrentWeights();
  float *ww = w->get_non_sparse<float>(n*classes->NumClasses());
  for(int i = 0; i < classes->NumClasses(); i++)
    classes->GetClass(i)->SetWeights(ww + i*n, n);
  delete learner;


  classes->Save(classNameOut);

  delete w;
  free(ww);
}


void LearnAttributeProbabilities(const char *trainSet, const char *classNameIn, const char *classNameOut, int maxImages) {
  fprintf(stderr, "Learning attribute detection score to probability conversion %s...\n", classNameOut);

  Classes *classes = new Classes();
  bool b = classes->Load(classNameIn);  assert(b);
  Dataset *dv = new Dataset(classes);
  b = dv->Load(trainSet); assert(b);
  dv->LearnClassAttributeDetectionParameters(maxImages);
  classes->Save(classNameOut);
  delete dv;
  delete classes;
}

void LearnAttributeUserProbabilities(const char *trainSet, const char *classNameIn, const char *classNameOut, 
			float gamma_class, float *gamma_cert) {
  fprintf(stderr, "Learning attribute question user response probabilities %s...\n", classNameOut);

  Classes *classes = new Classes();
  bool b = classes->Load(classNameIn);  assert(b);
  Dataset *dv = new Dataset(classes);
  b = dv->Load(trainSet);  assert(b);
  dv->LearnClassAttributeUserProbabilities(gamma_class, gamma_cert);
  classes->Save(classNameOut);
  delete dv;
  delete classes;
}

void LearnUserClickProbabilities(const char *trainSet, const char *classNameIn, const char *classNameOut) {
  fprintf(stderr, "Learning part click question user response probabilities %s...\n", classNameOut);

  Classes *classes = new Classes();
  bool b = classes->Load(classNameIn);  assert(b);
  Dataset *dv = new Dataset(classes);
  b = dv->Load(trainSet);  assert(b);
  dv->Randomize();
  dv->LearnUserClickProbabilities();
  classes->Save(classNameOut);
  delete dv;
  delete classes;
}

void LearnDetectionProbabilities(const char *trainSet, const char *classNameIn, const char *classNameOut, int maxDetectionImages) {
  fprintf(stderr, "Learning part detection score to probability conversion %s...\n", classNameOut);

  Classes *classes = new Classes();
  bool b = classes->Load(classNameIn); assert(b);
  Dataset *dv = new Dataset(classes);
  b = dv->Load(trainSet);  assert(b);
  dv->Randomize();
  dv->LearnPartDetectionParameters(maxDetectionImages);
  delete dv;
  classes->Save(classNameOut);
  delete classes;
}


/*
 * Build some codebooks for histogram features.  This includes a bag of words codebook using SIFT.
 * and codebooks for RGB color histograms and CIE color histograms
 */
void BuildCodebooks(const char *trainSet, const char *classNameIn, const char *classNameOut, 
		    const char *dictionaryOutPrefix, int maxImages, int ptsPerImage, int resize_image_width) {
  Classes *classes = new Classes();
  bool b = classes->Load(classNameIn);  assert(b);
  Dataset *dv = new Dataset(classes); 
  b = dv->Load(trainSet);  assert(b);
  dv->Randomize();

  
  // Add in the set of allowed transitions between parts/poses, and compute the mean offset between parent and child parts
  if(!classes->HasPartPoseTransitions()) {
    fprintf(stderr, "Computing initial part-aspect spatial offsets...");
    classes->AddSpatialTransitions(dv, 1, false, true); 
    fprintf(stderr, "...");
    classes->ComputePartPoseTransitionWeights(dv, false, true);
    fprintf(stderr, "done\n");
  }

  if(maxImages < dv->NumExamples()) {
    int *rand_split = RandSplit(maxImages, dv->NumExamples());
    Dataset *dv_new = dv->ExtractSubset(rand_split, 0);
    free(rand_split);
    delete dv;
    dv = dv_new;
  }

  FeatureWindow *feats = classes->FeatureWindows();
  for(int i = 0; i < classes->NumFeatureWindows(); i++) {
    char dictionaryOutFile[1000];
    sprintf(dictionaryOutFile, "%s.%d", dictionaryOutPrefix, i);
    if(!strncmp(feats[i].name, "HIST_", 5)) {
      FeatureDictionary *d = classes->FindCodebook(feats[i].name+strlen("HIST_"));
      if(d) 
	assert(d->NumWords() == feats[i].dim);
      else
	dv->BuildCodebook(dictionaryOutFile, feats[i].name+5, DESCRIPTOR_SPATIAL_WIDTH, DESCRIPTOR_SPATIAL_WIDTH, feats[i].dim, maxImages, ptsPerImage, strstr(feats[i].name,"HOG") ? (int)(.5+LOG2(feats[i].dim)) : 0, resize_image_width);
    } else if(!strncmp(feats[i].name, "FISHER_", 7)) {
      FisherFeatureDictionary *d = classes->FindFisherCodebook(feats[i].name+strlen("FISHER_"));
      int numWords = feats[i].dim/my_min(FISHER_PCA_DIMS,SQR(DESCRIPTOR_SPATIAL_WIDTH)*classes->Feature(feats[i].name+7)->numBins)/2;
      if(d) 
	assert(d->NumWords() == numWords);
      else
	dv->BuildFisherCodebook(dictionaryOutFile, feats[i].name+7, DESCRIPTOR_SPATIAL_WIDTH, DESCRIPTOR_SPATIAL_WIDTH, numWords, maxImages, ptsPerImage, FISHER_PCA_DIMS, resize_image_width);
    }
  }

  classes->Save(classNameOut);
  delete dv;
  delete classes;
}



void EvaluateTestset(const char *testSet, const char *classNameIn,  bool evaluatePartDetection, bool evaluateClassification, 
		     const char *predictionsOut, const char *imagesDirOut, const char *matfileOut,
		     const char *matlabProgressOut) {
  if(imagesDirOut)
    CreateDirectoryIfNecessary(imagesDirOut);
  Classes *classes = new Classes();
  bool b = classes->Load(classNameIn);  assert(b);
  Dataset *dataset = new Dataset(classes);
  b = dataset->Load(testSet);  assert(b);

  int *predictedClasses = matfileOut && evaluateClassification ? Create1DArray<int>(dataset->NumExamples()) : NULL;
  int *trueClasses = matfileOut && evaluateClassification ? Create1DArray<int>(dataset->NumExamples()) : NULL;
  double **classScores = matfileOut && evaluateClassification ? Create2DArray<double>(dataset->NumExamples(), classes->NumClasses()) : NULL;
  double *localizationLoss = matfileOut && evaluatePartDetection ? Create1DArray<double>(dataset->NumExamples()) : NULL;
  int ***predictedLocations = (matfileOut && evaluatePartDetection)||predictionsOut ? Create3DArray<int>(dataset->NumExamples(), classes->NumParts(), 5) : NULL;
  int ***trueLocations = matfileOut && evaluatePartDetection ? Create3DArray<int>(dataset->NumExamples(), classes->NumParts(), 5) : NULL;
  double *predictedLocationScores = matfileOut && evaluatePartDetection ? Create1DArray<double>(dataset->NumExamples()) : NULL;
  double **localizationLossComparedToUsers = matfileOut && evaluatePartDetection ? Create2DArray<double>(dataset->NumExamples(),classes->NumParts()) : NULL;
  

  fprintf(stderr, "Evaluating testset %s...\n", testSet);
  dataset->EvaluateTestset(evaluatePartDetection, evaluateClassification, imagesDirOut, predictedClasses, trueClasses, 
			   classScores, localizationLoss, predictedLocations, trueLocations, predictedLocationScores, 
			   matlabProgressOut, localizationLossComparedToUsers);
  if(predictionsOut) dataset->Save(predictionsOut);

  if(matfileOut) 
    SaveMatlabTest(matfileOut, classes->NumClasses(), dataset->NumExamples(), classes->NumParts(),
		   predictedClasses, trueClasses, classScores, localizationLoss,
		   predictedLocations, trueLocations, predictedLocationScores, localizationLossComparedToUsers);
}




void EvaluateTestset20Q(const char *testSet, const char *classNameIn, const char *matfileOut, int maxQuestions, 
			double timeInterval, int isCorrectWindow, bool stopEarly, QuestionSelectMethod method,
			bool disableClick, bool disableBinary, bool disableMultiple, bool disableCV, bool disableCertainty, 
                        const char *debugDir, 
			int debugNumClassPrint, bool debugProbabilityMaps, bool debugClickProbabilityMaps, int debugNumSamples,
			bool debugQuestionEntropies, bool debugMaxLikelihoodSolution, const char *matlabProgressOut) {
  Classes *classes = new Classes();
  bool b = classes->Load(classNameIn);  assert(b);
  Dataset *dataset = new Dataset(classes);
  b = dataset->Load(testSet);  assert(b);
  if(!timeInterval)
    maxQuestions = my_min(maxQuestions,classes->NumQuestions()+1);

  double *accuracy = Create1DArray<double>(maxQuestions);
  double ***perQuestionConfusionMatrix = Create3DArray<double>(maxQuestions, classes->NumClasses(), classes->NumClasses());
  double ***perQuestionClassProbabilities = Create3DArray<double>(dataset->NumExamples(), maxQuestions, classes->NumClasses());
  int **perQuestionPredictions = Create2DArray<int>(dataset->NumExamples(), maxQuestions);
  int **questionsAsked = Create2DArray<int>(dataset->NumExamples(), maxQuestions);
  int **responseTimes = Create2DArray<int>(dataset->NumExamples(), maxQuestions);
  int *numQuestionsAsked = Create1DArray<int>(dataset->NumExamples());
  int *gtClasses = Create1DArray<int>(dataset->NumExamples());
  
  // Run the visual 20 questions game with part clicks on each example in the test set
  fprintf(stderr, "Evaluating 20q on the testset %s...\n", testSet);
  dataset->EvaluateTestset20Q(maxQuestions, timeInterval, false, stopEarly, isCorrectWindow, method, accuracy, perQuestionConfusionMatrix,
			      perQuestionPredictions, perQuestionClassProbabilities, questionsAsked, responseTimes, numQuestionsAsked, gtClasses,
			      disableClick, disableBinary, disableMultiple, disableCV, disableCertainty, debugDir, debugNumClassPrint, debugProbabilityMaps, 
			      debugClickProbabilityMaps, debugNumSamples, debugQuestionEntropies, debugMaxLikelihoodSolution, 
			      matlabProgressOut);

  if(matfileOut)
    SaveMatlab20Q(matfileOut, maxQuestions, classes->NumClasses(), dataset->NumExamples(), 
		  accuracy, perQuestionConfusionMatrix, perQuestionPredictions, 
		  perQuestionClassProbabilities, questionsAsked, responseTimes, numQuestionsAsked, gtClasses);


  free(accuracy); free(perQuestionConfusionMatrix); free(perQuestionClassProbabilities); free(perQuestionPredictions); free(questionsAsked);
  free(responseTimes); free(numQuestionsAsked); free(gtClasses);
  delete dataset;
  delete classes;
}


void EvaluateTestsetInteractive(const char *testSet, const char *classNameIn, float stopThresh, const char *matfileOut, 
				const char *debugDir, bool debugImages, bool debugProbabilityMaps, const char *matlabProgressOut) {
  Classes *classes = new Classes();
  bool b = classes->Load(classNameIn);  assert(b);
  Dataset *dataset = new Dataset(classes);
  b = dataset->Load(testSet);  assert(b);
  int maxDrag = classes->NumParts()+1;

  double *aveLoss = Create1DArray<double>(maxDrag);
  double **partLoss = Create2DArray<double>(dataset->NumExamples(), maxDrag);
  int ***perDraggedPredictions = Create3DArray<int>(dataset->NumExamples(), maxDrag, 5);
  int **partsDragged = Create2DArray<int>(dataset->NumExamples(), maxDrag);
  double **dragTimes = Create2DArray<double>(dataset->NumExamples(), maxDrag);
  
  // Run the visual 20 questions game with part clicks on each example in the test set
  fprintf(stderr, "Evaluating interactive labeling on the testset %s...\n", testSet);
  dataset->EvaluateTestsetInteractive(maxDrag, stopThresh, debugDir, debugImages, debugProbabilityMaps,
				      aveLoss, partLoss, perDraggedPredictions, partsDragged, dragTimes, matlabProgressOut);

  if(matfileOut)
    SaveMatlabInteractive(matfileOut, maxDrag, dataset->NumExamples(), aveLoss, partLoss, perDraggedPredictions, partsDragged, dragTimes);

  free(aveLoss); free(partLoss); free(perDraggedPredictions); free(partsDragged);  free(dragTimes);
  delete dataset;
  delete classes;
}


void VisualizeDataset(const char *testSet, const char *classNameIn, const char *imagesDirOut, bool visualizePartModel, 
		      bool visualizeAttributeModels, bool showImageGallery, bool visualizeLabels, bool visualizeImageFeatures, 
		      int (*sort_cmp) ( const void *, const void * )) {
  Classes *classes = new Classes();
  bool b = classes->Load(classNameIn);  assert(b);
  Dataset *dataset = new Dataset(classes);
  b = dataset->Load(testSet);  assert(b);

  if(sort_cmp)
    dataset->Sort(sort_cmp);

  CreateDirectoryIfNecessary(imagesDirOut);

  char fname[1000];
  sprintf(fname, "%s/index.html", imagesDirOut);
  FILE *fout = fopen(fname, "w"); 
  assert(fout);
  fprintf(fout, "<html><body>\n");
  if(visualizePartModel) {
    ImageProcess *process = dataset->GetExampleData(0)->GetProcess(classes);
    int image_width = process->Image()->width, image_height = process->Image()->height;
    PartLocation *locs = PartLocation::NewPartLocations(classes,image_width,image_height,NULL,false);
    for(int id = classes->NumParts()-1; id >= 0; id--) {
      //if(!classes->GetPart(id)->NumParts()) continue;
      for(int k = 0; k < classes->GetPart(id)->NumPoses(); k++) {
        ObjectPose *pose = classes->GetPart(id)->GetPose(k);
        if(pose->IsNotVisible()) continue;
        for(int j = 0; j < classes->NumParts(); j++)
          if(process->GetPartInst(j)->GetNotVisiblePose())
	    locs[j].SetImageLocation(LATENT,LATENT,1,0,process->GetPartInst(j)->GetNotVisiblePose()->Model()->Name());
        locs[id].SetImageLocation(image_width/2,image_height/2, 4,0, pose->Name());
        process->GetPartInst(id)->GetPose(k)->SetPartLocationsAtIdealOffset(locs);
          
        sprintf(fname, "%s/%s", imagesDirOut, pose->Name());
        fprintf(stderr, "Making pose model visualization %s...\n", fname);

        process->VisualizePartModel(locs, fname);
        fprintf(fout, "<a href=\"%s.html\"><h2>Visualize Learned Pose Model \"%s\"</h2></a>\n", pose->Name(), pose->Name());
        fflush(fout);
      }
    }
    dataset->GetExampleData(0)->Clear();
    delete [] locs;
  }

  if(visualizeAttributeModels) {
    sprintf(fname, "%s/attributes", imagesDirOut);
    fprintf(stderr, "Making attribute model visualization %s...\n", fname);

    dataset->GetExampleData(0)->GetProcess(classes)->VisualizeAttributeModels(fname); 
    fprintf(fout, "<a href=\"attributes.html\"><h2>Visualize Learned Attribute Models</h2></a>\n");
    fflush(fout);
	dataset->GetExampleData(0)->Clear();
  }

  if(showImageGallery) {
    int numImagesPerGallery = 300;
    int num_galleries = ceil(dataset->NumExamples()/(double)numImagesPerGallery);
    fprintf(fout, "<br><br><h1>Training Images ");
    FILE *fouts[10000];
    for(int i = 0; i < num_galleries; i++) {
      char fname[1000];
      sprintf(fname, "%s/gallery_%d.html", imagesDirOut, i+1);
      fprintf(fout, "%s<a href=\"gallery_%d.html\">%d</a>", i ? "|" : "", i+1, i+1);
      fouts[i] = fopen(fname, "w");
      fprintf(fouts[i], "<br><br><h1>Training Images (click to view part-level features)</h1><br>\n<br><table>\n");
    }
    fflush(fout);
    
    int num = 0;
    int curr_ex = 0;

#ifdef USE_OPENMP
    omp_lock_t my_lock;
    omp_init_lock(&my_lock);
    //#pragma omp parallel for
#endif
    for(int ii = 0; ii < dataset->NumExamples(); ii++) {
#ifdef USE_OPENMP         
      omp_set_lock(&my_lock);
#endif
      int i = curr_ex++;
      FILE *fout = fouts[i/numImagesPerGallery];
#ifdef USE_OPENMP         
      omp_unset_lock(&my_lock);
#endif


      ImageProcess *process = dataset->GetExampleData(i)->GetProcess(classes);
      int width = 300;
      int images_per_col = 4;
	  char *line=(char*)malloc(100000), imgName[1000];

      sprintf(fname, "%s/index.html", imagesDirOut);
      GetRelativePath(dataset->GetExampleData(i)->GetImageName(), fname, imgName);
      if(visualizeImageFeatures) {
	char fname[1000];
	sprintf(fname, "%s/%08d_%s", imagesDirOut, i, process->Features()->Name());
	fprintf(stderr, "Making feature visualization for %s...\n", fname);
	if(dataset->GetExampleLabel(i)->NumObjects())
	  process->VisualizeFeatures(dataset->GetExampleLabel(i)->GetObject(0)->GetPartLocations(), fname);
	char *poseStr = (char*)malloc(100000);
	strcpy(poseStr, "");
	const char *pose;
	PartLocation *locs = dataset->GetExampleLabel(i)->GetObject(0)->GetPartLocations();
	for(int p = classes->NumParts()-1; p >= 0; p--) {
	  if(locs && classes->GetPart(p)->NumParts()) {
	    locs[p].GetImageLocation(NULL, NULL, NULL, NULL, &pose);
	    strcat(poseStr, "<br>");   
	    if(pose) strcat(poseStr, pose);
	  }
	}
	sprintf(line, "<td><center><a href=\"%08d_%s.html\"><img src=\"%08d_%s_original.png\" width=%d></a><br><b>%s</b>%s</center></td>", 
		i, process->Features()->Name(), i, process->Features()->Name(), width, 
		dataset->GetExampleLabel(i)->NumObjects() && dataset->GetExampleLabel(i)->GetObject(0)->GetClass() ? dataset->GetExampleLabel(i)->GetObject(0)->GetClass()->Name() : "", poseStr);
	//free(poseStr);
      } else if(visualizeLabels && dataset->GetExampleLabel(i)->NumObjects()) {
	char fname[1000], desc[10000];
	strcpy(desc, "");
	IplImage *img = cvCloneImage(process->Image());
        for(int o = 0; o < dataset->GetExampleLabel(i)->NumObjects(); o++) {
	  if(dataset->GetExampleLabel(i)->GetObject(o)->GetPartLocations()) {
	    PartLocation *locs = dataset->GetExampleLabel(i)->GetObject(o)->GetPartLocations();
	    process->Draw(img, locs);
	    for(int p = 0; p < classes->NumParts(); p++) {
	      const char *pose;
	      if(classes->GetPart(p)->NumParts()) {
		locs[p].GetImageLocation(NULL, NULL, NULL, NULL, &pose);
		sprintf(desc+strlen(desc), " <br>%s", pose);
	      }
	    }
	  }
	}
	sprintf(fname, "%s/%08d_%s_loc.png", imagesDirOut, i, process->Features()->Name());
	fprintf(stderr, "Visualizing part labels %s...\n", fname);
	cvSaveImage(fname, img);
	cvReleaseImage(&img);
	sprintf(line, "<td><a href=\"%08d_%s_loc.png\"><img src=\"%s\" width=%d></a>%s</td>", 
		i, process->Features()->Name(), imgName, width, desc);
      } else {
	sprintf(line, "<td><img src=\"%s\" width=%d></td>", imgName, width);
      }
      dataset->GetExampleData(i)->Clear();

      
#ifdef USE_OPENMP         
      omp_set_lock(&my_lock);
#endif
      if(num % images_per_col==0) fprintf(fout, "%s<tr>", num ? "</tr>" : "");
      fprintf(fout, "%s", line);
      fflush(fout);
      num++;
	  free(line);
#ifdef USE_OPENMP         
      omp_unset_lock(&my_lock);
#endif
    }
    for(int i = 0; i < num_galleries; i++) {
      fprintf(fouts[i], "</tr></table>\n\n");
      fclose(fouts[i]);
    }
  }
 
  fprintf(fout, "</body></html>\n");
  fclose(fout);

  delete dataset;
  delete classes;
}


void PlotResults(const char **matfiles, int numMatFiles, const char *pngOut, const char **labels, const char *title, 
		 const char *xLabel, const char *yLabel, int timeInterval) {
#ifdef HAVE_MATLAB  
  Engine *ep;
  mxArray *accuracy;
  MATFile *pmat;
  const mwSize *dims;
  char str[1000], legend[1000];
  const char *plotStyle[] = { "--", "-", "-.", "-v", "-p", "-s", "-x", "*", "+" };
  const char *plotColor[] = { "b", "r", "g", "c", "m", "y", "k" };
  int i, ma;

  strcpy(legend, "");
  ep = engOpen("\0");  assert(ep != NULL);
  sprintf(str, "figure(1); clf; hold on; title('%s'); xlabel('%s'); ylabel('%s');", title, xLabel, yLabel);
  engEvalString(ep, str);

  strcpy(legend, "");
  for(i = 0; i < numMatFiles; i++) {
    pmat = matOpen(matfiles[i], "r"); assert(pmat);
    accuracy=matGetVariable(pmat, "accuracy");  assert(accuracy != NULL && mxGetNumberOfDimensions(accuracy)==1);
    accuracy = engGetVariable(ep, "accuracy");
    dims = mxGetDimensions(accuracy);
    ma = dims[0]-1;

    sprintf(legend+strlen(legend), "%s'%s'", i ? ", " : "", labels[i]);
    sprintf(str, "plot(0:%d:%d, accuracy, '%s', 'LineWidth', 3, 'Color', '%s');", timeInterval, timeInterval*ma, 
	    plotStyle[i], plotColor[i]);
    engEvalString(ep, str);
    
    mxDestroyArray(accuracy);
    matClose(pmat);
  }

  sprintf(str, "legend({%s},'Location','Southeast')", legend);
  engEvalString(ep, str);

  sprintf(str, "saveas(1,'%s')", pngOut);
  engEvalString(ep, str);

  engClose(ep);
#else
  fprintf(stderr, "Can't plot without matlab support.  You must define -DHAVE_MATLABin the Makefile\n");
#endif
}



void EvaluateTestsetVOC(const char *classNameIn, const char *dirName, const char *predFileOut, const char *htmlDir, float overlap) {
  Classes *classes = new Classes();
  bool b = classes->Load(classNameIn); assert(b);

  int width = 300;
  int images_per_col = 4;
  char fname[1000];
  FILE *foutHtml = NULL;
  if(htmlDir) {
    CreateDirectoryIfNecessary(htmlDir);
    sprintf(fname, "%s/index.html", htmlDir);
    foutHtml = fopen(fname, "w");
    fprintf(foutHtml, "<html><body><h1>VOC Test Results</h1><table>\n");
  }

  int num = 0;
  char **fnames = ScanDir(dirName, ".jpg", &num);
  FILE *fout = fopen(predFileOut, "w");


#ifdef USE_OPENMP
  omp_lock_t my_lock;
  omp_init_lock(&my_lock);
  //#pragma omp parallel for
#endif
  for(int i = 0; i < num; i++) {
    char name[1000];
    char fname[1000];
    sprintf(fname, "%s/%s", dirName, fnames[i]);
    
    // Load image and run detector
    IplImage *img = cvLoadImage(fname);
    assert(img);
    ExtractFilename(fname, name);
    StripFileExtension(name);
    ImageProcess *process = new ImageProcess(classes, fname, IM_MAXIMUM_LIKELIHOOD, false, false);
    process->Detect();

    // Extract non-max suppressed bounding boxes
    int num_boxes, w, h;
    process->Features()->GetDetectionImageSize(&w, &h, 0, 0);
    CvRectScore *boxes = process->GetRootPart()->GetBoundingBoxes(&num_boxes);
    int num_boxes_nms = NonMaximalSuppression(boxes, num_boxes, overlap, w, h);
    char rects[50000];   strcpy(rects, "");

    // Output non-max suppressed bounding boxes
#ifdef USE_OPENMP
    omp_set_lock(&my_lock);
#endif
    for(int j = 0; j < num_boxes_nms; j++) {
      fprintf(stderr, "%s %f %d %d %d %d\n", name, boxes[j].score, boxes[j].rect.x, boxes[j].rect.y, boxes[j].rect.x+boxes[j].rect.width, boxes[j].rect.y+boxes[j].rect.height);
      fprintf(fout, "%s %f %d %d %d %d\n", name, boxes[j].score, boxes[j].rect.x, boxes[j].rect.y, boxes[j].rect.x+boxes[j].rect.width, boxes[j].rect.y+boxes[j].rect.height);
    }
#ifdef USE_OPENMP         
    omp_unset_lock(&my_lock);
#endif

    int r[10000], g[10000], b[10000];
    for(int k = num_boxes_nms-1; k >= 0; k--) {
      r[k]=rand()%255;
      g[k]=rand()%255;
      b[k]=rand()%255;
      cvRectangle(img, cvPoint(boxes[k].rect.x,boxes[k].rect.y), 
		  cvPoint(boxes[k].rect.x+boxes[k].rect.width,boxes[k].rect.y+boxes[k].rect.height), 
		  CV_RGB(r[k],g[k],b[k]), my_max(1,5-k));
    }
    for(int l = 0; l < num_boxes_nms; l++) 
      sprintf(rects+strlen(rects), "<br><font color=#%02x%02x%02x>%f</font>", r[l], g[l], b[l], boxes[l].score);

    free(boxes);

  
    if(foutHtml) {
      char fname3[1000], fname2[1000];
#ifdef USE_OPENMP
      omp_set_lock(&my_lock);
#endif
      if(num % images_per_col == 0) fprintf(foutHtml, "%s<tr>", num ? "</tr>" : "");
      strcpy(fname2, fnames[i]); StringReplaceChar(fname2, '/', '_'); StringReplaceChar(fname2, '\\', '_'); 
      sprintf(fname3, "%s/%s", htmlDir, fname2);
      fprintf(foutHtml, "<td><a href=\"../%s\"><img src=\"../%s\" width=%d></a>%s</td>", fname3, fname3, width, rects);
      fflush(foutHtml);
      num++;
#ifdef USE_OPENMP         
      omp_unset_lock(&my_lock);
#endif
      cvSaveImage(fname3, img);
    }

    delete process;
    cvReleaseImage(&img);
  }

  fclose(fout);
  if(foutHtml) fclose(foutHtml);
  free(fnames);
}




void EvaluateTestsetCaltechPedestrians(const char *classNameIn, const char *testSetDirName, 
				       const char *predDirOut, const char *htmlDir, float overlap) {
  Classes *classes = new Classes();
  bool b = classes->Load(classNameIn); assert(b);

  CreateDirectoryIfNecessary(predDirOut);
  if(htmlDir) CreateDirectoryIfNecessary(htmlDir);

  int width = 300;
  int images_per_col = 4;
  int i;

#ifdef USE_OPENMP
  omp_lock_t my_lock;
  omp_init_lock(&my_lock);
#endif
  for(i = 0; i < 100000; i++) {
    char vid_name[1000];
    sprintf(vid_name, "%s/V%03d", testSetDirName, i);

    int num = 0;
    bool finished = false;
    FILE *foutHtml = NULL;

#ifdef USE_OPENMP
#pragma omp parallel 
#endif
    {
      while(!finished) {
	char img_name[1000], htmlVidDir[1000], outVidDir[1000], out_name[1000];	
#ifdef USE_OPENMP
	omp_set_lock(&my_lock);
#endif
	int id = num*30 + 29;
	sprintf(img_name, "%s/I%05d.jpg", vid_name, id);
	IplImage *img = cvLoadImage(img_name);
	if(!img) {
	  finished = true;
	  omp_unset_lock(&my_lock);
	  break;
	}

	sprintf(outVidDir, "%s/V%03d", predDirOut, i);
	if(htmlDir) sprintf(htmlVidDir, "%s/V%03d", htmlDir, i);
	if(!num) {
	  CreateDirectoryIfNecessary(outVidDir);
	  if(htmlDir) {
	    CreateDirectoryIfNecessary(htmlVidDir);
	    char fname[1000];
	    sprintf(fname, "%s/index.html", htmlVidDir);
	    foutHtml = fopen(fname, "w");
	    fprintf(foutHtml, "<html><body><h1>Caltech Pedestrians Test Results</h1><table>\n");
	  }
	}
	num++;
#ifdef USE_OPENMP
	omp_unset_lock(&my_lock);
#endif

	// Load image and run detector
	assert(img);
	char name[1000];
	ExtractFilename(img_name, name);
	StripFileExtension(name);
	ImageProcess *process = new ImageProcess(classes, img_name, IM_MAXIMUM_LIKELIHOOD, false, false);
	process->Detect();

	// Extract non-max suppressed bounding boxes
	int num_boxes, w, h;
	process->Features()->GetDetectionImageSize(&w, &h, 0, 0);
	CvRectScore *boxes = process->GetRootPart()->GetBoundingBoxes(&num_boxes);
	int num_boxes_nms = NonMaximalSuppression(boxes, num_boxes, overlap, w, h);
	char rects[50000];   strcpy(rects, "");

	// Output non-max suppressed bounding boxes
	sprintf(out_name, "%s/I%05d.txt", outVidDir, id);
	FILE *fout = fopen(out_name, "w");
	for(int j = 0; j < num_boxes_nms; j++) {
	  fprintf(stderr, "%s %f %d %d %d %d\n", name, boxes[j].score, boxes[j].rect.x, boxes[j].rect.y, boxes[j].rect.x+boxes[j].rect.width, boxes[j].rect.y+boxes[j].rect.height);
	  fprintf(fout, "%d %d %d %d %f\n", boxes[j].rect.x, boxes[j].rect.y, boxes[j].rect.x+boxes[j].rect.width, boxes[j].rect.y+boxes[j].rect.height, boxes[j].score);
	}
	fclose(fout);

	int r[10000], g[10000], b[10000];
	for(int k = num_boxes_nms-1; k >= 0; k--) {
	  r[k]=rand()%255;
	  g[k]=rand()%255;
	  b[k]=rand()%255;
	  cvRectangle(img, cvPoint(boxes[k].rect.x,boxes[k].rect.y), 
		      cvPoint(boxes[k].rect.x+boxes[k].rect.width,boxes[k].rect.y+boxes[k].rect.height), 
		      CV_RGB(r[k],g[k],b[k]), my_max(1,5-k));
	}
	for(int l = 0; l < num_boxes_nms; l++) 
	  sprintf(rects+strlen(rects), "<br><font color=#%02x%02x%02x>%f</font>", r[l], g[l], b[l], boxes[l].score);

	free(boxes);

  
	if(foutHtml) {
	  char fname[1000], fname2[1000];
#ifdef USE_OPENMP
	  omp_set_lock(&my_lock);
#endif
	  if(num % images_per_col == 0) fprintf(foutHtml, "%s<tr>", num ? "</tr>" : "");
	  strcpy(fname2, img_name); StringReplaceChar(fname2, '/', '_'); StringReplaceChar(fname2, '\\', '_'); 
	  sprintf(fname, "%s/%s", htmlVidDir, fname2);
	  fprintf(foutHtml, "<td><a href=\"%s\"><img src=\"%s\" width=%d></a>%s</td>", fname2, fname2, width, rects);
	  fflush(foutHtml);
	  num++;
#ifdef USE_OPENMP         
	  omp_unset_lock(&my_lock);
#endif
	  cvSaveImage(fname, img);
	}

	delete process;
	cvReleaseImage(&img);
      }
    }
    if(num == 0) 
      break;
    else if(foutHtml)
      fclose(foutHtml);
  }
}

void ExtractImageFeatures(const char *datasetIn, const char *classNameIn, const char *featuresOut, bool extractPartLocalizedFeatures, int max_images, int image_resize_width) {
  Classes *classes = new Classes();
  bool b = classes->Load(classNameIn);  assert(b);
  int curr_ex = 0;
  int curr_write = 0;

  if(!classes->NumCodebooks() && !classes->NumFisherCodebooks()) {
    char cname[1000];
    sprintf(cname, "%s.codebook", classNameIn);
    BuildCodebooks(datasetIn, classNameIn, classNameIn, cname, 1000, 50, image_resize_width);
    delete classes;
    classes = new Classes();
    bool b = classes->Load(classNameIn);  assert(b);
  }


  Dataset *dataset = new Dataset(classes);
  b = dataset->Load(datasetIn);  assert(b);

  bool saveBinary = !strcmp(GetFileExtension(featuresOut), "bin");
  FILE *fout = fopen(featuresOut, saveBinary ? "wb" : "w");

  SparseVector **features = new SparseVector*[dataset->NumExamples()];
  memset(features, 0, sizeof(SparseVector*)*dataset->NumExamples());
  int *class_ids = new int[dataset->NumExamples()];

  if(max_images > 0 && dataset->NumExamples() > max_images) {
    int *rand_split = RandSplit(max_images, dataset->NumExamples());
    Dataset *new_dataset = dataset->ExtractSubset(rand_split, 0);
    delete dataset;
    dataset = new_dataset;
  }

#ifdef USE_OPENMP
  omp_lock_t my_lock;
  omp_init_lock(&my_lock);
  #pragma omp parallel for
#endif
  for(int ii = 0; ii < dataset->NumExamples(); ii++) {
#ifdef USE_OPENMP         
    omp_set_lock(&my_lock);
#endif
    int i = curr_ex++;
#ifdef USE_OPENMP         
    omp_unset_lock(&my_lock);
#endif
    fprintf(stderr, "Computing features for image %d of %d %s...\n", i+1, dataset->NumExamples(), dataset->GetExampleData(i)->GetImageName());

    // Extract features
    int n = (extractPartLocalizedFeatures ? classes->NumParts() : 1)*classes->NumWindowFeatures(), n2;
    float *f = new float[n];
    VFLOAT *fd = new VFLOAT[n];
    ImageProcess *process = dataset->GetExampleData(i)->GetProcess(classes);
    if(extractPartLocalizedFeatures)
      n2 = process->GetLocalizedFeatures(f, dataset->GetExampleLabel(i)->GetObject(0)->GetPartLocations());
    else {
      process->Features()->GetImage(image_resize_width);
      n2 = process->GetImageFeatures(f);
    }
    assert(n == n2);
    for(int j = 0; j < n; j++)
      fd[j] = f[j];
    class_ids[i] = dataset->GetExampleLabel(i)->GetObject(0)->GetClass()->Id()+1;
    features[i] = new SparseVector(fd, n, false, 1e-9);
    dataset->GetExampleData(i)->Clear();
    delete [] f;
    delete [] fd;

    // Save features to disk, preserving the ordering of the dataset examples
#ifdef USE_OPENMP         
    omp_set_lock(&my_lock);
#endif
    while(curr_write < dataset->NumExamples() && features[curr_write]) {
      int class_id = class_ids[curr_write];
      if(saveBinary) {
        fwrite(&class_id, sizeof(int), 1, fout);
        features[curr_write]->write(fout);
      } else {
        char * str = features[curr_write]->to_string();
        fprintf(fout, "%d %s\n", class_id, str);
        free(str);
      }
      delete features[curr_write];
      features[curr_write++] = NULL;
    }
#ifdef USE_OPENMP         
    omp_unset_lock(&my_lock);
#endif
  }

  fclose(fout);
  delete [] features; 
  delete [] class_ids; 
  delete dataset;
  delete classes;
}
