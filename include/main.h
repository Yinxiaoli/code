/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#ifndef __MAIN_H
#define __MAIN_H



/**
 * @file main.h
 * @brief Helper routines for training and testing
 */


/** @mainpage 
 *
 * @section intro_sec Introduction
 *
 * This toolbox contains a collection of routines for multiclass object detection, deformable part models, pose 
 * mixture models, localized attribute and classification models, online structured learning, probabilistic user models, and interactive 
 * annotation tools for labeling parts and object classes.  This toolbox was written by Steve Branson and
 * is implemented in C++. The documentation and usability will hopefully be improved soon.
 *
 *
 *
 * @section download Download
 * Source code download:
 * - <a href="http://vision.caltech.edu/~sbranson/code/code_v1.00.tgz">Version 1.0</a> June 23, 2013
 *
 * @section compile_sec Compilation
 * 
 * The only prerequisite is OpenCV (see https://help.ubuntu.com/community/OpenCV for installation in Ubuntu).  
 *   - <strong> Linux: </strong> It should compile simply by typing 'make'.  If you get errors linking OpenCV libraries, you may need to change the LIBS_CV flag in examples/Makefile.
 *   - <strong> Windows: </strong> Currently, this has only been tested in Visual Studio 2010.  Edit, paths_x64.props (for 64-bit builds) and/or paths.props (for 32-bit builds), changing the \<IncludePath\> and \<LibraryPath\> fields to include the appropriate location of your OpenCV directory.  Also check to make sure that the .lib files in the \<AdditionalDependencies\> section have the correct opencv library number.  Open code.sln, then select <em>Build</em> from the menu
 *
 *
 * @section overview Overview
 * 
 * @subsection object_detection Features, Object Recognition, and Object Detection 
 * The following features are included.  They can be used in conjunction with object recognition, sliding window detection, or deformable part models (e.g., localized versions of features are supported):
 *     - <strong> HOG: </strong> Dalal Triggs-style HOG detector, with tricks to compute them quickly over multiple orientations
 *     - <strong> Bag of Words SIFT: </strong> Sliding window detectors over vector quantized SIFT descriptors
 *     - <strong> Color Histograms: </strong> Sliding window detectors over RGB or CIE color histograms
 *     - <strong> Fisher Vectors: </strong> Fisher vector encoded SIFT or color features (Perronin et al. ECCV'2010) 
 *     - <strong> Spatial Pyramids: </strong> The above features can be stacked together in spatial pyramids, or multi-resolution pyramids
 * \image html features_small.png
 * See <a href="import_object_recognition_dataset_8cpp-example.html">import_object_recognition_dataset.cpp</a> for an example.
 *
 * @subsection dpm Deformable Part Models
 * Deformable part models, where each part has a sliding window appearance model.  This is similar to the model used by Felzenszwalb and Ramanan, but with an emphasis on semantically defined parts.  It includes the following features:
 *     - <strong> Per-Part Mixture Models: </strong> Each part can have a different discrete aspect (for example, the body can be in frontal view while the head can be in side view)
 *     - <strong> Predicting Visibility: </strong> Each part can be occluded independently
 * See <a href="import_birds200_8cpp-example.html">import_birds200.cpp</a> and <a href="train_detector_8cpp-example.html">train_detector.cpp</a> for an example.
 *
 * @subsection multiclass Multiclass Object Detection Using Shared Parts and Attributes
 * Sliding window part and attribute detectors, which can be shared among multiple classes (e.g., for subordinate classification) 
 * \image html attributes.png
 * See <a href="import_birds200_8cpp-example.html">import_birds200.cpp</a> and <a href="train_multiclass_detector_8cpp-example.html">train_multiclass_detector.cpp</a> for an example.
 *     
 * @subsection learning Online Structured SVM Learning
 * Fast structured SVM learning supporting very large datasets.  Examples can be added in online fashion via a network interface.  The network interface also allows examples to be classified or labeled interactively using the current model as the learner trains in online fashion.  Implementations of the following learning algorithms are included:
 *      - <strong> Object Detection: </strong> Train object detectors using customizable loss functions, for example using a soft loss function based on the overlap between predicted and ground truth bounding boxes.  
 *      - <strong> Deformable Part Models: </strong> Jointly learn appearance and spatial parameters for deformable part models while supporting customizable loss functions.
 *      - <strong> Multiclass Classification: </strong> Train multiclass classifiers, with customizable confusion costs for misclassifying any pair of classes
 *      - <strong> Joint Learning of Attribute Classifiers: </strong> Jointly train multiple attribute detectors while optimizing loss functions based on object class prediction
 * The documentation and API for this portion of the code will be improved in the next 1-2 weeks.  The sub-folder online_structured_svm/ in 
 * source code is self-contained, and one can use it independently from the computer vision code in this toolbox.  See <a href="structured_svm_multiclass_8cpp-example.html">structured_svm_multiclass.cpp</a> (which is used by <a href="import_object_recognition_dataset_8cpp-example.html">import_object_recognition_dataset.cpp</a>) for an example.
 *
 * @subsection interactive Interactive Computer Vision 
 * Ability to learn probability distributions modeling noise in the way people answer attribute questions or label the locations of parts. These can be combined with computer vision routines to create interactive interfaces
 * @subsubsection interactive_parts Interactive Part Labeling
 *   An interactive GUI for labeling part models, which displays in realtime the maximum likelihood location of all parts (e.g., the result of pictorial structure inference) as the user drags one or more parts.  See <a href="online_interactive_parts_server_8cpp-example.html">online_interactive_parts_server.cpp</a> for an example.
\htmlonly
<p><center><object classid="clsid:d27cdb6e-ae6d-11cf-96b8-444553540000" codebase="http://download.macromedia.com/pub/shockwave/cabs/flash/swflash.cab\#version=9,0,0,0" width="642" height="346" id="VideoPlayer" align="middle"> <param name="allowScriptAccess" value="*" /> <param name="allowFullScreen" value="true" /> <param name="movie" value="http://vision.ucsd.edu/visipedia/video/FLVPlayer.swf?video=interactive_label.flv&autoplay=false" /> <param name"quality" value="high" /><param name="bgcolor" value="\#ffffff" /> <embed src="http://vision.ucsd.edu/visipedia/video/FLVPlayer.swf?video=interactive_label.flv&autoplay=false" quality="high" bgcolor="\#000000" width="642" height="346" name="VideoPlayer" align="middle" allowScriptAccess="*" allowFullScreen="true" type="application/x-shockwave-flash" pluginspage="http://www.macromedia.com/go/getflashplayer" /> </object></center></p>
\endhtmlonly

 * @subsubsection visual_20q Visual 20 Questions Game
 * An interactive interface for classifying objects that are difficult for both humans and computers (e.g., allows non-experts to classify bird species).  The system uses a combination of computer vision (deformable part model localization and multiclass classification) and user responses (answers to yes/no or multiple choice questions or clicks on the locations of parts) to predict a probability distribution over object classes.  It optimizes an expected information gain criterion to intelligently select which question to ask next, progressively narrowing in on the true class.  See <a href="20q_server_8cpp-example.html">20q_server.cpp</a> for an example.
\htmlonly
<p><center><object classid="clsid:d27cdb6e-ae6d-11cf-96b8-444553540000" codebase="http://download.macromedia.com/pub/shockwave/cabs/flash/swflash.cab\#version=9,0,0,0" id="VideoPlayer" align="middle"> <param name="allowScriptAccess" value="*" /> <param name="allowFullScreen" value="true" /> <param name="movie" value="http://vision.ucsd.edu/visipedia/video/FLVPlayer.swf?video=v20q.flv&autoplay=false" /> <param name"quality" value="high" /><param name="bgcolor" value="\#ffffff" /> <embed src="http://vision.ucsd.edu/visipedia/video/FLVPlayer.swf?video=v20q.flv&autoplay=false" quality="high" bgcolor="\#000000" width="638" height="430" name="VideoPlayer" align="middle" allowScriptAccess="*" allowFullScreen="true" type="application/x-shockwave-flash" pluginspage="http://www.macromedia.com/go/getflashplayer" /> </object></center></p>
\endhtmlonly
 *
 *
 * @section citation Citation
 * This toolbox includes an implementation of the three papers listed below.  Please consider citing them if you use this toolbox:
 *
 \htmlonly
<font size="3">Branson S., Beijbom O., Belongie S., "Efficient Large-Scale Structured Learning", IEEE Conference on Computer Vision (CVPR), Portland, June 2013. <a href="http://vision.ucsd.edu/sites/default/files/cvpr2013_efficient_0.pdf">pdf</a></font>
<div id="cite1" style="display: block; margin-left: 20px; border: 1px solid #aaf; background-color: #dadafa;"> <pre>
@inproceedings { branson_online_interactive11,
	title = {Efficient Large-Scale Structured Learning},
	booktitle = {IEEE Conference on Computer Vision (CVPR)},
	year = {2013},
	address = {Portland, Oregon},
	author = {Steve Branson and Oscar Beijbom and Serge Belongie}
}
</pre></div> 
\endhtmlonly
 *
 *
\htmlonly
<font size="3">Branson S., Perona P., Belongie S., "Strong Supervision From Weak Annotation: Interactive Training of Deformable Part Models", IEEE International Conference on Computer Vision (ICCV), Barcelona, 2011. <a href="http://vision.ucsd.edu/sites/default/files/iccv2011_interactive_parts_2.pdf">pdf</a></font>
<div id="cite1" style="display: block; margin-left: 20px; border: 1px solid #aaf; background-color: #dadafa;"> <pre>
@inproceedings { branson_online_interactive11,
	title = {Strong Supervision From Weak Annotation:  Interactive Training of Deformable Part Models},
	booktitle = {IEEE International Conference on Computer Vision (ICCV)},
	year = {2011},
	address = {Barcelona, Spain},
	author = {Steve Branson and Pietro Perona and Serge Belongie}
}
</pre></div> 
\endhtmlonly
 *
\htmlonly
<br><p><font size="3">Wah C., Branson S., Perona P., Belongie S., "Multiclass Recognition and Part Localization with Humans in the Loop", IEEE International Conference on Computer Vision (ICCV), Barcelona, 2011. <a href="http://vision.ucsd.edu/sites/default/files/iccv2011_20q_parts_final.pdf">pdf</a></font>
<div id="cite2" style="display: block; margin-left: 20px; border: 1px solid #aaf; background-color: #dadafa;"> <pre>
@inproceedings { wah_multiclass11,
        title = {Multiclass Recognition and Part Localization with Humans in the Loop},
	booktitle = {IEEE International Conference on Computer Vision (ICCV)},
	year = {2011},
	address = {Barcelona, Spain},
	author = {Catherine Wah and Steve Branson and Pietro Perona and Serge Belongie}
}
</pre></div> \endhtmlonly
 *
 *
 *
 * @section getting_started_sec Getting Started
 * The best way to get started is to browse through the example code (click on the Examples tab).  These examples can be used without modification for many tasks (training part detectors, training 
 * multiclass classifiers, evaluating testsets, interactively labeling parts, the Visual 20 questions game); however, you may want to edit these examples to explore more advanced
 * usage scenarios.  You can obtain an example training/testing set by downloading http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz and unzipping it in the examples/ directory
 *
 */



#include "question.h"


/**
 * @brief Train part and pose detectors
 * 
 * The detector combines an appearance score and quadratic spatial score.  The appearance score for a part is the max over the appearance
 * scores of all possible poses of that part.  The spatial score combines a quadratic cost over the pixel offset between parent and child
 * parts and a pose transition cost.  It learns the parameters of the appearance and spatial model using SVM struct,
 * such that the ground truth part locations has higher score than every other location by (at least) some loss function,
 * where the loss function is a measure of how well the predicted location of a part overlaps with the groundtruth location of the part (bounding box intersection area divided by bounding box union area)
 *
 * @param trainSet The file name of the Dataset used for training
 * @param classNameIn The file name to read the Classes definition, which defines the set of parts, pose, and detection features
 * @param classNameOut The file name to output the Classes definition, which is redundant with the definition in classNameIn, but also contains the learned weights of the detector
 * @param lossType Type of detection loss to use for training
 * @param partLoss (optional) A numParts array of maximum localization loss associated with each part.  Leaving partLoss NULL is equivalent to defining each entry to be 1
 * @param C The regularization constant.  If set to 0 a default value is used
 * @param eps The accuracy level for approximate training.  If set to 0, a default value is used
 */
void TrainDetectors(const char *trainSet, const char *classNameIn, const char *classNameOut, 
		    PartDetectionLossType lossType=LOSS_PART_AVERAGE_AREA_UNION_OVER_INTERSECTION, double *partLoss=NULL, double C=0, double eps=0);

/**
 * @brief Train attribute detectors
 * 
 * The attribute detectors are assumed to be linear classifiers evaluated on ground truth labeled part locations
 *
 * @param trainSet The file name of the Dataset used for training
 * @param classNameIn The file name to read the Classes definition, which defines the set of parts, pose, and detection features
 * @param classNameOut The file name to output the Classes definition, which is redundant with the definition in classNameIn, but also contains the learned weights of the detector
 * @param trainJointly If false, it learns each attribute detector independently.  If true, it learns attribute detectors to maximize multi-class classification accuracy,
 *    assuming the score of each class is the sum of its attribute scores
 * @param classConfusionCost (optional) A numClassesXnumClasses class confusion cost for loss-sensitive learning, valid only when trainJointly is true.  classConfusionCost[y_gt][y_pred] is the loss associated with predicting class y_pred when the true class is y_gt.  If classConfusionCost is left NULL, by default we use the 0/1 loss matrix
 * @param C The regularization constant.  If set to 0, a default value is used
 * @param eps The accuracy level for approximate training.  If set to 0, a default value is used
 */
void TrainAttributes(const char *trainSet, const char *classNameIn, const char *classNameOut, bool trainJointly, double **classConfusionCost=NULL, double C=0, double eps=0, const char *validationFile=NULL, const char *optMethod=NULL);

void TrainMulticlass(const char *trainSet, const char *classNameIn, const char *classNameOut, double **classConfusionCost=NULL, double C=0, double eps=0);


/**
 * @brief Use cross-validation to learn parameters that transform attribute detection scores to probabilities
 * 
 * See Dataset::LearnClassAttributeDetectionParameters() for more information
 * Parameters are learned to maximize the log-likelihood of the validation set
 *
 * @param trainSet The file name of the Dataset used for training
 * @param classNameIn The file name to read the Classes definition, which defines the set of parts, pose, and detection features
 * @param classNameOut The file name to output the Classes definition, which is redundant with the definition in classNameIn, but also contains the learned weights of the detector
 * @param maxImages The maximum number of images to use to learn detection parameters
 *
 */
void LearnAttributeProbabilities(const char *trainSet, const char *classNameIn, const char *classNameOut, int maxImages);

/**
 * @brief Learn probability distributions for how users answer attribute questions
 * 
 * See Dataset::LearnClassAttributeUserProbabilities() for more information
 *
 * @param trainSet The file name of the Dataset used for training
 * @param classNameIn The file name to read the Classes definition, which defines the set of parts, pose, and detection features
 * @param classNameOut The file name to output the Classes definition, which is redundant with the definition in classNameIn, but also contains the learned weights of the detector
 * @param beta_class A beta prior with the probability distribution of \f$ p(u_a|c) \f$
 * @param beta_cert  An array of beta priors for each possible certainty value with the probability distribution of \f$ p(u_a,r_a) \f$
 *
 */
void LearnAttributeUserProbabilities(const char *trainSet, const char *classNameIn, const char *classNameOut, float beta_class, float *beta_cert);


/**
 * @brief Use cross-validation to learn parameters that transform part detection scores to probabilities
 * 
 * See Dataset::LearnPartDetectionParameters() for more information
 * Parameters are learned to maximize the log-likelihood of the validation set
 *
 * @param trainSet The file name of the Dataset used for training
 * @param classNameIn The file name to read the Classes definition, which defines the set of parts, pose, and detection features
 * @param classNameOut The file name to output the Classes definition, which is redundant with the definition in classNameIn, but also contains the learned weights of the detector
 * @param maxImages The maximum number of images to use to learn detection parameters
 *
 */
void LearnDetectionProbabilities(const char *trainSet, const char *classNameIn, const char *classNameOut, 
				 int maxImages);



/**
 * @brief Learn probability distributions for how users answer part click questions
 * 
 * See Dataset::LearnUserClickProbabilities() for more information
 *
 * @param trainSet The file name of the Dataset used for training
 * @param classNameIn The file name to read the Classes definition, which defines the set of parts, pose, and detection features
 * @param classNameOut The file name to output the Classes definition, which is redundant with the definition in classNameIn, but also contains the learned weights of the detector
 *
 */
void LearnUserClickProbabilities(const char *trainSet, const char *classNameIn, const char *classNameOut);




/**
 * @brief Learn codebooks using k-means.  
 *
 * The induced codebooks can be used for SIFT-based bag of words and RGB and CIE color histograms
 * 
 * @param trainSet The file name of the Dataset used for training
 * @param classNameIn The file name to read the Classes definition, which defines the set of parts, pose, and detection features
 * @param classNameOut The file name to output the Classes definition, which is redundant with the definition in classNameIn, but also contains the learned weights of the detector
 * @param dictionaryOutPrefix The prefix of the file name to output the learned dictionary.  The codebooks are outputed to files dictionaryOutPrefix.sift, dictionaryOutPrefix.rgb, dictionaryOutPrefix.cie
 * @param numSiftWords The number of words in the SIFT-based bag of words dictionary
 * @param numRGBWords The number of histogram bins for the RGB histogram
 * @param numCIEWords The number of histogram bins for the CIE histogram
 * @param maxImages The maximum number of images to use to construct the training set 
 * @param ptsPerImage The number of interest points to extract from each image
 */ 
void BuildCodebooks(const char *trainSet, const char *classNameIn, const char *classNameOut, 
		    const char *dictionaryOutPrefix, int maxImages, int ptsPerImage, int resize_image_width=0);


/**
 * @brief Run the 20 questions game with part clicks on every example in a testset.
 *
 * The results are outputed to a .mat file and can be plotted using matlab (see matlab/plot_results.m, matlab/plot_question_ordering.m)
 * @param testSet The file name of the testset
 * @param classNameIn The file name of the class definition file
 * @param matfileOut The file name of the .mat file used to output results
 * @param maxQuestions The maximum number of questions to ask.  If you want to measure results using human labor (in seconds), set maxQuestions=maxTime/timeInterval
 * @param timeInterval Set this to 0 if you want to measure results by the number of questions asked. If you want to measure results using human labor (in seconds), set this to some time interval (e.g. 1 second), such that classification accuracy will be plotted for time values spaced at that interval
 * @param isCorrectWindow Set this to 1 to measure regular classification accuracy.  Set to some value isCorrectWindow>1 to assume success if the true class is in the top isCorrectWindow ranked classes
 * @param stopEarly Normally set this to false (corresponds to Method 1 in ECCV'10).  Set this to true if you assume the user will stop the 20 questions game early if the true class is the top ranked class (corresponds to Method 2 in ECCV'10).
 * @param method The method used to select the next question to ask
 * @param disableClick Set to true to disable asking click questions
 * @param disableBinary Set to true to disable asking binary questions
 * @param disableMultiple Set to true to disable asking multiple choice questions
 * @param disableComputerVision Set to true to disable using computer vision
 * @param debugDir If non-null, generate an HTML visualization for each test image of the questions asked, and the evolution of class probabilities and
 *  part location probability maps as the user answers questions.  The html will be stored to debugDir/index.html, with lots of images in debugDir
 * @param debugNumClassPrint Print probabilities for the top debugNumClassPrint classes
 * @param debugProbabilityMaps Save probability maps after each question (can lead to very big files)
 * @param debugClickProbabilityMaps Save probability maps after each question (can lead to very big files)
 * @param debugNumSamples Save sampled part locations maps after each question (can lead to VERY VERY big files)
 * @param debugQuestionEntropies Print entropies for all questions
 * @param debugMaxLikelihoodSolution Draw the current max likelihood solution after every question
 * @param matlabProgressOut If non-null, periodically saves progress of all results to matlab file (for debugging while experiment is in progress)
 */
void EvaluateTestset20Q(const char *testSet, const char *classNameIn, const char *matfileOut, int maxQuestions, 
			double timeInterval, int isCorrectWindow, bool stopEarly, QuestionSelectMethod method,
			bool disableClick=false, bool disableBinary=false, bool disableMultiple=false, 
			bool disableComputerVision=false, bool disableCertainty=false, const char *debugDir=NULL,
			int debugNumClassPrint=10, bool debugProbabilityMaps=false, 
			bool debugClickProbabilityMaps=false, int debugNumSamples=0, bool debugQuestionEntropies=false,
			bool debugMaxLikelihoodSolution=true,const char *matlabProgressOut=NULL);



/**
 * @brief Run part detectors to predict the location of each part or run multi-class classifiers to predict object class.  The predicted part locations are stored
 * into the part locations field of each dataset example (over-writing any existing values)
 * @param testSet The file name of the testset
 * @param classNameIn The file name of the class definition file
 * @param predictionsOut The file name in which to store predicted part locations (in the same file format as 'testSet')
 * @param imagesDirOut If non-null, store an image visualization of the predicted bounding box of each part for each image
 * @param evaluatePartDetection If true, runs part detector on image
 * @param evaluateClassification If true, evaluates multi-class classifier.  If evaluatePartDetection is also on, it evaluates the classifiers
 * on the predicted location.  Otherwise it does it on the ground truth
 * @param matfileOut If non-null, store matlab matrices of class and part location predictions and scores
 * @param matlabProgressOut If non-null, periodically saves progress of all results to matlab file (for debugging while experiment is in progress)
 * @return The average loss
 * 
 */
void EvaluateTestset(const char *testSet, const char *classNameIn, bool evaluatePartDetection, bool evaluateClassification, 
		     const char *predictionsOut=NULL, const char *imagesDirOut=NULL, const char *matfileOut=NULL,
		     const char *matlabProgressOut=NULL);



/**
 * @brief Interactively label a testset, using the current part detector to speedup annotation
 * @param testSet The file name of the testset
 * @param classNameIn The file name of the class definition file
 * @param stopThresh When simulating users, the user is assumed to accept a part location when it is within stopThresh 
 *        standard deviations of their labeled location (standard deviation as measured from a validation set)
 * @param matfileOut If non-null, store matlab matrices of class and part location predictions and scores
 * @param debugDir If non-null, generate an HTML visualization for each test image of the parts labeled, and the evolution of 
 *  part location probability maps as the user answers questions.  The html will be stored to debugDir/index.html, with lots of images in debugDir
 * @param debugImages If true, generate an HTML visualization for each test image of the parts corrected
 * @param debugProbabilityMaps If true, display the evolution of part location probability maps as the user answers questions.
 * @param matlabProgressOut If non-null, periodically saves progress of all results to matlab file (for debugging while experiment is in progress)
 *
 */
void EvaluateTestsetInteractive(const char *testSet, const char *classNameIn, float stopThresh=1, const char *matfileOut=NULL, 
				const char *debugDir=NULL, bool debugImages=false,
				bool debugProbabilityMaps=false, const char *matlabProgressOut=NULL);


/**
 * @brief Visualize a dataset and/or learned detection models as an html file
 * @param testSet The file name of the testset
 * @param classNameIn The file name of the class definition file
 * @param imagesDirOut The output directory into which to store html and image files
 * @param visualizePartModel Create a visualization of the learned part detection model
 * @param visualizeAttributeModels Create a visualization of the learned attribute detector models
 * @param showImageGallery Draw an image gallery of the dataset images, organized by class
 * @param visualizeLabels Draw an image of the ground truth labels (visible when the user clicks on a gallery image)
 * @param visualizeImageFeatures Draw a visualization of the part-localized features for each iamge (visible when the user clicks on a gallery image)
 * @param sort_cmp If non-null, sort the images in the gallery using a custom function, e.g. PartLocationsPoseCmp(), PartLocationsClassCmp(), PartLocationsScoreCmp(), PartLocationsAspectRatioCmp()
 */
void VisualizeDataset(const char *testSet, const char *classNameIn, const char *imagesDirOut, 
		      bool visualizePartModel=true, bool visualizeAttributeModels=true, bool showImageGallery=true,
		      bool visualizeLabels=true, bool visualizeImageFeatures=false,
		      int ( * sort_cmp ) ( const void *, const void * ) = NULL);


//void PlotResults(const char **matfiles, int numMatFiles, const char *pngOut, const char **labels, 
//		 const char *title, const char *xLabel, const char *yLabel, int timeInterval);


/**
 * @brief Run an object detector on a directory of images, outputing VOC detection format predictions
 * @param classNameIn Name of classes definition file
 * @param dirName Directory to be scanned for test images
 * @param predFileOut Name of the file to output predictions
 * @param htmlDir if non-null, store an html visualization of predicted bounding boxes
 * @param overlap Percent overlap for Deva-Ramanan-style non-maximal suppression (highest scoring bounding boxes are greedily selected, then all
 *  boxes where the percent area of overlap is greater than this threshold are rejected).  If overlap==-1, then regular non-maximal suppression is used.
 */
void EvaluateTestsetVOC(const char *classNameIn, const char *dirName, const char *predFileOut, const char *htmlDir=NULL, float overlap=.4f);

/**
 * @brief Run an object detector on a set of images, outputing Caltech Pedestrian detection format predictions
 * @param classNameIn Name of classes definition file
 * @param testSetDirName Directory to be scanned for test images
 * @param predDirOut Name of the directory to output predictions
 * @param htmlDir if non-null, store an html visualization of predicted bounding boxes
 * @param overlap Percent overlap for greedy non-maximal suppression (highest scoring bounding boxes are greedily selected, then all
 *  boxes where the percent area of overlap is greater than this threshold are rejected).  If overlap==-1, then regular non-maximal suppression is used.
 */
void EvaluateTestsetCaltechPedestrians(const char *classNameIn, const char *testSetDirName, 
				       const char *predDirOut, const char *htmlDir, float overlap=.4f);



void ExtractImageFeatures(const char *datasetIn, const char *classNameIn, const char *featuresOut, bool extractPartLocalizedFeatures=false, int max_images=-1, int target_image_width=0);
#endif
