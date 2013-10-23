#include "classes.h"
#include "part.h"
#include "pose.h"
#include "attribute.h"
#include "class.h"
#include "question.h"
#include "dataset.h"
#include "datasetIO.h"
#include "fisher.h"

/*
 * @file import_birds200.cpp
 * @brief Import the CUB-200-2011 dataset
 */
 

#define IMAGE_GRID_SIZE 20
#define RGB_CODEBOOK_SIZE 100
#define CIE_CODEBOOK_SIZE 100
#define SIFT_CODEBOOK_SIZE 100
//#define SIFT_CODEBOOK_SIZE 128
#define ATTRIBUTE_WINDOW_SIZE 7
#define PART_WINDOW_SIZE 7
#define BODY_PART_WINDOW_SIZE 7   //Yinxiao changes here. before 0
//#define BODY_PART_WINDOW_SIZE 14
#define MIN_WIDTH 30
#define NON_VISIBLE_COST 15
#define VALIDATION_SET_SIZE 200
#define NUM_PART_FEATURES 1
#define USE_FISHER_FEATURES 1

#define NUM_ATTRIBUTE_FEATURE_SCALES 1
#define USE_SIFT

#define CLUSTER_BY_SEGMENTATION 1
#define PART_SEG_WIDTH (PART_WINDOW_SIZE*8)

#define NUM_POSES 7           // 7*2 mixture components for each part (multiplied by 2 for mirrored detectors)
#define NUM_HEAD_POSES 15     // 15*2 mixture components for the head (multiplied by 2 for mirrored detectors)
#define NUM_BODY_POSES 50     // 50*2 mixture components for the body (multiplied by 2 for mirrored detectors)

//#define NUM_POSES 3
//#define NUM_HEAD_POSES 5     // 15*2 mixture components for the head (multiplied by 2 for mirrored detectors)
//#define NUM_BODY_POSES 5     // 50*2 mixture components for the body (multiplied by 2 for mirrored detectors)


//#define MULTIPLE_VALIDATION_SPLITS 5

bool ImportCUBXBirds200(const char *datasetDir, const char *modelDefinitionOut, const char *trainsetOut, 
			const char *testsetOut, const char *validationSetOut, int numValidation, int num_poses, 
			int num_poses_head, int num_poses_body, int num_train=-1);

/**
 * @example import_birds200.cpp
 *
 * This example invokes dataset-specific import functions that create training and testing set files in our format.
 * Currently just imports Caltech-UCSD-Birds-200-2011 
 *
 * Example usage:
 * - Import from Caltech-UCSD-Birds-200-2011, where http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz is extracted into the directory CUB_200_2011, and http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011_segmentations.tgz is extracted into CUB_200_2011/segmentations
\htmlonly <div style="padding: 0.5em 1em; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; background-color: #eaeafa;">
 $ examples/bin/release_static/import_birds200.out CUB_200_2011 CUB_200_2011_data 
</div> \endhtmlonly
*
* The following examples can use this dataset:
* - train_localized_v20q.cpp - train a deformable part model, pose-normalized multiclass classifier, and human-in-the-loop recognition system
* - train_detector.cpp - train a deformable part model
* - train_multiclass_detector.cpp - train a pose-normalized multiclass classifier
*/
int main(int argc, const char **argv) {

  char datasetDir[1000], outputDir[1000], modelDefinitionOut[1000], trainsetOut[1000], testsetOut[1000], validationOut[1000];
  int num_train = -1;  // by default use the provided train/test split

  if(argc < 3) {
    fprintf(stderr, "USAGE: ./import_birds200.out CUB_200_2011 CUB_200_2011_data <num_train>\n");
    return -1;
  }
  if(argc >= 4) num_train = atoi(argv[3]);

  strcpy(datasetDir, argv[1]);
  strcpy(outputDir, argv[2]);
  sprintf(modelDefinitionOut, "%s/classes.txt", outputDir);
  sprintf(trainsetOut, "%s/train.txt", outputDir);
  sprintf(testsetOut, "%s/test.txt", outputDir);
  sprintf(validationOut, "%s/validation.txt", outputDir);
  CreateDirectoryIfNecessary(outputDir);

  ImportCUBXBirds200(datasetDir, modelDefinitionOut, trainsetOut, testsetOut, validationOut, 100, NUM_POSES*2, NUM_HEAD_POSES*2, NUM_BODY_POSES*2, num_train);

  return 0;
}


bool ImportCUBXBirds200(const char *datasetDir, const char *modelDefinitionOut, const char *trainsetOut, 
			const char *testsetOut, const char *validationSetOut, int numValidation, 
			int num_poses, int num_poses_head, int num_poses_body, int num_train) {
  char fname[1000], line[100000], qstr[1000];
  FeatureWindow attributeFeatures[2+NUM_ATTRIBUTE_FEATURE_SCALES], partFeatures[6], bodyFeatures;
  Classes *classes = new Classes();
  char vis[1000];

#if CLUSTER_BY_SEGMENTATION
  sprintf(fname, "%s/segmentations", datasetDir);
  if(!DirectoryExists(fname)) {
    fprintf(stderr, "ERROR: directory %s doesn't exist.  Download CUB-200-2011 segmentations or set CLUSTER_BY_SEGMENTATION=0\n", fname);
    return false;
  }
#endif

  //char trainsetSortedOut[1000];
  //sprintf(trainsetSortedOut, "%s.sorted", trainsetOut);

  attributeFeatures[0].w =  attributeFeatures[0].h = ATTRIBUTE_WINDOW_SIZE;
  attributeFeatures[0].name = StringCopy("HIST_CIE");  // StringCopy("FISHER_CIE");
  attributeFeatures[0].dx = attributeFeatures[0].dy = attributeFeatures[0].scale = attributeFeatures[0].orientation = 0;
  attributeFeatures[0].poseInd = -1;
  attributeFeatures[0].dim = CIE_CODEBOOK_SIZE;  // CIE_CODEBOOK_SIZE*48*2;
  attributeFeatures[NUM_ATTRIBUTE_FEATURE_SCALES+1] = attributeFeatures[NUM_ATTRIBUTE_FEATURE_SCALES] = attributeFeatures[0];
#ifdef USE_SIFT
  attributeFeatures[NUM_ATTRIBUTE_FEATURE_SCALES].name = StringCopy("HIST_SIFT");
  attributeFeatures[NUM_ATTRIBUTE_FEATURE_SCALES+1].name = StringCopy("HIST_SIFT");
#else
  attributeFeatures[NUM_ATTRIBUTE_FEATURE_SCALES].name = StringCopy("HIST_HOG");  
  attributeFeatures[NUM_ATTRIBUTE_FEATURE_SCALES+1].name = StringCopy("HIST_HOG");  
#endif
  attributeFeatures[NUM_ATTRIBUTE_FEATURE_SCALES].dim = SIFT_CODEBOOK_SIZE;  //SIFT_CODEBOOK_SIZE*64*2;
  attributeFeatures[NUM_ATTRIBUTE_FEATURE_SCALES+1].dim = SIFT_CODEBOOK_SIZE;  //SIFT_CODEBOOK_SIZE*64*2;
#if USE_FISHER_FEATURES
  attributeFeatures[0].name = StringCopy("FISHER_CIE");
  attributeFeatures[0].dim = CIE_CODEBOOK_SIZE*48*2;
#ifdef USE_SIFT
  attributeFeatures[NUM_ATTRIBUTE_FEATURE_SCALES].name = StringCopy("FISHER_SIFT");
  attributeFeatures[NUM_ATTRIBUTE_FEATURE_SCALES+1].name = StringCopy("FISHER_SIFT4");
#else
  attributeFeatures[NUM_ATTRIBUTE_FEATURE_SCALES].name = StringCopy("FISHER_HOG");
  attributeFeatures[NUM_ATTRIBUTE_FEATURE_SCALES+1].name = StringCopy("FISHER_HOG4");
#endif
  attributeFeatures[NUM_ATTRIBUTE_FEATURE_SCALES].dim = SIFT_CODEBOOK_SIZE*FISHER_PCA_DIMS*2;
  attributeFeatures[NUM_ATTRIBUTE_FEATURE_SCALES+1].dim = SIFT_CODEBOOK_SIZE*FISHER_PCA_DIMS*2;
#endif
  for(int f = 1; f < NUM_ATTRIBUTE_FEATURE_SCALES; f++) {
    attributeFeatures[f] = attributeFeatures[f-1];
    attributeFeatures[f].scale = f;
  }

  partFeatures[0].w = partFeatures[0].h = PART_WINDOW_SIZE;
  partFeatures[0].name = StringCopy("HOG");
  partFeatures[0].dx = partFeatures[0].dy = partFeatures[0].scale = partFeatures[0].orientation = 0;
  partFeatures[0].poseInd = -1;
  partFeatures[0].dim = PART_WINDOW_SIZE*PART_WINDOW_SIZE*classes->GetFeatureParams()->hogParams.numBins;
  partFeatures[2] = partFeatures[1] = partFeatures[0];
  partFeatures[1].scale = 3;
  partFeatures[2].scale = 6;
  partFeatures[4].w = partFeatures[4].h = 1;
  partFeatures[4].name = StringCopy("MASK");
  partFeatures[4].dx = partFeatures[4].dy = partFeatures[4].scale = partFeatures[4].orientation = 0;
  partFeatures[4].poseInd = -1;
  partFeatures[4].dim = 1;
  bodyFeatures = partFeatures[0];
  bodyFeatures.w = bodyFeatures.h = BODY_PART_WINDOW_SIZE;

  // Read the list of class names
  sprintf(fname, "%s/classes.txt", datasetDir);
  FILE *fin = fopen(fname, "r");
  if(!fin) {
    fprintf(stderr, "Couldn't open %s\n", fname);
    return false;
  }
  while(fgets(line, 99999, fin)) {
    chomp(line);
    const char *ptr = strstr(line, " ");
    int id = atoi(line);
    if(!ptr || id != classes->NumClasses()+1) {
      fprintf(stderr, "Error reading class definition file %s, at line %s\n", fname, line);
      fclose(fin);
      return false;
    }
    ObjectClass *cl = new ObjectClass(classes, ptr+1);
    char *ex[100], tmp[1000];
    for(int i = 0; i < 5; i++) {
      sprintf(tmp, "images/classes/%s/%d.jpg", ptr+1+4, i+1);
      ex[i] = StringCopy(tmp);
    }
    cl->SetExemplarImages(ex, 5);
    classes->AddClass(cl);
  }
  fclose(fin);

  int num_remove_parts = 2;  // remove the left and right legs
  int part_ind[] = { 0, 1, 2, 3, 4, 5, 6, -1, 7, 8, 9, -1, 10, 11, 12, 13, 14 };

  int num_pseudo_parts = 2;
  const char *pseudo_parts[] = { "16 head", "17 body" };
  int num_pseudo_poses[2] = { num_poses_head, num_poses_body };
  int num_child_parts[] = { 7, 7 };
  int head_parts[] = { 1, 4, 5, 6, 8, 9, 12 }; 
  int body_parts[] = { 0, 2, 3, 7, 10, 11, 13 };
  int *part_children[] = { head_parts, body_parts };
  int num_pseudo_parts_used = 0;
  const char *part_abbreviations[] = { "Ba", "Bi", "Be", "Br", "Cr", "Fo", "LE", "LW", "Na", "RE", "RW", "Ta", "Th", "He", "Bo" }; 
  //0 back; 1 beak; 2 belly; 3 breast; 4 crown; 5 forehead; 6 left eye; 7 left wing; 8 Nape; 9 right eye; 10 right wing; 11 tail; 12 throat; 13 head; 14 body  --Yinxiao


  // Read the list of part names
  char name[1000];
  sprintf(fname, "%s/parts/parts.txt", datasetDir);
  fin = fopen(fname, "r");
  if(!fin) {
    fprintf(stderr, "Couldn't open %s\n", fname);
    return false;
  }
  const char *ptr2;
  ObjectPose *poses[40][100];
  while((ptr2=fgets(line, 99999, fin))!=NULL || num_pseudo_parts_used < num_pseudo_parts) {
    chomp(line);
    int pseudo = -1;
    if(!ptr2) { pseudo = num_pseudo_parts_used;  ptr2 = pseudo_parts[num_pseudo_parts_used++]; }
    const char *ptr = strstr(ptr2, " ");
    int id = atoi(ptr2);
    if(part_ind[id-1] < 0) continue;

    if(!ptr || part_ind[id-1] != classes->NumParts()) {
      fprintf(stderr, "Error reading part definition file %s, at line %s\n", fname, line);
      fclose(fin);
      return false;
    }
    ObjectPart *part = new ObjectPart(ptr+1, false, part_abbreviations[part_ind[id-1]]);
    sprintf(vis, "images/parts/%s.png", ptr+1);
    part->SetVisualizationImage(vis);
    classes->AddPart(part);

    if(pseudo >= 0) {
      for(int i = 0; i < num_child_parts[pseudo]; i++) {
	part->AddPart(classes->GetPart(part_children[pseudo][i]));
      }
    }

    // Add poses and click part 
    sprintf(name, "%s not_visible", part->Name());
    ObjectPose *non_visible = new ObjectPose(name, false);
    part->AddPose(non_visible);
    classes->AddPose(non_visible);

    /*
    sprintf(name, "%s pose%d", part->Name(), 1);
    ObjectPose *pose = new ObjectPose(name, false);
    Attribute *a = new Attribute(ptr+1);
    a->SetFeatures(partFeatures,NUM_PART_FEATURES);
    pose->SetAppearanceModel(a);
    part->AddPose(pose);
    classes->AddPose(pose);
    */
    if(pseudo != -1) {
      for(int j = 0; j < num_pseudo_poses[pseudo]/2; j++) {
	sprintf(name, "%s pose%d", part->Name(), j+1);
	ObjectPose *pose = new ObjectPose(name, false);
#if BODY_PART_WINDOW_SIZE
	Attribute *a = new Attribute(ptr+1);
	a->SetFeatures(&bodyFeatures,1);
	pose->SetAppearanceModel(a);
#endif
	poses[part->Id()][j] = pose;
      }
    } else {
		for(int j = 0; j < num_poses/2; j++) {
			sprintf(name, "%s pose%d", part->Name(), j+1);
			ObjectPose *pose = new ObjectPose(name, false);
			Attribute *a = new Attribute(ptr+1);
			a->SetFeatures(partFeatures,NUM_PART_FEATURES);
	pose->SetAppearanceModel(a);
	poses[part->Id()][j] = pose;
      }
    }
  }

  if(classes->FindPart("left eye")) classes->FindPart("left eye")->SetFlipped(classes->FindPart("right eye"));
  if(classes->FindPart("right eye")) classes->FindPart("right eye")->SetFlipped(classes->FindPart("left eye"));
  if(classes->FindPart("left wing")) classes->FindPart("left wing")->SetFlipped(classes->FindPart("right wing"));
  if(classes->FindPart("right wing")) classes->FindPart("right wing")->SetFlipped(classes->FindPart("left wing"));
  for(int i = 0; i < classes->NumParts(); i++) {
    ObjectPart *part = classes->GetPart(i);
    for(int j = 0; j < (i >= classes->NumParts()-num_pseudo_parts ? num_pseudo_poses[i-(classes->NumParts()-num_pseudo_parts)] : num_poses)/2; j++) {
      ObjectPose *right = poses[part->GetFlipped() ? part->GetFlipped()->Id() : i][j]->FlippedCopy();
      part->AddPose(poses[i][j]);
      if(j == 0) {
	ObjectPart *clickPart = classes->CreateClickPart(part);
	ClickQuestion *q = new ClickQuestion(classes);
	sprintf(qstr, "Click on the %s", part->Name());
	q->SetText(qstr);
	q->SetNumSamples(50);
	q->SetPart(clickPart);
	clickPart->SetQuestion(q);
	sprintf(vis, "images/parts/%s.png", part->Name());
	q->SetVisualizationImage(vis);
	classes->AddQuestion(q);
      }
      part->AddPose(right);
      classes->AddPose(poses[i][j]);
      classes->AddPose(right);
    }
  }


  int base_parts = classes->NumParts() - num_pseudo_parts;
  
  const char *property_names[28] = { "has_bill_shape", "has_wing_color", "has_upperparts_color", 
				   "has_underparts_color", "has_breast_pattern", "has_back_color", 
				   "has_tail_shape", "has_upper_tail_color", "has_head_pattern", 
				   "has_breast_color", "has_throat_color", "has_eye_color", 
				   "has_bill_length", "has_forehead_color", "has_under_tail_color", 
				   "has_nape_color", "has_belly_color", "has_wing_shape", 
				   "has_size", "has_shape", "has_back_pattern", 
				   "has_tail_pattern", "has_belly_pattern", "has_primary_color", 
				   "has_leg_color", "has_bill_color", "has_crown_color", 
				   "has_wing_pattern" };
  const char *property_questions[28] = { "What is the bill shape?", "What is the wing color?", "What is the upperparts color?", 
					 "What is the underparts color?", "What is the breast color?", "What is the back color?", 
					 "What is the tail shape?", "What is the upper tail color?", "What is the head pattern?",
					 "What is the breast color?", "What is the throat color?", "What is the eye color?",
					 "What is the bill length?", "What is the forehead color?", "What is the under tail color", 
					 "What is the nape color?", "What is the belly color?", "What is the wing shape?", 
					 "What is the size of the bird?", "What is the shape of the bird?", "What is the back pattern?", 
					 "What is the tail pattern?", "What is the belly pattern?", "What is the primary color?", 
					 "What is the leg color?", "What is the bill color?", "What is the crown color?", 
					 "What is the wing pattern?" };
  int property_parts[28] = { 1, 7, 0, 
			     2, 3, 0,
			     11, 11, 13,
			     3, 12, 13,
			     1, 5, 11, 
			     8, 2, 10, 
			     14, 14, 0,
			     11, 2, 14,
			     2, 1, 4, 
			     11 };
    
  bool isMultipleChoice[28] = { true, false, false, 
			      false, true, false,
			      true, false, false,
			      false, false, false,
			      true, false, false,
			      false, false, true,
			      true, true, false,
			      true, true, false,
			      false, false, false,
			      true };
  Question *propertyQuestions[28];
  for(int i = 0; i < 28; i++) {
    propertyQuestions[i] = (isMultipleChoice[i] ? (Question*)(new MultipleChoiceAttributeQuestion(classes)) : 
			    (Question*)(new BatchQuestion(classes)));
    propertyQuestions[i]->SetText(property_questions[i]);
  }

  // Read the list of attribute names
  sprintf(fname, "%s/attributes/attributes.txt", datasetDir);
  fin = fopen(fname, "r");
  if(!fin) {
    fprintf(stderr, "Couldn't open %s\n", fname);
    return false;
  }
  while(fgets(line, 99999, fin)) {
    chomp(line);
    const char *ptr = strstr(line, " ");
    int id = atoi(line);
    if(!ptr || id != classes->NumAttributes()+1) {
      fprintf(stderr, "Error reading attribute definition file %s, at line %s\n", fname, line);
      fclose(fin);
      return false;
    }
    Attribute *a = new Attribute(ptr+1);
    a->SetFeatures(attributeFeatures,2+NUM_ATTRIBUTE_FEATURE_SCALES);
    const char *ptr2 = strstr(ptr, "::");
    assert(ptr2);
    char prop[1000]; strcpy(prop, ptr+1);
    prop[ptr2-(ptr+1)] = '\0';
    a->SetPropertyNames(prop, ptr2+2);
    classes->AddAttribute(a);

    sprintf(vis, "images/attributes/%s/%s.jpeg", prop, ptr2+2);
    a->SetVisualizationImage(vis);

    BinaryAttributeQuestion *q  = new BinaryAttributeQuestion(classes);
    sprintf(qstr, "%s?", a->Name());
    q->SetText(qstr);
    q->SetAttribute(a->Id());
    classes->AddQuestion(q);
    int property_ind = -1;
    for(int k = 0; k < 28; k++) {
      if(!strcmp(a->PropertyName(), property_names[k])) {
	property_ind = k;
	break;
      }
    }
    assert(property_ind >= 0);
    a->SetPart(classes->GetPart(property_parts[property_ind]));
    sprintf(vis, "images/parts/%s.png", classes->GetPart(property_parts[property_ind])->Name());
    q->SetVisualizationImage(vis);
    
    if(isMultipleChoice[property_ind]) {
      ((MultipleChoiceAttributeQuestion*)propertyQuestions[property_ind])->AddAttribute(a->Id());
    } else {
      ((BatchQuestion*)propertyQuestions[property_ind])->AddQuestion(q);
    }
    if(!propertyQuestions[property_ind]->GetVisualizationImageName())
      propertyQuestions[property_ind]->SetVisualizationImage(vis);
  }
  fclose(fin);

  for(int i = 0; i < 28; i++) 
    classes->AddQuestion(propertyQuestions[i]);


  // Read the list of certainty names
  sprintf(fname, "%s/attributes/certainties.txt", datasetDir);
  fin = fopen(fname, "r");
  if(!fin) {
    fprintf(stderr, "Couldn't open %s\n", fname);
    return false;
  }
  while(fgets(line, 99999, fin)) {
    chomp(line);
    const char *ptr = strstr(line, " ");
    int id = atoi(line);
    if(!ptr || id != classes->NumCertainties()+1) {
      fprintf(stderr, "Error reading certainty definition file %s, at line %s\n", fname, line);
      fclose(fin);
      return false;
    }
    classes->AddCertainty(ptr+1);
  }
  fclose(fin);


  
  // Read the list of class-attribute memberships
  sprintf(fname, "%s/attributes/class_attribute_labels_continuous.txt", datasetDir);
  fin = fopen(fname, "r");
  if(!fin) {
    fprintf(stderr, "Couldn't open %s\n", fname);
    return false;
  }
  for(int i = 0; i < classes->NumClasses(); i++) {
    char *ptr = fgets(line, 99999, fin);  assert(ptr);
    chomp(line);
    float *weights = (float*)malloc(sizeof(float)*classes->NumAttributes());
    for(int j = 0; j < classes->NumAttributes(); j++) {
      int n = sscanf(ptr, "%f", &weights[j]);  
      weights[j] = weights[j]/50.0f-1.0f;
      assert(n);
      ptr = strstr(ptr+1, " ");
    }
    classes->GetClass(i)->SetAttributeWeights(weights);
  }
  fclose(fin);


  // Read the list of all dataset image names
  char imgName[1000], segName[1000], bbox_name[1000], line_bbox[1000];
  Dataset *dataset = new Dataset(classes);
  if(!dataset->Load("all_images.tmp.txt")) {
    sprintf(fname, "%s/images.txt", datasetDir);
    sprintf(bbox_name, "%s/bounding_boxes.txt", datasetDir);
    fin = fopen(fname, "r");
    if(!fin) {
      fprintf(stderr, "Couldn't open %s\n", fname);
      return false;
    }
    FILE *fin_bbox = fopen(bbox_name, "r");
    if(!fin_bbox) {
      fprintf(stderr, "Couldn't open %s\n", bbox_name);
      return false;
    }
    while(fgets(line, 99999, fin)) {
      assert(fgets(line_bbox, 999, fin_bbox));
      chomp(line);
      const char *ptr = strstr(line, " ");
      int id = atoi(line);
      if(!ptr || id != dataset->NumExamples()+1) {
	fprintf(stderr, "Error reading images file %s, at line %s\n", fname, line);
	fclose(fin);
	return false;
      }
      int id_b;
      float x_b, y_b, w_b, h_b;
      assert(sscanf(line_bbox, "%d %f %f %f %f", &id_b, &x_b, &y_b, &w_b, &h_b) == 5 && id_b == id);

      StructuredExample *ex = new StructuredExample;
      PartLocalizedStructuredData *x = new PartLocalizedStructuredData();
      MultiObjectLabelWithUserResponses *y = new MultiObjectLabelWithUserResponses(x);
      PartLocalizedStructuredLabelWithUserResponses *l = new PartLocalizedStructuredLabelWithUserResponses(x);
      y->AddObject(l);
      l->SetBoundingBox((int)x_b, (int)y_b, (int)w_b, (int)h_b); 

      ex->x = x;   ex->y = y;
      sprintf(imgName, "%s/images/%s", datasetDir, ptr+1);
      sprintf(segName, "%s/segmentations/%s", datasetDir, ptr+1);
      StripFileExtension(segName);
      strcat(segName, ".png");
      x->SetImageName(StringCopy(imgName));
      x->SetSegmentationName(StringCopy(segName));
      IplImage *img = x->GetProcess(classes)->Image();
      if(!img) {
	fprintf(stderr, "Couldn't load image %s\n", imgName);
	return false;
      }
      fprintf(stderr, "Loading image %d: %s %dX%d\n", id, ptr+1, img->width, img->height);
      x->SetSize(img->width, img->height);
      x->Clear();
      dataset->AddExample(ex);
    }
    dataset->Save("all_images.tmp.txt");
    fclose(fin);

    //dataset->BuildCroppedImages("images", "cropped");	exit(0);
  }

  // Read the list of class labels for each image
  sprintf(fname, "%s/image_class_labels.txt", datasetDir);
  fin = fopen(fname, "r");
  if(!fin) {
    fprintf(stderr, "Couldn't open %s\n", fname);
    return false;
  }
  int num = 0;
  while(fgets(line, 99999, fin)) {
    chomp(line);
    const char *ptr = strstr(line, " ");
    int id = atoi(line);
    if(!ptr || id != num+1 || id-1 > dataset->NumExamples()) {
      fprintf(stderr, "Error reading image class labels file %s, at line %s\n", fname, line);
      fclose(fin);
      return false;
    }
    dataset->GetExampleLabel(num)->GetObject(0)->SetClassID(atoi(ptr+1)-1);
    num++;
  }
  fclose(fin);
  assert(num == dataset->NumExamples());

  
  // Read the train/test split
  int *split = (int*)malloc(dataset->NumExamples()*sizeof(int));
  sprintf(fname, "%s/train_test_split.txt", datasetDir);
  fin = fopen(fname, "r");
  if(!fin) {
    fprintf(stderr, "Couldn't open %s\n", fname);
    return false;
  }
  num = 0;
  while(fgets(line, 99999, fin)) {
    chomp(line);
    const char *ptr = strstr(line, " ");
    int id = atoi(line);
    if(!ptr || id != num+1 || id > dataset->NumExamples()) {
      fprintf(stderr, "Error reading train_test file %s, at line %s\n", fname, line);
      fclose(fin);
      return false;
    }
    split[num] = atoi(ptr+1);
    num++;
  }
  fclose(fin);
  assert(num == dataset->NumExamples());

  // Read in part locations
  sprintf(fname, "%s/parts/part_locs.txt", datasetDir);
  fin = fopen(fname, "r");
  if(!fin) {
    fprintf(stderr, "Couldn't open %s\n", fname);
    return false;
  }
  num = 0;
  PartLocation *locs= NULL;
  while(fgets(line, 99999, fin)) {
    int image_id, part_id, visible;
    float x, y;
    chomp(line);
    if(sscanf(line, "%d %d %f %f %d", &image_id, &part_id, &x, &y, &visible) != 5 || 
       image_id != num/(base_parts+num_remove_parts)+1 || part_id != (num%(base_parts+num_remove_parts))+1) {
      fprintf(stderr, "Error reading part_locs file %s, at line %s\n", fname, line);
      fclose(fin);
      return false;
    }
    if(part_id == 1) 
      locs = PartLocation::NewPartLocations(classes, dataset->GetExampleData(image_id-1)->Width(), 
					    dataset->GetExampleData(image_id-1)->Height(), NULL, false);
    if(part_ind[part_id-1] >= 0) {
      locs[part_ind[part_id-1]].SetImageLocation(x, y, LATENT, 0, classes->GetPart(part_ind[part_id-1])->GetPose(visible)->Name());
      if(part_id == base_parts+num_remove_parts) {
	// for the head and body, compute their locations as the average location of their parts
	for(int j = 0; j < num_pseudo_parts; j++) {
	  int numC = 0;
	  float x=0, y=0, xx, yy, tm=0;
	  int pose;
	  for(int i = 0; i < num_child_parts[j]; i++) {
	    locs[part_children[j][i]].GetImageLocation(&xx,&yy);
	    locs[part_children[j][i]].GetDetectionLocation(NULL,NULL,NULL,NULL,&pose);
	    if(pose > 0) {
	      x += xx; y += yy;
	      tm += locs[part_children[j][i]].GetResponseTime();
	      numC++;
	    }
	  }
	  if(numC) {
	    locs[base_parts+j].SetImageLocation(x/numC, y/numC, LATENT, 0, classes->GetPart(base_parts+j)->GetPose(1)->Name());
	    locs[base_parts+j].SetResponseTime(tm/numC);
	  } else
	    locs[base_parts+j].SetImageLocation(x, y, LATENT, 0, classes->GetPart(base_parts+j)->GetPose(0)->Name());
	}
	dataset->GetExampleLabel(image_id-1)->GetObject(0)->SetPartLocations(locs);
      }
    }
    num++;
  }
  fclose(fin);
  assert(num/(base_parts+num_remove_parts) == dataset->NumExamples() && num%(base_parts+num_remove_parts) == 0);

  // Read in part locations
  sprintf(fname, "%s/parts/part_click_locs.txt", datasetDir);
  fin = fopen(fname, "r");
  if(!fin) {
    fprintf(stderr, "Couldn't open %s\n", fname);
    return false;
  }
  PartLocation *click_locs[1000];
  memset(click_locs, 0, 1000*sizeof(PartLocation*));
  int curr_image = 0, curr_part = base_parts+num_remove_parts, user = 0;
  while(fgets(line, 99999, fin)) {
    int image_id, part_id, visible;
    float x, y, responseTime;
    chomp(line);
    if(!(sscanf(line, "%d %d %f %f %d %f", &image_id, &part_id, &x, &y, &visible, &responseTime) == 6 && 
	 ((curr_image == image_id && (part_id==curr_part || part_id==curr_part+1)) || 
	  (curr_image+1 == image_id && part_id == 1 && curr_part == base_parts+num_remove_parts)))) {
      fprintf(stderr, "Error reading part_click_locs file %s, at line %s\n", fname, line);
      fclose(fin);
      return false;
    }
    
    if(image_id != curr_image && user) {
      for(int i = 0; i < user; i++) {
        // for the head and body, compute their locations as the average location of their parts
        bool invalid = false;
        for(int j = 0; j < num_pseudo_parts; j++) {
          int numC = 0;
          float x=0, y=0, xx, yy, tm=0;
          int pose;
          for(int c = 0; c < num_child_parts[j]; c++) {
            click_locs[i][part_children[j][c]].GetImageLocation(&xx,&yy);
            click_locs[i][part_children[j][c]].GetDetectionLocation(NULL,NULL,NULL,NULL,&pose);
            if(pose > 0) {
      	      x += xx; y += yy;
	      tm += click_locs[i][part_children[j][c]].GetResponseTime();
              numC++;
            }
          }
          if(numC) {
            click_locs[i][base_parts+j].SetImageLocation(x/numC, y/numC, 1, 0, classes->GetClickPart(base_parts+j)->GetPose(1)->Name());
	    click_locs[i][base_parts+j].SetResponseTime(tm/numC);
          } else 
            invalid = true;
        }
        if(!invalid) {
          UserResponses *u = new UserResponses(dataset->GetExampleData(curr_image-1));
	  dataset->GetExampleLabel(curr_image-1)->GetObject(0)->AddUser(u);
	  u->SetPartLocations(click_locs[i]);
        }
      }
      memset(click_locs, 0, 1000*sizeof(PartLocation*));
      user = 0;
    }
    if(curr_part == part_id) {
      //user++;
    } else
      user = 0;
    curr_part = part_id;
    curr_image = image_id;
    if(!click_locs[user]) {
      click_locs[user] = PartLocation::NewPartLocations(classes, dataset->GetExampleData(curr_image-1)->Width(),
							dataset->GetExampleData(curr_image-1)->Height(), NULL, true);
      for(int i = 0; i < classes->NumParts(); i++)
	click_locs[user][i].SetImageLocation(LATENT, LATENT, LATENT, LATENT, classes->GetClickPart(i)->GetPose(0)->Name());
    }
    if(part_ind[part_id-1] >= 0) {
      click_locs[user][part_ind[part_id-1]].SetImageLocation(x, y, 1, 0, classes->GetClickPart(part_ind[part_id-1])->GetPose(visible)->Name());
      click_locs[user][part_ind[part_id-1]].SetResponseTime(responseTime);
    }
    user++;
  }
  if(user && curr_image) {
    for(int i = 0; i < user; i++) {
      dataset->GetExampleLabel(curr_image-1)->GetObject(0)->AddUser(new UserResponses(dataset->GetExampleData(curr_image-1)));
      dataset->GetExampleLabel(curr_image-1)->GetObject(0)->SetPartClickLocations(i, click_locs[i]);
    }
  }
  fclose(fin);

  

  sprintf(fname, "%s/attributes/image_attribute_labels.txt", datasetDir);
  fin = fopen(fname, "r");
  if(!fin) {
    fprintf(stderr, "Couldn't open %s\n", fname);
    return false;
  }
  num = 0;
  AttributeAnswer *attrResp = NULL;
  while(fgets(line, 99999, fin)) {
    int image_id, attribute_id, present, certainty;
    float respTime;
    chomp(line);
    if(sscanf(line, "%d %d %d %d %f", &image_id, &attribute_id, &present, &certainty, &respTime) != 5 || 
       image_id != num/classes->NumAttributes()+1 || attribute_id != (num%classes->NumAttributes())+1) {
      fprintf(stderr, "Error reading part_locs file %s, at line %s\n", fname, line);
      fclose(fin);
      return false;
    }
    if(attribute_id == 1) 
      attrResp = (AttributeAnswer*)malloc(sizeof(AttributeAnswer)*classes->NumAttributes());
    attrResp[attribute_id-1].answer = present;
    attrResp[attribute_id-1].certainty = certainty-1;
    attrResp[attribute_id-1].responseTimeSec = respTime;
    if(attribute_id == classes->NumAttributes()) {
      if(!dataset->GetExampleLabel(image_id-1)->GetObject(0)->NumUsers()) {
	UserResponses *u = new UserResponses(dataset->GetExampleData(image_id-1));
	dataset->GetExampleLabel(image_id-1)->GetObject(0)->AddUser(u);
      }
      dataset->GetExampleLabel(image_id-1)->GetObject(0)->SetAttributes(0, attrResp);
    }
    num++;
  }
  fclose(fin);
  assert(num/classes->NumAttributes() == dataset->NumExamples() && num%classes->NumAttributes() == 0);

  if(num_train >= 0) {
    free(split);
    srand(time(NULL));
    split = RandSplit(dataset->NumExamples()-num_train, dataset->NumExamples());
  }

  // Apply the train test split
  Dataset *trainset = dataset->ExtractSubset(split, 1);
  //trainset->Save("train.tmp.json");
  Dataset *testset = dataset->ExtractSubset(split, 0);
  //testset->Save("test.tmp.json");

  // Set all orientations to zero (no orientation used)
  trainset->AssignZeroOrientations();
  testset->AssignZeroOrientations();

  // Compute scales from the size of the bounding boxes around key points
  trainset->ApplyScaleConversion(MIN_WIDTH);
  testset->ApplyScaleConversion(MIN_WIDTH);

  bool assignChildren = true;
#if CLUSTER_BY_SEGMENTATION
  // Create a bunch of different poses by segmentation mask similarity
  char maskDir[1000];
  sprintf(maskDir, "%s.masks", modelDefinitionOut);
  CreateDirectoryIfNecessary(maskDir);
  float ***pose_clusters2 = trainset->ClusterPosesBySegmentationMasks(num_poses, PART_SEG_WIDTH, PART_SEG_WIDTH, true, maskDir);
  trainset->AssignPosesBySegmentationMasks(pose_clusters2, num_poses, PART_SEG_WIDTH, PART_SEG_WIDTH, false);
  testset->AssignPosesBySegmentationMasks(pose_clusters2, num_poses, PART_SEG_WIDTH, PART_SEG_WIDTH, false);
  free(pose_clusters2);
  assignChildren = false;
#endif

  for(int i = 0; i < num_pseudo_parts; i++) {
    // Create a bunch of different poses by clustering part location offsets
    int par = classes->NumParts()-num_pseudo_parts+i;
    float **pose_clusters = trainset->ClusterPosesByOffsets(par, num_pseudo_poses[i], NON_VISIBLE_COST, true);
    trainset->AssignPosesByOffsets(par, pose_clusters, num_pseudo_poses[i], NON_VISIBLE_COST, assignChildren);
    testset->AssignPosesByOffsets(par, pose_clusters, num_pseudo_poses[i], NON_VISIBLE_COST, assignChildren);
    free(pose_clusters);
  }

  // Split off a portion of the training set into the validation set
  int *rand_split = RandSplit(VALIDATION_SET_SIZE, trainset->NumExamples());
  Dataset *validation = trainset->ExtractSubset(rand_split, 0);
  Dataset *trainset_final = trainset->ExtractSubset(rand_split, 1);
  
  fprintf(stderr, "Computing initial part-aspect spatial offsets...");
  classes->AddSpatialTransitions(trainset_final, 1, false, true, true); 
  fprintf(stderr, "...");
  classes->ComputePartPoseTransitionWeights(trainset_final, false, true);
  fprintf(stderr, "done\n");
  fprintf(stderr, "Computing part click statistics...");
  trainset_final->LearnUserClickProbabilities();
  fprintf(stderr, "done\n");
  classes->SetDetectionLossMethod(LOSS_NUM_INCORRECT);

  attributeFeatures[0].w = attributeFeatures[0].h = attributeFeatures[1].w = attributeFeatures[1].h = IMAGE_GRID_SIZE;
  classes->SetFeatureWindows(attributeFeatures,NUM_ATTRIBUTE_FEATURE_SCALES+2);

  if(!classes->Save(modelDefinitionOut))
    return false;

  if(!trainset_final->Save(trainsetOut))
    return false;
  if(!testset->Save(testsetOut))
    return false;
  if(!validation->Save(validationSetOut))
    return false;

#ifdef MULTIPLE_VALIDATION_SPLITS
  for(int i = 0; i < MULTIPLE_VALIDATION_SPLITS; i++) {
    int *rand_split2 = RandSplit(VALIDATION_SET_SIZE, trainset->NumExamples());
    Dataset *validation2 = trainset->ExtractSubset(rand_split, 0);
    Dataset *trainset_final2 = trainset->ExtractSubset(rand_split, 1);
    char trainsetOut2[1000], validationSetOut2[1000];
    strcpy(trainsetOut2, trainsetOut); StripFileExtension(trainsetOut2);
    strcpy(validationSetOut2, validationSetOut); StripFileExtension(validationSetOut2);
    sprintf(trainsetOut2+strlen(trainsetOut2), "%d.txt", i+1);
    sprintf(validationSetOut2+strlen(validationSetOut2), "%d.txt", i+1);
    if(!trainset_final2->Save(trainsetOut2))
      return false;
    if(!validation2->Save(validationSetOut2))
      return false;
  }
#endif

  //trainset_final->Sort();
  //trainset_final->Save(trainsetSortedOut);

  delete dataset;
  delete trainset_final;
  delete validation;
  delete testset;
  delete trainset;
  free(split);
  free(rand_split);

  return true;
}
