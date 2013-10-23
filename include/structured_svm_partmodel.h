#ifndef __STRUCTURED_SVM_PARTMODEL_H
#define __STRUCTURED_SVM_PARTMODEL_H

#include "classes.h"
#include "class.h"
#include "imageProcess.h"
#include "part.h"
#include "structured_svm.h"

/**
 * @file structured_svm_partmodel.h
 * @brief Implements routines for loss-sensitive deformable part model training
 */



class PartLocalizedStructuredLabel;
class PartLocalizedStructuredData;

typedef enum {
  SAMPLE_ALL_POSES,
  SAMPLE_BY_BOUNDING_BOX,
  SAMPLE_RANDOMLY,
  SAMPLE_UPDATE_EACH_POSE_IMMEDIATELY
} MultiSampleMethod;

/**
 * @class PartLocalizedStructuredSVM
 * @brief Abstract class that can be extended by any structured SVM learning algorithm that uses part location data 
 */
class PartLocalizedStructuredSVM : public StructuredSVM {
 public:
  PartLocalizedStructuredSVM();

  /**
   * @brief Get a pointer to the object defining the set of all parts and poses 
   */
  Classes *GetClasses() { return classes; }

  virtual bool Load(const Json::Value &root);

  /**
   * @brief returns a new PartLocalizedStructuredLabel label
   * @param x a PartLocalizedStructuredData object associated with this label
   */
  virtual StructuredLabel *NewStructuredLabel(StructuredData *x);

  /**
   * @brief returns a new PartLocalizedStructuredData example
   */
  virtual StructuredData *NewStructuredData();
  
 protected:
  Classes *classes;  /**< Defines the set of all parts and poses */
};


/**
 * @class PartModelStructuredSVM
 * @brief Implements routines for loss-sensitive deformable part model training 
 */
class PartModelStructuredSVM : public PartLocalizedStructuredSVM {
 public:
  /******************* Functions overridden from StructuredSVM *************************/
  virtual SparseVector Psi(StructuredData *x, StructuredLabel *y);
  virtual double Inference(StructuredData *x, StructuredLabel *ybar, SparseVector *w, StructuredLabel *y_partial=NULL, StructuredLabel *y_gt=NULL, double w_scale=1);
  virtual double Loss(StructuredLabel *y_gt, StructuredLabel *y_pred);
  virtual double ImportanceSample(StructuredData *x, SparseVector *w, StructuredLabel *y_gt, struct _SVM_cached_sample_set *set, double w_scale);

  virtual Json::Value Save();
  virtual bool Load(const Json::Value &root);
  virtual void OnFinishedIteration(StructuredExample *ex);
  virtual bool AddInitialSamples();

  /******************* Customized functions and member variables *************************/
 public:
  PartModelStructuredSVM();
  virtual ~PartModelStructuredSVM();


  /**
   * @brief Set the Classes object defining the part model
   */
  void SetClasses(Classes *c) { classes = c; sizePsi=classes->NumWeights(true,false);  }

  /**
   * @brief A numParts array of losses for each part, where higher means bigger loss for inaccurate detection
   */
  void SetPartLosses(double *d) { 
    partLosses = d; 
    params.maxLoss = 0; 
    for(int i = 0; i < classes->NumParts(); i++) 
      params.maxLoss += partLosses[i];
  }

  virtual char *VisualizeExample(const char *htmlDir, StructuredExample *ex, const char *extraInfo=NULL);

  void OnFinishedPassThroughTrainset(); 

  void SetSamplingMethod(MultiSampleMethod m) { multiSampleMethod = m; }

  void SetWeights(double *d) { if (sum_w) { delete sum_w; } sum_w = new SparseVector(d, sizePsi, true);  *sum_w *= sum_w_scale; }

 protected:
  double *partLosses;
  bool useSoftPartLocations;
  MultiSampleMethod multiSampleMethod;
};


/**
 * @class PartLocalizedStructuredData
 * @brief Stores data x for a training example for a part detector.  
 */
class PartLocalizedStructuredData : public StructuredData {
protected:
  char *imagename;  /**< File name of the image on disk */
  ImageProcess *process;  /**< Object containing all data for image processing and part dection */
  SparseVector *cached_features, *part_features;  /**< A cached version of the features extracted at each part location */
  int *partInds, *vectorInds;
  double sqr;
  char *segmentationName;
  VFLOAT **class_attribute_dot_products;  
  int image_width; /**< Image width (allows converting to detection map coordinates without loading the image yet) */
  int image_height; /**< Image height (allows converting to detection map coordinates without loading the image yet) */  
  float image_scale;
  bool dontUpdate;
  PartLocation *gt_locs;

  friend class ClassAttributeStructuredSVM;
  friend class PartModelStructuredSVM;

public:
  PartLocalizedStructuredData() { 
    imagename = segmentationName = NULL; 
    process = NULL;
    cached_features = part_features = NULL;
    partInds = vectorInds = NULL;
    image_width = image_height = -1;
    image_scale = 1;
    sqr = 0;
    class_attribute_dot_products = NULL;
    dontUpdate = false;
    gt_locs = NULL;
  }

  ~PartLocalizedStructuredData() { 
    if(process) delete process;
    if(imagename) free(imagename);
    if(segmentationName) free(segmentationName);
    if(cached_features) delete cached_features;
    if(class_attribute_dot_products) free(class_attribute_dot_products);
    if(partInds) delete [] partInds;
    if(vectorInds) delete [] vectorInds;
    if(part_features) delete part_features;
  }

  /**
   * @brief Get the ImageProcess associated with this example
   * @param c the class model definition
   */
  ImageProcess *GetProcess(Classes *c) {
    if(!process) {
      process = new ImageProcess(c, imagename, IM_MAXIMUM_LIKELIHOOD, false, false, false);
      if(segmentationName) process->Features()->SetSegmentationName(segmentationName);
    }
    process->Features()->SetImageScale(image_scale);
    return process;
  }
  void SetProcess(ImageProcess *proc) {
    if(process) delete process;
    process = proc;
    process->Features()->SetImageScale(image_scale);
  }

  void SetImageScale(float s) { image_scale = s; }

  /**
   * @brief Get the filename of the image for this example
   */
  char *GetImageName() { return imagename; }

  const char *GetName() { return imagename; }

  /**
   * @brief Set the filename of the image for this example
   * @param s Set the file name of this image
   */
  void SetImageName(const char *s) { if(imagename) free(imagename); imagename = StringCopy(s); }
  void SetSegmentationName(const char *s) { if(segmentationName) free(segmentationName); segmentationName = StringCopy(s); }
  const char *GetSegmentationName() { return segmentationName; }

  /**
   * @brief Free all memory caches associated with this example
   */
  void Clear() {
    if(process) delete process;
    process = NULL;
  }

  /**
   * @brief Get the image width for this example
   */
  int Width() { return image_width; }

  /**
   * @brief Get the image height for this example
   */
  int Height() { return image_height; }

  /**
   * @brief Set the image width and height for this example
   * @param w the image width
   * @param h the image height
   */
  void SetSize(int w, int h) { image_width =w; image_height = h; }

  /**
   * @brief Load this example from a JSON encoding
   * @param x A JSON encoding of this object
   * @param s a pointer to the main learning object
   * @return true on success
   */
  bool load(const Json::Value &x, StructuredSVM *s) { 
    if(!x.isMember("imagename")) { fprintf(stderr, "Error reading PartModelStructuredData, imagename undefined\n"); return false; }
    image_width = x.get("width", -1).asInt();
    image_height = x.get("height", -1).asInt();
    image_scale = x.get("image_scale", 1).asFloat();
    char fname[100000];
    strcpy(fname, x["imagename"].asString().c_str());
    imagename = StringCopy(fname);
    if(x.isMember("segmentation")) {
      strcpy(fname, x["segmentation"].asString().c_str());
      segmentationName = StringCopy(fname);
    }
    return true;
  }

  /**
   * @brief Save this example into a JSON encoding
   * @param s a pointer to the main learning object
   * @return A JSON encoding of this object
   */
  Json::Value save(StructuredSVM *s) {
    Json::Value x;
    x["imagename"] = imagename;
    if(image_width >= 0) x["width"] = image_width;
    if(image_height >= 0) x["height"] = image_height;
    if(image_scale != 1) x["image_scale"] = image_scale;
    if(segmentationName) x["segmentation"] = segmentationName;
    return x;
  }
};



/**
 * @class PartLocalizedStructuredLabel
 * @brief Stores a label y for a training example for a deformable part model.  This is just a
 * list of part locations
 */
class PartLocalizedStructuredLabel : public StructuredLabel {
protected:
  PartLocation *part_locations, *gt_part_locations; /**< An array of locations of for part */
  int obj_class;  /**< The id of the class (the class index in classes) */
  Classes *classes; /**< Object defining the set of possible classes and attribute memberships */
  PartLocalizedStructuredData *x;  /**< The data example this label corresponds to */
  CvRect bounding_box;
  bool *attributes;
  bool hasBoundingBox;

  friend class PartModelStructuredSVM;
  friend class ClassAttributeStructuredSVM;
  friend class ObjectLabels;
public:
   /**
    * @brief Create a new PartLocalizedStructuredLabel
    * @param x the PartLocalizedStructuredData associated with this label
    */
 PartLocalizedStructuredLabel(StructuredData *x) : StructuredLabel(x) { 
    obj_class = -1; part_locations = gt_part_locations = NULL; classes = NULL; 
    this->x = (PartLocalizedStructuredData*)x; 
    attributes = NULL;
    hasBoundingBox = false;
    bounding_box = cvRect(0,0,0,0);
  }

  ~PartLocalizedStructuredLabel() { 
     if(gt_part_locations && gt_part_locations != part_locations) delete [] gt_part_locations; 
     if(part_locations) delete [] part_locations; 
     if(attributes) delete [] attributes;
  }


  /**
   * @brief Get the array of part locations for this label
   * @return A NumParts() array of part locations
   */
  PartLocation *GetPartLocations() {
    return part_locations;
  }
  PartLocation *GetGTPartLocations() {
    return gt_part_locations;
  }

  /**
   * @brief Set the array of part locations for this label
   * @param locs A NumParts() array of part locations
   */
  void SetPartLocations(PartLocation *locs) {
    if(part_locations && part_locations != gt_part_locations) delete [] part_locations;
    part_locations = locs;
  }

  /**
   * @brief Set the object class for this label
   * @param id the id of the class
   */
  void SetClassID(int id) { obj_class = id; }

  /**
   * @brief Get the object class id for this label
   */
  int GetClassID() { return obj_class; }

  void SetBoundingBox(int x, int y, int w, int h) { 
    bounding_box.x = x;
    bounding_box.y = y;
    bounding_box.width = w;
    bounding_box.height = h;
    hasBoundingBox = true;
  }
  void GetBoundingBox(int *x, int *y, int *w, int *h) { 
    *x = bounding_box.x;   
    *y = bounding_box.y;   
    *w = bounding_box.width;   
    *h = bounding_box.height;   
  }

  /**
   * @brief Get the object class for this label
   */
  ObjectClass *GetClass() { return obj_class >= 0 ? classes->GetClass(obj_class) : NULL; }

  /**
   * @brief Load this label from a JSON encoding
   * @param y A JSON encoding of this label
   * @param s a pointer to the main learning object
   * @return true on success
   */
  bool load(const Json::Value &y, StructuredSVM *s) { 
    classes = ((PartLocalizedStructuredSVM*)s)->GetClasses();
    
    PartLocalizedStructuredData *xx = (PartLocalizedStructuredData*)x;
    if(!y.isMember("part_locations") && y.isMember("objects")) {
      if(part_locations) delete [] part_locations;
      if(!(part_locations = classes->LoadPartLocations(y["objects"][Json::UInt(0)]["part_locations"], xx->Width(), xx->Height(), false)))
	return false;
    } else if(y.isMember("part_locations")) { 
      bool b1, b2, b3;
      if(!(b1=y.isMember("part_locations")) || !(b2=y["part_locations"].isArray()) || !(b3=((int)y["part_locations"].size() == classes->NumParts()))) { 
	fprintf(stderr, "Error reading PartLocalizedStructuredLabel: %s\n", !b1 ? "part_locations undefined" : (!b2 ? "part_locations is not an array" : "part location size is invalid")); 
	return NULL; 
      }
      if(y.isMember("bounding_box")) {
	hasBoundingBox = true;
	bounding_box.x = y["bounding_box"].get("x",0).asInt();
	bounding_box.y = y["bounding_box"].get("y",0).asInt();
	bounding_box.width = y["bounding_box"].get("width",0).asInt();
	bounding_box.height = y["bounding_box"].get("height",0).asInt();
      }
      if(part_locations) delete [] part_locations;
      if(!(part_locations = classes->LoadPartLocations(y["part_locations"], xx->Width(), xx->Height(), false)))
	return false;
    }
    gt_part_locations = part_locations;
    

    if(y.isMember("class") || (y.isMember("objects") && y["objects"][Json::UInt(0)].isMember("class"))) { 
      char cname[1000];
      ObjectClass *c;
      if(!y.isMember("class")) strcpy(cname, y["objects"][Json::UInt(0)]["class"].asString().c_str());
      else strcpy(cname, y["class"].asString().c_str());
      if((c=classes->FindClass(cname)) != NULL) {
	obj_class = c->Id();
      } else {
	fprintf(stderr, "Error reading ClassAttributeStructuredLabel, class '%s' not found\n", cname);
	//obj_class = atoi(cname+strlen("Class"))-1;
	return false;
      }
    }

    return true;
  }

  /**
   * @brief Save this label from a JSON encoding
   * @param s a pointer to the main learning object
   * @return A JSON encoding of this label
   */
  Json::Value save(StructuredSVM *s){
    classes = ((PartLocalizedStructuredSVM*)s)->GetClasses();
    Json::Value y;
    if(part_locations) y["part_locations"] = classes->SavePartLocations(part_locations);
    if(obj_class >= 0) y["class"] = classes->GetClass(obj_class)->Name();
    if(hasBoundingBox) {
      Json::Value r;
      r["x"] = bounding_box.x;
      r["y"] = bounding_box.y;
      r["width"] = bounding_box.width;
      r["height"] = bounding_box.height;
      y["bounding_box"] = r;
    }
    return y;
  }
};




#endif


