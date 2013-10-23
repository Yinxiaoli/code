#ifndef __STRUCTURED_SVM_MULTI_OBJECT_H
#define __STRUCTURED_SVM_MULTI_OBJECT_H

#include "structured_svm_partmodel.h"

/**
 * @file structured_svm_multi_object.h
 * @brief Implements routines for loss-sensitive training of systems for detecting multiple objects per image
 */


class MultiObjectLabel;


/**
 * @class MultiObjectStructuredSVM
 * @brief Implements routines for loss-sensitive deformable part model training 
 */
class MultiObjectStructuredSVM : public PartModelStructuredSVM {
  int max_objects_per_image;

 public:
  /******************* Functions overridden from StructuredSVM *************************/

  MultiObjectStructuredSVM() : PartModelStructuredSVM() { max_objects_per_image = 50; } 
  int GetMaxObjectsPerImage() { return max_objects_per_image; }
  void SetMaxObjectsPerImage(int o) { max_objects_per_image = o; }

  SparseVector Psi(StructuredData *x, StructuredLabel *y);
  double Inference(StructuredData *x, StructuredLabel *ybar, SparseVector *w, StructuredLabel *y_partial=NULL, StructuredLabel *y_gt=NULL, double w_scale=1);
  double Loss(StructuredLabel *y_gt, StructuredLabel *y_pred);
  double ImportanceSample(StructuredData *x, SparseVector *w, StructuredLabel *y_gt, struct _SVM_cached_sample_set *set, double w_scale);
  char *VisualizeExample(const char *htmlDir, StructuredExample *ex, const char *extraInfo=NULL);
  
  Json::Value Save();
  bool Load(const Json::Value &root);

  /**
   * @brief returns a new PartLocalizedStructuredLabel label
   * @param x a PartLocalizedStructuredData object associated with this label
   */
  virtual StructuredLabel *NewStructuredLabel(StructuredData *x);

  /**
   * @brief Create a dataset that converts each object occurrence to a single training example
   * @param d A multi-object dataset 
   */
  StructuredDataset *ConvertToSingleObjectDataset(StructuredDataset *d);
};


/**
 * @class MultiObjectLabel
 * @brief Stores a label y for a training example for a deformable part model.  This is a
 * list of 
 */
class MultiObjectLabel : public StructuredLabel {
  PartLocalizedStructuredLabel **objects;   /**< An array of num_objects objects in this image label */
  int num_objects;   /**< The number of objects in this image label */

  friend class PartModelStructuredSVM;
  friend class ClassAttributeStructuredSVM;

 public:

   /**
    * @brief Create a new MultiObjectLabel
    * @param x the PartLocalizedStructuredData associated with this label
    */
  MultiObjectLabel(StructuredData *x) : StructuredLabel(x) { 
    objects = NULL;
    num_objects = 0;
  }

  ~MultiObjectLabel() {
    Clear();
  }

  void Clear() {
    if(objects) {
      for(int i = 0; i < num_objects; i++) 
	delete objects[i];
      free(objects);
    }
    objects = NULL;
  }

  char *VisualizeExample(const char *htmlDir, StructuredExample *ex, const char *extraInfo);

  /**
   * @brief Create a new PartLocalizedStructuredLabel
   */
  virtual PartLocalizedStructuredLabel *NewObject() { return new PartLocalizedStructuredLabel(x); }
  
  /**
   * @brief Add a new PartLocalizedStructuredLabel
   * @param obj The new object to add
   */
  void AddObject(PartLocalizedStructuredLabel *obj) { 
    objects = (PartLocalizedStructuredLabel**)realloc(objects, sizeof(PartLocalizedStructuredLabel*)*(num_objects+1));
    objects[num_objects++] = obj;
  }

  /**
   * @brief Get the ith object in this image
   * @param i The index of the object to get
   */
  PartLocalizedStructuredLabel *GetObject(int i) { assert(i >= 0 && i < num_objects); return objects[i]; }  

  /**
   * @brief Get the number of objects in this image
   */
  int NumObjects() { return num_objects; }  

  /**
   * @brief Load this label from a JSON encoding
   * @param y A JSON encoding of this label
   * @param s a pointer to the main learning object
   * @return true on success
   */
  bool load(const Json::Value &y, StructuredSVM *s) { 
    if(y.isMember("part_locations")) {
      PartLocalizedStructuredLabel *obj = NewObject();
      if(!obj->load(y, s)) {
	delete obj; 
	return false;
      }
      AddObject(obj);
    }
    if(y.isMember("objects") && y["objects"].isArray()) {
      for(int i = 0; i < (int)y["objects"].size(); i++) {
	PartLocalizedStructuredLabel *obj = NewObject();
	if(!obj->load(y["objects"][i], s)) {
	  delete obj; 
	  return false;
	}
	if(y["objects"][i].get("isValid",true).asBool())
	  AddObject(obj);
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
    Json::Value y;
    Json::Value o(Json::arrayValue);
    for(int i = 0; i < num_objects; i++) 
      o[i] = objects[i]->save(s);
    y["objects"] = o;
    return y;
  }
};


#endif


