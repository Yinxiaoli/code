#ifndef __STRUCTURED_SVM_CLASS_ATTRIBUTE_H
#define __STRUCTURED_SVM_CLASS_ATTRIBUTE_H

#include "structured_svm_partmodel.h"
#include "class.h"

class ClassAttributeStructuredSVM;
class ClassAttributeStructuredData;
class ClassAttributeStructuredLabel;

/**
 * @file structured_svm_class_attribute.h
 * @brief  Implements routines for joint learning of attribute detectors, while optimizing loss-sensitive multiclass classification accuracy.
 */

/**
 * @class ClassAttributeStructuredSVM
 * @brief Implements routines for joint learning of attribute detectors, while optimizing loss-sensitive multiclass classification accuracy.
 * Attributes are localized by ground truth part locations.  Classes are assumed to share a common vocabulary of attributes, with
 * expert defined class-attribute memberships
 */
class ClassAttributeStructuredSVM : public PartLocalizedStructuredSVM {
 public:
  /******************* Functions overridden from StructuredSVM *************************/
  bool Load(const Json::Value &root);
  Json::Value Save();
  SparseVector Psi(StructuredData *x, StructuredLabel *y);
  double Inference(StructuredData *x, StructuredLabel *ybar, SparseVector *w, StructuredLabel *y_partial=NULL, StructuredLabel *y_gt=NULL, double w_scale=1);
  double Loss(StructuredLabel *y_gt, StructuredLabel *y_pred);

  void OnFinishedIteration(StructuredExample *ex);
  StructuredDataset *LoadDataset(const char *fname, bool getLock=true);

  double ImportanceSample(StructuredData *x, SparseVector *w, StructuredLabel *y_gt, struct _SVM_cached_sample_set *set, double w_scale=1);
  void MultiSampleUpdate(SVM_cached_sample_set *set, StructuredExample *ex, int R);


  /******************* Customized functions and member variables *************************/
 public:
  ClassAttributeStructuredSVM();
  ~ClassAttributeStructuredSVM();

  /**
   * @brief Set the Classes object defining the part model
   */
  void SetClasses(Classes *c) { classes = c; ComputeCaches(); }

  /**
   * @brief A numClassesXnumClasses array of losses: costs[y_gt][y_pred]
   */
  void SetClassConfusionCosts(double **costs) { classConfusionCosts = costs; }
  void SetClassConfusionCostsByAttribute();

  void SetTrainJointly(bool b) { trainJointly = b; }

 private:
  double **classConfusionCosts;
  double maxLoss;
  bool trainJointly;
  bool isTesting;

  int *attribute_feature_inds;
  VFLOAT **class_attributes;
  VFLOAT **class_binary_attributes;
  VFLOAT **class_attribute_dot_products;

  void ComputeCaches();
  void CreateSamples(struct _SVM_cached_sample_set *set, StructuredData *x, StructuredLabel *y_gt);
  void ComputeFeatureCache(PartLocalizedStructuredData *m_x, PartLocalizedStructuredLabel *m_y);
  double BinaryAttributeUpdate(StructuredData *x, SparseVector *w, StructuredLabel *y_gt, struct _SVM_cached_sample_set *set, double w_scale);
};



#endif


