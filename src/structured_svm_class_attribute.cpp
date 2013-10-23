#include "structured_svm_class_attribute.h"
#include "attribute.h"

#define DATASET_TOO_BIG_FOR_MEMORY

#ifndef DATASET_TOO_BIG_FOR_MEMORY
#define PRECOMPUTE_FEATURES
#endif
//#define DEBUG_MULTI_SAMPLE_UPDATE

ClassAttributeStructuredSVM::ClassAttributeStructuredSVM() : PartLocalizedStructuredSVM() {
  classConfusionCosts = NULL;
  attribute_feature_inds = NULL;
  class_attributes = NULL; 
  params.numCacheUpdatesPerIteration = 0;
  params.max_samples = 50000;
  params.maxCachedSamplesPerExample = 500000;
  params.numMultiSampleIterations = 20;

  params.method = SPO_DUAL_MULTI_SAMPLE_UPDATE_WITH_CACHE;
  //params.dontComputePsiGT = true;

#ifdef DATASET_TOO_BIG_FOR_MEMORY
  params.runMultiThreaded = -1;
#else
  runMultiThreaded = 0;
#endif

  //max_cache_sets_in_memory = 100000;//4*num_thr*2+100;//(int)(1.0e9 / (sizePsi*12));   // 1GB cache
  //max_cache_sets_in_memory = 4*num_thr*2+100;//(int)(1.0e9 / (sizePsi*12));   // 1GB cache
  params.updateFromCacheThread = true;
  params.canScaleW = true;
  trainJointly = true;
  isTesting = true;
}

ClassAttributeStructuredSVM::~ClassAttributeStructuredSVM() {
  if(classConfusionCosts) 
    free(classConfusionCosts);
  if(attribute_feature_inds)
    free(attribute_feature_inds);
  if(class_attributes)
    free(class_attributes);
}


double ClassAttributeStructuredSVM::Inference(StructuredData *x, StructuredLabel *ybar, SparseVector *w, 
					      StructuredLabel *y_partial, StructuredLabel *y_gt, double w_scale) {
  int bestclass=-1, first=1;
  double score=0, bestscore=-1;
  int num_attributes = classes->NumAttributes();
  isTesting = !y_gt;

  PartLocalizedStructuredData *m_x = (PartLocalizedStructuredData*)x;
  PartLocalizedStructuredLabel *m_ybar = (PartLocalizedStructuredLabel*)ybar;
  PartLocalizedStructuredLabel *m_y_gt = (PartLocalizedStructuredLabel*)y_gt;
  PartLocalizedStructuredLabel *m_y_partial = y_partial ? (PartLocalizedStructuredLabel*)y_partial : NULL;

  if(!m_x->cached_features) {
    //assert(y_gt);
    ComputeFeatureCache(m_x, (PartLocalizedStructuredLabel*)y_gt);
  }

  VFLOAT *attribute_scores = w->compute_region_scores(*m_x->cached_features, attribute_feature_inds, num_attributes);
  if(!trainJointly && !isTesting) {
    bestscore = 0;
    if(!m_ybar->attributes)
      m_ybar->attributes = new bool[num_attributes];
    if(!y_gt) {
      for(int i = 0; i < num_attributes; i++) {
	m_ybar->attributes[i] = attribute_scores[i] >= 0;
	bestscore += my_abs(attribute_scores[i]);
      }
    } else {
      int c = m_y_gt->obj_class;
      for(int i = 0; i < num_attributes; i++) {
	VFLOAT s = attribute_scores[i] - class_attributes[c][i];
	if(s < 0) { 
	  m_ybar->attributes[i] = false;
	  bestscore -= s;
	} else {
	  m_ybar->attributes[i] = true;
	  bestscore += s;
	}
      }
    }
  } else {
    for(int class_id = 0; class_id < classes->NumClasses(); class_id++) {
      // By default, compute ybar = max_y <w,Psi(x,y)>, but it y_partial is non-null,
      // only consider labels that agree with y_partial
      m_ybar->obj_class=class_id;
      if(y_partial && m_y_partial->obj_class != class_id)
	score = -INFINITY;
      else {
	score = y_gt ? Loss(y_gt, ybar) : 0;
	for(int k = 0; k < num_attributes; k++) 
	  score += attribute_scores[k]*class_attributes[class_id][k];
        score *= w_scale;
      }
      
      // Keep track of the highest coring class
      if(score > bestscore || first) {
	bestscore=score;
	bestclass=class_id;
	first=0;
      }
    }
    m_ybar->obj_class = bestclass;
  }

  delete [] attribute_scores;

  return bestscore;
}

void ClassAttributeStructuredSVM::CreateSamples(struct _SVM_cached_sample_set *set, StructuredData *x, StructuredLabel *y_gt) {
  if(!set->psi_gt) {
    set->psi_gt = Psi(x, y_gt).ptr();
    set->psi_gt_sqr = set->psi_gt->dot(*set->psi_gt);
  }

  if(!set->num_samples) {
    // Add a sample set that includes all classes 
    int cl_gt = ((PartLocalizedStructuredLabel*)y_gt)->obj_class;
    PartLocalizedStructuredData *m_x = (PartLocalizedStructuredData*)x;
    int num_classes = classes->NumClasses();
    
    for(int i = 0; i < num_classes; i++) {
      if(i == cl_gt) continue;
      PartLocalizedStructuredLabel *ybar = (PartLocalizedStructuredLabel*)NewStructuredLabel(x);
      ybar->obj_class = i;
      SVM_cached_sample *sample = SVM_cached_sample_set_add_sample(set, ybar);

      // Optionally set these things, so they don't have to be computed later
      sample->psi = NULL;//Psi(x, ybar).ptr();
      sample->loss = Loss(y_gt, ybar);
      sample->sqr = m_x->class_attribute_dot_products[cl_gt][cl_gt]+m_x->class_attribute_dot_products[i][i] -
                    2*m_x->class_attribute_dot_products[i][cl_gt];   // <set->psi_gt-sample->psi,set->psi_gt-sample->psi>
      sample->dot_psi_gt_psi = m_x->class_attribute_dot_products[cl_gt][i];  // <set->psi_gt,sample->psi>
#ifdef DEBUG_MULTI_SAMPLE_UPDATE
      sample->psi = Psi(x, ybar).ptr();
      SparseVector dpsi = *sample->psi - *set->psi_gt;
      double sqr = dpsi.dot(dpsi);
      double dot_psi_gt_psi = set->psi_gt->dot(*sample->psi);
      if(sqr) assert(sqr/sample->sqr > .999999999 && sqr/sample->sqr < 1.00000001);
      if(dot_psi_gt_psi) assert(dot_psi_gt_psi/sample->dot_psi_gt_psi > .999999999 && dot_psi_gt_psi/sample->dot_psi_gt_psi < 1.00000001);
#endif
    }
  }
}

double ClassAttributeStructuredSVM::BinaryAttributeUpdate(StructuredData *x, SparseVector *w, StructuredLabel *y_gt, struct _SVM_cached_sample_set *set, double w_scale) {
  int cl_gt = ((PartLocalizedStructuredLabel*)y_gt)->obj_class;
  int num_attributes = classes->NumAttributes();
  int num_parts = classes->NumParts();
  double score = 0;
  int i; 
  PartLocalizedStructuredData *m_x = (PartLocalizedStructuredData*)x;

  params.debugLevel = 3;

  if(!set->num_samples) {
    VFLOAT *sqrs = m_x->part_features->compute_region_scores(*m_x->part_features, m_x->partInds, num_parts);
    for(i = 0; i < num_attributes; i++) {
      SVM_cached_sample *sample = SVM_cached_sample_set_add_sample(set, NULL);
      int part = classes->GetAttribute(i)->Part()->Id();
      sample->psi = NULL;
      sample->loss = my_abs(class_binary_attributes[cl_gt][i]);
      sample->sqr = sqrs[part]*4;
      sample->dot_psi_gt_psi = -sample->sqr;  // <set->psi_gt,sample->psi>
    }
    delete [] sqrs;
  }

  if(params.runMultiThreaded) 
    Lock();

  // Equivalent to training liblinear-like binary attribute classifiers, where each attribute is trained independently (but we 
  // take an update step for each attribute at the same time)
  VFLOAT *attribute_scores = sum_w->compute_region_scores(*m_x->cached_features, attribute_feature_inds, num_attributes);
  VFLOAT *multipliers = new VFLOAT[num_attributes], dalpha;
  set->score_gt = 0;
  set->alpha = 0;
  set->loss = 0;
  VFLOAT sum_w_scale2 = sum_w_scale/num_attributes;
  for(int i = 0; i < num_attributes; i++) {
    SVM_cached_sample *s = &set->samples[i];
    if(s->psi) { delete s->psi; s->psi = NULL; }
    s->slack = my_abs(class_attributes[cl_gt][i])*sum_w_scale2 - 2*attribute_scores[i]*class_binary_attributes[cl_gt][i];
    if(s->slack > 0) {
      set->loss += my_abs(class_attributes[cl_gt][i]);
      score += my_abs(class_attributes[cl_gt][i])*sum_w_scale2 - attribute_scores[i]*class_binary_attributes[cl_gt][i];
    }
    dalpha = s->sqr ? s->slack / my_max(s->sqr,.0000000001) : 0;
    dalpha = my_min(1-s->alpha,my_max(-s->alpha,dalpha));
    s->alpha += dalpha;
    set->alpha += s->alpha;
    multipliers[i] = 2*dalpha*class_binary_attributes[cl_gt][i];
    set->score_gt += attribute_scores[i]*class_binary_attributes[cl_gt][i];
    if(i == 0)
      set->slack_after = (s->slack - dalpha*(s->sqr))/sum_w_scale2;
  }
  if(set->psi_gt) { delete set->psi_gt; set->psi_gt = NULL; }
  SparseVector dw = SparseVector(*m_x->cached_features);
  dw.multiply_regions(attribute_feature_inds, multipliers, classes->NumAttributes());
  *sum_w += dw;

  delete [] multipliers;
  delete [] attribute_scores;

  m_x->dontUpdate = true;
  
  if(params.runMultiThreaded) 
    Unlock();

  return score;
}

double ClassAttributeStructuredSVM::ImportanceSample(StructuredData *x, SparseVector *w, StructuredLabel *y_gt, struct _SVM_cached_sample_set *set, double w_scale) {
  int first=1;
  double score,bestscore=0;
  int cl = ((PartLocalizedStructuredLabel*)y_gt)->obj_class, i;
  int num_attributes = classes->NumAttributes();
  int num_classes = classes->NumClasses();
  PartLocalizedStructuredData *m_x = (PartLocalizedStructuredData*)x;
  if(!m_x->cached_features)
    ComputeFeatureCache(m_x, (PartLocalizedStructuredLabel*)y_gt);

  if(!trainJointly)
    return BinaryAttributeUpdate(x, w, y_gt, set, w_scale); 

  CreateSamples(set, x, y_gt);

  /*
  if(!set->psi_gt) {
    set->psi_gt = Psi(x, y_gt).ptr();
    set->psi_gt_sqr = set->psi_gt->dot(*set->psi_gt, useWeights);
  }
  */

  if(params.runMultiThreaded) {
    Lock();
    w = sum_w;
    w_scale = 1.0/(sum_w_scale);
  }

  VFLOAT *attribute_scores = w->compute_region_scores(*m_x->cached_features, attribute_feature_inds, num_attributes);
  
  set->score_gt = 0;
  for(int k = 0; k < num_attributes; k++) 
    set->score_gt += attribute_scores[k]*w_scale*class_attributes[cl][k];

  for(i = 0; i < set->num_samples; i++) {
    SVM_cached_sample *sample = &set->samples[i];
    int cl_i = ((PartLocalizedStructuredLabel*)sample->ybar)->obj_class;
    score = sample->loss;
    for(int k = 0; k < num_attributes; k++) 
      score += attribute_scores[k]*w_scale*class_attributes[cl_i][k];
    sample->slack = score-set->score_gt;
    sample->dot_w = (sample->slack - sample->loss)*sum_w_scale;
#ifdef DEBUG_MULTI_SAMPLE_UPDATE
    double dot_w_real = sum_w->dot(*sample->psi)-sum_w->dot(*set->psi_gt);
    if(dot_w_real) assert(dot_w_real/sample->dot_w > .999999999 && dot_w_real/sample->dot_w < 1.00000001);
#endif
    if(score > bestscore || first) {
      bestscore=score;
      first=0;
      cl = i;
    }
  }
  set->dot_sum_w_psi_gt = set->score_gt*sum_w_scale;
  qsort(set->samples, set->num_samples, sizeof(SVM_cached_sample), SVM_cached_sample_cmp);

  if(params.runMultiThreaded) {
    m_x->dontUpdate = false;
    MultiSampleUpdate(set, trainset->examples[set->i], params.numMultiSampleIterations);
    m_x->dontUpdate = true;
    Unlock();
  }
  if(set->psi_gt) { delete set->psi_gt; set->psi_gt = NULL; }

  delete [] attribute_scores;

  return bestscore;
}


// Iteratively optimize alpha parameters for a set of samples 's'.  This is a faster version of 
// StructuredSVM::MultiSampleUpdate for an class-attribute SVM, that uses the fact that the dot product 
// of psi(x,c1) and psi(x,c2) just phi^2(x) times the dot product of the class-attribute labels for c1 and c2,
// to avoid having to evaluate any dot products.  Thus if dot_w=dot(w,psi(x,c1)) has been precomputed in 
// ImportanceSample(), the time to evaluate this multi sample update is just the time for one vector addition
// plus some other terms that are independent of the feature space dimensionality
void ClassAttributeStructuredSVM::MultiSampleUpdate(SVM_cached_sample_set *set, StructuredExample *ex, int R) {
  VFLOAT dot_w_u = 0;   // dot_w_u = <w,u>
  VFLOAT L_i = 0;       // L_i = <w,-u_i/(lambda*t)> + \sum_ybar alpha_{i,ybar} loss(y_i,ybar)
  int i, j, r;
  PartLocalizedStructuredData *x = (PartLocalizedStructuredData*)ex->x;
  if((!set->num_samples && !set->alpha) || x->dontUpdate) return;

  bool changed = false;
  int cl_gt = ((PartLocalizedStructuredLabel*)ex->y)->obj_class;
  int num_attributes = classes->NumAttributes();
  int num_classes = classes->NumClasses();
  VFLOAT *attr_weights = new VFLOAT[num_attributes+num_classes+2*set->num_samples];
  VFLOAT *dot_u_v_no_gt = attr_weights + num_attributes;
  VFLOAT *dot_u_v_orig = dot_u_v_no_gt + num_classes;
  VFLOAT *old_alphas = dot_u_v_orig + set->num_samples;
  for(i = 0; i < num_attributes; i++) 
    attr_weights[i] = 0;
  for(i = 0; i < num_classes; i++) 
    dot_u_v_no_gt[i] = 0;

  for(i = 0; i < set->num_samples; i++) {
    int cl_i = ((PartLocalizedStructuredLabel*)set->samples[i].ybar)->obj_class;
    old_alphas[i] = set->samples[i].alpha;
    if(set->samples[i].alpha) {
      dot_w_u += set->samples[i].alpha*set->samples[i].dot_w/sum_w_scale;
      for(j = 0; j < num_classes; j++) 
        dot_u_v_no_gt[j] += set->samples[i].alpha*x->class_attribute_dot_products[cl_i][j];
    }
  }
  for(i = 0; i < set->num_samples; i++) {
    int cl_i = ((PartLocalizedStructuredLabel*)set->samples[i].ybar)->obj_class;
    dot_u_v_orig[i] = set->alpha*(x->class_attribute_dot_products[cl_gt][cl_gt]-x->class_attribute_dot_products[cl_gt][cl_i]) +
      dot_u_v_no_gt[cl_i]-dot_u_v_no_gt[cl_gt];
  }
  L_i = set->D_i+dot_w_u;

  VFLOAT D_i = set->D_i;
  VFLOAT sum_alpha = set->alpha;
  VFLOAT u_i_sqr = set->u_i_sqr;
  VFLOAT dot_w_u_before = dot_w_u;

#ifdef DEBUG_MULTI_SAMPLE_UPDATE
  if(!set->u_i) set->u_i = new SparseVector;
  SparseVector sum_w_no_u = *sum_w + *set->u_i - *set->psi_gt*set->alpha;
#endif

  for(r = 0; r < R; r++) {
    for(i = 0; i < set->num_samples; i++) {
      SVM_cached_sample *s = &set->samples[i];
      int cl_i = ((PartLocalizedStructuredLabel*)s->ybar)->obj_class;
      VFLOAT dot_u_v = sum_alpha*(x->class_attribute_dot_products[cl_gt][cl_gt]-x->class_attribute_dot_products[cl_gt][cl_i]) +
                          dot_u_v_no_gt[cl_i]-dot_u_v_no_gt[cl_gt]; // <u_i,v>
      VFLOAT dot = s->dot_w - dot_u_v + dot_u_v_orig[i]; // <sum_w,v>

#ifdef DEBUG_MULTI_SAMPLE_UPDATE
      if(set->u_i) {
	SparseVector w_sum_new = sum_w_no_u + set->psi_gt->mult_scalar(sum_alpha, useWeights) - *set->u_i;
	SparseVector u_new = *set->u_i  - (*set->psi_gt*sum_alpha);
	double dot_u_v_real = u_new.dot(*s->psi-*set->psi_gt);
	double dot_real = w_sum_new.dot(*s->psi-*set->psi_gt);
	//fprintf(stderr, "t=%d, i=%d, j=%d, dot_u_v=%lg:%lg, dot=%lg:%lg\n", (int)t, set->i, i, dot_u_v_real, dot_u_v, dot_real, dot);
	if(dot_u_v) assert(dot_u_v_real/dot_u_v > .999999999 && dot_u_v_real/dot_u_v < 1.00000001);
	if(dot) assert(dot_real/dot > .999999999 && dot_real/dot < 1.00000001);
      }
#endif

      VFLOAT scale=1, new_sum_alpha;
      VFLOAT dalpha = (dot + s->loss*(sum_w_scale)) / my_max(s->sqr,.0000000001);
      if(sum_alpha < 1 || dalpha < 0) {
	// alpha expand: solve for the optimal amount 'dalpha' to increase s->alpha
	// and then scale all set->samples[:].alpha (the value of 'dalpha' and 'scale' 
	// that maximizes the increase in the dual objective). 
	dalpha = my_min(1-sum_alpha, my_max(-s->alpha,dalpha));
	new_sum_alpha = sum_alpha*scale + dalpha;
      } else {
	// alpha swap: solve for the optimal amount 'dalpha' to increase s->alpha
	// while scaling down all set->samples[:].alpha, such that we preserve sum_k{s->samples[:].alpha}=1
	// (choose the value of 'dalpha' that maximizes the increase in the dual objective)
	VFLOAT e = dot/(sum_w_scale) + s->loss;
	VFLOAT sqr = u_i_sqr + 2*dot_u_v + s->sqr;
	dalpha = (e-L_i)*(sum_w_scale) / my_max(sqr,.00000000001);
	dalpha = my_min(1-s->alpha, my_max(-s->alpha,dalpha));
	scale = 1-dalpha;
	new_sum_alpha = 1;
      }
      assert(scale >= 0 && new_sum_alpha >= -0.000000001 && new_sum_alpha <= 1.000000001);
      new_sum_alpha = my_min(1,my_max(new_sum_alpha,0));

      if(dalpha != 0 || scale != 1) {
	changed = true;
	for(j = 0; j < num_classes; j++) 
	  dot_u_v_no_gt[j] = dot_u_v_no_gt[j]*scale + dalpha*x->class_attribute_dot_products[cl_i][j];
	for(j = 0; j < set->num_samples; j++) 
	  set->samples[j].alpha *= scale;
	s->alpha += dalpha;

	// Keep track of L_i, D_i, u_i_sqr, dot_w_u, dot_u_psi_gt using inexpensive online updates
	sum_alpha = new_sum_alpha;
        dot_w_u = scale*dot_w_u + (dalpha*dot - scale*(scale-1)*u_i_sqr - (2*scale*dalpha-dalpha)*dot_u_v - s->sqr*SQR(dalpha)) / (sum_w_scale);
	u_i_sqr = SQR(scale)*u_i_sqr + 2*scale*dalpha*dot_u_v + s->sqr*SQR(dalpha);
	set->dot_u_psi_gt = scale*set->dot_u_psi_gt + dalpha*(s->dot_psi_gt_psi - set->psi_gt_sqr);
	D_i = scale*D_i + dalpha*s->loss;
	L_i = dot_w_u + D_i;
	assert(!isnan(L_i));

#ifdef DEBUG_MULTI_SAMPLE_UPDATE
	*set->u_i = (*set->u_i * scale) + (*s->psi*dalpha);
	SparseVector w_sum_new = sum_w_no_u + set->psi_gt->mult_scalar(sum_alpha, useWeights) - *set->u_i;
	SparseVector u_new = *set->u_i  - (*set->psi_gt*sum_alpha);
	double dot_w_u_real = w_sum_new.dot(u_new)/(sum_w_scale);
	double dot_u_psi_gt_real = u_new.dot(*set->psi_gt, useWeights);
	double u_i_sqr_real = u_new.dot(u_new, useWeights);
	/*fprintf(stderr, "t=%d, i=%d, j=%d, scale=%f, dalpha=%f, dot_w_u=%lg:%lg, dot_u_psi_gt=%lg:%lg, u_i_sqr=%lg:%lg\n", 
		(int)t, set->i, i, (float)scale, (float)dalpha, 
		dot_w_u_real, dot_w_u,  dot_u_psi_gt_real, set->dot_u_psi_gt,  u_i_sqr_real, u_i_sqr);*/
	assert(dot_w_u_real/dot_w_u > .999999999 && dot_w_u_real/dot_w_u < 1.00000001);
	assert(dot_u_psi_gt_real/set->dot_u_psi_gt > .999999999 && dot_u_psi_gt_real/set->dot_u_psi_gt < 1.00000001);
	assert(u_i_sqr_real/u_i_sqr > .999999999 && u_i_sqr_real/u_i_sqr < 1.00000001);
#endif
      }
    }
  }

  double d_sum_w_sqr = 2*(dot_w_u_before-dot_w_u)*sum_w_scale + set->u_i_sqr - u_i_sqr;
  if(changed) {
    for(i = 0; i < set->num_samples; i++) {
      int cl_i = ((PartLocalizedStructuredLabel*)set->samples[i].ybar)->obj_class;
      double dalpha = set->samples[i].alpha - old_alphas[i];
      for(j = 0; j < num_attributes; j++) 
	attr_weights[j] += dalpha*(class_attributes[cl_gt][j]-class_attributes[cl_i][j]);
    }
    sum_w->add_in_weighted_regions(*x->cached_features, attribute_feature_inds, attr_weights, classes->NumAttributes());
  }
  sum_w_sqr += d_sum_w_sqr;

#ifdef DEBUG_MULTI_SAMPLE_UPDATE
  double sum_w_sqr_real = sum_w->dot(*sum_w, useWeights);
  //fprintf(stderr, "t=%d, i=%d, sum_w_sqr=%lg:%lg\n", (int)t, set->i, sum_w_sqr_real, sum_w_sqr);
  assert(sum_w_sqr_real/sum_w_sqr > .99999 && sum_w_sqr_real/sum_w_sqr < 1.0001);
#endif

  set->u_i_sqr = u_i_sqr;
  set->alpha = sum_alpha;
  sum_alpha_loss += D_i - set->D_i;
  sum_dual = -sum_w_sqr/(2*sum_w_scale) + sum_alpha_loss;
  regularization_error = (sum_w_sqr/SQR(sum_w_scale))*params.lambda/2;
  set->D_i = D_i;

  SVM_cached_sample *s = &set->samples[0];
  int cl_i = ((PartLocalizedStructuredLabel*)s->ybar)->obj_class;
  VFLOAT dot_u_v = sum_alpha*(x->class_attribute_dot_products[cl_gt][cl_gt]-x->class_attribute_dot_products[cl_gt][cl_i]) +
                   dot_u_v_no_gt[cl_i]-dot_u_v_no_gt[cl_gt]; // <u_i,v>
  VFLOAT dot = s->dot_w - dot_u_v + dot_u_v_orig[0]; // <sum_w,v>
  set->slack_after = dot/sum_w_scale + s->loss;

  delete [] attr_weights;

  //fprintf(stderr, "sum dual is %lg, dual_change=%lg, D=%lg\n", sum_dual, -d_sum_w_sqr/(2*sum_w_scale) + set->D_i - D_i_orig, set->D_i);
}



SparseVector ClassAttributeStructuredSVM::Psi(StructuredData *x, StructuredLabel *y) {
  PartLocalizedStructuredData *m_x = (PartLocalizedStructuredData*)x;
  PartLocalizedStructuredLabel *m_y = (PartLocalizedStructuredLabel*)y;

  if(!m_x->cached_features) 
    ComputeFeatureCache(m_x, m_y);

  // Let class_attributes[c] be an array of size numAttributes, with each entry 1 or -1 
  // indicating attribute memberships.  We compute features by taking the part localized
  // extracted features m_x->cached_features and then multiplying them according to attribute memberships
  SparseVector retval(*m_x->cached_features);
  if(trainJointly || isTesting)
    retval.multiply_regions(attribute_feature_inds, class_attributes[m_y->obj_class], classes->NumAttributes());
  else
    retval.multiply_regions(attribute_feature_inds, class_binary_attributes[m_y->obj_class], classes->NumAttributes());

  return retval;
}

double ClassAttributeStructuredSVM::Loss(StructuredLabel *y_gt, StructuredLabel *y_pred) {
  // Computes the loss of prediction y_pred against the correct label y_gt. 
  PartLocalizedStructuredLabel *m_y_gt = (PartLocalizedStructuredLabel*)y_gt;
  PartLocalizedStructuredLabel *m_y_pred = (PartLocalizedStructuredLabel*)y_pred;

  if(trainJointly || isTesting) {
    if(classConfusionCosts)
      return classConfusionCosts[m_y_gt->obj_class][m_y_pred->obj_class];
    else
      return m_y_gt->obj_class == m_y_pred->obj_class ? 0 : 1;
  } else {
    int c = m_y_gt->obj_class;
    VFLOAT loss = 0;
    for(int i = 0; i < classes->NumAttributes(); i++) {
      if(m_y_pred->attributes[i] && class_attributes[c][i] < 0) 
        loss -= class_attributes[c][i];
      else if(!m_y_pred->attributes[i] && class_attributes[c][i] > 0) 
        loss += class_attributes[c][i];
    }
    return loss;
  }
}


void ClassAttributeStructuredSVM::OnFinishedIteration(StructuredExample *ex) {
  // Since memory can be a scarce resource, free all memory for sliding window features and
  // detection associated with this example after a training iteration finishes.  We still keep 
  // the part localized features cached, for much faster computation of Psi() and Inference() 
  // on the next iteration.  Therefore, we require that main memory is large enough to hold 
  // sizePsiXnumTrainingExamplesX8 bytes.  If it is not, consider free'ing x->cached_features
  // (or never storing it at all)
  PartLocalizedStructuredData *xx = (PartLocalizedStructuredData*)ex->x;
  if(xx->process) {
    delete xx->process; 
    xx->process = NULL;
  }
  if(xx->part_features && xx->cached_features) {
    delete xx->cached_features;
    xx->cached_features = NULL;
  }
  if(ex->set && ex->set->psi_gt) {
    delete ex->set->psi_gt;
    ex->set->psi_gt = NULL;
  }

#ifdef DATASET_TOO_BIG_FOR_MEMORY
  if(xx->part_features) {
    //if(debugLevel > 1) fprintf(stderr, "Freeing features %s %d...\n", xx->imagename, isTesting);
    delete xx->part_features;
    xx->part_features = NULL;
  }
#endif
}

void ClassAttributeStructuredSVM::ComputeCaches() {
  int i, j;
  int num_attributes = classes->NumAttributes(), num_classes = classes->NumClasses();

  // class_attributes is a numClassesXnumAttributes array of values between 1 and -1 
  if(class_attributes)
    free(class_attributes);
  class_attributes = (VFLOAT**)malloc(2*num_classes*(2*sizeof(VFLOAT*)+sizeof(VFLOAT)*(num_attributes+num_classes)));
  class_binary_attributes = (VFLOAT**)(class_attributes+classes->NumClasses());
  VFLOAT *ptr = (VFLOAT*)(class_binary_attributes+classes->NumClasses());
  for(i = 0; i < num_classes; i++, ptr += 2*num_attributes) {
    class_attributes[i] = ptr;
    class_binary_attributes[i] = ptr+num_attributes;
    ObjectClass *c = classes->GetClass(i);
    int cl;
    for(j = 0; j < classes->NumAttributes(); j++) {
      cl = c->Id();
      assert(cl >= 0);
      class_attributes[i][j] = classes->GetClass(cl)->GetAttributeWeight(j);
      class_binary_attributes[i][j] = class_attributes[i][j] < 0 ? -1 : (class_attributes[i][j] > 0 ? 1 : 0);
    }
  }

  if(!trainJointly) {
    for(j = 0; j < classes->NumAttributes(); j++) {
      VFLOAT sum = 0;
      for(i = 0; i < num_classes; i++) 
	sum += class_attributes[i][j];
      sum /= num_classes;
      for(i = 0; i < num_classes; i++) { 
	class_attributes[i][j] -= sum;
	class_binary_attributes[i][j] = class_attributes[i][j] < 0 ? -1 : (class_attributes[i][j] > 0 ? 1 : 0);
      }
    }
  }


  // attribute_feature_inds is a numAttributesArray storing the cumulative sum of the number of features for each attribute
  if(attribute_feature_inds)
    free(attribute_feature_inds);
  attribute_feature_inds = (int*)malloc(sizeof(int)*classes->NumAttributes());
  int n = 0;
  for(j = 0; j < num_attributes; j++) {
    n += classes->GetAttribute(j)->NumWeights();
    attribute_feature_inds[j] = n;
  }
  sizePsi = classes->NumWeights(false, true);
}

void ClassAttributeStructuredSVM::ComputeFeatureCache(PartLocalizedStructuredData *m_x, PartLocalizedStructuredLabel *m_y) {
  bool firstTime = false;

  if(classes->AllAttributesHaveSaveFeatures()) {
    if(!m_x->part_features) {
      // Better memory compression if all attributes for a given part share the same feature space
      // Just extract a vector of features for each part
      m_x->partInds = new int[classes->NumParts()+1];
      m_x->vectorInds = new int[classes->NumParts()+1];
      PartLocation *locs = m_x->gt_locs ? m_x->gt_locs : m_y->GetPartLocations();
      float *tmp_features = (float*)malloc((sizePsi+2)*sizeof(VFLOAT));
      ImageProcess *process = m_x->GetProcess(classes);
      FeatureWindow *featureWindows = classes->GetAttribute(0)->Features();
      int numWindows = classes->GetAttribute(0)->NumFeatureTypes();
      int n = process->GetLocalizedFeatures(tmp_features, locs, featureWindows, numWindows, m_x->partInds);
      if(params.debugLevel > 1) fprintf(stderr, "Precomputing features (same %d) %s...\n", n, m_x->imagename);
      m_x->partInds[classes->NumParts()] = n;
      m_x->part_features = new SparseVector(tmp_features, n);
      m_x->part_features->NonSparseIndsToSparseInds(m_x->partInds, m_x->vectorInds, classes->NumParts()+1);
      free(tmp_features);
      process->Clear();
      firstTime = true;
    }

    // Now for each attribute, copy the corresponding part feature vector, and expand them
    // into a vector of attribute features (it will be free'd in OnFinishIteration())
    m_x->cached_features = new SparseVector();
    for(int i = 0; i < classes->NumAttributes(); i++) {
      if(classes->GetAttribute(i)->NumFeatureTypes()) {
	int part = classes->GetAttribute(i)->Part()->Id();
	m_x->cached_features->AppendSparseRegion(*m_x->part_features, m_x->vectorInds[part], 
		 m_x->vectorInds[part+1]-m_x->vectorInds[part], m_x->partInds[part], 
		 m_x->partInds[part+1]-m_x->partInds[part]);
      }
    }
    
  } else {
    // Otherwise, feature space is the concatenation of all attribute feature spaces
    if(params.debugLevel > 1) fprintf(stderr, "Precomputing features (different) %s...\n", m_x->imagename);
    ImageProcess *process = m_x->GetProcess(classes);
    VFLOAT *tmp_features = (VFLOAT*)malloc((sizePsi+2)*sizeof(VFLOAT));
    PartLocation *locs = m_x->gt_locs ? m_x->gt_locs : m_y->GetPartLocations();
    assert(locs);
    int n = process->GetFeatures(tmp_features, locs, NULL, false, true);
    assert(n == sizePsi);
    m_x->cached_features = new SparseVector(tmp_features, sizePsi);
    free(tmp_features);
    process->Clear();
    firstTime = true;
  }

  if(firstTime) {
    m_x->sqr = m_x->cached_features->dot(*m_x->cached_features);

    int num_attributes = classes->NumAttributes(), num_classes = classes->NumClasses();
    VFLOAT *sqrs = m_x->cached_features->compute_region_scores(*m_x->cached_features, attribute_feature_inds, num_attributes); 
    m_x->class_attribute_dot_products = (VFLOAT**)malloc(num_classes*(sizeof(VFLOAT*)+sizeof(VFLOAT)*num_classes));
    VFLOAT *ptr = (VFLOAT*)(m_x->class_attribute_dot_products+num_classes);
    for(int i = 0; i < num_classes; i++, ptr += num_classes) {
      m_x->class_attribute_dot_products[i] = ptr;
      for(int j = 0; j < num_classes; j++) {
	m_x->class_attribute_dot_products[i][j] = 0;
	for(int k = 0; k < num_attributes; k++) 
	  m_x->class_attribute_dot_products[i][j] += class_attributes[i][k]*class_attributes[j][k]*sqrs[k];
      }
    }
    delete [] sqrs;
  }

}

// Set confusion cost between to classes to be proportional to the Euclidean distance between their class-attribute vectors
void ClassAttributeStructuredSVM::SetClassConfusionCostsByAttribute() {
  int num_attributes = classes->NumAttributes(), num_classes = classes->NumClasses();
  classConfusionCosts = (VFLOAT**)malloc(num_classes*(sizeof(VFLOAT*)+sizeof(VFLOAT)*num_classes));
  VFLOAT *ptr = (VFLOAT*)(classConfusionCosts+num_classes);
  int i, j;
  VFLOAT maxDist = 0;
  for(i = 0; i < num_classes; i++, ptr += num_classes) {
    classConfusionCosts[i] = ptr;
    for(j = 0; j < num_classes; j++) {
      classConfusionCosts[i][j] = 0;
      for(int k = 0; k < num_attributes; k++) 
	classConfusionCosts[i][j] += SQR(class_attributes[i][k]-class_attributes[j][k]);
      classConfusionCosts[i][j] = sqrt(classConfusionCosts[i][j]);
      if(classConfusionCosts[i][j] > maxDist)
	maxDist = classConfusionCosts[i][j];
    }
  }

  // Normalize to be between 0 and 1
  for(i = 0; i < num_classes; i++) 
    for(j = 0; j < num_classes; j++) 
      classConfusionCosts[i][j] /= maxDist;
}

StructuredDataset *ClassAttributeStructuredSVM::LoadDataset(const char *fname, bool getLock) {
  StructuredDataset *d = StructuredSVM::LoadDataset(fname, getLock);
  if(!d) return NULL;

  for(int i = 0; i < d->num_examples; i++) {
    PartLocalizedStructuredData *m_x = (PartLocalizedStructuredData*)d->examples[i]->x;
    PartLocalizedStructuredLabel *m_y = (PartLocalizedStructuredLabel*)d->examples[i]->y;
    m_x->gt_locs = m_y->GetPartLocations();
  }

  bool precompute = false;
#ifdef PRECOMPUTE_FEATURES
  precompute = true;
#endif

  if(precompute) {
    // Extract precomputed features at the ground truth part locations
    if(getLock) Lock();

#ifdef USE_OPENMP
#pragma omp parallel
#endif
    for(int i = 0; i < d->num_examples; i++) {
      PartLocalizedStructuredData *m_x = (PartLocalizedStructuredData*)d->examples[i]->x;
      PartLocalizedStructuredLabel *m_y = (PartLocalizedStructuredLabel*)d->examples[i]->y;
      ComputeFeatureCache(m_x, m_y);
      OnFinishedIteration(d->examples[i]);
    }

    if(getLock) Unlock();
  }

  return d;
}



Json::Value ClassAttributeStructuredSVM::Save() {
  Json::Value root;
  
  root["classes"] = classes->GetFileName();

  SparseVector *w = GetCurrentWeights(false);
  float *ww = w->get_non_sparse<float>(GetSizePsi());
  classes->SetWeights(ww, false, true); 
  classes->Save();
  free(ww);

  if(classConfusionCosts) {
    Json::Value c;
    int n = 0;
    int num_classes = classes->NumClasses();
    for(int i = 0; i < num_classes; i++) {
      for(int j = 0; j < num_classes; j++) {
	if(classConfusionCosts[i][j]) {
	  Json::Value o;
	  o["c_gt"] = i;
	  o["c_pred"] = j;
	  o["loss"] = classConfusionCosts[i][j];
	  c[n++] = o;
	}
      }
    }
    root["Class Confusion Costs"] = c;
  }

  return root;
}

bool ClassAttributeStructuredSVM::Load(const Json::Value &root) {
  if(!PartLocalizedStructuredSVM::Load(root)) 
    return false;

  if(root.isMember("Class Confusion Costs") && root["Class Confusion Costs"].isArray()) {
    int num_classes = classes->NumClasses();
    classConfusionCosts = (double**)malloc((num_classes+1)*(sizeof(double*)+(num_classes+1)*sizeof(double)));
    double *ptr = (double*)(classConfusionCosts+(num_classes+1));
    for(int i = 0; i <= num_classes; i++, ptr += (num_classes+1)) {
      classConfusionCosts[i] = ptr;
      for(int j = 0; j <= num_classes; j++)
	classConfusionCosts[i][j] = 0;
    }
    const Json::Value a = root["Class Confusion Costs"];
    for(int i = 0; i < (int)a.size(); i++) {
      int c_gt = a[i].get("c_gt",-1).asInt();
      int c_pred = a[i].get("c_pred",-1).asInt();
      double l = a[i].get("loss",0).asDouble();
      if(c_gt >= num_classes || c_pred >= num_classes || c_gt < 0 || c_pred < 0) {
	fprintf(stderr, "Error reading Class Confusion Costs\n");
	return false;
      }
      classConfusionCosts[c_gt][c_pred] = l;
    }
  }

  ComputeCaches();

  return true;
}

