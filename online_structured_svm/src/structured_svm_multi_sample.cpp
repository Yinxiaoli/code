#include "structured_svm.h"


double StructuredSVM::ImportanceSample(StructuredData *x, SparseVector *w, StructuredLabel *y_gt, struct _SVM_cached_sample_set *set, double w_scale) {
  //char str[1000]; OptimizationMethodToString(method, str);
  //fprintf(stderr, "ERROR: method %s is not supported, because ImportanceSample() is not implemented\n", str);
  //exit(0);
    
  // Usually, the user of this API should define a custom importance sampling routine, which adds a set of predicted labels with non-zero slack.  
  // If unimplemented, we simply add a single most violated label in each iteration, which is appended to the set of most violated labels
  // from previous iterations.
  StructuredLabel *ybar = NewStructuredLabel(x);
  double retval = Inference(x, ybar, w, NULL, y_gt, w_scale);
  SVM_cached_sample_set_add_sample(set, ybar);
  SVM_cached_sample s = set->samples[0];
  set->samples[0] = set->samples[set->num_samples-1];
  set->samples[set->num_samples-1] = s;

  return retval;
}


void StructuredSVM::SVM_cached_sample_set_compute_features(SVM_cached_sample_set *set, StructuredExample *ex) {
  for(int j = 0; j < set->num_samples; j++) {
    if(!set->samples[j].psi && !set->samples[j].sqr && set->samples[j].ybar) {
      if(!set->psi_gt) {
	set->psi_gt = Psi(trainset->examples[set->i]->x, trainset->examples[set->i]->y).ptr();
	set->psi_gt_sqr = set->psi_gt->dot(*set->psi_gt);
      }

      set->samples[j].psi = Psi(ex->x, set->samples[j].ybar).ptr();
      set->samples[j].loss = Loss(ex->y, set->samples[j].ybar);
      set->samples[j].dot_psi_gt_psi = set->psi_gt->dot(*set->samples[j].psi);
      set->samples[j].sqr = set->psi_gt_sqr - 2*set->samples[j].dot_psi_gt_psi + set->samples[j].psi->dot(*set->samples[j].psi);
      set->samples[j].dot_psi_gt_psi = set->psi_gt->dot(*set->samples[j].psi);
      set->samples[j].dot_w = sum_w ? sum_w->dot(*set->samples[j].psi) - set->score_gt*sum_w_scale : 0;
      assert(!isnan(set->samples[j].sqr));
    }
  }
}



// Iteratively optimize alpha parameters for a set of samples 's'.  Implements 'Multi-Sample Dual Update Step' of writeup
void StructuredSVM::MultiSampleUpdate(SVM_cached_sample_set *set, StructuredExample *ex, int R) {
  if(params.allSamplesOrthogonal) {
    MultiSampleUpdateOrthogonalSamples(set, ex, R);
    return;
  }

  VFLOAT dot_w_u = 0;   // dot_w_u = <w,u>
  VFLOAT L_i = 0;       // L_i = <w,-u_i/(lambda*t)> + \sum_ybar alpha_{i,ybar} loss(y_i,ybar)
  VFLOAT s_u = 1;
  int j, r;

  //if(!set->num_samples && !set->alpha) return;

  if(!set->u_i) {
    if(!u_i_buff) {
      u_i_buff = (VFLOAT*)malloc(sizeof(VFLOAT)*sizePsi);
      for(int i = 0; i < sizePsi; i++) u_i_buff[i] = 0;
    }
    set->u_i = new SparseVector;
  } else {
    dot_w_u = (sum_w->dot(*set->u_i)-sum_w->dot(*set->psi_gt)*set->alpha)/(sum_w_scale);
    L_i = set->D_i+dot_w_u;
  }
  double dot_w_u_orig = dot_w_u, u_i_sqr_orig = set->u_i_sqr;
  
  // For computational reasons, we want each iteration of the main loop over set->samples to run
  // in time proportional to the number of non-zero entries in v=set->samples[j]->psi as
  // opposed to sizePsi, since v will often be very sparse
  // (e.g., for multiclass classification and mixture models it is very sparse).  Our vector
  // algebra operations have the following properties:
  //   1) Scalar multiplication takes time proportional to the number of non-zero entries
  //   2) Addition, subtraction, and dot products of 2 sparse vectors takes time proportional
  //      to the sum of the number of non-zero entries in both vectors
  //   3) Addition, subtraction, and dot products of a sparse vector and a non-sparse vector
  //      are faster, taking time proportional to the number of non-zero entries in the sparse vector
  // Since sum_w and u_i will not be sparse, we want to avoid doing any scalar multiplies (1) of sum_w
  // and u_i, and we want to avoid any operations between sum_w and u_i (2).  We therefore decompose 
  // w_sum and u_i as follows:
  //    u_i = s_u*u_i_scaled
  //    sum_w = sum_w_without_u_i - u_i
  //    u_i_sqr = <u_i,u_i>
  // We can maintain computation of u_i_scaled, s_u, sum_w_minus_u_i, u_i_sqr, L_i, and dot_w_u
  // using only operation (3) in each iteration.  In each iteration, it requires computing only one 
  // sparse dot product:
  //   <u_i_scaled,v>
  // and one sparse vector addition and scalar multiply:
  //   u_i_scaled = u_i_scaled + v*(step_size/s_u)
  // All other operations are arithmetic operations between scalars.
  SparseVector *sum_w_without_u_i = sum_w;
  SparseVector *u_i_scaled = set->u_i;
  if(set->alpha || params.runMultiThreaded) {
    if(set->alpha) {
      *sum_w_without_u_i += *set->u_i;
      *sum_w_without_u_i -= set->psi_gt->mult_scalar(set->alpha);
    }
    double score_gt_without_u = sum_w_without_u_i->dot(*set->psi_gt);
    for(j = 0; j < set->num_samples; j++) 
      set->samples[j].dot_w = sum_w_without_u_i->dot(*set->samples[j].psi)-score_gt_without_u;
  }
  VFLOAT D_i_orig = set->D_i;
  VFLOAT dot_u_w_without_u = dot_w_u + set->u_i_sqr/(sum_w_scale);

  if(set->alpha && set->u_i_sqr) {
    // Can compute the optimal scale on u_i inexpensively
    s_u = 1 + (L_i*sum_w_scale) / set->u_i_sqr;
    s_u = my_min(1.0/set->alpha, my_max(0, s_u));
    
    set->alpha *= s_u;
    dot_u_w_without_u *= s_u;
    dot_w_u = s_u*dot_w_u - (s_u*(s_u-1)*set->u_i_sqr) / (sum_w_scale);
    set->u_i_sqr *= SQR(s_u);
    set->dot_u_psi_gt *= s_u;
    set->D_i *= s_u;
    L_i = dot_w_u + set->D_i;

    set->drift_bits += s_u <= 0 ? MAX_DRIFT_BITS : my_abs(LOG2(s_u));
    if(set->drift_bits > MAX_DRIFT_BITS) {
      // To avoid numerical precision issues, recompute set->u_i_sqr, set->dot_u_psi_gt
      set->drift_bits = 0;
      *set->u_i *= s_u;
      double d_u_gt = set->u_i->dot(*set->psi_gt);
      set->dot_u_psi_gt = d_u_gt - set->alpha*set->psi_gt_sqr;
      set->u_i_sqr = set->u_i->dot(*set->u_i) - 2*set->alpha*d_u_gt + SQR(set->alpha)*set->psi_gt_sqr;
      for(int jj = 0; jj < set->num_samples; jj++)
        set->samples[jj].alpha *= s_u;
      s_u = 1;
    }
  }
  
  u_i_scaled->make_non_sparse(true, sizePsi, true, u_i_buff);  // convert to a non-sparse vector, to accelerate -= and dot() operations
  for(r = 0; r < R; r++) {
    for(j = 0; j < set->num_samples; j++) {
      SVM_cached_sample *s = &set->samples[j];

      VFLOAT dot_u_v = u_i_scaled->dot(*s->psi)*s_u - set->dot_u_psi_gt - set->alpha*s->dot_psi_gt_psi;  // <u_i,v>
      VFLOAT dot = s->dot_w - dot_u_v;   // (lambda*t)<w,v>

#ifdef DEBUG_MULTI_SAMPLE_UPDATE
      SparseVector w_sum_new = *sum_w + set->psi_gt->mult_scalar(set->alpha) - u_i_scaled->mult_scalar(s_u);
      SparseVector u_new = *u_i_scaled*s_u - (*set->psi_gt*set->alpha);
      double dot_u_v_real = u_new.dot(*s->psi - *set->psi_gt);
      double dot_real = w_sum_new.dot(*s->psi - *set->psi_gt);
      fprintf(stderr, "t=%d, i=%d, j=%d, dot_u_v=%lg:%lg, dot=%lg:%lg\n", (int)t, set->i, j, dot_u_v_real, dot_u_v, dot_real, dot);
      assert(!dot_u_v || (dot_u_v_real/dot_u_v > .999999999 && dot_u_v_real/dot_u_v < 1.00000001));
      assert(!dot || (dot_real/dot > .999999999 && dot_real/dot < 1.00000001));
#endif

      VFLOAT scale=1, new_alpha;
      VFLOAT dalpha = (dot + s->loss*(sum_w_scale)) / my_max(s->sqr,.0000000001);
      if(set->alpha < 1 || dalpha < 0) {
	// alpha expand: solve for the optimal amount 'dalpha' to increase s->alpha
	// and then scale all set->samples[:].alpha (the value of 'dalpha' and 'scale' 
	// that maximizes the increase in the dual objective). 
	dalpha = my_min(1-set->alpha, my_max(-s->alpha*s_u,dalpha));
	/*if(set->u_i_sqr && set->alpha) {
	  scale = 1 + (L_i*sum_w_scale - dalpha*dot_u_v) / set->u_i_sqr;
	  scale = my_min((1-dalpha)/set->alpha, my_max(0, scale));
	  if(s->alpha*s_u*scale+dalpha < 0) dalpha = -s->alpha*s_u*scale;
	  }*/
	new_alpha = set->alpha*scale + dalpha;
      } else {
	// alpha swap: solve for the optimal amount 'dalpha' to increase s->alpha
	// while scaling down all set->samples[:].alpha, such that we preserve sum_k{s->samples[:].alpha}=1
	// (choose the value of 'dalpha' that maximizes the increase in the dual objective)
	VFLOAT e = dot/(sum_w_scale) + s->loss;
	VFLOAT sqr = set->u_i_sqr + 2*dot_u_v + s->sqr;
	dalpha = (e-L_i)*(sum_w_scale) / my_max(sqr,.00000000001);
	dalpha = my_min(1-s->alpha*s_u, my_max(-s->alpha*s_u,dalpha));
	scale = 1-dalpha;
	//assert(scale > 0 && scale <= 1);
	new_alpha = 1;
      }
      if(!(scale >= 0 && new_alpha >= 0 && new_alpha <= 1.000000001)) {
	fprintf(stderr, "scale=%f, new_alpha=%f, dalpha=%f, set->alpha=%f, u=%f\n", (float)scale, (float)new_alpha, (float)dalpha, (float)set->alpha, (float)set->u_i_sqr);
      }
      assert(scale >= 0 && new_alpha >= -0.000000001 && new_alpha <= 1.000000001);
      new_alpha = my_min(1,my_max(new_alpha,0));

      if(dalpha != 0 || scale != 1) {
	s_u *= scale;
	set->drift_bits += s_u <= 0 ? MAX_DRIFT_BITS : my_abs(LOG2(scale));
	if(set->drift_bits >= MAX_DRIFT_BITS) {
	  // To avoid numerical precision problems, set s_u=1 and recompute some values manually
	  set->drift_bits = 0;
	  u_i_scaled->make_non_sparse(false, -1, false, u_i_buff);
	  *u_i_scaled *= s_u;
	  double d_u_gt = u_i_scaled->dot(*set->psi_gt), alpha = set->alpha*scale;
	  set->dot_u_psi_gt = d_u_gt - alpha*set->psi_gt_sqr;
	  set->u_i_sqr = u_i_scaled->dot(*u_i_scaled) - 2*alpha*d_u_gt + SQR(alpha)*set->psi_gt_sqr;
	  set->D_i = scale*set->D_i;
	  dot_u_w_without_u = sum_w_without_u_i->dot(*u_i_scaled-(*set->psi_gt*alpha))/(sum_w_scale);
	  dot_w_u = dot_u_w_without_u - set->u_i_sqr/(sum_w_scale);
	  dot_u_v = u_i_scaled->dot(*s->psi) - set->dot_u_psi_gt - alpha*s->dot_psi_gt_psi; 
	  dot = s->dot_w - dot_u_v;
	  for(int jj = 0; jj < set->num_samples; jj++)
	    set->samples[jj].alpha *= s_u;
	  scale = 1;
	  s_u = 1;
	  u_i_scaled->make_non_sparse(true, sizePsi, true, u_i_buff);
	}

	*u_i_scaled += (*s->psi * (dalpha/s_u));
	s->alpha += dalpha/s_u;

	// Keep track of L_i, D_i, u_i_sqr, dot_w_u, dot_u_psi_gt using inexpensive online updates
	set->alpha = new_alpha;
	dot_u_w_without_u = scale*dot_u_w_without_u + dalpha*s->dot_w/(sum_w_scale);
        dot_w_u = scale*dot_w_u + (dalpha*dot - scale*(scale-1)*set->u_i_sqr - (2*scale*dalpha-dalpha)*dot_u_v - s->sqr*SQR(dalpha)) / (sum_w_scale);
	set->u_i_sqr = SQR(scale)*set->u_i_sqr + 2*scale*dalpha*dot_u_v + s->sqr*SQR(dalpha);
	set->dot_u_psi_gt = scale*set->dot_u_psi_gt + dalpha*(s->dot_psi_gt_psi - set->psi_gt_sqr);
	set->D_i = scale*set->D_i + dalpha*s->loss;
	L_i = dot_w_u + set->D_i;
	assert(!isnan(L_i));
#ifdef DEBUG_MULTI_SAMPLE_UPDATE
	SparseVector w_sum_new = *sum_w + set->psi_gt->mult_scalar(set->alpha) - u_i_scaled->mult_scalar(s_u);
	SparseVector u_new = *u_i_scaled*s_u - (*set->psi_gt*set->alpha);
	double dot_w_u_real = w_sum_new.dot(u_new)/(sum_w_scale);
	double dot_u_psi_gt_real = u_new.dot(*set->psi_gt);
	double u_i_sqr_real = u_new.dot(u_new);
	double dot_u_w_without_u_real = sum_w->dot(u_new)/(sum_w_scale);
	fprintf(stderr, "t=%d, i=%d, j=%d, scale=%f, dalpha=%f, s_u=%f, dot_w_u=%lg:%lg, dot_u_psi_gt=%lg:%lg, u_i_sqr=%lg:%lg\n", (int)t, set->i, j, (float)scale, (float)dalpha, (float)s_u, dot_w_u_real, dot_w_u,  dot_u_psi_gt_real, set->dot_u_psi_gt,  u_i_sqr_real, set->u_i_sqr);
	assert(dot_w_u_real/dot_w_u > .999999999 && dot_w_u_real/dot_w_u < 1.00000001);
	assert(dot_u_psi_gt_real/set->dot_u_psi_gt > .999999999 && dot_u_psi_gt_real/set->dot_u_psi_gt < 1.00000001);
	assert(u_i_sqr_real/set->u_i_sqr > .999999999 && u_i_sqr_real/set->u_i_sqr < 1.00000001);
	if(dot_u_w_without_u) assert(dot_u_w_without_u_real/dot_u_w_without_u > .999999999 && dot_u_w_without_u_real/dot_u_w_without_u < 1.00000001);
#endif
      }
    }
  }

  
  // Update sum_w, sum_dual, sum_w_sqr, and regularization_error, taking into account the new value of u_i
  set->u_i->make_non_sparse(false, -1, false, u_i_buff);
  *set->u_i *= s_u;
  *sum_w -= *set->u_i;   // Add u_i back into (lambda*t)w
  *sum_w += set->psi_gt->mult_scalar(set->alpha);
  double d_sum_w_sqr = 2*(dot_w_u_orig-dot_u_w_without_u)*sum_w_scale + set->u_i_sqr + u_i_sqr_orig;
  sum_dual += -d_sum_w_sqr/(2*sum_w_scale) + set->D_i - D_i_orig;
  sum_alpha_loss += set->D_i - D_i_orig;
  sum_w_sqr += d_sum_w_sqr;
  regularization_error = (sum_w_sqr/SQR(sum_w_scale))*params.lambda/2;
  for(j = 0; j < set->num_samples; j++)
    set->samples[j].alpha *= s_u;
  set->slack_after = set->num_samples ? sum_w->dot(*set->samples[0].psi-*set->psi_gt)/(sum_w_scale) + set->samples[0].loss : 0;
  

#ifdef DEBUG_MULTI_SAMPLE_UPDATE
  double sum_w_sqr_real = sum_w->dot(*sum_w);
  fprintf(stderr, "t=%d, i=%d, sum_w_sqr=%lg:%lg\n", (int)t, set->i, sum_w_sqr_real, sum_w_sqr);
  assert(sum_w_sqr_real/sum_w_sqr > .99999 && sum_w_sqr_real/sum_w_sqr < 1.0001);
#endif

  //fprintf(stderr, "sum dual is %lg, dual_change=%lg, D=%lg\n", sum_dual, -d_sum_w_sqr/(2*sum_w_scale) + set->D_i - D_i_orig, set->D_i);
}



typedef struct {
  VFLOAT alpha, val, e, d;
  int ind;
} SortableSample;

int SortableSample_cmp(const void * a, const void * b) {
  VFLOAT f = ((SortableSample*)b)->val - ((SortableSample*)a)->val;
  return f < 0 ? -1 : (f > 0 ? 1 : 0);
}



// Special faster update for when all samples are orthogonal to one another, for example multiclass SVMs
// or pose mixture models
void StructuredSVM::MultiSampleUpdateOrthogonalSamples(SVM_cached_sample_set *set, StructuredExample *ex, int R) {
  if(!set->num_samples && !set->alpha)
    return;
  SortableSample *vals = new SortableSample[set->num_samples+1];
  VFLOAT *alpha = new double[(set->num_samples+1)];
  VFLOAT gt_sqr = set->psi_gt_sqr;
  VFLOAT c = 0, d_e = 0, d_e2 = 0, c_bound = 0, sum_d = 0, sum_alpha = 0, k = 0, v = 0;
  int i, j, m;

  int num = my_min(params.max_samples, set->num_samples);
  for(i = 0; i < num; i++) {
    SVM_cached_sample *s = &set->samples[i];
    vals[i].alpha = s->alpha;
    vals[i].e = s->dot_w + s->loss*(sum_w_scale);
    vals[i].d = 1.0 / (s->sqr-gt_sqr);
    vals[i].val = vals[i].alpha + (vals[i].e)*vals[i].d;
    vals[i].ind = i;
    alpha[i] = s->alpha;
  }

  // The derivative of the dual w.r.t. this sample is
  //   dD/dalpha_i=<w,v_i>+loss_i-(alpha_i*v_i^2/sum_w_scale)
  //              =(vals[i].e)/sum_w_scale
  // were v_i = Psi(x,y_gt)-Psi(x,ybar_i)
  // If we were not subject to the constraints alpha_i>=0 and sum_alpha<=1, then we would reach the
  // global maximum if we take one Newton step alpha' = alpha - H^-1 \nabla
  // The 2nd partial derivative of the dual
  //   dD/(dalpha_i*dalpha_j)=-1/sum_w_scale <v_i,v_j>
  // So the hessian matrix is
  //   H=-1/(sum_w_scale)*[Psi(x,y_gt)^2*1+D], where 1 is a mXm 
  // matrix of ones and D is a mXm diagonal matrix with diagonal elements Psi(x,ybar_i)^2. 
  //   H^-1 = -sum_w_scale[D^-1 - 1/(g+1) * Psi(x,y_gt)^2 * D^-1 1 D^-1]
  //        = -sum_w_scale[diag(d) - k d d^T ], where
  // where g = trace(D^-1 * Psi(x,y_gt)^2*1) = Psi(x,y_gt)^2 * sum_i vals[i].d, and d is a
  // m-vector with elements vals[i].d=1/Psi(x,ybar_i)^2, and k=1/(g+1)*Psi(x,y_gt)^2
  // This corresponds to an update:
  //   alpha' = alpha + (diag(d) - k d d^T ) * e
  // where e is an n-vector of elements vals[i].e and c=k d^T e.  Then
  //   alpha_i' = alpha_i + vals[i].d*(vals[i].e-c)
  // However, to handle the constraint alpha_i>=0, we can sort each sample by
  //    alpha_i + vals[i].d*vals[i].e, which will imply that if alpha_i'<0 then
  // alpha_j'<0 for all j>i. 
  // To handle the constraint (sum_i alpha_i)<=1, we can solve for the value of c where
  // (sum_i alpha_i')=1:
  //   c_bound = [sum_i ( alpha_i + vals[i].d*vals[i].e) - 1] / [sum_i vals[i].d]
  // This effectively makes it such that after the update, dD/dalpha_i will be the same for all samples for which alpha_i>0
  qsort(vals, num, sizeof(SortableSample), SortableSample_cmp); // sort by val in descending order
  v = set->psi_gt_sqr*set->alpha;  // increase in slack for each example vals[i].e i=1...m from setting alpha_j=0 for j > m 
  for(m = 0; m < num && vals[m].val+vals[m].d*(v-c) > 0; m++) {
    sum_d += vals[m].d;
    sum_alpha += vals[m].alpha;
    k = set->psi_gt_sqr/(1+set->psi_gt_sqr*sum_d);
    v -= set->psi_gt_sqr*vals[m].alpha;
    d_e += vals[m].e*vals[m].d;
    d_e2 = d_e + v*sum_d;
    c_bound = (d_e2+sum_alpha-1)/sum_d;
    c = my_max(d_e2*k, c_bound);
  }

  sum_alpha = 0;
  for(j = 0; j < m; j++) {
    vals[j].alpha = vals[j].val+vals[j].d*(v-c);
    assert(vals[j].alpha >= -0.00001);
    vals[j].alpha = my_min(1, my_max(0,vals[j].alpha));
    sum_alpha += vals[j].alpha;
  }
  for(j = m; j < num; j++) {
    assert(vals[j].val-vals[j].d*c <= 0.00001);
    vals[j].alpha = 0;
  }
  assert(sum_alpha <= 1.00001);
  sum_alpha = my_min(1, my_max(0,sum_alpha));

  for(i = 0; i < num; i++)
    alpha[vals[i].ind] = vals[i].alpha;

  set->u_i_sqr = 0;
  double d_sum_w_sqr = 0;
  double D_i = 0;
  for(i = 0; i < set->num_samples; i++) {
    double dalpha = alpha[i] - set->samples[i].alpha;
    if(dalpha) {
      d_sum_w_sqr += -2*dalpha*(set->samples[i].dot_w+set->dot_sum_w_psi_gt) + set->psi_gt_sqr*SQR(dalpha);
      *sum_w -= (*set->samples[i].psi*dalpha);
    }
    set->samples[i].alpha = alpha[i];
    set->u_i_sqr += set->psi_gt_sqr*(SQR(alpha[i]) + SQR(sum_alpha));
    D_i += alpha[i]*set->samples[i].loss;
  }
  set->dot_u_psi_gt = -sum_alpha*set->psi_gt_sqr;
  double dalpha = sum_alpha-set->alpha;
  if(dalpha) {
    d_sum_w_sqr += 2*dalpha*set->dot_sum_w_psi_gt + set->psi_gt_sqr*SQR(dalpha);
    *sum_w += (*set->psi_gt*dalpha);
  }
  set->alpha = sum_alpha;
  sum_alpha_loss += D_i - set->D_i;
  sum_w_sqr += d_sum_w_sqr;
  sum_dual = -sum_w_sqr/(2*sum_w_scale) + sum_alpha_loss;
  regularization_error = (sum_w_sqr/SQR(sum_w_scale))*params.lambda/2;
  set->D_i = D_i;

  set->slack_after = set->num_samples ? (sum_w->dot(*set->samples[0].psi)-sum_w->dot(*set->psi_gt))/(sum_w_scale) + set->samples[0].loss : 0;

  delete [] alpha;
  delete [] vals;
}



void free_SVM_cached_sample(SVM_cached_sample *s) {
  if(s->ybar) delete s->ybar;
  if(s->psi) delete s->psi;
  s->ybar = NULL;
  s->psi = NULL;
}


bool ReadString(char *str, FILE *fin) {
  int len;
  return fread(&len, sizeof(int), 1, fin) && fread(str, sizeof(char), len, fin);
}

bool WriteString(char *str, FILE *fout) {
  int len=strlen(str)+1;
  return fwrite(&len, sizeof(int), 1, fout) && fwrite(str, sizeof(char), len, fout);
}





void read_SVM_cached_sample(SVM_cached_sample *s, FILE *fin, StructuredSVM *svm, StructuredData *x, bool readFull) {
  char str[100000];
  ReadString(str, fin);
  s->ybar = svm->NewStructuredLabel(x);
  Json::Reader reader;
  Json::Value v;
  bool b = reader.parse(str, v);
  assert(b);
  s->ybar->load(v, svm);

  s->psi = NULL;
  clear_SVM_cached_sample(s);
  if(readFull) {
    bool hasPsi;
    bool b = (fread(&s->loss, sizeof(VFLOAT), 1, fin) && fread(&s->alpha, sizeof(VFLOAT), 1, fin) && fread(&s->sqr, sizeof(VFLOAT), 1, fin) &&
	      fread(&s->slack, sizeof(VFLOAT), 1, fin) && fread(&s->dot_psi_gt_psi, sizeof(VFLOAT), 1, fin) && 
	      fread(&s->dot_w, sizeof(VFLOAT), 1, fin) && fread(&hasPsi, sizeof(bool), 1, fin));
    assert(b);
    if(hasPsi) {
      s->psi = new SparseVector;
      s->psi->read(fin);
    }
  }
}


void write_SVM_cached_sample(SVM_cached_sample *s, FILE *fout, StructuredSVM *svm, bool saveFull) {
  char str[100000];
  Json::FastWriter writer;
  Json::Value v = s->ybar->save(svm);
  strcpy(str, writer.write(v).c_str());
  WriteString(str, fout);
  
  if(saveFull) {
    bool hasPsi = s->psi != NULL;
    bool b = (fwrite(&s->loss, sizeof(VFLOAT), 1, fout) && fwrite(&s->alpha, sizeof(VFLOAT), 1, fout) && fwrite(&s->sqr, sizeof(VFLOAT), 1, fout) &&
	      fwrite(&s->slack, sizeof(VFLOAT), 1, fout) && fwrite(&s->dot_psi_gt_psi, sizeof(VFLOAT), 1, fout) && 
	      fwrite(&s->dot_w, sizeof(VFLOAT), 1, fout) && fwrite(&hasPsi, sizeof(bool), 1, fout));
    assert(b);
    if(hasPsi) 
      s->psi->write(fout);
  }
}



void clear_SVM_cached_sample_set(SVM_cached_sample_set *s) {
  if(s->u_i)
    delete s->u_i;
  s->u_i = NULL;
  if(s->ybar)
    delete s->ybar;
  s->ybar = NULL;
  s->num_samples = 0;
  s->num_evicted_samples = 0;
  s->alpha = 0;
  s->loss = 0;
  s->slack_before = 0;
  s->slack_after = 0;
  s->psi_gt_sqr = 0;
  s->D_i = 0;
  s->dot_u_psi_gt = 0;
  s->u_i_sqr = 0;
  s->drift_bits = 0;
  s->lock = false;
  s->numIters = 0;
  s->sumSlack = 0;
}

SVM_cached_sample_set *new_SVM_cached_sample_set(int i, SparseVector *psi_gt) {
  SVM_cached_sample_set *retval = (SVM_cached_sample_set*)malloc(sizeof(SVM_cached_sample_set));
  retval->samples = retval->evicted_samples = NULL;
  retval->u_i = NULL;
  retval->ybar = NULL;
  retval->psi_gt = psi_gt;
  retval->i = i;
  retval->prev = retval->next = NULL;
  retval->inMemory = false;
  clear_SVM_cached_sample_set(retval);
  return retval;
}


void free_SVM_cached_sample_set(SVM_cached_sample_set *s) {
  int i;
  for(i = 0; i < s->num_samples; i++)
    free_SVM_cached_sample(&s->samples[i]);
  if(s->samples) free(s->samples);
  for(i = 0; i < s->num_evicted_samples; i++)
    free_SVM_cached_sample(&s->evicted_samples[i]);
  if(s->evicted_samples) free(s->evicted_samples);
  if(s->psi_gt) delete s->psi_gt;
  if(s->u_i) delete s->u_i;
  if(s->ybar) delete s->ybar;
  free(s);
}

SVM_cached_sample_set *read_SVM_cached_sample_set(FILE *fin, StructuredSVM *svm, StructuredData *x, bool readFull) {
  int i;
  bool hasFull;
  bool b = (fread(&hasFull, sizeof(bool), 1, fin)); assert(b);
  b = (fread(&i, sizeof(int), 1, fin)); assert(b);
  SVM_cached_sample_set *s = new_SVM_cached_sample_set(i);
  b = (fread(&s->num_samples, sizeof(int), 1, fin)); assert(b);
  s->samples = (SVM_cached_sample*)realloc(s->samples, sizeof(SVM_cached_sample)*(s->num_samples+1));
  for(int i = 0; i < s->num_samples; i++) {
    read_SVM_cached_sample(&s->samples[i], fin, svm, x, hasFull);
    if(!readFull) 
      clear_SVM_cached_sample(&s->samples[i]);
  }
  b = (fread(&s->num_evicted_samples, sizeof(int), 1, fin)); assert(b);
  if(s->num_evicted_samples)
    s->evicted_samples = (SVM_cached_sample*)realloc(s->evicted_samples, sizeof(SVM_cached_sample)*(s->num_evicted_samples+1));
  else if(s->evicted_samples) {
    free(s->evicted_samples);
    s->evicted_samples = NULL;
  }
  for(int i = 0; i < s->num_evicted_samples; i++) {
    read_SVM_cached_sample(&s->evicted_samples[i], fin, svm, x, hasFull);
    if(!readFull) 
      clear_SVM_cached_sample(&s->evicted_samples[i]);
  }
  s->psi_gt = new SparseVector;  
  s->psi_gt->read(fin);

  if(hasFull) {
    b = (fread(&s->alpha, sizeof(double), 1, fin));  assert(b);
    b = (fread(&s->loss, sizeof(double), 1, fin));  assert(b);
    b = (fread(&s->slack_before, sizeof(double), 1, fin));  assert(b);
    b = (fread(&s->slack_after, sizeof(double), 1, fin));  assert(b);
    b = (fread(&s->psi_gt_sqr, sizeof(double), 1, fin));  assert(b);

    b = (fread(&s->dot_sum_w_psi_gt, sizeof(double), 1, fin));  assert(b);
    b = (fread(&s->D_i, sizeof(VFLOAT), 1, fin));  assert(b);
    b = (fread(&s->dot_u_psi_gt, sizeof(VFLOAT), 1, fin));  assert(b);
    b = (fread(&s->u_i_sqr, sizeof(double), 1, fin));  assert(b);
    b = (fread(&s->drift_bits, sizeof(double), 1, fin));  assert(b);
    b = (fread(&s->sumSlack, sizeof(double), 1, fin));  assert(b);
    b = (fread(&s->numIters, sizeof(int), 1, fin));  assert(b);
    bool has_u_i;
    b = (fread(&has_u_i, sizeof(bool), 1, fin));  assert(b);
    if(has_u_i) {
      s->u_i = new SparseVector;  
      s->u_i->read(fin);
    }
    if(!readFull)
      clear_SVM_cached_sample_set(s);
  }
  s->psi_gt_sqr = s->psi_gt->dot(*s->psi_gt);
  return s;
}

void write_SVM_cached_sample_set(SVM_cached_sample_set *s, FILE *fout, StructuredSVM *svm, bool fullWrite) {
  bool b = (fwrite(&fullWrite, sizeof(bool), 1, fout));  assert(b);
  b = (fwrite(&s->i, sizeof(int), 1, fout)); assert(b);
  b = (fwrite(&s->num_samples, sizeof(int), 1, fout)); assert(b);
  for(int i = 0; i < s->num_samples; i++)
    write_SVM_cached_sample(&s->samples[i], fout, svm, fullWrite);
  b = (fwrite(&s->num_evicted_samples, sizeof(int), 1, fout)); assert(b);
  for(int i = 0; i < s->num_evicted_samples; i++)
    write_SVM_cached_sample(&s->evicted_samples[i], fout, svm, fullWrite);
  s->psi_gt->write(fout);

  if(fullWrite) {
    b = (fwrite(&s->alpha, sizeof(double), 1, fout));  assert(b);
    b = (fwrite(&s->loss, sizeof(double), 1, fout));  assert(b);
    b = (fwrite(&s->slack_before, sizeof(double), 1, fout));  assert(b);
    b = (fwrite(&s->slack_after, sizeof(double), 1, fout));  assert(b);
    b = (fwrite(&s->psi_gt_sqr, sizeof(double), 1, fout));  assert(b);

    b = (fwrite(&s->dot_sum_w_psi_gt, sizeof(double), 1, fout));  assert(b);
    b = (fwrite(&s->D_i, sizeof(VFLOAT), 1, fout));  assert(b);
    b = (fwrite(&s->dot_u_psi_gt, sizeof(VFLOAT), 1, fout));  assert(b);
    b = (fwrite(&s->u_i_sqr, sizeof(double), 1, fout));  assert(b);
    b = (fwrite(&s->drift_bits, sizeof(double), 1, fout));  assert(b);
    b = (fwrite(&s->sumSlack, sizeof(double), 1, fout));  assert(b);
    b = (fwrite(&s->numIters, sizeof(int), 1, fout));  assert(b);
    bool has_u_i = s->u_i != NULL;
    b = (fwrite(&has_u_i, sizeof(bool), 1, fout));  assert(b);
    if(has_u_i) s->u_i->write(fout);
  }
}


void clear_SVM_cached_sample(SVM_cached_sample *s) {
  if(s->psi) delete s->psi;
  s->psi = NULL;
  s->loss = 0;
  s->alpha = 0;
  s->slack = 0;
  s->sqr = 0;
  s->dot_w = 0;
  s->dot_psi_gt_psi = 0;
}

SVM_cached_sample *SVM_cached_sample_set_add_sample(SVM_cached_sample_set *s, StructuredLabel *ybar) {
  s->samples = (SVM_cached_sample*)realloc(s->samples, sizeof(SVM_cached_sample)*(s->num_samples+1));
  SVM_cached_sample *retval = s->samples+s->num_samples;
  retval->ybar = ybar;
  retval->psi = NULL;
  clear_SVM_cached_sample(retval);
  s->num_samples++;

  return retval;
}

//#define DEBUG_MULTI_SAMPLE_UPDATE

int int_compare (const void * a, const void * b)
{
  return ( *(int*)a - *(int*)b );
}

int SVM_cached_sample_cmp(const void * a, const void * b) {
  double d = ((SVM_cached_sample*)b)->slack - ((SVM_cached_sample*)a)->slack;
  return d < 0 ? -1 : (d > 0 ? 1 : 0);
}

int SVM_cached_sample_alpha_cmp(const void * a, const void * b) {
  double d = ((SVM_cached_sample*)b)->alpha - ((SVM_cached_sample*)a)->alpha;
  return d < 0 ? -1 : (d > 0 ? 1 : 0);
}

int example_ave_slack_cmp(const void *a, const void *b) {
  SVM_cached_sample_set *set1 = (*((StructuredExample**)a))->set;
  SVM_cached_sample_set *set2 = (*((StructuredExample**)b))->set;
  double v1 = set1 && set1->numIters ? set1->sumSlack/set1->numIters : -10000000;
  double v2 = set2 && set2->numIters ? set2->sumSlack/set2->numIters : -10000000;
  double d = v2-v1;
  return d < 0 ? -1 : (d > 0 ? 1 : 0);
}

int example_slack_cmp(const void *a, const void *b) {
  SVM_cached_sample_set *set1 = (*((StructuredExample**)a))->set;
  SVM_cached_sample_set *set2 = (*((StructuredExample**)b))->set;
  double v1 = set1 ? set1->slack_before : -10000000;
  double v2 = set2 ? set2->slack_before : -10000000;
  double d = v2-v1;
  return d < 0 ? -1 : (d > 0 ? 1 : 0);
}

int example_alpha_cmp(const void *a, const void *b) {
  SVM_cached_sample_set *set1 = (*((StructuredExample**)a))->set;
  SVM_cached_sample_set *set2 = (*((StructuredExample**)b))->set;
  double v1 = set1 ? set1->alpha : 0;
  double v2 = set2 ? set2->alpha : 0;
  double d = v2-v1;
  return d < 0 ? -1 : (d > 0 ? 1 : example_ave_slack_cmp(a,b));
}

int example_suggest_cmp(const void *a, const void *b) {
  SVM_cached_sample_set *set1 = (*((StructuredExample**)a))->set;
  SVM_cached_sample_set *set2 = (*((StructuredExample**)b))->set;
  double v1 = set1 ? set1->slackSuggest : 0;
  double v2 = set2 ? set2->slackSuggest : 0;
  double d = v2-v1;
  return d < 0 ? -1 : (d > 0 ? 1 : example_ave_slack_cmp(a,b));
}

