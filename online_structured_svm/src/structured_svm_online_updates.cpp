#include "structured_svm.h"
#include "structured_svm_train_statistics.h"
#include "structured_svm_train_choose_example.h"



int StructuredSVM::AddExample(StructuredData *x, StructuredLabel *y) {
  int retval = -1;
  Lock();
  hasConverged = false;

  // Add copy of ex to list of examples
  assert(trainset);
  exampleIdsToIndices = (int*)realloc(exampleIdsToIndices, sizeof(int)*(numExampleIds+1));
  exampleIndicesToIds = (int*)realloc(exampleIndicesToIds, sizeof(int)*(numExampleIds+1));
  exampleIdsToIndices[numExampleIds] = trainset->num_examples;
  exampleIndicesToIds[trainset->num_examples] = numExampleIds;
  retval = numExampleIds++;
  trainset->AddExample(CopyExample(x,y));
  chooser->CreateTrainingExampleQueues(trainset->num_examples);
  Unlock();

  // Note: Updating sum_dual and regularization_error will happen automatically via a call to
  // SetSumWScale()

  return retval;
}




bool StructuredSVM::RemoveExample(int id) {
  Lock();
  if(id >= numExampleIds || id < 0 || exampleIdsToIndices[id] < 0) {
    Unlock();
    return false;
  }

  int ind = exampleIdsToIndices[id];

  hasConverged = false;

  StructuredExample *ex = trainset->examples[ind];
  while(trainset->examples[ind]->set && trainset->examples[ind]->set->lock) {
    Unlock();
    usleep(100000);
    Lock();
  }

  // The effect of setting the dual parameters for this example to zero: set->alpha = 0
  if(ex->set && ex->set->u_i) 
    RemoveExample(ex->set);

  // The effect of changing the dual objective to be over n-1 examples instead of n examples
  if(n == trainset->num_examples) {
    n--;
    SetSumWScale(params.lambda*n);
  }

  // Update data structures and variables to remove this example.  To remove the example at index ind, we will
  // move the example at index trainset->num_examples-1 to index ind, then decrement trainset->num_examples
  stats->RemoveExample(ind);
  chooser->RemoveExample(ind);
  
  trainset->examples[ind] = trainset->examples[trainset->num_examples--];
  trainset->examples[ind]->set->i = ind;
  exampleIdsToIndices[exampleIndicesToIds[trainset->num_examples-1]] = ind;
  exampleIndicesToIds[ind] = exampleIndicesToIds[trainset->num_examples-1];
  exampleIdsToIndices[id] = -1;
  delete ex;


  Unlock();

  return true;
}

void StructuredSVM::RemoveExample(SVM_cached_sample_set *set) {
  assert(set->u_i);

  double d_sum_w_sqr = 2*(sum_w->dot(*set->u_i) - sum_w->dot(*set->psi_gt)*set->alpha) + set->u_i_sqr;
  *sum_w += *set->u_i;   // Add u_i back into (lambda*t)w
  *sum_w -= set->psi_gt->mult_scalar(set->alpha);

  sum_dual += -d_sum_w_sqr/(2*sum_w_scale) - set->D_i;
  sum_alpha_loss -= set->D_i;
  sum_w_sqr += d_sum_w_sqr;
  regularization_error = (sum_w_sqr/SQR(sum_w_scale))*params.lambda/2;
}


bool StructuredSVM::RelabelExample(int id) {
  Lock();
  if(id >= numExampleIds || id < 0 || exampleIdsToIndices[id] < 0) {
    Unlock();
    return false;
  }

  int ind = exampleIdsToIndices[id];
  StructuredExample *ex = trainset->examples[ind];
  while(ex->set && ex->set->lock) {
    Unlock();
    usleep(100000);
    Lock();
  }

  OnRelabeledExample(ex);
  Unlock();

  return true;
}

void StructuredSVM::OnRelabeledExample(StructuredExample *ex) {
  hasConverged = false;

  if(ex->set) {
    // Update computation of psi_gt=Psi(x,y) using the new label, and update its effect on w^2
    SparseVector *psi_gt = Psi(ex->x, ex->y).ptr();
    SparseVector d_gt = (*psi_gt - *ex->set->psi_gt);
    double d_sum_w_sqr = SQR(ex->set->alpha)*d_gt.dot(d_gt) + 2*ex->set->alpha*sum_w->dot(d_gt);
    *sum_w += d_gt*ex->set->alpha;
    sum_w_sqr += d_sum_w_sqr;
    regularization_error = sum_w_sqr/SQR(sum_w_scale)*params.lambda/2;
    sum_dual -= d_sum_w_sqr/(2*sum_w_scale);
    delete ex->set->psi_gt;
    ex->set->psi_gt = psi_gt;

    // Update the loss Loss(y,ybar) with respect to the new label and all samples ybar
    double D_i_old = ex->set->D_i;
    double d_u_gt = ex->set->u_i->dot(*ex->set->psi_gt);
    ex->set->psi_gt_sqr = ex->set->psi_gt->dot(*ex->set->psi_gt);
    ex->set->dot_u_psi_gt = d_u_gt - ex->set->alpha*ex->set->psi_gt_sqr;
    ex->set->u_i_sqr = ex->set->u_i->dot(*ex->set->u_i) - 2*ex->set->alpha*d_u_gt + SQR(ex->set->alpha)*ex->set->psi_gt_sqr;
    ex->set->D_i = 0;
    for(int j = 0; j < ex->set->num_samples; j++) {
      double los = Loss(ex->y, ex->set->samples[j].ybar);
      ex->set->D_i += ex->set->samples[j].alpha*los;
      ex->set->samples[j].loss = los;
      if(ex->set->samples[j].psi) {
        ex->set->samples[j].dot_psi_gt_psi = ex->set->psi_gt->dot(*ex->set->samples[j].psi);
        ex->set->samples[j].sqr = ex->set->psi_gt_sqr - 2*ex->set->samples[j].dot_psi_gt_psi + ex->set->samples[j].psi->dot(*ex->set->samples[j].psi);
        ex->set->samples[j].dot_psi_gt_psi = ex->set->psi_gt->dot(*ex->set->samples[j].psi);
        ex->set->samples[j].dot_w = sum_w ? sum_w->dot(*ex->set->samples[j].psi) - ex->set->score_gt*sum_w_scale : 0;
      }
    }
    for(int j = 0; j < ex->set->num_evicted_samples; j++) {
      if(ex->set->evicted_samples[j].psi) {
        double los = Loss(ex->y, ex->set->evicted_samples[j].ybar);
        ex->set->D_i += ex->set->evicted_samples[j].alpha*los;
      }
    }

    sum_dual += ex->set->D_i-D_i_old;
    sum_alpha_loss += ex->set->D_i-D_i_old;

    MultiSampleUpdate(ex->set, ex);
    ex->set->score_gt = sum_w->dot(*ex->set->psi_gt)/sum_w_scale;
  }
}



