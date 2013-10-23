
#include "structured_svm_partmodel.h"

extern StructuredSVM *g_learner;
int main_classify(int argc, char* argv[]);


int main(int argc, char* argv[]) {
  g_learner = new PartModelStructuredSVM;
  return main_classify(argc, argv);
}
