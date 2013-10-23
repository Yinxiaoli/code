/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#include "dataset.h"
#include "main.h"


#include <assert.h>


int main(int argc, const char **argv) {
  int num[1000], it[1000], num2, it2, i=0, j=0;
  double tm[1000], dual[1000], tm2, dual2, loss, svm_err;
  char time_file[1000], res_file[1000], modelfile[1000], testset[1000], model[1000], predictionsFile[1000];

  if(argc < 3) {
    fprintf(stderr, "USAGE:\n  ./plot_online_structured_learning <testset_file> <model_name>");
    return -1;
  }

  

  strcpy(testset, argv[1]);
  strcpy(model, argv[2]);

  sprintf(time_file, "%s.times", model);
  FILE *fin = fopen(time_file, "r");
  assert(fin);
  sprintf(res_file, "%s.results", model);
  char line[10000];
  while(fgets(line, 9999, fin) && sscanf(line, "%d %lf %d %lf\n", &num[i], &tm[i], &it[i], &dual[i])>=3) {
    if(!i || num[i] != num[i-1]) i++;
	else if(i) { tm[i-1]=tm[i]; it[i-1]=it[i]; dual[i-1]=dual[i];  }
  }
  fclose(fin);

  fin = fopen(res_file, "r");
  while(fin && fgets(line, 9999, fin) && sscanf(line, "%d %lf %d %lf %lf %lf\n", &num2, &tm2, &it2, &dual2, &loss, &svm_err)==6) {
    assert(num[j] == num2 && tm[j] == tm2 && it[j] == it2 && dual[j] == dual2); 
    j++;
  }
  if(fin) fclose(fin);

  FILE *fout = fopen(res_file, "a");
  for(; j < i; j++) {
    StructuredSVM *svm = new PartModelStructuredSVM;

    sprintf(modelfile, "%s.%d", model, num[j]);
    sprintf(predictionsFile, "%s.%d.pred", model, num[j]);
    bool b = svm->Load(modelfile, false);
    assert(b);
	
	if(argc > 3) {
		Classes *classes = ((PartModelStructuredSVM*)svm)->GetClasses();
		Dataset *dataset = dataset = new Dataset(classes);
		dataset->Load(argv[3]);
		dataset->LearnUserClickProbabilities();
		classes->SetDetectionLossMethod(LOSS_NUM_INCORRECT);
		classes->Save("CUB_200_2011_data/classes.spatial.user");
		delete dataset;
	}


    double loss = svm->Test(testset, predictionsFile, NULL, &svm_err);
    fprintf(fout, "%d %lf %d %lf %lf %lf\n", num[j], tm[j], it[j], dual[j], loss, svm_err);
    fflush(fout);
  }
  fclose(fout);
  return 0;
}


 
