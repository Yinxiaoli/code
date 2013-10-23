#include "visualizationTools.h"


void BuildGallery(const char **images, int numImages, const char *outFileName, const char *title, 
		  const char *header, const char **thumbs, const char **imageDescriptions, int numThumbs) {
  const char *htmlTemplate = "examples/html/vis/gallery.html";
  const char *imageTemplate = "examples/html/vis/image_templ.html";
  char relPath[1000], outFolderName[1000], sys[1000];
  ExtractPathname(outFileName, outFolderName);
  strcat(outFolderName, "/");
  if(!FileExists(htmlTemplate)) {
    htmlTemplate = "html/vis/gallery.html";
    imageTemplate = "html/vis/image_templ.html";
    if(!FileExists(htmlTemplate)) {
      fprintf(stderr, "Couldn't find %s\n", htmlTemplate);
      return;
    }
    sprintf(sys, "cp -r html/js html/css %s", outFolderName);
  } else
    sprintf(sys, "cp -r examples/html/js examples/html/css %s", outFolderName);
  int res = system(sys);

  char *image_templ = ReadStringFile(imageTemplate);
  assert(image_templ);

  int i, len = 1;;
  char **imageHTMLs = (char**)malloc(sizeof(char*)*numImages);
  for(i = 0; i < numImages; i++) {
    char imageTitle[1000];
    ExtractFilename(images[i], imageTitle);
    char *imageStr = StringReplace(image_templ, "INSERT_IMAGE_TITLE_HERE", imageTitle);
    char *tmp = StringReplace(imageStr, "INSERT_IMAGE_NAME_HERE", images[i]);
    free(imageStr);  imageStr = tmp;
    tmp = StringReplace(imageStr, "INSERT_IMAGE_THUMB_NAME_HERE", thumbs ? thumbs[i] : images[i]);
    free(imageStr);  imageStr = tmp;
    tmp = StringReplace(imageStr, "INSERT_IMAGE_DESCRIPTION_HERE", imageDescriptions ? imageDescriptions[i] : "");
    free(imageStr);  imageStr = tmp;
    imageHTMLs[i] = imageStr;
    len += strlen(imageStr);
  }
  char *imageHtml = (char*)malloc(sizeof(char)*len);
  len = 0;
  for(i = 0; i < numImages; i++) {
    strcpy(imageHtml+len, imageHTMLs[i]);
    len += strlen(imageHTMLs[i]);
    free(imageHTMLs[i]);
  }


  char numThumbsStr[1000];
  char *html = ReadStringFile(htmlTemplate);
  assert(html);
  char *tmp = StringReplace(html, "INSERT_TITLE_HERE", title ? title : "");
  free(html);  html = tmp;
  tmp = StringReplace(html, "INSERT_HEADER_HERE", header ? header : "");
  free(html);  html = tmp;
  sprintf(numThumbsStr, "%d", numThumbs);
  tmp = StringReplace(html, "INSERT_NUM_THUMBS_HERE", numThumbsStr);
  free(html);  html = tmp;
  tmp = StringReplace(html, "INSERT_IMAGES_HERE", imageHtml);
  free(html);  html = tmp;

  FILE *fout = fopen(outFileName, "w");
  fprintf(fout, "%s",  html); 
  fclose(fout);

  free(imageHtml);
  free(html);
  free(image_templ);
  free(imageHTMLs);
}

void BuildConfusionMatrix(int *predLabels, int *gtLabels, int numExamples, const char *outFileName,
			  const char **classNames, const char *title, const char *header, const char **imageNames, 
			  const char **linkNames, int *classGroups, int numGroups, int confMatWidth) {
  const char *htmlTemplate = "examples/html/vis/confMat.html";
  if(!FileExists(htmlTemplate)) {
    htmlTemplate = "html/vis/confMat.html";
    if(!FileExists(htmlTemplate)) {
      fprintf(stderr, "Couldn't find %s\n", htmlTemplate);
      return;
    }
  }

  int numClasses = 0, i, j;
  for(i = 0; i < numExamples; i++) {
    if(gtLabels[i] >= numClasses)
      numClasses = gtLabels[i]+1;
    if(predLabels[i] >= numClasses)
      numClasses = predLabels[i]+1;
  }
  
  double **confMat = (double**)malloc(numClasses*(sizeof(double*)+numClasses*sizeof(double)+2*sizeof(int)));
  double *ptr = (double*)(confMat + numClasses);
  for(i = 0; i < numClasses; i++, ptr += numClasses) {
    confMat[i] = ptr;
    for(j = 0; j < numClasses; j++) 
      confMat[i][j] = 0;
  }
  int *class_pred_counts = (int*)ptr; 
  int *class_gt_counts = class_pred_counts+numClasses; 
  for(i = 0; i < numClasses; i++)
    class_pred_counts[i] = class_gt_counts[i] = 0;
  for(i = 0; i < numExamples; i++) {
    class_pred_counts[predLabels[i]]++;
    class_gt_counts[gtLabels[i]]++;
    confMat[gtLabels[i]][predLabels[i]]++;
  }
  
  int numLinks=0;
  Json::Value nodes, links, root;
  for(i = 0; i < numClasses; i++) {
    Json::Value node;
    node["id"] = i;
    node["name"] = classNames[i];
    node["group"] = classGroups ? classGroups[i] : 0;
    nodes[i] = node;
    if(!imageNames) {
      for(j = 0; j < numClasses; j++) {
	Json::Value link;
	if(confMat[i][j]) {
	  link["source"] = i;
	  link["target"] = j;
	  link["value"] = confMat[i][j];
	  links[numLinks++] = link;
	}
      }
    }
  }
  
  if(imageNames) {
    for(j = 0; j < numExamples; j++) {
      Json::Value link;
      link["source"] = gtLabels[j];
      link["target"] = predLabels[j];
      link["value"] = 1;
      link["img"] = imageNames[j];
      if(linkNames) link["href"] = linkNames[j];
      links[numLinks++] = link;
    }
  } 
  root["nodes"] = nodes;
  root["links"] = links;

  char jsonFile[1000], jsonFileShort[1000], baseName[1000], widthStr[200];
  Json::StyledWriter writer;
  sprintf(widthStr, "%d", confMatWidth);
  strcpy(baseName, outFileName);
  StripFileExtension(baseName);
  sprintf(jsonFile, "%s.json", baseName);
  ExtractFilename(jsonFile, jsonFileShort);
  FILE *fout = fopen(jsonFile, "w");
  fprintf(fout, "%s",  writer.write(root).c_str()); 
  fclose(fout);
  
  char *html = ReadStringFile(htmlTemplate);
  assert(html);
  char *tmp = StringReplace(html, "INSERT_TITLE_HERE", title);
  free(html);  html = tmp;
  tmp = StringReplace(html, "INSERT_HEADER_HERE", header ? header : "");
  free(html);  html = tmp;
  tmp = StringReplace(html, "INSERT_WIDTH_HERE", widthStr);
  free(html);  html = tmp;
  tmp = StringReplace(html, "conf.json", jsonFileShort);
  free(html);  html = tmp;

  fout = fopen(outFileName, "w");
  fprintf(fout, "%s",  html); 
  fclose(fout);

  free(confMat);
  free(html);
}


ExampleVisualization *AllocateExampleVisualization(const char *fname, const char *thumb, const char *description, double loss) {
  ExampleVisualization *retval = (ExampleVisualization*)malloc(sizeof(ExampleVisualization) + (fname ? strlen(fname)+1 : 0) + 
							       (thumb ? strlen(thumb)+1 : 0) + (description ? strlen(description)+1 : 0));
  memset(retval, 0, sizeof(ExampleVisualization));
  char *ptr = (char*)(retval+1);
  if(fname) {
    retval->fname = ptr;
    strcpy(retval->fname, fname);
    ptr += strlen(fname)+1;
  }
  if(thumb) {
    retval->thumb = ptr;
    strcpy(retval->thumb, thumb);
    ptr += strlen(thumb)+1;
  }
  if(description) {
    retval->description = ptr;
    strcpy(retval->description, description);
    ptr += strlen(description)+1;
  }
  retval->loss = loss;

  return retval;
}
