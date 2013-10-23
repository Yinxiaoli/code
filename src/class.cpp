/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#include "class.h"
#include "imageProcess.h"
#include "attribute.h"
#include "classes.h"

ObjectClass::ObjectClass(Classes *c, const char *n) {
  name = n ? StringCopy(n) : NULL;
  classes = c;
  attributeWeights = NULL;
  attributePositiveUserProbs = attributeNegativeUserProbs = NULL;
  attributeUserProbs = NULL;
  id = -1;

  num_exemplar_images = 0;
  exemplar_images = NULL;
  wikipedia_url = NULL;
  weights = NULL;
  num_weights = 0;
}

ObjectClass::~ObjectClass() {
  if(name) StringFree(name);
  if(attributeWeights) free(attributeWeights);
  if(attributePositiveUserProbs) free(attributePositiveUserProbs);
  if(attributeNegativeUserProbs) free(attributeNegativeUserProbs);
  if(attributeUserProbs) free(attributeUserProbs);
  if(wikipedia_url) free(wikipedia_url);
  for(int i = 0; i < num_exemplar_images; i++) free(exemplar_images[i]);
  if(exemplar_images) free(exemplar_images);
  if(weights) free(weights);
};

Json::Value ObjectClass::Save() {
  Json::Value root;
  if(name) root["name"] = name;
  root["id"] = id;
  if(wikipedia_url) root["wikipedia"] = wikipedia_url;
  root["meta"] = meta;

  if(exemplar_images) {
    Json::Value exemplars;
    for(int i = 0; i < num_exemplar_images; i++) 
      exemplars[i] = exemplar_images[i];
    root["exemplars"] = exemplars;
  }

  if(attributeWeights) {
    Json::Value w;
    for(int i = 0; i < classes->NumAttributes(); i++) 
      w[i] = attributeWeights[i];
    root["attributesWeights"] = w;
  }
  if(weights) {
    FILE *fout = classes->GetBinaryWeightsFile();
    if(!fout) {
      Json::Value w;
      for(int i = 0; i < num_weights; i++) 
        w[i] = weights[i];
      root["weights"] = w;
    } else {
      // Save weights in a separate binary file
      long seek = ftell(fout);
      root["weights_seek_u"] = (unsigned int)((seek & 0xFFFFFFFF00000000L)>>32);
      root["weights_seek_l"] = (unsigned int)(seek & 0x00000000FFFFFFFFL);
      root["num_weights"] = num_weights;
      int n = fwrite(weights, sizeof(float), num_weights, fout);
      assert(n == num_weights);
    }
  }

  if(attributePositiveUserProbs && attributeNegativeUserProbs) {
    Json::Value w;
    for(int i = 0; i < classes->NumCertainties(); i++) {
      Json::Value c, p, n;
      c["certainty"] = i;
      for(int j = 0; j < classes->NumAttributes(); j++) {
	p[j] = attributePositiveUserProbs[i][j];
	n[j] = attributeNegativeUserProbs[i][j];
      }
      c["probsPos"] = p;
      c["probsNeg"] = n;
      w[i] = c;
    }
    root["attributeUserProbs"] = w;
  }

  if(attributeUserProbs) {
    Json::Value a;
    for(int j = 0; j < classes->NumAttributes(); j++) 
      a[j] = attributeUserProbs[j];
    root["attributeUserProbsNoCertainty"] = a;
  }

  return root;
}

bool ObjectClass::Load(const Json::Value &root) {
  attributePositiveUserProbs = (float**)malloc(classes->NumCertainties()*(sizeof(float*)+sizeof(float)*classes->NumAttributes()));
  attributeNegativeUserProbs = (float**)malloc(classes->NumCertainties()*(sizeof(float*)+sizeof(float)*classes->NumAttributes()));
  float *aptrP = (float*)(attributePositiveUserProbs+classes->NumCertainties());
  float *aptrN = (float*)(attributeNegativeUserProbs+classes->NumCertainties());
  for(int j = 0; j < classes->NumCertainties(); j++, aptrP += classes->NumAttributes(), aptrN += classes->NumAttributes()) {
    attributePositiveUserProbs[j] = aptrP;
    attributeNegativeUserProbs[j] = aptrN;
    for(int i = 0; i < classes->NumAttributes(); i++) {
      attributePositiveUserProbs[j][i] = attributeNegativeUserProbs[j][i] = 0;
    }
  }

  name = root.isMember("name") ? StringCopy(root.get("name","").asString().c_str()) : NULL;
  id = root.get("id",-1).asInt();
  wikipedia_url = root.isMember("wikipedia") ? StringCopy(root.get("wikipedia","").asString().c_str()) : NULL;
  meta = root["meta"];

  if(root.isMember("exemplars") && root["exemplars"].isArray()) {
    char tmp[1000];
    for(int i = 0; i < (int)root["exemplars"].size(); i++) {
      strcpy(tmp, root["exemplars"][i].asString().c_str());
      exemplar_images = (char**)realloc(exemplar_images, sizeof(char*)*(num_exemplar_images+1));
      exemplar_images[num_exemplar_images++] = StringCopy(tmp);
    }
  }
  if(root.isMember("attributesWeights") && root["attributesWeights"].isArray() && 
     (int)root["attributesWeights"].size()==classes->NumAttributes()) {
    Json::Value a = root["attributesWeights"];
    attributeWeights = (float*)malloc(sizeof(float)*classes->NumAttributes());
    for(int i = 0; i < (int)a.size(); i++) {
      attributeWeights[i] = a[i].asFloat();
    }
  }
  if(root.isMember("weights") && root["weights"].isArray()) {
    Json::Value a = root["weights"];
    num_weights = (int)a.size();
    weights = (float*)malloc(sizeof(float)*num_weights);
    for(int i = 0; i < num_weights; i++) 
      weights[i] = a[i].asFloat();
  } else if(root.isMember("weights_seek_u")) {
    num_weights = root["num_weights"].asInt();
    weights = (float*)malloc(sizeof(float)*num_weights);
    FILE *fin = classes->GetBinaryWeightsFile();
    assert(fin);
    fseek(fin, (long)(((unsigned long)(root["weights_seek_u"].asUInt())<<32) | ((unsigned long)root["weights_seek_l"].asUInt())),  SEEK_SET);
    if(fread(weights, sizeof(float), num_weights, fin) != num_weights) {
      fprintf(stderr, "Error loading feature weights from binary file\n");
      return false;
    }
  }

  if(root.isMember("attributeUserProbs") && root["attributeUserProbs"].isArray()) {
    for(int i = 0; i < (int)root["attributeUserProbs"].size(); i++) {
      Json::Value c = root["attributeUserProbs"][i];
      if(!c.isObject() || !c.isMember("certainty") || 
	 !c.isMember("probsPos") || !c.isMember("probsNeg") ||
	 (int)c["probsPos"].size() != classes->NumAttributes() || 
	 (int)c["probsNeg"].size() != classes->NumAttributes()) {
	fprintf(stderr, "Error parsing attributeUserProbs\n");
	return false;
      }
      int ind = c["certainty"].asInt();
      Json::Value p = c["probsPos"], n = c["probsNeg"];
      int j;
      for(j = 0; j < classes->NumAttributes(); j++) 
	attributePositiveUserProbs[ind][j] = p[j].asFloat();
      for(j = 0; j < classes->NumAttributes(); j++) 
	attributeNegativeUserProbs[ind][j] = n[j].asFloat();
    }
  }

  if(root.isMember("attributeUserProbsNoCertainty") && root["attributeUserProbsNoCertainty"].isArray()) {
    attributeUserProbs = (float*)malloc(sizeof(float)*classes->NumAttributes());
    Json::Value c = root["attributeUserProbsNoCertainty"];
    if((int)c.size() != classes->NumAttributes()) {
      fprintf(stderr, "Error parsing attributeUserProbsNoCertainty\n");
      return false;
    }
    for(int j = 0; j < classes->NumAttributes(); j++) 
      attributeUserProbs[j] = c[j].asFloat();
  }
    
  return true;
}


bool ObjectClass::ResolveLinks(Classes *classes) {
  return true;
}

float ObjectClass::GetAttributeUserProbability(int attr, int v, int cert, float gamma) { 
  float p = v ? attributeUserProbs[attr] : 1-attributeUserProbs[attr]; 
  //return p;
  float pp = p ? exp(gamma*classes->GetCertaintyWeights()[cert]*log(p)) : 0;
  return pp; 
}
