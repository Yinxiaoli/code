/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/

#ifndef __CLASS_H
#define __CLASS_H

#include "util.h"

/**
 * @file class.h
 * @brief An object category class
 */

class Classes;

/**
 * @class ObjectClass
 * 
 * @brief An object class, e.g. a particular species of bird
 *
 */
class ObjectClass {
  char *name;  /**< The name of the class */
  int id;  /**< The id of the class. This should be an index into classes->classes */
  Classes *classes; /**< Pointer to a Classes object, which defines the set of possible classes, parts, and attributes */

  float *attributeWeights;  /**< numAttributes size array indicating class attribute memberships */
  float *weights;
  int num_weights;

  char **exemplar_images;
  int num_exemplar_images;
  char *wikipedia_url;

  Json::Value meta;

  /**
   * numCertaintiesXnumAttributes size array indicating the percentage of time a user indicates that this 
   * class has each attribute.  p(a,r|c) probability of an attribute response and certainty level for a given class
   */
  float **attributePositiveUserProbs;  

  /**
   * numCertaintiesXnumAttributes size array indicating the percentage of time a user indicates that this 
   * class has each attribute.  p(^a,r|c) probability of an attribute response and certainty level for a given class
   */
  float **attributeNegativeUserProbs;  

  /**
   * numAttributes size array indicating the percentage of time a user indicates that this 
   * class has each attribute  p(a|c).  Unlike attributePositiveUserProbs and attributeNegativeUserProbs,
   * this ignores certainty level
   */
  float *attributeUserProbs;

public:

  /**
   * @brief Constructor
   * 
   * @param classes  The classes definition file, which defines the set of possible classes, parts, and attributes
   * @param name If non-null, the name of this class e.g. "Blue Jay"
   */
  ObjectClass(Classes *classes, const char *name=NULL);

  /**
   * @brief Destructor
   */
  ~ObjectClass();

  void SetWeights(float *w, int n) {
    weights = (float*)realloc(weights, sizeof(float)*n);
    memcpy(weights, w, n*sizeof(float));
    num_weights = n;
  }
  float *GetWeights() { return weights; }

  /**
   * @brief Load definition of this class from a JSON object
   * 
   * @param root A JSON encoding of this class
   * @return true if successful
   */
  bool Load(const Json::Value &root);

  /**
   * @brief Save a definition of this class to a JSON object
   * 
   * @return A JSON encoding of this class
   */
  Json::Value Save();
  
  /**
   * @brief Check if this class has an attribute
   * 
   * @param i The attribute index
   * @return The attribute weight, where 1 indicates presence and -1 indicates absence
   */
  float GetAttributeWeight(int i) { return attributeWeights[i]; }

  /**
   * @brief Set the array of attribute weights, where a weight 1 indicates presence and -1 indicates absence
   * @param w a NumAttributes() array of attribute weights
   */
  void SetAttributeWeights(float *w) { attributeWeights = w; }

  /**
   * @brief Get the class id
   * 
   * @return The id of this class
   */
  int Id() { return id; }

  /**
   * @brief Get the name of this class
   * 
   * @return The name of this class (e.g. "Blue Jay")
   */
  char *Name() { return name; }

  /**
   * @brief Set the name of this class
   * 
   * @param n The name of this class (e.g. "Blue Jay")
   */
  void SetName(const char *n) { name=StringCopy(n); }

  /**
   * @brief Get the user response probabilities for answering whether this class has this attribute
   * 
   * @return a numCertaintiesXnumAttributes size array indicating the percentage of time a user indicates that this 
   * class has each attribute.  p(a,r|c) probability of an attribute response and certainty level for a given class
   */
  float **GetAttributePositiveUserProbs() { return attributePositiveUserProbs; }

  /**
   * @brief Get the user response probabilities for answering whether this class does not have this attribute
   * 
   * @return a numCertaintiesXnumAttributes size array indicating the percentage of time a user indicates that this 
   * class does not have each attribute.  p(^a,r|c) probability of an attribute response and certainty level for a given class
   */
  float **GetAttributeNegativeUserProbs() { return attributeNegativeUserProbs; }

  
  /**
   * @brief Get the filename of the ith exemplar image for this class
   * @param i an index between 0 an NumExemplarImages()-1
   */
  char *GetExemplarImageName(int i) { return exemplar_images[i]; }

  /**
   * @brief Get the array of filenames of exemplar images for this class
   * @return a NumExemplarImages() array of exemplar image names for this class
   */
  char **GetExemplarImages() { return exemplar_images; }

  /**
   * @brief Get the number of exemplar images for this class
   */
  int NumExemplarImages() { return num_exemplar_images; }

  /**
   * @brief Set the array of filenames of exemplar images for this class
   * @param e an array of n exemplar image names for this class
   * @param n the number of exemplar images
   */
  void SetExemplarImages(char **e, int n) { 
    exemplar_images=(char**)malloc(sizeof(char*)*n); 
    for(int i=0; i < n; i++)
      exemplar_images[i] = StringCopy(e[i]);
    num_exemplar_images=n; 
  }

  /**
   * @brief Get the name of the wikipedia article corresponding to this object class
   */
  char *GetWikipediaUrl() { return wikipedia_url; }

  /**
   * @brief Set the name of the wikipedia article corresponding to this object class
   * @param w the wikipedia article url
   */
  void SetWikipediaArticle(const char *w) { wikipedia_url = StringCopy(w); }


  /**
   * @brief Set the array of attribute user probabilities, which defines for each attribute the probability that a user would say this class has that attirbute
   * @param a an array of size NumAttributes() of attribute user probabilities
   */
  void SetAttributeUserProbabilities(float *a) { attributeUserProbs = a; }

  /**
   * @brief Get the array of attribute user probabilities, which defines for each attribute the probability that a user would say this class has that attribute
   */
  float *GetAttributeUserProbabilities() { return attributeUserProbs; }

  /**
   * @brief Get the probability that a user would say this class has an attribute for a given certainty level
   * @param attr the attribute index 
   * @param v 1 indicates attribute presence, 0 indicates absence
   * @param cert the certainty level
   * @param gamma constant used for probability calculations 
   */
  float GetAttributeUserProbability(int attr, int v, int cert, float gamma=1);

  void AddMeta(const char *name, const char *val) { meta[name] = val; }
  const char *GetMeta(const char *name) { return meta.get(name,"").asString().c_str(); }


  Classes *GetClasses() { return classes; }
  void SetClasses(Classes *c) { classes = c; }

 private:
  /**
   * @brief Resolve pointers to other objects, parts, or attributes
   *
   * Typically, this is called after all classes, parts, and attribute definitions have been loaded
   * 
   * @param classes Classes definition object
   * @return true on success
   */
  bool ResolveLinks(Classes *classes);

  /**
   * @brief Set the id of this class
   * 
   * @param i The new class id
   */
  void SetId(int i) { id = i; }

  friend class Classes;



 public:
  //deprecated
  /// @cond
  char *LoadFromString(char *str);
  char *ToString(char *str);
  bool ResolveLinksOld(Classes *c);
  /// @endcond
};


#endif
