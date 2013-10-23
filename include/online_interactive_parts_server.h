#ifndef __ONLINE_INTERACTIVE_PARTS_SERVER_H
#define __ONLINE_INTERACTIVE_PARTS_SERVER_H

#include "util.h"

class Classes;
class QuestionAskingSession;

#define HACK_FOR_JOURNAL_PAPER_USER_STUDY

#ifdef HACK_FOR_JOURNAL_PAPER_USER_STUDY
bool ParseUserStudyResults(int user, int image, int predClass, Classes *classes, Json::Value& response, float sessionTime, QuestionAskingSession *session=NULL, int setFound=-1);
#endif

#include "online_interactive_server.h"
#include "structured_svm_partmodel.h"
#include "interactiveLabel.h"

/**
 * @file online_interactive_parts_server.h
 * @brief Enables interactive labeling and online learning of deformable part models, with a network interface for online learning and active labeling.
 */

/**
 * @class PartModelStructuredLearnerRpc
 * @brief Enables interactive labeling and online learning of deformable part models.
 *
 * This mostly just uses the same general purpose structured learning and active labeling
 * routines inherited from StructuredLearnerRpc, but also enables special routines for a realtime
 * interactive labeling interface specialized for deformable part models, in which parts are sequentially
 * labeled one at a time.  The network protocol is something like:
 *  - Get the initial max-likelihood part locations where image_filename is the path to a new image, and part_locations_string1 is in the form of PartLocalizedStructuredLabel::write()
\htmlonly <div style="padding: 0.5em 1em; border: 1px solid #aaf; background-color: #dadafa;">
<font color="blue">Client:</font> {"jsonrpc":"2.0","id":1,"method":"classify_example","x":"image_filename"}
</div> \endhtmlonly
\htmlonly <div style="padding: 0.5em 1em; border: 1px solid #faa; background-color: #fadada;">
<font color="red">Server:</font> {"id":1,"session_id":session_id,"y":"part_locations_string1"}
</div><br> \endhtmlonly 
 *
 *  - As the user drags the part part_id1 to pixel location (x,y), interactively display the max-likelihood solution conditioned on that part location: 
\htmlonly <div style="padding: 0.5em 1em; border: 1px solid #aaf; background-color: #dadafa;">
<font color="blue">Client:</font> {"jsonrpc":"2.0","id":2,"method":"preview_part_locations","session_id":"session_id","part":part_id1,"x":x,"y":y}
</div> \endhtmlonly
\htmlonly <div style="padding: 0.5em 1em; border: 1px solid #faa; background-color: #fadada;">
<font color="red">Server:</font> {"id":2,"session_id":"session_id","y":"part_locations_string2"}
</div> \endhtmlonly
 * The user may keep dragging part_id1, producing a series of requests in the same format as above \htmlonly <br> \endhtmlonly
 *
 *  - When the user releases part_id1, finalize its position at (x,y) and receive the max-likelihood solution conditioned on that part location:
\htmlonly <div style="padding: 0.5em 1em; border: 1px solid #aaf; background-color: #dadafa;">
<font color="blue">Client:</font> {"jsonrpc":"2.0","id":3,"method":"label_part","session_id":"session_id","part":"part_id1,,"x":x,"y":y}
</div> \endhtmlonly
\htmlonly <div style="padding: 0.5em 1em; border: 1px solid #faa; background-color: #fadada;">
<font color="red">Server:</font> {"id":3,"session_id":"session_id","y":"part_locations_string3"}
</div> \endhtmlonly 
 * The user can click and drag other parts, producing similar command sequences as the two commands above \htmlonly <br> \endhtmlonly
 *
 *  - The client asks the server to add a new training example with the verified set of part locations, the server sends back the index of the added training example: 
\htmlonly <div style="padding: 0.5em 1em; border: 1px solid #aaf; background-color: #dadafa;">
<font color="blue">Client:</font> {"jsonrpc":"2.0","id":4,"method":"add_example","session_id":"session_id","y":"part_locations_string4"}
</div> \endhtmlonly
\htmlonly <div style="padding: 0.5em 1em; border: 1px solid #faa; background-color: #fadada;">
<font color="red">Server:</font> {"id":4,"session_id":"session_id","index":"ind"}
</div> <br> \endhtmlonly
 * 
 * There are many other network commands possible that are inherited from StructuredLearnerRpc
 */
class PartModelStructuredLearnerRpc : public StructuredLearnerRpc {
 protected:
  /// @cond
  Json::Value *classesJSON, *partsJSON, *posesJSON, *attributesJSON, *certaintiesJSON, *questionsJSON;
  Classes *classes;
  /// @endcond

 public:
  PartModelStructuredLearnerRpc();
  ~PartModelStructuredLearnerRpc();

 /**
   * @brief Create a new PartModelStructuredLearnerRpc
   * @param s the PartModelStructuredSVM to use for training/testing
   */
  PartModelStructuredLearnerRpc(PartModelStructuredSVM *s) : StructuredLearnerRpc(s) { Init(); };
  void Init();

 protected:
  virtual void parse_command_line_arguments(int argc, const char **argv);
  virtual void AddMethods();

  /**
   * @brief Preprocess the image (run part detectors)
   *
   * The parameters to the RPC are:
   *  - session_id: Session id returned by new_session()
   *  - x: String encoding of an example in the format of PartLocalizedStructuredData::read() (the image file name)

   * The return values are:
   *  - y: A string encoding of the predicted label y in the format of PartLocalizedStructuredLabel::read()
   *  - score: The score of the predicted label y
   *  - session_id: A string encoding of the session id.  The client should pass this as a parameter to all future accesses to x";
   *
   * @param root JSON array storing parameters
   * @param response JSON array into which return values are stored
   */
  bool Preprocess(const Json::Value& root, Json::Value& response);

  /**
   * @brief The user should call this function each time they finish labeling a new part
   *
   * The parameters to the RPC are:
   *  - session_id: Session id returned by new_session()
   *  - part: The name of the part being labeled
   *  - x: The x pixel location of the center of the part
   *  - y: The y pixel location of the center of the part
   *  - scale: Optional scale of the part
   *  - rotation: Optional rotation of the part
   *  - pose: Optional pose name of the part
   *
   * The return values are:
   *  - y: A string encoding of the predicted label y in the format of PartLocalizedStructuredLabel::read()
   *  - score: The score of the predicted label y
   *  - session_id: A string encoding of the session id.  The client should pass this as a parameter to all future accesses to x";
   *
   * @param root JSON array storing parameters
   * @param response JSON array into which return values are stored
   */
  bool LabelPart(const Json::Value& root, Json::Value& response);

  /**
   * @brief The user should call this function when they want to preview the maximum likelihood solution of all 
   * parts when one new part is moved.
   *
   * The parameters to the RPC are:
   *  - session_id: Session id returned by new_session()
   *  - part: The name of the part being labeled
   *  - x: The x pixel location of the center of the part
   *  - y: The y pixel location of the center of the part
   *  - scale: Optional scale of the part
   *  - rotation: Optional rotation of the part
   *  - pose: Optional pose name of the part
   *
   * The return values are:
   *  - y: A string encoding of the predicted label y in the format of PartLocalizedStructuredLabel::read()
   *  - score: The score of the predicted label y
   *  - session_id: A string encoding of the session id.  The client should pass this as a parameter to all future accesses to x";
   *
   * @param root JSON array storing parameters
   * @param response JSON array into which return values are stored
   * 
   */
  bool PreviewPartLocations(const Json::Value& root, Json::Value& response);

  /**
   * @brief The user should call this function when they want to preview the maximum likelihood solution of all 
   * parts when one new part is moved.
   *
   * The parameters to the RPC are:
   * - classes: If true, get the list of all classes";
   * - questions: If true, get the list of all questions";
   * - parts: If true, get the list of all parts";
   * - poses: If true, get the list of all poses";
   * - certainties: If true, get the list of all certainties";
   * - attributes: If true, get the list of all attributes";
   *
   * The return values are:
   * - classes: An array of classes structs
   * - questions: An array of questions structs
   * - parts: An array of part structs
   * - poses: An array of poses structs
   * - certainties: An array of certainties structs
   * - attributes: An array of attributes structs
   *
   * @param root JSON array storing parameters
   * @param response JSON array into which return values are stored
   * 
   */
  bool GetDefinitions(const Json::Value& root, Json::Value& response);

  /// @cond
  bool GetClasses(const Json::Value& root, Json::Value& response);
  bool GetQuestions(const Json::Value& root, Json::Value& response);
  bool GetParts(const Json::Value& root, Json::Value& response);
  bool GetPoses(const Json::Value& root, Json::Value& response);
  bool GetAttributes(const Json::Value& root, Json::Value& response);
  bool GetCertainties(const Json::Value& root, Json::Value& response);
  /// @endcond

  float ***all_group_points;
  float nonVisibleCost;
  bool useMirroredPoses;
  Dataset *dataset;

  PartLocalizedStructuredData *FindNearestExampleByPose(PartLocation *locs, int restrictClass, int w, int h);

 private:
  bool ComputeClasses();
  bool ComputeQuestions();
  bool ComputeParts();
  bool ComputePoses();
  bool ComputeAttributes();
  bool ComputeCertainties();

  void BuildDatasetPoseCache();
};

class InteractivePartsData : public PartLocalizedStructuredData {
  InteractiveLabelingSession *session;
public:
  InteractivePartsData() { session = NULL; }
  void SetSession(InteractiveLabelingSession *s) { session = s; }
  InteractiveLabelingSession *GetSession() { return session; }
  virtual ~InteractivePartsData() { if(session) delete session; session = NULL; }
};

#endif
