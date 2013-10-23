#ifndef __20Q_SERVER_H
#define __20Q_SERVER_H

#include "question.h"
#include "online_interactive_parts_server.h"

/**
 * @file 20q_server.h
 * @brief Implements a remote procedure call interface for a 20 questions game style interface for interactively
 * identifying the class of an object using a combination of computer vision and part and attribute questions
 */



/**
 * @class Visual20qRpc
 * @brief Implements a remote procedure call interface for a 20 questions game style interface for interactively
 * identifying the class of an object using a combination of computer vision and part and attribute questions
 */
class Visual20qRpc : public PartModelStructuredLearnerRpc {

public:
  Visual20qRpc();

protected:

  /**
   * @brief Parse command line arguments
   * @param argc number of command line arguments
   * @param argv the command line arguments
   */
  void parse_command_line_arguments(int argc, const char **argv);

  /**
   * @brief Register all available methods, making them available over the network as remote procedure calls
   */
  virtual void AddMethods();

  /**
   * @brief Preprocess an image (run sliding window detectors, multiclass classifiers, entropy computations)
   *
   * The parameters to the RPC are:
   *  - session_id: Session id returned by new_session()
   *  - x: String encoding of an example in the format of PartModelStructuredData::read() (the image file name)
   *  - num_classes: The number of classes to return
   *  - show_parts: If true, return the most likely part configuration
   *  - question_select_method: Optional parameter, defining the criterion used to predict which question to ask.  Can be one of 'information_gain' (select most informative question), 'time' (select question that minimize the expected to to identify the true class), 'random' (select a random question)
   *  - debug: Optional parameter, if true generate extra debug information in the form of an html page in the sessions directory
   *  - debug_num_class_print: Optional parameter, the number of top ranked classes to show in the html debug page
   *  - debug_probability_maps: Optional parameter, if true generate image probability maps of where we think parts are located
   *  - debug_click_probability_maps: Optional parameter, if true generate image probability maps of where we think the user would click on for a part click question
   *  - debug_num_samples: Optional parameter, if true generate images depicting random part location samples used by internal algorithms
   *  - debug_question_entropies: Optional parameter, if true display the expected information gain for each question
   *  - debug_max_likelihood_solution: Optional parameter, if true generate an image of the maximum likelihood part locations
   *  - disable_click: Optional parameter, if true disable questions asking the user to click on the location of a part
   *  - disable_binary: Optional parameter, if true disable binary yes/no questions
   *  - disable_multiple: Optional parameter, if true disable multiple choice and multi-select questions
   *  - disable_computer_vision: Optional parameter, if true disable computer vision algorithms
   *  - top_classes: An array of the top ranked classes, of size num_classes.  Each entry contains a pair class_id,prob
   * The return values are:
   *  - parts: A string encoding of the predicted most likely part locations in the format of PartModelStructuredLabel::read() (only return if show_parts=true)
   *  - parts_score: The score of the predicted part locations (only return if show_classes=true)
   *  - session_id: A string encoding of the session id.  The client should pass this as a parameter to all future accesses to x";
   *
   * @param root JSON array storing parameters
   * @param response JSON array into which return values are stored
   */
  bool Preprocess(const Json::Value& root, Json::Value& response);

  /**
   * @brief Provide the answer to a question, causing the class probabilities and part location probabilities to change
   *
   * The parameters to the RPC are:
   *  - session_id: Session id returned by new_session()
   *  - question_id: The id of the question being answered";
   *  - answer: 0 or 1 for binary questions, an array of 0 or 1 values for multi-select questions, or a struct with values x,y,scale,orientation,pose for part click questions
   *  - certainty: The certainty (confidence level) of the user's response
   *  - response_time: The time in seconds of the user's response
   *  - num_classes: The number of classes to return, in the ranked list of most likely classes
   *  - show_parts: If true, return the most likely part configuration
   *  
   * The return values are:
   *  - parts: A string encoding of the predicted most likely part locations in the format of PartModelStructuredLabel::read() (only return if show_parts=true)
   *  - parts_score: The score of the predicted part locations (only return if show_classes=true)
   *  - session_id: A string encoding of the session id.  The client should pass this as a parameter to all future accesses to x";
   *
   * @param root JSON array storing parameters
   * @param response JSON array into which return values are stored
   */
  bool AnswerQuestion(const Json::Value& root, Json::Value& response);

  /**
   * @brief Get the next question to ask (typically the question with highest expected information gain)
   *
   * The parameters to the RPC are:
   *  - session_id: Session id returned by new_session()
   *  
   * The return values are:
   *  - question_id: The id of the question to ask
   *  - session_id: A string encoding of the session id.  The client should pass this as a parameter to all future accesses to x";
   *
   * @param root JSON array storing parameters
   * @param response JSON array into which return values are stored
   */
  bool NextQuestion(const Json::Value& root, Json::Value& response);

  /**
   * @brief The user wants to directly say 
   *
   * The parameters to the RPC are:
   *  - session_id: Session id returned by new_session()
   *  - class_id: The id of the class being labeled
   *  - answer: 0 or 1, where 1 indicates the class is the true class
   *  - num_classes: The number of classes to return, in the ranked list of most likely classes
   *  - show_parts: If true, return the most likely part configuration
   *  
   * The return values are:
   *  - parts: A string encoding of the predicted most likely part locations in the format of PartModelStructuredLabel::read() (only return if show_parts=true)
   *  - parts_score: The score of the predicted part locations (only return if show_classes=true)
   *  - session_id: A string encoding of the session id.  The client should pass this as a parameter to all future accesses to x";
   *
   * @param root JSON array storing parameters
   * @param response JSON array into which return values are stored
   */
  bool VerifyClass(const Json::Value& root, Json::Value& response);

private:
  void ReturnClassesIfNecessary(QuestionAskingSession *session, const Json::Value& root, Json::Value& response);
};


#endif
