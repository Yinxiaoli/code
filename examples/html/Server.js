;(function ( namespace, undefined ) {  
    
    var start_time = null;
    
    var classes = null, parts = null, poses = null, attributes = null, certainties = null;
    var session_id = null, session_dir = null, question_id = null;
    var num_preview_classes=200;

    var user_study_id = -1;
    var user_study_image_id = -1;
    var user_study_next_image_url = null;
    var showScore = true;
    var totalLoss = 0;
    var debugPages = "";
    var user_study_start_image = -1;
	
    var requestURL = "request.php";

    var htmlDir = "examples/html";

    var debug_json = false;

	var question = null;
      
	var question_select_method="information_gain";
	var disable_computer_vision=false;
	var disable_binary=true;
	var disable_click=false;
	var disable_multiple=false;
	
        var debug_keep_big_images = false;
	var debug=false;
	var debug_num_class_print=10;
	var debug_num_samples=0;
	var debug_probability_maps=true;
	var debug_click_probability_maps=false;
	var debug_max_likelihood_solution=true;
	var debug_question_entropies=false;

	namespace.GetSessionID = function(){
		return session_id;
	}
	namespace.SetSessionID = function(id){
		session_id = id;
	}
	namespace.GetSessionDir = function(){
		return session_dir;
	}
	namespace.SetSessionDir = function(dir){
		session_dir = dir;
	}

     namespace.ParseArguments = function() {
	var query = window.location.search.substring(1);
	var pairs = query.split("&");

	for (var i=0;i<pairs.length;i++) {
            var pos = pairs[i].indexOf('=');
            if (pos >= 0) {
		var argname = pairs[i].substring(0,pos);
		var value = pairs[i].substring(pos+1);
		if(argname.indexOf("debug_num_samples") == 0) debug_num_samples = parseInt(value);
		else if(argname.indexOf("debug_click_probability_maps") == 0) debug_click_probability_maps = value == 'true' || value == '1';
		else if(argname.indexOf("debug_max_likelihood_solution") == 0) debug_max_likelihood_solution = value == 'true' || value == '1';
		else if(argname.indexOf("debug_probability_maps") == 0) debug_probability_maps = value == 'true' || value == '1';
		else if(argname.indexOf("debug_num_class_print") == 0) debug_num_class_print = parseInt(value);
		else if(argname.indexOf("debug_question_entropies") == 0) debug_question_entropies = value == 'true' || value == '1';
		else if(argname.indexOf("debug_keep_big_images") == 0) debug_keep_big_images = value == 'true' || value == '1';
		else if(argname.indexOf("debug") == 0) debug = value == 'true' || value == '1';
		else if(argname.indexOf("disable_binary") == 0) disable_binary = value == 'true' || value == '1';
		else if(argname.indexOf("disable_click") == 0) disable_click = value == 'true' || value == '1';
		else if(argname.indexOf("disable_computer_vision") == 0) disable_computer_vision = value == 'true' || value == '1';
		else if(argname.indexOf("disable_multiple") == 0) disable_multiple = value == 'true' || value == '1';
		else if(argname.indexOf("select_by") == 0) question_select_method = value;
		else if(argname.indexOf("user_id") == 0) {
		    user_study_id = parseInt(value);
		    window.Log.LoggingEnabled = true;
		} else if(argname.indexOf("image_start") == 0) user_study_start_image = parseInt(value);
            }
	}
    };
 
    namespace.IsUserStudy = function() { return user_study_id >= 0; }
	
	function JSONRPCRequest(JSONrequest, callback) {
		var args = new Array();
		args[0] = new Array();
		args[0].name = "json";
		args[0].value = escape(JSONrequest);
		httpRequest("GET", requestURL, true, args, function(JSONtext) {
			if(callback){
				callback(JSONtext);
			}
		});
	}

    // Read all class and question definitions from the server 
    namespace.LoadDefinitions = function() {
	var params = ',"classes":true,"questions":true,"certainties":true,"parts":true';
	if(user_study_id >= 0) {
	    params += ',"user_study_id":' + user_study_id;
	    if(user_study_start_image >= 0) params += ',"user_study_start_image":' + user_study_start_image;
	}
		JSONRPCRequest('{"method":"get_definitions","jsonrpc":"2.0"' + params + '}', function(JSONtext) {
			var definitions = eval('(' + JSONtext + ')');
			classes = definitions['classes'];
			parts = definitions['parts'];
			poses = definitions['poses'];
			attributes = definitions['attributes'];
			certainties = definitions['certainties'];
			questions = definitions['questions'];

		        window.Server.SynchDebugDiv();
		    
		    if('user_study_image_id' in definitions)
			user_study_image_id = definitions['user_study_image_id'];
		    if('nextImage' in definitions)
			UploadURL(definitions['nextImage']);
		});
    }

    // Ask the server to start a new session for the image 'image_name', causing initial computer vision preprocessing to occur
    namespace.StartSession = function() {
    	
    	if(window.Log.LoggingEnabled){
			window.Log.BeginningSession();
		}
    
		JSONRPCRequest('{"method":"new_session","jsonrpc":"2.0","mkdir":true}', function(JSONtext) {
			var res = eval('(' + JSONtext + ')');
			top_classes = res['top_classes'];
			session_id = res['session_id'];
			session_dir = res['session_dir'];
			UploadImage(); // delegates to PrepocessImage after upload finishes
		}); 
    }

    

    // Ask the server to start a new session for the image 'image_name', causing initial computer vision preprocessing to occur
    namespace.PreprocessImage = function(c, b) {
    	
    	if(window.Log.LoggingEnabled){
			window.Log.SendStartSessionInfo();
		}
    	
    	// Tell the Question to reset
		window.Question.reset()
		
		$("#upload-status-text").html("Preprocessing Image...");
		$("#upload-animation").fadeIn(125);
		
		window.UserImage.setSource(window.UserImage.getImageName());
		
	        if(debug) this.ShowDebugDiv();
		

		var req = {};
		var x = {};
		
		x["imagename"] = htmlDir + '/' + window.UserImage.getImageName();
		req["jsonrpc"] = "2.0";
		req["method"] = "initialize_20q";
		req["session_id"] = session_id;
		req["num_classes"] = num_preview_classes;
		req["question_select_method"] = question_select_method;
		req["x"] = x;
		req["debug"] = debug || user_study_id >= 0;
		req["debug_click_probability_maps"] = debug_click_probability_maps;
		req["debug_max_likelihood_solution"] = debug_max_likelihood_solution;
		req["debug_probability_maps"] = debug_probability_maps;
		req["debug_num_class_print"] = debug_num_class_print;
		req["debug_num_samples"] = debug_num_samples;
		req["debug_question_entropies"] = debug_question_entropies;
		req["debug_keep_big_image"] = debug_keep_big_images;
		req["disable_binary"] = disable_binary;
		req["disable_click"] = disable_click;
		req["disable_computer_vision"] = disable_computer_vision;
		req["disable_multiple"] = disable_multiple;
		
		JSONRPCRequest(JSON.stringify(req), function(JSONtext) {
			var res = eval('(' + JSONtext + ')');
			if(!res['top_classes']) {
				alert('Bad response to PreprocessImage(): ' + JSONtext);
			} 
			else {
				$("#upload-status-text").html("Preprocessing Finished");
				$("#upload-animation").hide();
				$("#upload-box").hide();
				$("#canvas").fadeIn(500);
				top_classes = res['top_classes'];
				
				if(window.Log.LoggingEnabled){
					window.Log.SendResultsInfo(top_classes.map(function(c){return parseInt(c.class_id);}));
				}
				
				window.Results.UpdateTopClasses(classes, top_classes);
				GetNextQuestion();
				if(c){
					c.innerHTML = b;
				}
			}
		});
    }
	
	namespace.RemoveClassFromConsideration = function(classId){
		
		if(window.Log.LoggingEnabled){
			window.Log.SendRemovalInfo(classId);
		}

	    this.VerifyClass(classId, 0);
	}
	namespace.VerifyClass = function(classId, answer){
		
	    var params = ',"session_id":"'+session_id+'", "answer" : ' + answer + ', "class_id" : ' + classId + ', "num_classes" : ' + num_preview_classes;
	    if(user_study_id >= 0)
		params += ',"user_study_id":' + user_study_id + ',"user_study_image_id":' + user_study_image_id;

		JSONRPCRequest('{"jsonrpc":"2.0","method":"verify_class"' + params + '}', function(JSONtext) {
			var res = eval('(' + JSONtext + ')');
			if(res['top_classes']) {
				window.Question.increment(); // WHY increment here?
				top_classes = res['top_classes'];
				
				if(window.Log.LoggingEnabled){
					window.Log.SendResultsInfo(top_classes.map(function(c){return parseInt(c.class_id);}));
				}
				
				window.Results.UpdateTopClasses(classes, top_classes);
			}
		    
		    if('user_study_image_id' in res)
			user_study_image_id = res['user_study_image_id'];
		    if('loss' in res) 
			window.Server.ShowScorePage(('nextImage' in res) ? res['nextImage'] : null, res["taxonomicalLoss"], res["sessionTime"],
					   res["loss"], res["predictedClass"], res["trueClass"], res["timeFactor"], res["type"], res["sumLoss"]);
		});
	
	}
	
    // Ask the server to compute which question to query the user
    function GetNextQuestion() {
		JSONRPCRequest('{"jsonrpc":"2.0","method":"next_question","session_id":"'+session_id+'"}', function(JSONtext) {
			var res = eval('(' + JSONtext + ')');
			if(res['question_id'] < 0) {
				//alert('Bad response to GetNextQuestion(): ' + JSONtext);
			} 
			else {
				question_id = parseInt(res['question_id']);
				question = questions[question_id];
				
				if(window.Log.LoggingEnabled){
					window.Log.SendStartQuestionInfo(question_id, question['type']);
				}
				
				window.Question.UpdateQuestion(question, certainties);
			}
		});
    }

    // Submit an answer to the current question
    namespace.SubmitAnswer = function(certainty) {
		$("#questionDiv").fadeOut(500, function() {});
		
		if(window.Log.LoggingEnabled){
			if(question['type'] == 'binary' || question['type'] == 'multiple_choice' || question['type'] == 'batch') {
				window.Log.SendEndQuestionInfo(question_id,  question['type'], window.Question.getSelected(), certainty);
			}
			else{
				window.Log.SendEndQuestionInfo(question_id, question['type'], [window.UserImage.getPositionX(), window.UserImage.getPositionY()], certainty);
			}
		}
		
		answer = {};
		answer['method'] = "answer_question";
		answer['session_id'] = session_id;
		answer['jsonrpc'] = '2.0';
		answer['question_id'] = question_id;
		answer['response_time'] = EndTiming();
		answer['num_classes'] = num_preview_classes;
		if(question['type'] == 'binary') {
			answer['answer'] = window.Question.getSelected();
			answer['certainty'] = certainty;
		} 
		else if(question['type'] == 'multiple_choice' || question['type'] == 'batch') {
			answer['answers'] = window.Question.getSelected();
			answer['certainty'] = certainty;
		} 
		else if(question['type'] == 'part_click') {
			answer['x'] = window.UserImage.getPositionX(); 
			answer['y'] = window.UserImage.getPositionY(); 
			answer['visible'] = certainty; 
			window.Question.incrementClickQuestions();
		}
		json_args = JSON.stringify(answer);
		
		JSONRPCRequest(json_args, function(JSONtext) {
			var res = eval('(' + JSONtext + ')');
			top_classes = res['top_classes'];
			
			if(window.Log.LoggingEnabled){
				window.Log.SendResultsInfo(top_classes.map(function(c){return parseInt(c.class_id);}));
			}
			
			window.Question.increment();
			window.Results.UpdateTopClasses(classes, top_classes);
			GetNextQuestion();
		});
    }

    function OnMouseExit() { 
      window.UserImage.setSource(window.UserImage.getImageName());
    }

    namespace.Debug = function() {
	return debug;
    }

    namespace.SynchDebugDiv = function() {
      debugDiv = document.getElementById('debugDiv'); 
      debugLink = document.getElementById('debugLink'); 
      if(debug) {
        if(debug_max_likelihood_solution) {
          var mlLink = document.createElement('a');
          var mlLinkText=document.createTextNode('Max Likelihood Solution');
          mlLink.appendChild(mlLinkText);
          mlLink.className = "debugLink";
          mlLink.onmouseover = function() {
	    window.UserImage.setSource(session_dir + "/" + session_id + "_ml_q" + window.Question.getNumClickQuestions() + ".png");
          };
          mlLink.onmouseout = OnMouseExit;
          debugDiv.appendChild(document.createTextNode("  "));
          debugDiv.appendChild(mlLink);
        }
        if(debug_probability_maps) {
          for(var i = 0; i < parts.length; i++) {
            var partLink = document.createElement('a');
            var partLinkText=document.createTextNode(parts[i]["part_name"]);
            partLink.appendChild(partLinkText);
            partLink.i = i;
            partLink.className = "debugLink";
            partLink.onmouseover = function() {
              window.UserImage.setSource(session_dir + "/" + session_id + "_" + window.Question.getNumClickQuestions() + "_" + parts[this.i]["part_name"].replace(/ /g,"_") + "_heat.png");
            }
            partLink.onmouseout = OnMouseExit;
            debugDiv.appendChild(document.createTextNode("  "));
            debugDiv.appendChild(partLink);
          }
        }
      }
    }

    namespace.ShowDebugDiv = function() {
        debugDiv.style.visibility = 'visible';
        debugLink.style.visibility = 'visible';
        var base = session_dir + "/" + session_id;
        debugLink.href = session_dir + "/" + session_id/*base.replace(/\//g,"_")*/ + ".html";
    }
	
	function initReq(request, reqType, url, bool, args, callback) {
      try {
        var argStr = "";
        if(args) {
            for(var i = 0; i < args.length; i++) {
                argStr += (i ? "&" : "") + args[i].name + "=" + args[i].value;
            }
        }
        if(reqType.toLowerCase() != "post" && argStr.length > 0)
          url += "?" + argStr;
        request.open(reqType, url, bool);
        //alert("here2");

        // Specify the function that will handle the HTTP response
        request.onreadystatechange = function() {
            if (request.readyState == 4) {
                if (request.status == 200) {
                    if(callback) callback(request.responseText);
                    else cookie = request.responseText;
                } else {
                    alert("Status " + request.status + " returned editing page " + url);
                    if(callback) callback(null);
                }
            }
        };

        // If the reqType param is POST, then the
        // fifth argument to the function is the POSTed data
        if (reqType.toLowerCase() == "post") {
            // Set the Content-Type header for a POST request
            request.setRequestHeader("Content-Type", "application/x-www-form-urlencoded; charset=UTF-8");
            var postContent = "";//"\r\n--" + boundary + "\r\n";
            postContent += argStr;
            request.send(postContent);
        } else {
            request.send(null);
        }
        if(!bool)
            cookie = request.responseText;
      } catch (errv) {
        alert("The application cannot contact the server " + url + " at the moment. " +
              "Please try again in a few seconds.\n" +
              "Error detail: " + errv.message);
      }
    };

    function httpRequest(reqType,url,asynch,args, callback) {
      var request = null;

      // Mozilla-based browsers
      if (window.XMLHttpRequest) {
        request = new XMLHttpRequest();
      } else if (window.ActiveXObject) {
        request = new ActiveXObject("Msxml2.XMLHTTP");
        if (!request) {
            request = new ActiveXObject("Microsoft.XMLHTTP");
        }
      }

      // Request could still be null if neither ActiveXObject
      // initialization succeeded
      if (request) {
        initReq(request, reqType, url, asynch, args, callback);
        return;
      } else {
        alert("Your browser does not permit the use of all " +
              "of this application's features!");
      }  
      callback(null);
    }
    
    function GetCurrentTime() {
      var my_current_timestamp = new Date();
      return my_current_timestamp.getTime();
    }

    namespace.StartTiming = function() {
      start_time = GetCurrentTime();
    }

    function EndTiming() {
      var end_time = GetCurrentTime();
      var time_difference = (end_time - start_time) / 1000;
      return time_difference;
    }

    namespace.ShowScorePage = function(nextImgUrl, taxoLoss, sessTime, loss, predClass, trueClass, timeFactor, type, sumLoss) {
	var header = document.getElementById('scorePageHeader');
	var score = document.getElementById('scorePageComputation');
	var str = 'Predicted ' + classes[predClass]['class_name'].replace(/\_/g," ") + ' when the true class is ' + classes[trueClass]['class_name'].replace(/\_/g," ") + ' (taxonomical distance of ' + taxoLoss + ')' + ' in ' + sessTime + ' seconds!'

	if(type === "control") {
	    totalLoss += loss;
	    var d = session_dir + "/" + session_id + ".html";
	    debugPages += '<br><a href="' + d + '" target="_blank">' + str + '</a>';
	}
	if(showScore) {
	    if(predClass == trueClass) header.innerHTML = 'Predicted ' + classes[trueClass]['class_name'].replace(/\_/g," ") + ' correctly in ' + sessTime + ' seconds!'; 
	    else header.innerHTML = str; 
	
	    score.innerHTML = 'Your loss was ' + loss + ', computed as ' + taxoLoss + 'bits + ' + sessTime + 'sec / ' + timeFactor + 'sec/bits';
	} else {
	    header.innerHTML = 'Predicted ' + classes[predClass]['class_name'].replace(/\_/g," ") + ' in ' + sessTime + ' seconds!'; 
	    score.innerHTML = "We won't tell you yet if you're right";
	}
	user_study_next_image_url = nextImgUrl;
	$("#scorePageDiv").fadeIn(500);

	var img1 = document.getElementById('scorePagePredImg');
	img1.src = classes[predClass]['class_images'][0];
	var c1 = document.getElementById('scorePagePredClass');
	c1.innerHTML = classes[predClass]['class_name'].replace(/\_/g," ");

	var img2 = document.getElementById('scorePageTrueImg');
	img2.style.display = showScore ? 'block' : 'none';
	img2.src = classes[trueClass]['class_images'][0];
	var c2 = document.getElementById('scorePageTrueClass');
	c2.style.display = showScore ? 'block' : 'none';
	c2.innerHTML = classes[trueClass]['class_name'].replace(/\_/g," ");

	var next = document.getElementById('scorePageNextImageButton');
	next.style.display = nextImgUrl ? 'block' : 'none';

	var finished = document.getElementById('scorePageFinished');
	finished.innerHTML = "<h1>All Done! Your Total Loss Was " + (user_study_start_image >= 0 ? sumLoss : totalLoss) + '</h1><p style="font-size:60%">' + debugPages + '</p>'; 
	finished.style.display = !nextImgUrl ? 'block' : 'none';
    }

   namespace.ScorePageNextImage = function() {
       $("#scorePageDiv").fadeOut(500);
       UploadURL(user_study_next_image_url);
   }
     
})(window.Server = window.Server || {});  

