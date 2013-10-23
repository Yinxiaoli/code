;(function ( namespace, undefined ) {  
	
	// turned on in the url
	namespace.LoggingEnabled = false;
	
	function basicErrorHandler(response){
		if("error" in response){
			alert(response["error"]);
		}
	}
	
	function queryLogger(query_data, callback){
		
		
		
		jqXHR = $.post("logger.php",
			query_data,
			function(responseText){
				
				// JSON decode
				if(callback){
					callback(jQuery.parseJSON(responseText));
				}
			}
		)
		.fail(function (xhr, textStatus, thrownError){
				
			alert("Status: " + xhr.status);
			alert("Error: " + thrownError);
			//window.location.href = urlForError;
				
		});
			
		
	}
	
	// This is a unique name given to the photo : <user_id>_<img_id>.jpg
	var imageName = "";
	namespace.SetImageName = function(name){
		imageName = name.slice(0,-4);
	}
	
	// Keep track of the start time of the session
	var startSessionTime = 0;
	namespace.BeginningSession = function(){
		startSessionTime = Date.now();
	}
	
	namespace.SendStartSessionInfo = function(){
	
		var query = {"action" : "logStartSession",
					 "sessionId" : window.Server.GetSessionID(), 
					 "sessionImgName" : window.UserImage.getImageName(),
					 "imageName": imageName,
					 "time": startSessionTime}
		var callback = basicErrorHandler;
	
		queryLogger(query, callback);
	
	}
	
	namespace.SendEndSessionInfo = function(selection){
	
		var query = {"action" : "logEndSession",
					 "imageName": imageName,
					 "selection" : selection, 
					 "time": Date.now()}
		var callback = basicErrorHandler;
	
		queryLogger(query, callback);
	
	}
	
	namespace.SendStartQuestionInfo = function(questionId, questionType){
		var query = {"action" : "logStartQuestion",
					 "questionId" : questionId, 
					 "questionType" : questionType,
					 "imageName": imageName,
					 "time": Date.now()}
		var callback = basicErrorHandler;
	
		queryLogger(query, callback);
	}
	
	namespace.SendEndQuestionInfo = function(questionId, questionType, answer, certainty){
		var query = {"action" : "logEndQuestion",
					 "questionId" : questionId, 
					 "questionType" : questionType,
					 "imageName": imageName,
					 "time": Date.now(),
					 "answer" : answer,
					 "certainty" : certainty}
		var callback = basicErrorHandler;
	
		queryLogger(query, callback);
	}
	
	namespace.SendResultsInfo = function(classes){
		var query = {"action" : "logResultsReceived",
					 "imageName": imageName,
					 "classes" : classes,
					 "time": Date.now()}
		var callback = basicErrorHandler;
	
		queryLogger(query, callback);
	}
	
	namespace.SendRemovalInfo = function(category){
		var query = {"action" : "logSpeciesRemoved",
					 "imageName": imageName,
					 "class" : category,
					 "time": Date.now()}
		var callback = basicErrorHandler;
	
		queryLogger(query, callback);
	}
	
	namespace.SendDetailViewInfo = function(category){
		var query = {"action" : "logViewedSpeciesDetail",
					 "imageName": imageName,
					 "class" : category,
					 "time": Date.now()}
		var callback = basicErrorHandler;
	
		queryLogger(query, callback);
	}
	    
})(window.Log = window.Log || {});  

