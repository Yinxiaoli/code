<?php
	
	// we need the session id and the image name for each log
	
	// new session logging would pass though here
	function createLog($userId, $imageId){
	
		$log_name = null;
		$ver = -1;
		foreach (glob("logs/$userId/$imageId/*.txt") as $filename) {
			$matches =array();
			$count = preg_match("#log_(\d+).txt#", $filename, $matches);
			if(intval($matches[1]) > $ver){
				$ver = intval($matches[1]);
				$log_name = $filename;
			}
		}
		
		$ver++;
		$log_name = "log_$ver.txt";
		
		return $log_name;
		
	}
	
	// all other logging other than new session would pass through here
	// assumes that the latest version of the log is the correct one
	function getLogPath($userId, $imageId){
	
		$log_name = null;
		$ver = -1;
		foreach (glob("logs/$userId/$imageId/*.txt") as $filename) {
			$matches =array();
			$count = preg_match("#log_(\d+).txt#", $filename, $matches);
			if(intval($matches[1]) > $ver){
				$ver = intval($matches[1]);
				$log_name = $filename;
			}
		}
		return $log_name;
		
	}
	
	function getLogContents($log_path){
		if(is_file($log_path)){
			return  json_decode(file_get_contents($log_path), true);
		}
		return array();
	}
	
	function putLogContents($log_path, $contents){
		file_put_contents($log_path, json_encode($contents));
	}
	
	
	// this will store any feedback
	$returnArray = array();
	
	try {
	
		$action = $_REQUEST["action"];
		$returnArray = array();
		switch($action){
		
			case "logStartSession": {
			
				// store the start time (the time at which the user uploaded the image)
				// store the session id
				// store the image name
				
				$sessionId = $_REQUEST["sessionId"];
				$sessionImgName = $_REQUEST["sessionImgName"];
				$imageName = $_REQUEST["imageName"];
				$time = $_REQUEST["time"];
				
				// parse the user name and image id
				$pieces = explode("_", $imageName);
				$userId = $pieces[0];
				$imageId = $pieces[1];
				
				// has this user done any images yet?
				if (!is_dir("./logs/$userId/")){
					mkdir("./logs/$userId/");
				}
				
				// is this the first time for this image?
				if (!is_dir("./logs/$userId/$imageId/")){
					mkdir("./logs/$userId/$imageId/");
				}
				
				# get the log name for this session
				$log_name = createLog($userId, $imageId);
				$log_path = "logs/$userId/$imageId/$log_name";
				$log_data = getLogContents($log_path);
				$log_data["start_session"] = array("sessionId" => $sessionId,
												   "sessionImgName" => $sessionImgName,
												   "imageName" => $imageName,
												   "time" => $time);
				
				putLogContents($log_path, $log_data);
				
			}break;
			
			case "logEndSession":{
				// store the end time and the species selected
				
				$imageName = $_REQUEST["imageName"];
				$time = $_REQUEST["time"];
				$selection = $_REQUEST["selection"];
				
				// parse the user name and image id
				$pieces = explode("_", $imageName);
				$userId = $pieces[0];
				$imageId = $pieces[1];
				
				$log_path = getLogPath($userId, $imageId);
				
				if(!$log_path){
					$returnArray = array('error' => "unable to find log file");
				}
				else{
				
					$log_data = getLogContents($log_path);
					$log_data["end_session"] = array("time" => $time,
													 "selection" => $selection);
					
					putLogContents($log_path, $log_data);
				}
				
			} break;
			
			case "logStartQuestion": {
				// store the time and question id
				
				$imageName = $_REQUEST["imageName"];
				$questionId = $_REQUEST["questionId"];
				$time = $_REQUEST["time"];
				$type = $_REQUEST["questionType"];
				
				// parse the user name and image id
				$pieces = explode("_", $imageName);
				$userId = $pieces[0];
				$imageId = $pieces[1];
				
				$log_path = getLogPath($userId, $imageId);
				
				if(!$log_path){
					$returnArray = array('error' => "unable to find log file");
				}
				else{
				
					//$log_path = "logs/$userId/$imageId/$log_name";
					$log_data = getLogContents($log_path);
					if(!array_key_exists("questions", $log_data)){
						$log_data["questions"] = array();
					}
					$log_data["questions"][] = array("id" => $questionId,
													 "type" => $type,
													 "time" => $time,
													 "stage" => "start");
					
					putLogContents($log_path, $log_data);
				}
								
			}break;
			
			case "logEndQuestion":{
				// store the time and the question id and answer
				
				$imageName = $_REQUEST["imageName"];
				$questionId = $_REQUEST["questionId"];
				$time = $_REQUEST["time"];
				$answer = $_REQUEST["answer"];
				$certainty = $_REQUEST["certainty"];
				$type = $_REQUEST["questionType"];
				
				// parse the user name and image id
				$pieces = explode("_", $imageName);
				$userId = $pieces[0];
				$imageId = $pieces[1];
				
				$log_path = getLogPath($userId, $imageId);
				
				if(!$log_path){
					$returnArray = array('error' => "unable to find log file");
				}
				else{
				
					//$log_path = "logs/$userId/$imageId/$log_name";
					$log_data = getLogContents($log_path);
					if(!array_key_exists("questions", $log_data)){
						$log_data["questions"] = array();
					}
					$log_data["questions"][] = array("id" => $questionId,
													 "type" => $type,
													 "time" => $time,
													 "stage" => "end",
													 "answer" => $answer,
													 "certainty"=>$certainty);
					
					putLogContents($log_path, $log_data);
				}
				
			} break;
			
			case "logResultsReceived":{
			
				// store the results 
				$imageName = $_REQUEST["imageName"];
				$classes = $_REQUEST["classes"];
				$time = $_REQUEST["time"];
				
				// parse the user name and image id
				$pieces = explode("_", $imageName);
				$userId = $pieces[0];
				$imageId = $pieces[1];
				
				$log_path = getLogPath($userId, $imageId);
				
				if(!$log_path){
					$returnArray = array('error' => "unable to find log file");
				}
				else{
				
					$log_data = getLogContents($log_path);
					if(!array_key_exists("results", $log_data)){
						$log_data["results"] = array();
					}
					$log_data["results"][] = array("classes" => $classes,
												   "time" => $time);
					
					putLogContents($log_path, $log_data);
				}
				
				
			} break;
			
			case "logSpeciesRemoved":{
				// store the time and the species removed

				$imageName = $_REQUEST["imageName"];
				$class = $_REQUEST["class"];
				$time = $_REQUEST["time"];
				
				// parse the user name and image id
				$pieces = explode("_", $imageName);
				$userId = $pieces[0];
				$imageId = $pieces[1];
				
				$log_path = getLogPath($userId, $imageId);
				
				if(!$log_path){
					$returnArray = array('error' => "unable to find log file");
				}
				else{
				
					$log_data = getLogContents($log_path);
					if(!array_key_exists("results", $log_data)){
						$log_data["removals"] = array();
					}
					$log_data["removals"][] = array("class" => $class,
												    "time" => $time);
					
					putLogContents($log_path, $log_data);
				}
				
			} break;
			
			case "logViewedSpeciesDetail":{
				// store that a species was clicked on
				
				$imageName = $_REQUEST["imageName"];
				$class = $_REQUEST["class"];
				$time = $_REQUEST["time"];
				
				// parse the user name and image id
				$pieces = explode("_", $imageName);
				$userId = $pieces[0];
				$imageId = $pieces[1];
				
				$log_path = getLogPath($userId, $imageId);
				
				if(!$log_path){
					$returnArray = array('error' => "unable to find log file");
				}
				else{
				
					$log_data = getLogContents($log_path);
					if(!array_key_exists("results", $log_data)){
						$log_data["detail_views"] = array();
					}
					$log_data["detail_views"][] = array("class" => $class,
												        "time" => $time);
					
					putLogContents($log_path, $log_data);
				}
				
			} break;
			
			case "viewLog":{
				
				$imageName = $_REQUEST["imageName"];
				
				// parse the user name and image id
				$pieces = explode("_", $imageName);
				$userId = $pieces[0];
				$imageId = $pieces[1];
				
				$log_path = getLogPath($userId, $imageId);
				if(!$log_path){
					$returnArray = array('error' => "unable to find log file");
				}
				else{
					$returnArray = getLogContents($log_path);
				}
				
			} break;
			
			case "logWhatBirdData":{
				// store that a species was clicked on
				
				$userId = $_REQUEST["user_id"];
				$imageId = $_REQUEST["image_id"];
				$speciesName = $_REQUEST["speciesName"];
				$questionCount = $_REQUEST["questionCount"];
				$time = $_REQUEST["time"];
				
				// has this user done any images yet?
				if (!is_dir("./logs/$userId/")){
					mkdir("./logs/$userId/");
				}
				
				// is this the first time for this image?
				if (!is_dir("./logs/$userId/$imageId/")){
					mkdir("./logs/$userId/$imageId/");
				}
				
				# get the log name for this session
				$log_name = createLog($userId, $imageId);
				$log_path = "logs/$userId/$imageId/$log_name";
				$log_data = getLogContents($log_path);
				$log_data["data"] = array("name" => $speciesName,
										   "question_count" => $questionCount,
										   "time" => $time);
				
				putLogContents($log_path, $log_data);
				
			} break;
			
			default : {
				throw new Exception("Invalid action : $action");
			}
		}
	}
	
	catch (Exception $e) {
		
	}
	
	echo json_encode($returnArray);						

?>