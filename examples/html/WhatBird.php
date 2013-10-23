<?php
header("Cache-Control: no-cache, must-revalidate"); // HTTP/1.1
header("Expires: Sat, 26 Jul 1997 05:00:00 GMT"); // Date in the past
?>

<html>
  <head>		
    <meta http-equiv="Content-Type" content="text/html;charset=utf-8" />
    <title>WhatBird Test</title>
    
    <link rel="stylesheet" href="http://code.jquery.com/ui/1.10.0/themes/base/jquery-ui.css" />
	<link href="//netdna.bootstrapcdn.com/twitter-bootstrap/2.3.0/css/bootstrap-combined.min.css" rel="stylesheet">
            
    
  </head>

	<body>
		<div class="container">
			<h1> Please follow the instructions</h2>
			<div class = "thumbnail">
				<img id="testImage" />
			</div>
			<div id="step1" class="well">
				<h2>Step 1</h2> 
				<h3>Please identify the bird species in the image above by clicking on the button below. You may refer back to this image.</h3>
				<button class="btn btn-large btn-primary" onclick="goToWhatBird()">Go To WhatBird</button>
			</div>
			<div id="step2" class="well">
				<h2>Step 2</h2>
				<h3>Submit the species name here</h3>
				<div>
					<label for="species">Species: </label>
					<input id="species" />
				</div>
				<div>
					<label for="questions">Number of Questions Answered: </label>
					<input id="questions" />
				</div>
				<button class="btn btn-large btn-danger" onclick="submitSpecies()">Submit</button>
				</div>
			</div>
			<div id="step3" class="well">
				<h2>Step 3</h2> 
				<h3>Please close the whatbird.com tabs that opened in your browser.</h3>
				<div id="nextImage">
					<h3> Click the button below to go to the next image.</h3>
					<button class="btn btn-large btn-danger" onclick="stepOne()">Next Photo</button>
				</div>
				<div id="finished">
					<h3> All done! Please refer back to the email for the next instructions.</h3>
				</div>
			</div>
		</div>
		<script src="http://ajax.googleapis.com/ajax/libs/jquery/1.8.3/jquery.min.js"></script>
		<script src="http://ajax.googleapis.com/ajax/libs/jqueryui/1.10.0/jquery-ui.min.js"></script>
		<script src="http://netdna.bootstrapcdn.com/twitter-bootstrap/2.3.0/js/bootstrap.min.js"></script>
            
		<script>
			
			window.timer = null;
			window.user_id = null;
			window.image_num = null;
			window.iterations = 0;
			window.total_iterations = 5;
			window.testInterface = false;
			
			function goToWhatBird(){
				window.timer = Date.now();
				window.open('http://identify.whatbird.com/mwg/_/0/attrs.aspx', '_blank');
  				window.focus();
  				
  				stepTwo();
			}
			
			function submitSpecies(){
				
				var species = $("#species").val();
				if ( ! /^[a-zA-Z'\- ]+$/.test(species)){
					alert("It seems like there is an issue with the species name, please use letters, the apostrophe (') and the hyphen (-)");
					return;
				}
				
				var question_count = $("#questions").val();
				if (! /^[0-9]+$/.test(question_count) ){
					alert("Please use only numbers for the question count");
					return;
				}
				
				time = Date.now() - window.timer;
				console.log("It took " + time/1000.0 + " seconds");
				
				// submit result to server
				if(!window.testInterface){
					sendDataToLog(species, question_count, time)
				}
				else{
					// don't submit to server
				}
				
				// Go to next step
				stepThree();
				
			}
			
			function getUrlVars() {
				var vars = {};
				var parts = window.location.href.replace(/[?&]+([^=&]+)=([^&]*)/gi, function(m,key,value) {
					vars[key] = value;
				});
				return vars;
			}
			
			function getUrlForImage(image_num){
				var base = "http://vasuki.ucsd.edu/demo/pilot/";
				//var base = "https://dl.dropbox.com/sh/rc3ycp6tymiyc8m/_wT19pArOF/";
				url = base + window.user_id + "/whatbird/" + window.user_id + "_" + image_num + ".jpg";
				return url;
			}
			
			function incrementImage(){
				window.image_num = 1 * image_num + 1
				window.iterations += 1;
			}
			
			function stepOne(){
				
				if(! window.testInterface){
					i = new Image;
					i.onload = function(){
						//console.log("here");
						$("#testImage").attr("src", i.src);
					}
					i.src = getUrlForImage(window.image_num);
				}
				else{
					$("#testImage").attr("src", "http://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcQIAiSGyCEZKle-gMI1L3vnigo8wW_Rtksn_mNHwWAnhQnvVDev");
				}
				
				// hide the other steps
				$("#step2").addClass("hidden");
				$("#step3").addClass("hidden");
				
				// show ourself
				$("#step1").removeClass("hidden");
			}
			
			function stepTwo(){
				
				// hide the other steps
				$("#step1").addClass("hidden");
				$("#step3").addClass("hidden");
				
				// clear the fields
				$("#species").val("");
				$("#questions").val("");
				
				// show ourself
				$("#step2").removeClass("hidden");
				
			}
			
			function stepThree(){
				
				// hide the other steps
				$("#step1").addClass("hidden");
				$("#step2").addClass("hidden");
				
				incrementImage();
				
				if(! window.testInterface){
					if (window.iterations >= window.total_iterations){
						$("#nextImage").addClass("hidden");
						$("#finished").removeClass("hidden");
					}
					else{
						$("#nextImage").removeClass("hidden");
						$("#finished").addClass("hidden");
					}
				}
				else{
					$("#nextImage").addClass("hidden");
					$("#finished").removeClass("hidden");
				}
				// show ourself
				$("#step3").removeClass("hidden");
				
			}
			
			function sendDataToLog(speciesName, questionCount, time){
				
				var query = {"action" : "logWhatBirdData",
					 		 "user_id" : window.user_id,
					 		 "image_id" : window.image_num,
					 		 "speciesName" : speciesName,
					 		 "questionCount" : questionCount,
					 		 "time" : time};
				
				$.post("logger.php", query);
			}
		
			$(document).ready(function() {
				
				urlParams = getUrlVars();
				
				
				if("test" in urlParams){
					var test = 1 * urlParams["test"];
					if(test){
						window.testInterface = true;
					}
				}
				else{
					if("user_id" in urlParams){
						window.user_id = urlParams["user_id"];
					}
					if("image_num" in urlParams){
						window.image_num = urlParams["image_num"];
					}
				}				
				
				stepOne();
				
			});
		</script>
		
	</body>
</html>

