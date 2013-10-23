<?php
header("Cache-Control: no-cache, must-revalidate"); // HTTP/1.1
header("Expires: Sat, 26 Jul 1997 05:00:00 GMT"); // Date in the past
?>

<html>
  <head>		
    <meta http-equiv="Content-Type" content="text/html;charset=utf-8" />
    <title>Visipedia</title>
    <link rel="stylesheet" href="20q.css">
  </head>

	<body>
	  <div id="mainInstructions">
		<form name="uploadForm" id="uploadForm" enctype="multipart/form-data" action="request.php" method="POST">
		  <input type="hidden" name="MAX_FILE_SIZE" value="4000000" />
		  <input type="hidden" id="uploadJSON" name="json" />
			Upload an image of a bird: 
		  <input name="uploadedfile" id="uploadfile" type="file" onchange="window.Server.StartSession(); return false;" /><br />
		  <div id="debugDiv" name="debugDiv">
			<a id="debugLink" name="debugLink" target="_blank"> View Debug Output</a>
		  </div>
		</form>
	  </div>
	
	  <div id="upload-box"> 
		<div> 
		  <p id="upload-status-text_p">
			<span id="upload-status-text"> 
		  Drag and Drop an Image of a Bird...
			</span>
			<img id="upload-animation" src="images/loading.gif" />
		  </p>
		  <p id="upload-details"> 
		This tool will help you identify the species
		  </p> 
		</div> 
	  </div> 
	
	  <div id="imageContainer" >
		<div id="canvas" >
		  <b></b><img name="mainImage" id="mainImage" alt="" /> 
		</div> 
	  </div> 
	
	  <div id="drop-box-overlay"> 
		<h1>Drop image anywhere to upload...</h1> 
	  </div>

	  <div id="magnifyDiv" width=200> 
	  <div id="magnifyLabel"></div>
	  <div id="magnifyInstructions">Click to consider choosing...</div>
	  <img id="magnifyImage" width=200>
	  </div>
	
	  <div id='questionDiv'></div> 

	  <div id='classVerificationDiv'></div> 

	  <div id='scorePageDiv' display='none'>
	       <center>
	       <h1 id='scorePageHeader'> </h1>
	       <div id='scorePageComputation'> </div>
	       <br>
	       <table><tr>
		  <td><center>True Class<br><img id='scorePageTrueImg' class="scoreImage"><div id='scorePageTrueClass'></center></td>
		  <td><center>Predicted Class<br><img id='scorePagePredImg' class="scoreImage"><div id='scorePagePredClass'></center></td>
	       </tr></table>
	       <button type="button" id='scorePageNextImageButton' style="font-size:150%" onclick="JavaScript:window.Server.ScorePageNextImage();">Go to the next image</button>
	       <div id='scorePageFinished' display='none'> All Done!</div>
	       </center>
	  </div> 
	
	  <div id="classesDiv">
            <center>
                <font size=+2><b>Top Ranked Bird Species</b></font>
                <table><tr>
                  <td width="50px"><img top="-20px" src="images/controls/left.png" id="scrollLeft" onmouseover="window.Results.scrollClassDiv(-10,-10)" onmouseout="window.Results.stopTimer()" onmousedown="window.Results.stopTimer(); window.Results.scrollClassDiv(0,-500)" ></td>
                  <td>
                    <div id="classesDiv1" class="classResults" ></div>
                    <div id="classesDiv2" class="classResults" ></div>
                  </td>
                  <td width="50px"><img src="images/controls/right.png" id="scrollRight" onmouseover="window.Results.scrollClassDiv(10,10)" onmouseout="window.Results.stopTimer()" onmousedown="window.Results.stopTimer(); window.Results.scrollClassDiv(0,500)"></td>
                </tr></table>
	    </center>
	  </div>
	
	  <div id="infoDiv" name="infoDiv"></div>
	
		<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.6.3/jquery.min.js"></script> 
		<script>!window.jQuery && document.write(unescape('%3Cscript src="/public/scripts/libs/jquery-1.6.3.min.js"%3E%3C/script%3E'))</script> 
		
		<script src="https://ajax.googleapis.com/ajax/libs/jqueryui/1.8.16/jquery-ui.min.js"></script> 
		<script>!jQuery.ui && document.write(unescape('%3Cscript src="/public/scripts/libs/jquery-ui-1.8.16.custom.min.js"%3E%3C/script%3E'))</script> 
		
		<script src="image_upload.js"></script>
		<script src="raphael.js"></script>
		
		<script src="Server.js"></script>
		<script src="Question.js"></script>
		<script src="Results.js"></script>
		<script src="UserImage.js"></script>
		<script src="Log.js"></script>
		<script src="Verification.js"></script>
		
		
		<script>
			
			function getUrlVars() {
				var vars = {};
				var parts = window.location.href.replace(/[?&]+([^=&]+)=([^&]*)/gi, function(m,key,value) {
					vars[key] = value;
				});
				return vars;
			}
		
			$(document).ready(function() {
				
				urlParams = gParamId = getUrlVars();
				
				if("log" in urlParams){
					if(urlParams["log"] == "true"){
						window.Log.LoggingEnabled = true;
					}
				}
				
				
				window.Server.ParseArguments();		     
				if(!window.Server.IsUserStudy()) initDragAndDrop();
				window.Question.Initialize();
				window.Results.Initialize();
				window.UserImage.Initialize();
				window.Server.LoadDefinitions();
				window.Verification.Initialize();
			});
		</script>
		
	</body>
</html>

