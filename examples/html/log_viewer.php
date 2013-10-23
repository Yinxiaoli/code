<?php
	header("Cache-Control: no-cache, must-revalidate"); // HTTP/1.1
	header("Expires: Sat, 26 Jul 1997 05:00:00 GMT"); // Date in the past
?>

<html>
	<body>
		<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.6.3/jquery.min.js"></script> 
		<script>!window.jQuery && document.write(unescape('%3Cscript src="/public/scripts/libs/jquery-1.6.3.min.js"%3E%3C/script%3E'))</script> 
		
		<script src="https://ajax.googleapis.com/ajax/libs/jqueryui/1.8.16/jquery-ui.min.js"></script> 
		<script>!jQuery.ui && document.write(unescape('%3Cscript src="/public/scripts/libs/jquery-ui-1.8.16.custom.min.js"%3E%3C/script%3E'))</script> 
		
		<script src="underscore-min.js"></script>
		
		<script type="text/javascript">
			
			function getUrlVars() {
				var vars = {};
				var parts = window.location.href.replace(/[?&]+([^=&]+)=([^&]*)/gi, function(m,key,value) {
					vars[key] = value;
				});
				return vars;
			}
			
			$(document).ready(function() {
				
				imageName = gParamId = getUrlVars()["log"];
			
				var query = {"action" : "viewLog",
					 		 "imageName" : imageName};
				$.post("logger.php",
					    query,
					    function(responseText){
						
							// JSON decode
							data = jQuery.parseJSON(responseText);
							
							// restructure the data
							structured = []
							times = []
							for (section in data){
								console.log(section);
								if (section == "start_session"){
									structured.push("start session");
									times.push(data[section]["time"]);
								}
								else if(section == "end_session"){
									structured.push("end session");
									times.push(data[section]["time"]);
								}
								else if(section == "results"){
									for (event in data[section]){
										structured.push("received new results");
										times.push(data[section][event]["time"]);
									}
								}
								else if(section == "questions"){
									for (event in data[section]){
										if(data[section][event]["stage"] == "start"){
											structured.push("received question");
										}
										else{
											structured.push("answered question");
										}
										times.push(data[section][event]["time"]);
									}
								}
								else if(section == "detail_views"){
									for (event in data[section]){
										structured.push("viewed details");
										times.push(data[section][event]["time"]);
									}
								}
								else if(section == "removals"){
									for (event in data[section]){
										structured.push("removed category");
										times.push(data[section][event]["time"]);
									}
								}
							}
							
							comb = _.zip(structured, times);
							org = _.sortBy(comb, function(ele){ return ele[1];});
							
							for(event in org){
								document.write("<p>" + org[event][0] + "</p>");
							}
							
						}
					);
			});
		</script>
	</body>
</html>