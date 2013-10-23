;(function ( namespace, undefined ) {  
	
	
	var classesDiv = null
	
    var numClassRankUpdates = 0;
	var topClassesTable = null;
	var top_classes = null;
	
	namespace.Initialize = function(){
		classesDiv = document.getElementById('classesDiv');
		$("#classesDiv").hide();
	}
	
	function RemoveChildren(par) { 
      while (par.hasChildNodes()) {
        par.removeChild(par.lastChild);
      }
    }

    var timer1;
    namespace.scrollClassDiv = function(dep1, dep2) {
      var divId = numClassRankUpdates % 2 == 0 ? "classesDiv2" : "classesDiv1";
      var scroll_container = document.getElementById(divId);
      scroll_container.scrollLeft += dep2;
      document.getElementById("scrollLeft").style.display = scroll_container.scrollLeft <= 0 ? "none" : "inline";
      document.getElementById("scrollRight").style.display = scroll_container.scrollLeft >= scroll_container.scrollWidth-scroll_container.clientWidth ? "none" : "inline";
      timer1 = setTimeout('window.Results.scrollClassDiv('+dep1+',' + dep1 + ')', 30);
    }
    namespace.stopTimer = function() {
      clearTimeout(timer1);
    }

    namespace.getScroll = function() {
      var divId = numClassRankUpdates % 2 == 0 ? "classesDiv2" : "classesDiv1";
      var scroll_container = document.getElementById(divId);
      return scroll_container.scrollLeft;
    }


	
	// Draw the row of top-ranked classes
    namespace.UpdateTopClasses = function(classes, new_top_classes) {
      	
      	top_classes = new_top_classes;
      	
		var oldTable = topClassesTable;
		
		topClassesTable = document.createElement('table');
		topClassesTable.className = 'classesTable';
		var body = document.createElement('tbody');
		topClassesTable.appendChild(body);
		
		var imgRow = document.createElement('tr');
		var nameRow = document.createElement('tr');
		body.appendChild(imgRow);
		body.appendChild(nameRow);
		
		var x_img = [];
		for(var i=0; top_classes && i < top_classes.length; i++) {
			
			var id = parseInt(top_classes[i].class_id);
			
			var imgCol = document.createElement('td');
			imgCol.className = 'classesImageCell';
			var nameCol = document.createElement('td');
			nameCol.className = 'classesImageCell';
			if(classes[id]['class_images'] && classes[id]['class_images'].length) {
				var a = document.createElement('a');
				//a.href = "http://vasuki.ucsd.edu/mediawiki/index.php/Category:" + classes[id]['class_name'];
				//a.target="_blank";
			    
			        a.style.cursor = 'pointer';
				a.cl = classes[id];
				a.cl_id = id;
				a.onclick = function() {
				    window.Verification.VerifyClass(this.cl, this.cl_id);
				    return true;
				};
			    
				var img = document.createElement('img'), img_url;
				if('img_src' in top_classes[i]){
					img_url = top_classes[i].img_src;
				}
				else{
					img_url = classes[id]['class_images'][0];
				}
				img.src = img_url;
				img.height = 80;
				img.width = 80;
				imgCol.appendChild(a);
				a.appendChild(img);
				window.Verification.InitMagnifyClass(img, img_url, classes[id], true);
			}
			var cellText = document.createTextNode(classes[id]['class_name'].replace(/\_/g," "));
			nameCol.appendChild(cellText);

			//if(window.Server.Debug()) {
			    nameCol.appendChild(document.createElement('br'));
			    nameCol.appendChild(document.createTextNode('prob='+top_classes[i].prob.toFixed(3)));
			//}
			
			nameCol.appendChild(document.createElement('br'));
			x_img[i] = document.createElement('img');
			x_img[i].src = "images/no.png";
			x_img[i].className = "removeClass";
			x_img[i].width = x_img[i].height = 14;
			x_img[i].i = id;
			x_img[i].onclick = function() {
				
				window.Server.RemoveClassFromConsideration(this.i);
				
				return true;
			};
			//nameCol.appendChild(x_img[i]);
			
			imgRow.appendChild(imgCol);
			nameRow.appendChild(nameCol);
		}
		
		$("#classesDiv").show();

		if(numClassRankUpdates % 2 == 0) {
			document.getElementById("classesDiv1").appendChild(topClassesTable);
                        document.getElementById("classesDiv1").scrollLeft = 0;
                        document.getElementById("classesDiv2").scrollLeft = 0;
		        $("#classesDiv2").fadeOut(500, function() { $("#classesDiv1").fadeIn(500); RemoveChildren(document.getElementById("classesDiv2")); } );
		} 
		else {
			document.getElementById("classesDiv2").appendChild(topClassesTable);
                        document.getElementById("classesDiv2").scrollLeft = 0;
                        document.getElementById("classesDiv1").scrollLeft = 0;
			$("#classesDiv1").fadeOut(500, function() { $("#classesDiv2").fadeIn(500); RemoveChildren(document.getElementById("classesDiv1")); } );
		}
                document.getElementById("scrollLeft").style.display = "none";
                document.getElementById("scrollRight").style.display = "inline";
		numClassRankUpdates = numClassRankUpdates+1;
    }
     
})(window.Results = window.Results || {});  

