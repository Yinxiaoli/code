;(function ( namespace, undefined ) {  
	
	
	var mainImage = null
	
	var canvas = null, canvasImage = null;
	
	var image_name = null;
	var pos_x, pos_y, hasClick=false;
	var partbox = null;
	
	var part_diameter = '5';
    var auto_submit_part_clicks = false;
	
	var partbox_style = { "stroke" : "#00FFFF", "fill":"#00FFFF",  "fill-opacity":".5", "radius" : 7, "stroke-width":2, "cursor":"pointer" };
    var image_width = 400.0;
	
	namespace.setImageName = function(im_name){
		image_name = im_name;
	}
	
	namespace.getImageName = function(){
		return image_name;
	}
	
	namespace.Initialize = function(){
		
		mainImage = document.getElementById('mainImage'); 
     
		$("#canvas").hide();
		
		canvas = Raphael("canvas", image_width, image_width);
		mainImage.style.display = "none";
		canvasImage = canvas.image(mainImage.src, 0, 0, image_width, image_width);
		mainImage.onload = function() {
			zoom = (mainImage.width > mainImage.height ? image_width/mainImage.width : image_width/mainImage.height);
			canvasImage.attr({"src":mainImage.src, "width":(mainImage.width*zoom),"height":(mainImage.height*zoom)});
		}
		partbox = canvas.circle(0, 0, partbox_style.radius);
		partbox.attr(partbox_style);
		partbox.hide();
		partbox.drag(
			function(dx,dy) { 
				pos_x = (this.anchor_x+dx)/zoom; 
				pos_y = (this.anchor_y+dy)/zoom; 
				partbox.attr({"cx":(pos_x*zoom),"cy":(pos_y*zoom)}); 
			},
			function() { 
				this.anchor_x = partbox.attr("cx"); 
				this.anchor_y = partbox.attr("cy"); 
			},
			null
		);
		
	}
	
	namespace.getPositionX = function(){
		return pos_x;
	}
	
	namespace.getPositionY = function(){
		return pos_y;
	}
	
	namespace.setSource = function(src){
		mainImage.src = src
	}
	
	namespace.Reset = function(){
		mainImage.style.cursor = "default";
		canvasImage.attr({"cursor" : "default"});
		canvasImage.node.onclick = null;
	}
	
	function ClickPart() {
		partbox.show();
		hasClick = true;
		window.Question.EnableButtons(hasClick);
    }
	
	namespace.PreparePartClick = function(){
		canvasImage.attr({"cursor" : "crosshair"});
		canvasImage.node.onclick = function(e) { 
			c = document.getElementById('canvas');
			pos_x = (e.clientX-findPosX(c))/zoom;  
			pos_y = (e.clientY-findPosY(c))/zoom;
			partbox.attr({"cx":(pos_x*zoom), "cy":(pos_y*zoom)});
			ClickPart();
		};
		
		partbox.hide();
		hasClick = false;
		//window.Question.EnableButtons(hasClick); // I don't think this is needed here
	} 
	
	namespace.HidePartBox = function(){
		partbox.hide();
	}
	
	function findPosX(obj){
		var curleft = 0;
		if(obj.offsetParent)
			while(1) 
			{
			  curleft += obj.offsetLeft;
			  if(!obj.offsetParent)
				break;
			  obj = obj.offsetParent;
			}
		else if(obj.x)
			curleft += obj.x;
		return curleft;
	}
	function findPosY(obj){
		var curtop = 0;
		if(obj.offsetParent)
			while(1)
			{
			  curtop += obj.offsetTop;
			  if(!obj.offsetParent)
				break;
			  obj = obj.offsetParent;
			}
		else if(obj.y)
			curtop += obj.y;
		return curtop;
	  }
    
})(window.UserImage = window.UserImage || {});  

