;(function ( namespace, undefined ) {  
	
        var magnifyDiv = null;
        var magnifyImage = null;
        var magnifyLabel = null;
        var magnifyInstructions = null;
	var classVerificationDiv = null;
	var choicesTable = null;

	var numCols = 5;
	var numRows = 3;
	var isShowing = false;
	var id = -1;
    
    
	
	namespace.Initialize = function(){
		classVerificationDiv = document.getElementById('classVerificationDiv');
		$("#classVerificationDiv").hide();

		magnifyDiv = document.getElementById('magnifyDiv');
		magnifyLabel = document.getElementById('magnifyLabel');
		magnifyInstructions = document.getElementById('magnifyInstructions');
		magnifyImage = document.getElementById('magnifyImage');
		magnifyDiv.style.visibility = 'hidden';
		magnifyDiv.style.pointerEvents = 'none';
	}

	namespace.reset = function(){
                $("#classVerificationDiv").fadeOut(500);
		isShowing = false;
        }

	namespace.InitMagnifyClass = function(img_el, img_url, cl, adjustScroll) {
	    img_el.onmouseover = function() {
		var pos = getAbsolutePosition(img_el, document.body);
		magnifyDiv.style.position = 'absolute';
		magnifyDiv.style.zIndex = 10000;
		magnifyDiv.style.display = 'block';
		
		magnifyImage.src = img_url;
		var headerSize = 0;
		if(cl) {
		    magnifyLabel.style.display='block';
		    magnifyInstructions.style.display='block';
		    magnifyLabel.innerHTML = cl['class_name'].replace(/\_/g," ");
		    headerSize = magnifyLabel.clientHeight + magnifyInstructions.clientHeight;
		} else {
		    magnifyLabel.style.display='none';
		    magnifyInstructions.style.display='none';
		}
		magnifyImage.onload = function() {
		    magnifyDiv.style.left = (pos.x+(img_el.clientWidth-magnifyImage.width)/2 - (adjustScroll ? window.Results.getScroll() : 0)) + 'px';
		    magnifyDiv.style.top  = (pos.y+(img_el.clientHeight-magnifyImage.height)/2-headerSize) + 'px';
		}
		magnifyDiv.style.visibility = 'visible';
            };
	    img_el.onmouseout = function() {
		magnifyDiv.style.visibility = 'hidden';
	    }
	}  

	function RemoveChildren(par) { 
	    while (par.hasChildNodes()) {
		par.removeChild(par.lastChild);
	    }
	}

    
	function ClickYes() { 
	    choicesTable.style.display = 'none';
	    isShowing = false;
	    $("#classesDiv").fadeOut(500);
	    $("#classVerificationDiv").fadeOut(500);
	    window.Server.VerifyClass(window.Verification.GetClassId(), 1);
	    
	    if(window.Log.LoggingEnabled){
			window.Log.SendEndSessionInfo(id);
		}
	    
	}
	function ClickNo() { 
	    isShowing = false;
	    window.Server.RemoveClassFromConsideration(window.Verification.GetClassId());
	    $("#classVerificationDiv").fadeOut(500, function() { $("#questionDiv").fadeIn(500); } );
	}
	function ClickUnsure() { 
	    isShowing = false;
	    $("#classVerificationDiv").fadeOut(500, function() { $("#questionDiv").fadeIn(500); } );
	}

	namespace.GetClassId = function(cl) {
	    return id;
	}

	namespace.InitVerifyClass = function(cl) {
	
	    RemoveChildren(classVerificationDiv);

	    var classLabel = document.createElement('h1');
	    classLabel.innerHTML = 'Is your bird a "' + cl['class_name'].replace(/\_/g," ") + '"?';
	    classVerificationDiv.appendChild(classLabel);
	    classVerificationDiv.appendChild(document.createElement('br'));

	    var classLabel2 = document.createElement('div');
	    classLabel2.className = 'instructionsText';
	    classLabel2.innerHTML = 'Here are some example images:';
	    classVerificationDiv.appendChild(classLabel2);
	    
	    var currDiv = null;
	    numImages = cl['class_images'].length;
	    if(numImages > numCols*numRows)
		numImages = numCols*numRows;
	    var table = document.createElement('table');
	    table.className = 'verificationTable';
	    var tbody = document.createElement('tbody');
	    var imgRow = null;
	    classVerificationDiv.appendChild(table);
	    table.appendChild(tbody);
	    for(var i = 0; i < numImages; i++) {
		if(i % numCols == 0) {
		    imgRow = document.createElement('tr');
		    tbody.appendChild(imgRow);
		}
		var imgCol = document.createElement('td');
		imgCol.className = 'classesImageCell';
		imgRow.appendChild(imgCol);
		var img = document.createElement('img');
		img.className = 'verifyClassImage';
		img.src = cl['class_images'][i];
		this.InitMagnifyClass(img, cl['class_images'][i], null, false);
		imgCol.appendChild(img);
	    }


	    choicesTable = document.createElement('table');
	    choicesTable.className = "verifyButtonsTable";
	    var choicesTableBody = document.createElement('tbody');
	    var choicesRow = document.createElement('tr');
	    var yes = new Button('Yes!', ClickYes, 'buttonYes', false);
	    var no = new Button('Definite No', ClickNo, 'buttonNo', false);
	    var unsure = new Button('Still Unsure', ClickUnsure, 'buttonUnsure', false);
	    classVerificationDiv.appendChild(choicesTable);
	    choicesTable.appendChild(choicesTableBody);
	    choicesTableBody.appendChild(choicesRow);
	    choicesRow.appendChild(yes.td);
	    choicesRow.appendChild(no.td);
	    choicesRow.appendChild(unsure.td);

	}

	namespace.VerifyClass = function(cl, i) {
	    id = i;
	    
	    if(window.Log.LoggingEnabled){
			window.Log.SendDetailViewInfo(id);
		}
	    
	    if(!isShowing) { 
			isShowing = true;
			$("#questionDiv").fadeOut(500, function() { 
				window.Verification.InitVerifyClass(cl);
				$("#classVerificationDiv").fadeIn(500, function() {}) 
			});
	    } 
	    else {
			$("#classVerificationDiv").fadeOut(500, function() { 
				window.Verification.InitVerifyClass(cl);
				$("#classVerificationDiv").fadeIn(500, function() {}) 
			});
	    }
	}
    

    function getAbsolutePosition(e, root) {
	var retval = new Array();
	retval.x = retval.y = 0;
	while(e != null && e != root) {
            retval.x += e.offsetLeft;
            retval.y += e.offsetTop;
            e = e.offsetParent;
	}
	return retval;
    }


})(window.Verification = window.Verification || {});  
