
function addEvent(obj, evType, fn){
	if(obj.addEventListener)
	    obj.addEventListener(evType, fn, true)
	if(obj.attachEvent)
	    obj.attachEvent("on"+evType, fn)
}
function removeEvent(obj, type, fn){
	if(obj.detachEvent){
		obj.detachEvent('on'+type, fn);
	}else{
		obj.removeEventListener(type, fn, false);
	}
}

function initDragAndDrop() {
    // Add drag handling to target elements
    addEvent(document.body, "dragenter", onDragEnter, false);
    addEvent(document.getElementById("drop-box-overlay"), "dragleave", onDragLeave, false);
    addEvent(document.getElementById("drop-box-overlay"), "dragover", noopHandler, false);
    addEvent(document.getElementById("drop-box-overlay"), "drop", onDrop, false);
}

function noopHandler(evt) {
	evt.stopPropagation();
	evt.preventDefault();
}

function onDragEnter(evt) {
	$("#drop-box-overlay").fadeIn(125);
}

function onDragLeave(evt) {
	/*
	 * We have to double-check the 'leave' event state because this event stupidly
	 * gets fired by JavaScript when you mouse over the child of a parent element;
	 * instead of firing a subsequent enter event for the child, JavaScript first
	 * fires a LEAVE event for the parent then an ENTER event for the child even
	 * though the mouse is still technically inside the parent bounds. If we trust
	 * the dragenter/dragleave events as-delivered, it leads to "flickering" when
	 * a child element (drop prompt) is hovered over as it becomes invisible,
	 * then visible then invisible again as that continually triggers the enter/leave
	 * events back to back. Instead, we use a 10px buffer around the window frame
	 * to capture the mouse leaving the window manually instead. (using 1px didn't
	 * work as the mouse can skip out of the window before hitting 1px with high
	 * enough acceleration).
	 */
	if(evt.pageX < 10 || evt.pageY < 10 || $(window).width() - evt.pageX < 10  || $(window).height - evt.pageY < 10) {
		$("#drop-box-overlay").fadeOut(125);
		$("#drop-box-prompt").fadeOut(125);
	}
}

function showMessageBox(statusStr, detailsStr, animation) {
	$("#mainImage").fadeOut(125);
	$("#upload-box").fadeIn(125);
        if(statusStr) $("#upload-status-text").html(statusStr);
	if(detailsStr) $("#upload-details").html(detailsStr);

        if(animation) $("#upload-animation").fadeIn(125);
        else $("#upload-animation").hide();
}

var extension = null;
function onDrop(evt) {

	if(window.Log && window.Log.LoggingEnabled){
		window.Log.BeginningSession();
	}

	// Consume the event.
	noopHandler(evt);
	
	// Hide overlay
	$("#drop-box-overlay").fadeOut(0);

	// Get the dropped files.
	var files = evt.dataTransfer.files;
	
	// If anything is wrong with the dropped files, exit.
	if(typeof files == "undefined" || files.length == 0) {
              var html = evt.dataTransfer.getData("text/html");
              var ind, ind2, ind3, type;
              if(html) {
                ind = html.indexOf("<img");
                if(ind < 0) ind = html.indexOf("<IMG");
                ind3 = html.indexOf("src");
                if(ind3 < 0) ind3 = html.indexOf("SRC");
                if(ind >= 0) {
                  html = html.substring(ind);
                  ind = html.indexOf("</img");
                  if(ind < 0) ind = html.indexOf("</IMG");
                  if(ind >= 0) html = html.substring(0,ind);
                  ind = html.indexOf("src");
                  if(ind < 0) ind = html.indexOf("SRC");
                  if(ind >= 0) {
                    html = html.substring(ind);
                    if((ind = html.indexOf("data:image")) >= 0 || (ind = html.substring(ind).indexOf("DATA:IMAGE")) >= 0) {
                      type = html.substring(ind+5).split(";")[0];
                      extension = type.split("/").pop();
                      // data is base64 encoded
                      html = html.substring(ind);
                      if((ind=html.indexOf(","))>=0) {
                        html = html.substring(ind+1);
                        if((ind=html.indexOf("\""))>=0)
                          html = html.substring(0,ind);
                        $.ajax({
                          type: 'POST',
                            url: 'request.php',
                            data: html,// Just send the Base64 content in POST body
                            processData: false, // No need to process
                            timeout: 60000, // 1 min timeout
                            dataType: 'text', // Pure Base64 char data
                            beforeSend: function onBeforeSend(xhr, settings) {
                              // Put the important file data in headers
                              xhr.setRequestHeader('x-file-name', 'image' + "." + extension);
                              xhr.setRequestHeader('x-file-type', type);

                              showMessageBox("Uploading Image...", "", true);
                            },
                            error: onError,
                            success: onUploadComplete
                        });
                      }
                    } else if((ind = html.indexOf("http")) >= 0 || (ind = html.indexOf("HTTP")) >= 0) {
                       var url = html.substring(ind).split("\"")[0];
		       UploadURL(url);
                    }
                  }
                }
              }
              return;
        } 

        uploadFile(files[0], 1);
}

function UploadURL(url) {
    showMessageBox("Uploading Image...", "", true);
    if(window.Log && window.Log.LoggingEnabled){
    	
		window.Log.BeginningSession();
    	
		var s = url.split("/");
		window.Log.SetImageName(s[s.length-1]);
    }
    $.get('request.php', {'image_url':url}, onUploadComplete);
}

function onUploadComplete(responseStr) {
        response = $.parseJSON(responseStr);

        // If the parse operation failed (for whatever reason) bail
        if(!response || typeof response == "undefined" || !response["session_id"]) {
            // Error, update the status with a reason as well.
            var details = "The server was unable to process the upload. Invalid server response " + responseStr;
            showMessageBox("Upload <span style='color: red;'>failed</span>", details, false);
            return;
        }

        // Update status
        showMessageBox("Upload Finished!", "", false);
        window.Server.SetSessionID(response["session_id"]);
        window.Server.SetSessionDir(response["session_dir"]);
        if(response["extension"]){
        	extension = response["extension"];
        }
        var image_name = window.Server.GetSessionDir() + "/" + window.Server.GetSessionID() + "." + extension;
        $("#questionDiv").fadeOut(500, function() {});
        $("#classVerificationDiv").fadeOut(500, function() {});
        window.UserImage.setImageName(image_name);
        window.Server.PreprocessImage();
}

function onError(XMLHttpRequest, textStatus, errorThrown) {
        var details;
        if(textStatus == "timeout")
             details = "Upload was taking too long and was stopped.";
        else
             details = "An error occurred while uploading the image: " + textStatus;
        showMessageBox("Upload <span style='color: red;'>failed</span>", details, false);
}

function uploadFile(file, totalFiles) {
    
    // Capture the name of the file here
    if(window.Log && window.Log.LoggingEnabled){
		window.Log.SetImageName(file.name)
	}
    
    var reader = new FileReader();
	
    // Handle errors that might occur while reading the file (before upload).
    reader.onerror = function(evt) {
	var message;
		
	// REF: http://www.w3.org/TR/FileAPI/#ErrorDescriptions
	switch(evt.target.error.code) {
	case 1:
	    message = file.name + " not found.";
	    break;
	case 2:
	    message = file.name + " has changed on disk, please re-try.";
	    break;
				
	case 3:
	    messsage = "Upload cancelled.";
	    break;
				
	case 4:
	    message = "Cannot read " + file.name + ".";
	    break;
				
	case 5:
	    message = "File too large for browser to upload.";
	    break;
	}
		
        showMessageBox(message, "", false);
    }
	
    // When the file is done loading, POST to the server.
    reader.onloadend = function(evt){
	var data = evt.target.result;
		
	// Make sure the data loaded is long enough to represent a real file.
	if(data.length > 128){
	    /*
	     * Per the Data URI spec, the only comma that appears is right after
	     * 'base64' and before the encoded content.
	     */
	    var base64StartIndex = data.indexOf(',') + 1;
            extension = file.name.split('.').pop();
	    
	    /*
	     * Make sure the index we've computed is valid, otherwise something 
	     * is wrong and we need to forget this upload.
	     */
	    if(base64StartIndex < data.length) {
		$.ajax({
			type: 'POST',
			    url: 'request.php',
			    data: data.substring(base64StartIndex), // Just send the Base64 content in POST body
			    processData: false, // No need to process
			    timeout: 60000, // 1 min timeout
			    dataType: 'text', // Pure Base64 char data
			    beforeSend: function onBeforeSend(xhr, settings) {
			      // Put the important file data in headers
			      xhr.setRequestHeader('x-file-name', file.name);
			      xhr.setRequestHeader('x-file-size', file.size);
			      xhr.setRequestHeader('x-file-type', file.type);
						
                              showMessageBox("Uploading Image...", "", true);
			    },
			    error: onError, 
			    success: onUploadComplete
		        });
	    }
	}
    };

    // Start reading the image off disk into a Data URI format.
    reader.readAsDataURL(file);
}


function UploadImage() {
	
	if(window.Log && window.Log.LoggingEnabled){
		window.Log.SetImageName($("uploadfile").val())
	}
	
    var form = document.uploadForm;
    var iframe = document.createElement("iframe");
    var detectWebKit = RegExp(" AppleWebKit/").test(navigator.userAgent);
    showMessageBox("Uploading...", "", true);
  
    iframe.setAttribute("id","ajax-temp");
    iframe.setAttribute("name","ajax-temp");
    iframe.setAttribute("width","0");
    iframe.setAttribute("height","0");
    iframe.setAttribute("border","0");
    iframe.setAttribute("style","width: 0; height: 0; border: none;");
    form.parentNode.appendChild(iframe);
    window.frames['ajax-temp'].name="ajax-temp";
    var image_name = window.Server.GetSessionDir() + "/" + window.Server.GetSessionID() + "." + document.getElementById("uploadfile").value.split('.').pop();
    window.UserImage.setImageName(image_name);
    extension = image_name.split('.').pop();
    var doUpload = function(){
		//JSONvis += '<font color=#00FF00>Upload image ' + image_name + '</font>\n<br/>\n<br/>';
	
		removeEvent(document.getElementById('ajax-temp'),"load", doUpload);
		var cross = "javascript: ";
		cross += "window.Server.PreprocessImage();"
		document.getElementById('ajax-temp').src = cross;
		if(detectWebKit)
			remove(document.getElementById('ajax-temp'));
		else
			setTimeout(function(){ 
						showMessageBox("Upload <span style='color: red;'>failed</span>", "Upload was taking too long and was stopped.", false);
				remove(document.getElementById('ajax-temp'))
				}, 250);
    }
    addEvent(document.getElementById('ajax-temp'),"load", doUpload);
    form.setAttribute("target","ajax-temp");
    form.setAttribute("action","request.php");
    form.setAttribute("method","post");
    form.setAttribute("enctype","multipart/form-data");
    form.setAttribute("encoding","multipart/form-data");
    document.getElementById('uploadJSON').value ='{"extension":"' + extension + '","session_id":"' + window.Server.GetSessionID() + '","session_dir":"' + window.Server.GetSessionDir() + '"}'; 
    form.submit();
}
