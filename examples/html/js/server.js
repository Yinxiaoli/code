/************ Server Communication ****************************/

var JSONvis = "";

function JSONRPCRequest(JSONrequest, callback, requestURL) {
    var args = new Array();
    args[0] = new Array();
    args[0].name = "json";
    args[0].value = escape(JSONrequest);
    httpRequest("GET", gParameters.requestURL, true, args, function(JSONtext) {
        JSONvis += '<font color=#0000FF>' + JSONrequest + '</font>\n<br/>\n<br/>' + 
            '<font color=#FF0000>' + JSONtext + '</font>\n<br/>\n<br/>';
        if(gParameters.debug_json)
            document.getElementById('infoDiv').innerHTML = JSONvis;
        
        if(callback)
            callback(JSONtext);
    });
}

function LoadTrainset(onLoaded, sortBy, getX, getY, getSuggestedLabel, maxExamples, startFrom) {
    var t = this;
    var extra_args = '';
    
    if (typeof sortBy != 'undefined' ) extra_args += ',"sortBy":"' + sortBy + '"';
    if (typeof getX != 'undefined' ) extra_args += ',"getX":' + getX;
    if (typeof getY != 'undefined' ) extra_args += ',"getY":' + getY;
    if (typeof getSuggestedLabel != 'undefined' ) extra_args += ',"getSuggestedLabel":' + getSuggestedLabel;
    if (typeof maxExamples != 'undefined' ) extra_args += ',"maxExamples":' + maxExamples;
    if (typeof startFrom != 'undefined' ) extra_args += ',"startFrom":' + startFrom;

    JSONRPCRequest('{"method":"get_trainset","jsonrpc":"2.0"' + extra_args + '}', function(JSONtext) {
	var len = JSONtext.length;
        var trainset = eval('(' + JSONtext + ')');
	if(onLoaded) onLoaded(trainset);
    });
}

// Read all class and question definitions from the server 
function LoadDefinitions(onLoaded) {
    var t = this;
    JSONRPCRequest('{"method":"get_definitions","jsonrpc":"2.0","classes":false,"questions":false,"certainties":false,"parts":true,"poses":true}', function(JSONtext) {
        var definitions = eval('(' + JSONtext + ')');
        t.classes = definitions['classes'];
        t.parts = definitions['parts'];
        t.poses = definitions['poses'];
        t.attributes = definitions['attributes'];
        t.certainties = definitions['certainties'];
        t.questions = definitions['questions'];

        t.poses_by_name = {};
        if(t.poses) {
          for(var i = 0; i < t.poses.length; i++) {
            t.poses_by_name[t.poses[i].pose_name] = t.poses[i];
          }
        }
	if(onLoaded) onLoaded();
    });
}
		  
// Ask the server to start a new session for the image 'image_name', causing initial computer vision preprocessing to occur
function StartSession(onFinished) {
    var t = this;
    t.num_click_questions = 0;
    JSONRPCRequest('{"method":"new_session","jsonrpc":"2.0","mkdir":true}', function(JSONtext) {
        var res = eval('(' + JSONtext + ')');
        t.top_classes = res['top_classes'];
        t.session_id = res['session_id'];
        t.session_dir = res['session_dir'];
        if(onFinished) onFinishised(t);
    });
}

    
// Finalize the location of a part
function FinalizePartLocation(y, part_id, onFinalized) {
    var loc = {};
    loc.x = y.part_locations[part_id].x;
    loc.y = y.part_locations[part_id].y;
    loc.part = y.part_locations[part_id].part;
    
    answer = JSON.parse(JSON.stringify(loc));
    answer['method'] = "label_part";
    answer['session_id'] = session_id;
    answer['jsonrpc'] = '2.0';
    answer['response_time'] = EndTiming();
    json_args = JSON.stringify(answer);

    JSONRPCRequest(json_args, function(JSONtext) {
        var res = eval('(' + JSONtext + ')');
        y.part_locations = res["y"]["part_locations"];
        if(onFinalized) onFinalized(y);
    });
}

// Preview part locations
function PreviewPartLocation(y, part_id, onFinished) {
    var loc = {};
    loc.x = part_locs[part_id].x;
    loc.y = part_locs[part_id].y;
    loc.part = part_locs[part_id].part;
    
    answer = eval('(' + JSON.stringify(loc) + ')');
    answer['method'] = "preview_part_locations";
    answer['session_id'] = session_id;
    answer['jsonrpc'] = '2.0';
    json_args = JSON.stringify(answer);
    
    if(isPreviewingPartLocations) {
        queuedJsonRequest = json_args;
    } else {
        PreviewPartLocationRequest(y, json_args, onFinished);
    }
}

function PreviewPartLocationRequest(y, json_args, onFinished) {
    isPreviewingPartLocations = true;
    JSONRPCRequest(json_args, function(JSONtext) {
        var res = eval('(' + JSONtext + ')');
        y.part_locations = res["y"]["part_locations"];
        if(onFinished) onFinished(y);
        isPreviewingPartLocations = false;
        if(queuedJsonRequest) {
            PreviewPartLocationRequest(queuedJsonRequest);
            queuedJsonRequest = false;
        }
    });
}

// Ask the server to start a new session for the image 'image_name', causing initial computer vision preprocessing to occur
function PreprocessImage(c, b) {
    $("#upload-status-text").html("Preprocessing Image...");
    $("#upload-animation").fadeIn(125);

    //image_name = basename(document.getElementById("uploadfile").value);
    mainImage.src = image_name;

    if(gParameters.debug) {
        debugDiv.style.visibility = 'visible';
        debugLink.style.visibility = 'visible';
        var base = session_dir + "/" + session_id;
        debugLink.href = session_dir + "/" + base.replace(/\//g,"_") + ".html";
    }

    var req = {};
    var x = {};
    x["imagename"] = gParameters.htmlDir + '/' + image_name;
    req["jsonrpc"] = "2.0";
    req["method"] = "initialize_interactive_parts";
    req["session_id"] = session_id;
    req["x"] = x;
    req["debug"] = gParameters.debug;
    req["debug_max_likelihood_solution"] = gParameters.debug_max_likelihood_solution;
    req["debug_probability_maps"] = gParameters.debug_probability_maps;
      
    JSONRPCRequest(JSON.stringify(req), function(JSONtext) {
        var res = eval('(' + JSONtext + ')');
        if(!res['y']) {
            alert('Bad response to PreprocessImage(): ' + JSONtext);
        } else {
          $("#upload-status-text").html("Preprocessing Finished");
          $("#upload-animation").hide();
          $("#upload-box").hide();
          $("#canvas").fadeIn(500);
	  part_locs = res["y"]["part_locations"];
          isAnchored = [];
          for(var i = 0; i < parts.length; i++) {
            isAnchored[i] = null;
            partCircles[i].attr(isAnchored[i] ? { 'fill':'#FF0000', 'stroke':'#FF0000' } : partStyles[i]);
          }
          UpdatePartLocations();
          if(c) c.innerHTML = b;
        }
      });
}

/************************************************************************************/
    // HTTP communication helper functions

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
