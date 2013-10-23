<?php

$ip = "localhost";//"vision602";
$port = "8086";
$max_width = "300"; 
$max_height = "300"; 
$SESSIONS_DIR = "sessions/";

function SafeFileName($fname) {
  $replace="_";
  $pattern="/([[:alnum:]_\.-]*)/";
  $fname=str_replace(str_split(preg_replace($pattern,$replace,$fname)),$replace,$fname);
  return $fname;
}

function JSONRequest($json) {
	global $ip, $port;

	$fh = fopen("sessions/history", "a");
        fwrite($fh, date("F j, Y, g:i a") . " " . $_SERVER['REMOTE_ADDR'] . ": " . $json . "\n");
        fclose($fh);
	$query = str_replace('\\"', '"', $json);

        // Create a TCP Stream Socket
	$socket = socket_create(AF_INET, SOCK_STREAM, SOL_TCP);
	if ($socket === false){
		header("Content-type: text/plain");
    $errorcode = socket_last_error();
    $errormsg = socket_strerror($errorcode);
		echo("Socket Creation Failed: [$errorcode] $errormsg");
    $fh = fopen("sessions/history", "a"); fwrite($fh, "Socket Creation Failed [$errorcode] $errormsg\n"); fclose($fh);
		exit;
	}
  

	// Connect to the Visipedia server.
	$result = socket_connect($socket, $ip, $port);
	if ($result === false){
		header("Content-type: text/plain");
    $errorcode = socket_last_error();
    $errormsg = socket_strerror($errorcode);
		echo("Connection Failed: [$errorcode] $errormsg");
    $fh = fopen("sessions/history", "a"); fwrite($fh, "Connection Failed: [$errorcode] $errormsg\n"); fclose($fh);
		exit;
	}
  

	// Write to Visipedia Server requesting updated class probablities
	socket_write($socket, $query, strlen($query));

	// Read from socket
	$line = socket_read($socket, 500024, PHP_NORMAL_READ);
	$output = $line;

	// Close and return.
	socket_close($socket);
  
	$query = str_replace('\\"', '"', $json);
  
  
	return $output;
}

if(isset($_FILES['uploadedfile']['tmp_name']) && isset($_POST['json'])) {
        // Client uploaded an image with a form.  Assume new_session was already called.  Save the image
	$query = $_POST['json'];
	$query_decode = json_decode(str_replace('\\"', '"', $query), true);

        if($query_decode["session_id"] && $query_decode["session_dir"]) {
	  $sess_id = $query_decode["session_id"];
	  $sess_dir = $SESSIONS_DIR . '/' . SafeFileName($sess_id);//$query_decode["session_dir"];
	  $extension = SafeFileName($query_decode["extension"]);
          $src_path = $_FILES['uploadedfile']['tmp_name'];
	  $orig_path = $sess_dir . '/' . SafeFileName($sess_id) . "_orig." . SafeFileName($extension);
          if(strcasecmp($extension,"jpg") && strcasecmp($extension,"jpeg") && strcmp($extension,"png"))
	    $extension = "jpg";
	  $target_path = $sess_dir . '/' . SafeFileName($sess_id) . "." . $extension;
          $size=GetImageSize( $src_path ); 
	
  	  $fh = fopen("sessions/history", "a"); fwrite($fh, date("F j, Y, g:i a") . " " . $_SERVER['REMOTE_ADDR'] . ": " . $src_path . " " . $target_path . " " . $_FILES['uploadedfile']['tmp_name'] . " " . $query . "\n"); fclose($fh);

          copy($src_path, $orig_path); 
          exec("convert -size {$size[0]}x{$size[1]} $orig_path -thumbnail {$max_width}x{$max_height} $target_path"); 
        }
} elseif(is_array($_POST) && isset($_SERVER['HTTP_X_FILE_NAME'])) {
        // Client posted a raw image in base64 format.  Create a session and save the image
        $json = JSONRequest('{"method":"new_session","jsonrpc":"2.0","id":0,"mkdir":true}');
	if($json) {
	          $query = str_replace('\\"', '"', $json);
		  $res = json_decode($json, true);
		  if($res && $res["session_id"] && $res["session_dir"]) {
		  	  $sess_id = $res["session_id"];
                          $sess_dir = $SESSIONS_DIR . '/' . SafeFileName($sess_id);//$res["session_dir"];
                          $info = pathinfo($_SERVER['HTTP_X_FILE_NAME']);
                          $orig_path = $sess_dir . '/' . $sess_id . "_orig." . $info['extension'];
                          $extension = $info['extension'];
          		  if(strcasecmp($extension,"jpg") && strcasecmp($extension,"jpeg") && strcmp($extension,"png"))
	    		    $extension = "jpg";
			  $target_path = $sess_dir . '/' . $sess_id . "." . $extension;
			  $fh = fopen("sessions/history", "a"); fwrite($fh, "Save image HTTP_X $target_path\n"); fclose($fh);
		file_put_contents($orig_path, base64_decode(file_get_contents("php://input")));
        		  $size=GetImageSize( $orig_path ); 
			  if ($size[0] > $max_width || $size[1] > $max_height || strcasecmp($info['extension'],$extension)) {
	        	     exec("convert -size {$size[0]}x{$size[1]} $orig_path -thumbnail {$max_width}x{$max_height} $target_path"); 
                 $fh = fopen("sessions/history", "a"); fwrite($fh, "convert -size {$size[0]}x{$size[1]} $src_path -thumbnail {$max_width}x{$max_height} $orig_path $target_path\n"); fclose($fh);
			  } else {
			    //exec("ln -s $sess_id" ."_orig.$extension $target_path");
          copy($orig_path, $target_path);
          $fh = fopen("sessions/history", "a"); fwrite($fh, $sess_id . "_orig." . $extension ."\n"); fwrite($fh, $extension2 ."\n"); fclose($fh);
			  }
			  echo $json;
			  exit;
	          }
	}
        header("Content-type: text/plain");
        echo("new_session failed");
        exit;
} elseif(is_array($_GET) && isset($_GET['image_url'])) {
        // Client posted a raw image in base64 format.  Create a session and save the image
        $json = JSONRequest('{"method":"new_session","jsonrpc":"2.0","id":0,"mkdir":true}');
        $url = $_GET['image_url'];
        if($json) {
                  $query = str_replace('\\"', '"', $json);
                  $res = json_decode($json, true);
                  if($res && $res["session_id"] && $res["session_dir"]) {
                          $sess_id = $res["session_id"];
                          $sess_dir = $SESSIONS_DIR . '/' . SafeFileName($sess_id);//$res["session_dir"];
                          $info = pathinfo($url);
                          $extension = (strlen($info['extension']) ? $info['extension'] : "jpg");
                          $orig_path = $sess_dir . '/' . $sess_id . "_orig." . $extension;
                          $extension2 = $extension;
          		  if(strcasecmp($extension,"jpg") && strcasecmp($extension,"jpeg") && strcmp($extension,"png"))
	    		    $extension = "jpg";
                          $target_path = $sess_dir . '/' . $sess_id . "." . $extension;
			  $fh = fopen("sessions/history", "a"); fwrite($fh, "Save image wget $target_path $url\n"); fclose($fh);
                          exec("wget --no-check-certificate " . $url . " -O " . $orig_path);
                          $size=GetImageSize( $orig_path );
                          if ($size[0] > $max_width || $size[1] > $max_height || strcasecmp($extension2,$extension)) {
                             exec("convert -size {$size[0]}x{$size[1]} $orig_path -thumbnail {$max_width}x{$max_height} $target_path");
                             $fh = fopen("sessions/history", "a"); fwrite($fh, "convert -size {$size[0]}x{$size[1]} $src_path -thumbnail {$max_width}x{$max_height} $orig_path $target_path\n"); fclose($fh);
                          } else {
			    //exec("ln -s $sess_id" ."_orig.$extension $target_path");
          copy($orig_path, $target_path);
          $fh = fopen("sessions/history", "a"); fwrite($fh, "copy " . $sess_id . "_orig." . $extension ."\n".$extension2 ."\n");  fclose($fh);
			  }
                          $res["extension"] = $extension;
                          echo json_encode($res);
                          exit;
                  }
        }
        header("Content-type: text/plain");
        echo("new_session failed");
        exit;
} else if(isset($_GET['json'])) {
        echo JSONRequest($_GET['json']);
}

?>
