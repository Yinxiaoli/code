/* 
Copyright (C) 2010-11 Steve Branson

This code may not be redistributed without the consent of the authors.
*/
 
#include "20q_server.h"
#include "main.h"

/**
 * @example 20q_server.cpp
 *
 * This is a server that implements 
 *  -# The visual 20q game (multiclass classification and part localization, while intelligently choosing questions to pose to a human user to identify the true class as quickly as possible)
 *  -# Interactive labeling of a deformable part model
 *
 * The server listens on the network for incoming requests.  See html/20q.html and html/interactive_label.html for sample client applications
 *
 * The server takes as input the model file learned using \ref train_localized_v20q.cpp
 *
 * Example usage:
 *   - Start a new server on port 8086, where a client can connect in to add training examples:
\htmlonly <div style="padding: 0.5em 1em; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; background-color: #eaeafa;">
 $ examples/bin/release_static/20q_server.out -P 8086 -c CUB_200_2011_data/classes.txt.v20q
</div> \endhtmlonly
 * 
 * A web-based client for interactive part labeling is available in the directory examples/html/interactive_label.html.  
 * For this to work, it requires a webserver to be setup.  The steps to do this are:
 *   - <strong> Linux </strong> (Ubuntu):
 *    -# Install Apache, PHP, and imagemagick
\htmlonly <div style="padding: 0.5em 1em; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; background-color: #eaeafa;">
  $ sudo apt-get install apache2 php5 libapache2-mod-php5 imagemagick wget
</div> \endhtmlonly
 *    -# Make the server html directory accessible to the web server:
\htmlonly <div style="padding: 0.5em 1em; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; background-color: #eaeafa;">
 $ ln -s /home/steve/visipedia_toolbox/examples/html /var/www/visipedia
</div> \endhtmlonly
 *   - <strong> Windows: </strong>
 *    -# Download and install Apache: http://httpd.apache.org/docs/2.0/platform/windows.html#down
 *    -# Download and install PHP (make sure you download the one that works with Apache): http://windows.php.net/download/.  Make sure you install socket support (PHP_SOCKETS in php.ini)
 *    -# Download and install Imagemagick:  http://www.imagemagick.org/script/binary-releases.php#windows
 *    -# Download and install wget: http://gnuwin32.sourceforge.net/packages/wget.htm
 *    -# Make sure you can run the commands 'convert' and 'wget' from DOS prompt, and if not add the paths for Imagemagick and wget into your windows system path 
 *    -# Edit Apache config file C:\\Program Files (x86)\\Apache Software Foundation\\Apache2.2\\conf\\httpd.conf, adding the following lines:
\htmlonly
<div style="padding: 0.5em 1em; border-top: 1px solid #eed; border-bottom: 1px solid #eed; background-color: #f4f4ea;">
#BEGIN PHP INSTALLER EDITS - REMOVE ONLY ON UNINSTALL
<br>PHPIniDir "C:\Program Files (x86)\PHP\"
<br>LoadModule php5_module "C:\Program Files (x86)\PHP\php5apache2_2.dll"
<br>#END PHP INSTALLER EDITS - REMOVE ONLY ON UNINSTALL
<br>
<br>&ltDirectory "C:\Users\Steve\Documents\visipedia_toolbox\examples\html"&gt
<br>&nbsp;&nbsp;&nbsp;&nbsp;    Order Allow,Deny
<br>&nbsp;&nbsp;&nbsp;&nbsp;    Allow from All
<br>&nbsp;&nbsp;&nbsp;&nbsp;    # Any other directory-specific stuff
<br>&lt/Directory&gt
<br>DocumentRoot "C:\Users\Steve\Documents\visipedia_toolbox\examples\html"
</div>
\endhtmlonly
 * where C:\\Users\\Steve\\Documents\\visipedia_toolbox is the directory of this toolbox and C:\\Program Files (x86)\\PHP is the directory where PHP was installed
 *    -# Restart Apache 
*/

/*
 */
#include "dataset.h"
#include "classes.h"
int main(int argc, const char **argv) {
  Visual20qRpc v;
  return v.main(argc, argv);
}

