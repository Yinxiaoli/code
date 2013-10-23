

function DrawableContainer(idName, imgSrc, canEdit, id, onLoad, width, height) {
    this.Init = function () {
        this.canEdit = canEdit;
        this.image = document.createElement('img');
        this.image.style.display = "none";
	this.children = [];
	this.ind = parseInt(id);

        var t = this;
        this.image.onload = function () {
            if (!width) {
                width = t.image.width;
                height = t.image.height;
            }
            t.canvas = Raphael(idName, width, canEdit ? height + 30 : height);
            t.canvasImage = t.canvas.image(imgSrc, 0, 0, width, height);
            t.zoom = (t.image.width > t.image.height ? width / t.image.width : height / t.image.height);
            t.canvasImage.attr({ "src": t.image.src, "width": (t.image.width * t.zoom), "height": (t.image.height * t.zoom) });
	    if(onLoad) onLoad(t);
        }
        this.image.src = imgSrc;
    }

    this.AddElement = function(e) {
	this.children[this.children.length] = e;
    }

    this.Draw = function() {
	for(var i = 0; i < this.children.length; i++) 
	    this.children[i].Draw();
    }

    this.Init();
}

function DrawablePartMotion(container, yFrom, yTo, model, style) {
    this.Init = function() {
	this.motionLines = [];
	for(var i = 0; i < model.parts.length; i++) {
	    this.motionLines[i] = container.canvas.path("M10,20L30,40");
	    this.motionLines[i].attr(style.drag);
	}
    }

    this.Draw = function() {
	for(var i = 0; i < model.parts.length; i++) {
            var loc1 = yFrom.part_locations[i];
            var loc2 = yTo.part_locations[i];
            if(loc1.visible && loc2.visible) {
		this.motionLines[i].attr({"path" : "M" + (loc1.x*container.zoom) + " " + (loc1.y*container.zoom) + 
					"L" +  (loc2.x*container.zoom) + " " + (loc2.y*container.zoom)});  
		this.motionLines[i].show();
            } else
		this.motionLines[i].hide();
	}
    }

    this.Init();
}

function DrawablePartInstance(container, y, model, style) {
    this.Draw = function() {
      var numNonVisible = 0;
      for(var i = 0; i < model.parts.length; i++) {
        var loc = this.isAnchored[i] ? this.isAnchored[i] : y.part_locations[i];
        if(!loc.visible && !this.partCircles[i].dragging && container.canEdit) {
          loc.x = (130+numNonVisible*20)/container.zoom
          loc.y = (container.image.height*container.zoom+15)/container.zoom;
          numNonVisible++;
        }
        this.partCircles[i].attr({"cx":(loc.x*container.zoom),"cy":(loc.y*container.zoom)});
        if(this.partKeys[i]) this.partKeys[i].attr({"x":(loc.x*container.zoom),"y":(loc.y*container.zoom)});
        if(this.partLabels[i]) {
	  this.partLabels[i].attr({"x":(loc.x*container.zoom),"y":(loc.y*container.zoom-style.circle.radius-5)});
          if(style.show_pose_name) {
            if(style.show_score) this.partLabels[i].attr("text", model.poses_by_name[part_locs[i].pose].pose_name + "(" + y.part_locations[i].score + ")");
            else this.partLabels[i].attr("text", model.poses_by_name[y.part_locations[i].pose].pose_name);
          }
        }
        if(this.partLines[i]) {
          var parent_loc = this.isAnchored[model.parts[i].parent_id] ? this.isAnchored[model.parts[i].parent_id] : y.part_locations[model.parts[i].parent_id];
          if(loc.visible && parent_loc.visible) {
            this.partLines[i].attr({"path" : "M" + (loc.x*container.zoom) + " " + (loc.y*container.zoom) + 
				    "L" +  (parent_loc.x*container.zoom) + " " + (parent_loc.y*container.zoom)});  
            this.partLines[i].show();
          } else
            this.partLines[i].hide();
        }
      }
    }

    this.Init = function() {
	this.partShapes = [];
	this.partCircles = [];
	this.partLines = [];
	this.partKeys = [];
	this.partLabels = [];
	this.partStyles = [];
	this.isAnchored = [];

	for(var i = 0; i < model.parts.length; i++) {
	    this.isAnchored[i] = null;
            this.partShapes[i] = container.canvas.set();
            if(model.parts[i].parent_id >= 0) {
		this.partLines[i] = container.canvas.path("M10,20L30,40");
		this.partLines[i].attr(style.line);
		this.partShapes[i].push(this.partLines[i]);
            } else
		this.partLines[i] = null;

            var ii = i;
            var color1 = style.colors1[i%15];
            var color2 = model.parts.length > 15 ? style.colors1[i/15] : style.colors2[i%2];
            this.partStyles[i] = { 'fill':color1, 'stroke':color2 };
	    this.partCircles[i] = container.canvas.circle(0, 0, style.circle.radius);
            this.partCircles[i].attr(style.circle);
            this.partCircles[i].attr(this.partStyles[i]);
            this.partShapes[i].push(this.partCircles[i]);
            this.partLabels[i] = container.canvas.text(0,0,model.parts[i].part_name);

            this.partShapes[i].i = i;
            this.partCircles[i].i = this.partCircles[i].node.i = i;

            if(model.parts[i].abbreviation) {
		this.partKeys[i] = container.canvas.text(0,0,model.parts[i].abbreviation);
		this.partKeys[i].attr(style.key);
		this.partKeys[i].attr({"fill" : color2});
		this.partKeys[i].i = this.partKeys[i].node.i = i;
		this.partShapes[i].push(this.partKeys[i]);
            }

            this.partLabels[i].attr(style.text);
            this.partShapes[i].push(this.partLabels[i]);
            if(!style.show_pose_name) this.partLabels[i].hide();
	}
	
	if(container.canEdit)
	    this.InitDraggableControls();
    }

    this.MakeInteractive = function(session) {
	this.onPartMoved = FinalizePartLocation;
	this.onPartDrag = PreviewPartLocation;
    }

    this.InitEditControls = function() {
	this.nonvisibleRect = container.canvas.rect(0,0, 0,0);
	this.nonvisibleRect.attr(style.nonvisible_rect);
	this.nonvisibleLabel = container.canvas.text(5,0,"Nonvisible parts:");
	this.nonvisibleLabel.attr(style.nonvisible_text);
        this.nonvisibleLabel.attr({"x":5,"y":(container.image.height*container.zoom)+20});
        this.nonvisibleRect.attr({"x":0,"y":(container.image.height*container.zoom),"width":(container.image.width*container.zoom),"height":30});
    
	this.dragLine = container.canvas.path();
	this.dragLine.attr(style.drag);
    }

    this.InitDraggableControls = function() {
	for(var i = 0; i < model.parts.length; i++) {
          var ii = i;
          var t = this;
	    this.partShapes[i].attr("cursor", "pointer");

          var start = function () {
              var ii = this.i;
              this.pt_x = y.part_locations[ii].x;
              this.pt_y = y.part_locations[ii].y;
              t.partCircles[ii].dragging = true;
              t.dragLine.attr("path", "M" + (this.pt_x*container.zoom) + " " + (this.pt_y*container.zoom) + "L" +  (this.pt_x*container.zoom) + " " + (this.pt_y*container.zoom));  
              t.dragLine.show();
          },
          move = function (dx, dy) {
              var ii = this.i;
              y.part_locations[ii].x = this.pt_x + dx/container.zoom;
              y.part_locations[ii].y = this.pt_y + dy/container.zoom;
              y.part_locations[ii].visible = y.part_locations[ii].y < container.image.height;
              t.isAnchored[ii] = y.part_locations[ii];
              if(t.onPartDrag) t.onPartDrag(y, ii, container.Draw);
              t.dragLine.attr("path", "M" + (this.pt_x*container.zoom) + " " + (this.pt_y*container.zoom) + 
			      "L" +  (y.part_locations[ii].x*container.zoom) + " " + (y.part_locations[ii].y*container.zoom));  
	      t.Draw();
          },
          up = function () {
              var ii = this.i;
              t.partCircles[ii].dragging = false;
              if(t.onPartMoved) t.onPartMoved(y, ii, container.Draw);
              t.dragLine.hide();
          },
	  hover = function (e) {
              t.partCircles[this.i].attr(style.circle_over);
              t.partLabels[this.i].attr("text", model.poses_by_name[y.part_locations[this.i].pose].pose_name);
              t.partLabels[this.i].show();
          },
          exit = function (e) {
	      t.partCircles[this.i].attr(t.isAnchored[this.i] ? { 'fill':'#FF0000', 'stroke':'#FF0000' } : t.partStyles[this.i]);
	      if(!style.show_pose_name) t.partLabels[this.i].hide();
          };

          this.partCircles[i].node.onmouseup = this.partCircles[i].node.onmouseout = exit;
          this.partCircles[i].node.onmouseover = hover;
	  if(model.parts[i].abbreviation) {
	      this.partKeys[i].i = i;
	      this.partKeys[i].node.onmouseup = this.partKeys[i].node.onmouseout = exit;
	      this.partKeys[i].node.onmouseover = hover;
	  }
          this.partShapes[i].drag(move, start, up);
      }
    }

    if(container.canEdit) 
	this.InitEditControls();
    this.Init();
}



function DefaultDrawStyle() {
    this.colors1 = ["#FF0000", "#800000", "#00FF00", "#008000", "#FFBF4A", "#000080", "#FFFF00", "#626200",
                     "#00FFFF", "#006262", "#FF00FF", "620062", "#FFFFFF", "#000000", "#44200F"];
    this.colors2 = ["#000000", "#FFFFFF"];
    this.circle = { "radius": 7, "stroke-width": 2 };
    this.circle_over = { "stroke": "#00FF00", "fill": "#00FF00" };
    this.line = { "stroke-width": 3, "stroke": "#0000FF" };
    this.drag = { "stroke-width": 5, "stroke": "#00FF00" };
    this.text = { "font-family": "Times New Roman", "font-size": 16, "font-style": "normal",
        "font-weight": "normal", "text-anchor": "middle", "fill": "#FF0000", "fill-opacity": 1
    };
    this.key = { "font-family": "Times New Roman", "font-size": 11, "font-style": "normal",
        "font-weight": "normal", "text-anchor": "middle", "fill": "#0000FF", "fill-opacity": 1
    };
    this.nonvisible_text = { "font-family": "Times New Roman", "font-size": 16, "font-style": "normal",
        "font-weight": "normal", "text-anchor": "start", "fill": "#000000", "fill-opacity": 1
    };
    this.nonvisible_rect = { "stroke-width": 1, "stroke": "#6060A0", "fill": "#A0A0FF" };
    this.show_pose_name = false;
    this.show_score = false;
}


function ThumbnailDrawStyle() {
    this.colors1 = ["#FF0000", "#800000", "#00FF00", "#008000", "#FFBF4A", "#000080", "#FFFF00", "#626200",
                     "#00FFFF", "#006262", "#FF00FF", "620062", "#FFFFFF", "#000000", "#44200F"];
    this.colors2 = ["#000000", "#FFFFFF"];
    this.circle = { "radius": 4, "stroke-width": 1 };
    this.circle_over = { "stroke": "#00FF00", "fill": "#00FF00" };
    this.line = { "stroke-width": 2, "stroke": "#0000FF" };
    this.drag = { "stroke-width": 3, "stroke": "#00FF00" };
    this.text = { "font-family": "Times New Roman", "font-size": 9, "font-style": "normal",
        "font-weight": "normal", "text-anchor": "middle", "fill": "#FF0000", "fill-opacity": 1
    };
    this.key = { "font-family": "Times New Roman", "font-size": 6, "font-style": "normal",
        "font-weight": "normal", "text-anchor": "middle", "fill": "#0000FF", "fill-opacity": 1
    };
    this.nonvisible_text = { "font-family": "Times New Roman", "font-size": 9, "font-style": "normal",
        "font-weight": "normal", "text-anchor": "start", "fill": "#000000", "fill-opacity": 1
    };
    this.nonvisible_rect = { "stroke-width": 1, "stroke": "#6060A0", "fill": "#A0A0FF" };
    this.show_pose_name = false;
    this.show_score = false;
}

var defaultDrawStyle = new DefaultDrawStyle();
var thumbnailDrawStyle = new ThumbnailDrawStyle();
