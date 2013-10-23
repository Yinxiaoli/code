;(function ( namespace, undefined ) {  
	
	var questionDiv = null
	
	var isFirst = true;
	var isSelected = null;
	var choicesTable=null, nextButtons = new Array(), choicesRow=null;
	var numQuestions = 0;
	
	var num_click_questions = 0;
	
	var numChoicesPerRow = 6;
	
	var extraInstructions = new Array();
    extraInstructions["binary"] = "Select yes or no. If the applicable part isn't visible, make your best guess, then select \"Can't Tell\".";
    extraInstructions["multiple_choice"] = "Select one. If the applicable part isn't visible, make your best guess, then select \"Can't Tell\".";
    extraInstructions["batch"] = "Select at least one. If the applicable part isn't visible, make your best guess, then select \"Can't Tell\".";
    extraInstructions["part_click"] = "Click on the applicable part in the uploaded image to the left.  If the part is not visible in the image, click 'Not Visible'.";
    
    var question = null;
    var certainties = null;
    
    namespace.Initialize = function(){
    	questionDiv = document.getElementById('questionDiv'); 
    	$("#questionDiv").hide();
    }
	
	namespace.getSelected = function(type){
		if(question['type'] == 'binary') {
			return isSelected[0];
		} else if(question['type'] == 'multiple_choice' || question['type'] == 'batch') {
			return isSelected;
		}
		else{
			alert("Error with getSelected. Should not be here");
		}
	}
	
	namespace.reset = function(){
	
		numQuestions = 0;
		num_click_questions = 0;
		$("#questionDiv").fadeOut(500, function() {
			while (questionDiv.hasChildNodes())
			questionDiv.removeChild(questionDiv.firstChild);
		});
	
	}
	
	namespace.increment = function(){
		numQuestions++;
	}
	
	namespace.incrementClickQuestions = function(){
		num_click_questions++;
	}

	namespace.getNumClickQuestions = function(){
		return num_click_questions;
	}
	namespace.getNumkQuestions = function(){
		return numQuestions;
	}

    // Draw the row of top-ranked classes
    namespace.UpdateQuestion = function(new_question, new_certainties) {
    	
    	question = new_question;
    	certainties = new_certainties;
    	
		isFirst = false;
		isSelected = new Array;
		selected = new Array();
		var td = new Array;
		var tdLab = new Array;
		var tip = new Array;  
		
		var infoDiv = document.createElement('div');
		infoDiv.className = 'infobox';
		
		// The HTML where the question being posed is described
		var qDiv = document.createElement('div');
		qDiv.className = 'questionText';
		qDiv.innerHTML = question['part_visualization'] ? '<table><tr><td><img width=100 src="'+question['part_visualization']+'"/></td><td>' +  question['question_text'] + '</td></tr></table>' : question['question_text'];
		
		infoDiv.appendChild(qDiv);
        
		while (questionDiv.hasChildNodes()){
			questionDiv.removeChild(questionDiv.firstChild);
		}

		// The HTML for additional instructions
		var instructionsDiv = document.createElement('div');
		instructionsDiv.className = 'instructionsText';
		instructionsDiv.innerHTML = extraInstructions[question['type']];
		infoDiv.appendChild(instructionsDiv);
		questionDiv.appendChild(infoDiv);
        
      
		// A table with a selectable list of attribute check responses
		if(question['type'] == 'binary' || question['type'] == 'multiple_choice' || question['type'] == 'batch') {
			var attributeTable = document.createElement('table');
			attributeTable.className = 'attributeTable';
			var attributeTableBody = document.createElement('tbody');
			attributeTable.appendChild(attributeTableBody);
			var col = 0, row = 0;
			var currRow = null;
			attributeInputs = new Array;
			var choices = question['choices'];
			var maxSelect = 8;
		        var numChoicesPerRow2 = numChoicesPerRow;
		        var maxWidth = null;
			if(question['type'] == 'binary') {
				choices = eval('(' + '[{"attribute_value":"yes"},{"attribute_value":"no"}]' + ')');
				maxSelect = 1;
			} 
			else if(question['type'] == 'multiple_choice'){
	  			maxSelect = 1;
	  		}
		        if(choices.length > numChoicesPerRow*2) {
			    numChoicesPerRow2 = Math.floor((choices.length+1)/2);
			    maxWidth = '70px';
			}
			for(var i = 0; i < choices.length; i++) {
				if(col == 0) {
					if(row != 0){
				  		attributeTableBody.appendChild(currRow);
				  	}
					row++;
					currRow = document.createElement('tr');
			  	}
			  	col++;
				isSelected[i] = 0;
				td[i] = document.createElement('td');
				td[i].className = 'attributeCell';
				if(choices[i]["attribute_visualization"]) {
					var img = document.createElement('img');
					img.className = 'attributeImage';
					img.src = choices[i]["attribute_visualization"];
				        if(maxWidth) img.style.maxWidth = maxWidth;
					td[i].appendChild(img);
				}
				tdLab[i] = document.createElement('div');
				tdLab[i].className = 'attributeLabel';
				tdLab[i].innerHTML = choices[i]["attribute_value"].replace(/\_/g," ");
				td[i].appendChild(document.createElement('br'));
				td[i].appendChild(tdLab[i]);
				td[i].i = i; 
				td[i].clickAttribute = function() {
					var i = this.i;
					isSelected[i] = isSelected[i] ? 0 : 1;
					td[i].className = isSelected[i] ? 'attributeCell selected' : 'attributeCell';
					tdLab[i].className = isSelected[i] ? 'attributeLabel selected' : 'attributeLabel';
					
					if(isSelected[i]) {
						selected[selected.length] = i;
						if(selected.length > maxSelect){
							td[selected[0]].clickAttribute();
						}
					} 
					else {
						for(var j = 0; j < selected.length; j++) {
							if(selected[j] == i) {
								selected.splice(j,1);
								break;
							}
						}
					}
					namespace.EnableButtons();
				};
				td[i].onclick = td[i].clickAttribute;
				
				if(choices[i]["tooltip"]){
					AddTip(td[i], choices[i]["tooltip"]);          
				}
				
				currRow.appendChild(td[i]);
				
				if(col >= numChoicesPerRow2){
					col = 0;
				}
			} 
	
			attributeTableBody.appendChild(currRow);
			questionDiv.appendChild(attributeTable);
			window.UserImage.Reset();
		} 
		else {
			window.UserImage.PreparePartClick();
		}
      
		AddButtonControls();	
		namespace.EnableButtons();	
		window.UserImage.HidePartBox();
		
		$("#questionDiv").fadeIn(500, function() { window.Server.StartTiming(); } );
		
    }
   
    
      
    // Instructions for selecting the confidence of an answer
    function AddButtonControls() {  
		// Buttons for selecting the confidence of an answer and going 
		// to the next question
		choicesTable = document.createElement('table');
		choicesRow = document.createElement('tr');
		var tBody = document.createElement('tbody');
		tBody.appendChild(choicesRow);
		choicesTable.appendChild(tBody);
		if(question['type'] != 'part_click') {
			for(var i = 0; i < certainties.length; i++) {
				nextButtons[i] = new Button(certainties[i], window.Server.SubmitAnswer, 'buttonNext', certainties[i]);
				choicesRow.appendChild(nextButtons[i].td);
			}
		} 
		else{
			nextButtons[0] = new Button('Not Visible', window.Server.SubmitAnswer, 'buttonNext', false);
			choicesRow.appendChild(nextButtons[0].td);
			//if(!gParameters.auto_submit_part_clicks) {
			nextButtons[1] = new Button('Next', window.Server.SubmitAnswer, 'buttonNext', true);
			choicesRow.appendChild(nextButtons[1].td);
			//}
		}
		questionDiv.appendChild(choicesTable);
    }
      
   namespace.EnableButtons = function(hasClick) {
   		hasClick = hasClick || false;
		if(question['type'] != 'part_click') {
			e = selected.length > 0;
			for(var i = 0; i < certainties.length; i++) {
			  nextButtons[i].SetEnabled(e || certainties[i] == "not_visible" || certainties[i] == "not visible");
			}
		} 
		else {
			nextButtons[0].SetEnabled(true);
			nextButtons[1].SetEnabled(hasClick);
		} 
    };
    
    
	
     
})(window.Question = window.Question || {});  

function Button(name, func, id, args) {
    
	this.element = document.createElement('span');
	this.element.className = 'button';
	this.element.id = id;
	this.icon = document.createElement('span');
	this.icon.className = 'icon';
	this.icon.innerHTML = name;
	this.element.appendChild(this.icon);
	this.element.func = func;
	this.element.args = args;
	this.element.isEnabled = true;
	this.td = document.createElement('td');
	this.td.appendChild(this.element);
	
	this.element.onclick = function() {
		if(this.isEnabled)
			this.func(this.args);
		};
		
		this.SetEnabled = function(e) {
		this.element.isEnabled = e;
		this.element.className = e ? 'button' : 'buttonDis';
	};
	
};
