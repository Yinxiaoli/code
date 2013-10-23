import sys, os
import json
from pprint import pprint

def processLog(userId, imageId, ver=None):
	
	path = "logs/" + str(userId) + "/" + str(imageId) + "/"
	
	if not ver:
		min_ver = -1
		dirList=os.listdir(path)	
		for fname in dirList:
			v = int(fname.split("_")[1].split(".")[0])
			if v > min_ver:
				min_ver = v
				ver = v
	
	log_path = path + "log_" + str(ver) + ".txt"
	
	with open(log_path) as f:
		data = json.load(f)
	
	timeline = []
	times = []
	
	questionData = []
	
	for section in data:
		
		if section == "start_session" :
			timeline.append("start session")
			times.append(data[section]["time"])
		
		elif section == "end_session" :
			timeline.append("end session")
			times.append(data[section]["time"])
		
		elif section == "results" :
			for event in data[section] :
				timeline.append("received new results") 
				times.append(event["time"])
			
		
		elif section == "questions" :
			for event in data[section] :
				if event["stage"] == "start" :
					timeline.append("received question");
				else:
					timeline.append("answered question")
				
				times.append(event["time"])
			
			questions = data[section]
			questions.sort(key=lambda ele: ele["time"])
			for q in questions:
				if q["stage"] == "start" :
					qd = {"start" : q["time"], "id" : q["id"]}
					questionData.append(qd)
				else:
					qd = questionData[-1];
					if ( qd["id"] != q["id"]):
						print "Error Question Mismatch!"
				 	else:
				 		qd["end"] = q["time"]
			
		
		elif section == "detail_views" :
			for event in data[section] :
				timeline.append("viewed details")
				times.append(event["time"])
			
		
		elif section == "removals" :
			for event in data[section] :
				timeline.append("removed category")
				times.append(event["time"])
	
	comb = zip(timeline, times)
	comb.sort(key=lambda tup: tup[1]) 
	pprint(comb)
	
	pprint(questionData)
	for question in questionData:
		if "end" in question:
			time = int((int(question["end"]) - int(question["start"])) / 1000.0)
			print "Question %d took %d seconds" % (int(question["id"]), time)
			
	if "end_session" in data:
		total_time = int((int(data["end_session"]["time"]) - int(data["start_session"]["time"])) / 1000.0)
		print "Total Time: %d seconds" % (total_time) 	
			