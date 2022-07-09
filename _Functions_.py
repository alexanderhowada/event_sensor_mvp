import time
import numpy as np
import json
from mpi4py import MPI
import time
import sys

def SortFunc(a):
	return a[u'Confidence']

def SortFunc2(a):
	return a[u'Type']

def PrintEmotions(json_data):
	Emotions = json_data[u'FaceDetails'][0][u'Emotions']
	Emotions.sort(key=SortFunc2)
	for index in np.arange(7):
		print Emotions[index][u'Type'], "\t","{:.2f}".format(Emotions[index][u'Confidence'])

def GetEmotions(json_data):
	Emotions = []
	Emotions.append({u'Type': u'ANGRY', u'Confidence'     : 0})
	Emotions.append({u'Type': u'CALM', u'Confidence'      : 0})
	Emotions.append({u'Type': u'CONFUSED', u'Confidence'  : 0})
	Emotions.append({u'Type': u'DISGUSTED', u'Confidence' : 0})
	Emotions.append({u'Type': u'HAPPY', u'Confidence'     : 0})
	Emotions.append({u'Type': u'SAD', u'Confidence'       : 0})
	Emotions.append({u'Type': u'SURPRISED', u'Confidence' : 0})

	for Face in json_data[u'FaceDetails']:
		Face[u'Emotions'].sort(key=SortFunc2)
		for index in range(7):
			Emotions[index][u'Confidence'] = Emotions[index][u'Confidence'] + Face[u'Emotions'][index][u'Confidence']
#			print Face

#	Emotions = json_data[u'FaceDetails'][0][u'Emotions']
	return Emotions

def GetRectangle(json_data, shape):
	result = []
	for face in json_data[u'FaceDetails']:
		Coord = face[u'BoundingBox']
		frame_y_size = shape[0]
		frame_x_size = shape[1]
		Top = int(Coord[u'Top']*frame_y_size)
		Width = int(Coord[u'Width']*frame_x_size)
		Left = int(Coord[u'Left']*frame_x_size)
		Height = int(Coord[u'Height']*frame_y_size)
	#	temp = [u if u >= 0 else 0 for u in [Left, Top, Width+Left,Top + Height] ]
#		temp = [u if u >= 0 else 0 for u in [Left, Top, Width ,Height] ]
		result.append(Left)
		result.append(Top)
		result.append(Width)
		result.append(Height)
	return np.array(result, dtype='i')

def SaveRekognition(Database, table_name, json_data):
	for face in json_data['FaceDetails']:
		face[u'Emotions'].sort(key=SortFunc2)
		temp_string = " (Eyeglasses,Sunglasses,Gender,ANGRY,CALM,CONFUSED,DISGUSTED,HAPPY,SAD,SURPRISED,AgeHigh,AgeLow,Smile,Mustache,Beard) VALUES("
		Command = "INSERT INTO " + table_name + temp_string
		temp_value = 0

		###EyeGlasses
		if face['Eyeglasses']['Value'] == True:
			temp_value = face['Eyeglasses']['Confidence'] 
		else:
			temp_value = 100. - face['Eyeglasses']['Confidence']
		Command = Command + str(temp_value)+','
		###SunGlasses
		if face['Sunglasses']['Value'] == True:
			temp_value = face['Sunglasses']['Confidence'] 
		else:
			temp_value = 100. - face['Sunglasses']['Confidence']
		Command = Command + str(temp_value)+','
		###Gender - Male = 1, Female = 0
		if face['Gender']['Value'] == "Male":
			temp_value = face['Gender']['Confidence'] 
		else:
			temp_value = 100. - face['Gender']['Confidence']
		Command = Command + str(temp_value)+','
		###ANGRY,CALM,CONFUSED,DISGUSTED,HAPPY,SAD,SURPRISED
		for emotion in face['Emotions']:
			temp_value = emotion['Confidence']
			Command = Command + str(temp_value) + ','
			print emotion['Type'], emotion['Confidence']
		###AgeRange
		temp_value = face['AgeRange']['High']
		Command = Command + str(temp_value) + ','
		temp_value = face['AgeRange']['Low']
		Command = Command + str(temp_value) + ','
		###Smile
		if face['Smile']['Value'] == True:
			temp_value = face['Smile']['Confidence']
		else:			
			temp_value = 100.0 - face['Smile']['Confidence']
		Command = Command + str(temp_value) + ','
		###Mustache
		if face['Mustache']['Value'] == True:
			temp_value = face['Mustache']['Confidence']
		else:			
			temp_value = 100.0 - face['Mustache']['Confidence']  
		Command = Command + str(temp_value) + ','
		###Beard
		if face['Beard']['Value'] == True:
			temp_value = face['Beard']['Confidence']
		else:			
			temp_value = 100.0 - face['Beard']['Confidence']    
		Command = Command + str(temp_value) + ');'

		Database.Exec(Command)

def GenerateDummyResults(shape):
	json_data = {u'FaceDetails': []}
	for i in np.arange(np.random.randint(4)):
		json_data[u'FaceDetails'].append( {u'Emotions': [], u'BoundingBox' : {}} )

		rndArray = np.random.random(7)
		rndArray = rndArray/np.sum(rndArray)*100.
		json_data[u'FaceDetails'][i][u'Emotions'].append({u'Type': u'CALM', u'Confidence'      : rndArray[0]})
		json_data[u'FaceDetails'][i][u'Emotions'].append({u'Type': u'DISGUSTED', u'Confidence' : rndArray[1]})
		json_data[u'FaceDetails'][i][u'Emotions'].append({u'Type': u'SURPRISED', u'Confidence' : rndArray[2]})
		json_data[u'FaceDetails'][i][u'Emotions'].append({u'Type': u'HAPPY', u'Confidence'     : rndArray[3]})
		json_data[u'FaceDetails'][i][u'Emotions'].append({u'Type': u'ANGRY', u'Confidence'     : rndArray[4]})
		json_data[u'FaceDetails'][i][u'Emotions'].append({u'Type': u'CONFUSED', u'Confidence'  : rndArray[5]})
		json_data[u'FaceDetails'][i][u'Emotions'].append({u'Type': u'SAD', u'Confidence'       : rndArray[6]})

		rndArray = np.random.random(3)
		rndArray = rndArray/np.sum(rndArray)
		json_data[u'FaceDetails'][i][u'BoundingBox'][u'Top'] = rndArray[0]
		json_data[u'FaceDetails'][i][u'BoundingBox'][u'Width'] = rndArray[1]
		rndArray = np.random.random(3)
		rndArray = rndArray/np.sum(rndArray)
		json_data[u'FaceDetails'][i][u'BoundingBox'][u'Left'] = rndArray[0]
		json_data[u'FaceDetails'][i][u'BoundingBox'][u'Height'] = rndArray[1]
	return json_data

def ProbeRecv(Comm, buf, size, Type, source_, tag_, sleepTime, timeOut = 0):
	StartTime = time.time()
	flag = Comm.Iprobe(source = source_, tag = tag_)
	while flag == False:
		flag = Comm.Iprobe(source = source_, tag = tag_)
		time.sleep(sleepTime)
		if time.time() - StartTime > timeOut and timeOut > 0:
			return False
	Comm.Recv([buf, size, Type], source = source_, tag = tag_)
	return True


def ProbeRecvStatus(Comm, buf, Type, source_, tag_, sleepTime, status,timeOut = 0):
	StartTime = time.time()
	flag = Comm.Iprobe(source = source_, tag = tag_, status = status)
	while flag == False:
		flag = Comm.Iprobe(source = source_, tag = tag_, status = status)
		time.sleep(sleepTime)
		if time.time() - StartTime > timeOut and timeOut > 0:
			return status
	Comm.Recv([buf, status.Get_elements(Type), Type], source = source_, tag = tag_)
	return status

def GetInput():
	Input = ""
	try:
		Input = raw_input()
	except:
		print "stuff"
		pass
	return Input

def IsFloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def PlotBar(x, y):
	plt.bar(xx[2:], CumEmotions[2:], color=np.random.rand(3), width = 1, align='center')
	plt.xticks(xx[2:], my_xticks, rotation = 45)
