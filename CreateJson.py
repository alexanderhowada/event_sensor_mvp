import numpy as np
import json
import random
import numpy as np

def SortFunc(a):
	return a[u'Confidence']

def PrintEmotions(json_data):
	Emotions = json_data[u'FaceDetails'][0][u'Emotions']
	Emotions.sort(key=SortFunc, reverse=True)
	for index in [0, 1, 2]:
		print Emotions[index][u'Type'], "\t","{:.2f}".format(Emotions[index][u'Confidence'])

def GetRectangle(json_data, shape):
	Coord = json_data[u'FaceDetails'][0][u'BoundingBox']
	frame_y_size = shape[0]
	frame_x_size = shape[1]
	Top = int(Coord[u'Top']*frame_y_size)
	Width = int(Coord[u'Width']*frame_x_size)
	Left = int(Coord[u'Left']*frame_x_size)
	Height = int(Coord[u'Height']*frame_y_size)
	return np.array([Top, Left, Width+Left,Top + Height], dtype='i')


def GenerateDummyResults(data, shape):

	json_data = {u'FaceDetails': []}
	json_data[u'FaceDetails'].append( {u'Emotions': [], u'BoundingBox' : {}} )

	rndArray = np.random.random(7)
	rndArray = rndArray/np.sum(rndArray)
	json_data[u'FaceDetails'][0][u'Emotions'].append({u'Type': u'CALM', u'Confidence'      : rndArray[0]})
	json_data[u'FaceDetails'][0][u'Emotions'].append({u'Type': u'DISGUSTED', u'Confidence' : rndArray[1]})
	json_data[u'FaceDetails'][0][u'Emotions'].append({u'Type': u'SURPRISED', u'Confidence' : rndArray[2]})
	json_data[u'FaceDetails'][0][u'Emotions'].append({u'Type': u'HAPPY', u'Confidence'     : rndArray[3]})
	json_data[u'FaceDetails'][0][u'Emotions'].append({u'Type': u'ANGRY', u'Confidence'     : rndArray[4]})
	json_data[u'FaceDetails'][0][u'Emotions'].append({u'Type': u'CONFUSED', u'Confidence'  : rndArray[5]})
	json_data[u'FaceDetails'][0][u'Emotions'].append({u'Type': u'SAD', u'Confidence'       : rndArray[6]})

	rndArray = np.random.random(3)
	rndArray = rndArray/np.sum(rndArray)
	json_data[u'FaceDetails'][0][u'BoundingBox'][u'Top'] = rndArray[0]
	json_data[u'FaceDetails'][0][u'BoundingBox'][u'Width'] = rndArray[1]
	rndArray = np.random.random(3)
	rndArray = rndArray/np.sum(rndArray)
	json_data[u'FaceDetails'][0][u'BoundingBox'][u'Left'] = rndArray[0]
	json_data[u'FaceDetails'][0][u'BoundingBox'][u'Height'] = rndArray[1]
	return json_data

json_data = []
json_data = GenerateDummyResults((1,2,3))
PrintEmotions(json_data)
print GetRectangle(json_data, (480,640,3))
