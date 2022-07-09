import cv2 as opencv
import numpy as np
import json

def PrintEmotions(json_data):
	Emotions = json_data[u'FaceDetails'][0][u'Emotions']
	Emotions.sort(key=SortFunc, reverse=True)
	for index in [0, 1, 2]:
		print Emotions[index][u'Type'], "\t","{:.2f}".format(Emotions[index][u'Confidence'])

def GetRectangle(json_data):
	Coord = json_data[u'FaceDetails'][0][u'BoundingBox']
	return [ Coord[u'Width'], Coord[u'Top'], Coord[u'Height'], Coord[u'Left'] ]

frame = opencv.imread("temp.jpeg")

opencv.imshow("frame", frame)

data = []
with open("data.json") as json_file:
	data.append(json.load(json_file))

#print data[0][u'FaceDetails'][0][u'Emotions']
Rectangle = GetRectangle(data[0])
#print Rectangle	
print frame.shape[:2]


while True:
	key = opencv.waitKey(100)
	if key == ord('q'):
		break


