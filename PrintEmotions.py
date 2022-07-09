import numpy as np
import json
import _Functions_ as FCN

def SortFunc(a):
	return a[u'Confidence']

def GetEmotions(json_data):
	Emotions = json_data[u'FaceDetails'][0][u'Emotions']
	print type(json_data[u'FaceDetails'])
	Emotions.sort(key=SortFunc, reverse = True)
	return Emotions

data = []
with open("peoples.json", "r") as json_file:
	data.append(json.load(json_file))

print FCN.SaveRekognition(1, "table_name", data[0])

