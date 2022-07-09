import numpy as np
import cv2 as opencv
import boto3
import botocore
from PIL import Image
import io
import json

def SortFunc2(a):
	return a[u'Type']

def PrintEmotions(json_data):
	Emotions = json_data[u'FaceDetails'][0][u'Emotions']
	Emotions.sort(key=SortFunc2)
	for index in np.arange(7):
		print Emotions[index][u'Type'], "\t","{:.2f}".format(Emotions[index][u'Confidence'])

def GetEmotions(json_data):
	Emotions = json_data[u'FaceDetails'][0][u'Emotions']
	Emotions.sort(key=SortFunc2)
	return Emotions

def GetRectangle(json_data, shape):
	Coord = json_data[u'FaceDetails'][0][u'BoundingBox']
	frame_y_size = shape[0]
	frame_x_size = shape[1]
	Top = int(Coord[u'Top']*frame_y_size)
	Width = int(Coord[u'Width']*frame_x_size)
	Left = int(Coord[u'Left']*frame_x_size)
	Height = int(Coord[u'Height']*frame_y_size)
#	print [Left, Top, Width,Height]
	temp = [u if u >= 0 else 0 for u in [Left, Top, Width+Left,Top + Height] ]
	return np.array(temp, dtype='i')

def GenerateDummyResults(shape):
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

def ProbeRecv(Comm, buf, size, Type, source_, tag_, sleepTime, timeOut = 0):
	StartTime = time.time()

	flag = Comm.Iprobe(source = source_, tag = tag_)
	while flag == False:
		flag = Comm.Iprobe(source = source_, tag = tag_)
		time.sleep(sleepTime)
		if time.time() - StartTime > timeOut:
			return False
	Comm.Recv([buf, size, Type], source = source_, tag = tag_)
	return True
		

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

s3_connection = boto3.resource('s3')
bucketName = "videojpg"
bucket = s3_connection.Bucket(bucketName)
client = boto3.client("rekognition")

frame = opencv.imread("peopleEmotions.jpeg", 1)
shape = frame.shape
opencv.imshow("imagem", frame)
key = opencv.waitKey(0)
opencv.destroyAllWindows()

#buf = np.empty( shape, np.int )
buf = frame.reshape(shape, order="C")
image = Image.fromarray(buf)
stream = io.BytesIO()
if 'exif' in image.info:
	exif=image.info['exif']
	image.save(stream,format=image.format, exif=exif)
else:
	image.save(stream, format="jpeg")#
image_binary = stream.getvalue()
stream.close()

result = client.detect_faces( Image={"Bytes":image_binary} , Attributes=['ALL', 'DEFAULT'] )
with open("data.json", 'w') as outfile:
	json.dump(result, outfile)
PrintEmotions(result)
Emotions = GetEmotions(result)
