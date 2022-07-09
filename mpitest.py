import boto3
import botocore
import cv2 as opencv
import numpy as np
import datetime
import json
from PIL import Image
import io
from mpi4py import MPI

def SortFunc(a):
	return a[u'Confidence']

def PrintEmotions(json_data):
	Emotions = json_data[u'FaceDetails'][0][u'Emotions']
	Emotions.sort(key=SortFunc, reverse=True)
	for index in [0, 1, 2]:
		print Emotions[index][u'Type'], "\t","{:.2f}".format(Emotions[index][u'Confidence'])

Comm = MPI.COMM_WORLD	
rank = Comm.Get_rank()
print rank

WebCam = opencv.VideoCapture(0)
framesPerSecond = WebCam.get(opencv.CAP_PROP_FPS)
timePerFrame = int(1000./framesPerSecond)
figName = "temp.jpeg"
Success, frame = WebCam.read(0)
shape = frame.shape
WebCam.release()
print shape[0]*shape[1]*shape[2], 480*640*3

if rank == 0: ### CAM STUFF

	shape_ = np.array([shape[0], shape[1], shape[2]])

	Comm.Send(shape_, dest=1, tag = 0)
	ArraySize = shape[0]*shape[1]*shape[2]

	opencv.imshow("asdf", frame)
	while True:
		key = opencv.waitKey(1)
		if key == ord('q'):
			break
#	buf = np.arange(ArraySize, dtype=np.uint8)
	buf = frame.reshape(ArraySize, order = "C")
	print np.sum(buf)
	Comm.Send(buf, dest = 1, tag = 0)


else: ##rekognition

	shape = np.empty(3, dtype=np.int)
	Comm.Recv(shape, source = 0, tag = 0)
	ArraySize = shape[0]*shape[1]*shape[2]
	buf = np.empty(ArraySize, dtype=np.uint8)
	Comm.Recv(buf, source = 0, tag = 0)
	buf = np.reshape(buf, shape, order = "C")

	print np.sum(buf)
	opencv.imshow("cam", buf)
	while True:
		key = opencv.waitKey(1)
		if key == ord('q'):
			break
	opencv.destroyAllWindows()


