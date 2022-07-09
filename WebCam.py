import boto3
import botocore
import cv2 as opencv
import numpy as np
import json
from PIL import Image
import io
from mpi4py import MPI
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random as rnd
import time
import sys
import Tkinter as tkinter

import _SQLite_ as SQLite
import _Functions_ as FCN
import _Gui_

import time
import numpy as np
import json
from mpi4py import MPI
import time
import sys
import argparse


Comm = MPI.COMM_WORLD	
rank = Comm.Get_rank()

KeepGoing = np.array([1], dtype ='i')
BoundingBox = np.array([0, 0, 0, 0], dtype='i')

_MAINCAM_RANK_ = 2
_MASTER_RANK_ = 0
_CAMWORKER_RANK_ = 1
_PLOT_RANK_ = 3

Database = SQLite._SQLite_("Database.db")

face_cascade = opencv.CascadeClassifier('haarcascade_frontalface_default.xml')
Face_scale = 1.3

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=False, help="path to video")
args = vars(ap.parse_args())

if rank == _MAINCAM_RANK_: ### CAM STUFF

	InputMedia = 0
	TimeToSave = 1e300
	framesPerSecond = 1
	timePerFrame = 1
	if args['video'] is not None:
		InputMedia = args['video']
	WebCam = opencv.VideoCapture(InputMedia)
	if args['video'] is not None:
		TimeToSave = 1e300
		framesPerSecond = WebCam.get(opencv.CAP_PROP_FPS)
		timePerFrame = int(1000./framesPerSecond)

	Success, frame = WebCam.read()

#	colored
	shape = np.array(frame.shape, dtype='i')
	ArraySize_color = np.prod(shape)
	for dest_ in [_PLOT_RANK_]:
		Comm.Send(shape, dest = dest_, tag = 0)

	frame = opencv.cvtColor(frame, opencv.COLOR_BGR2GRAY)
	shape = np.array(frame.shape, dtype=np.int)

	Buf_2 = np.empty(3, dtype='i')
	ArraySize_gray = int(np.prod(shape))

	#gray scale
	for dest_ in [_CAMWORKER_RANK_]:
		Comm.Send(shape, dest = dest_, tag = 0)

	KeepFrame = frame
	key = "a"

	time_counter = 0
	Waiting = False
#	while WebCam.isOpened() or Waiting == True:
	while True:

		Success, frame = WebCam.read()

		try:
			frame_BW = opencv.cvtColor(frame, opencv.COLOR_BGR2GRAY)
			faces = face_cascade.detectMultiScale(frame_BW, 1.20, 5)
			if len(faces) > 0:
				frame_temp = np.empty(frame.shape, dtype=np.uint8)
				np.copyto(frame_temp, frame)
				for (x,y,w,h) in faces:
					img = opencv.rectangle(frame_temp,(x,y),(x+w,y+h),(255,0,0),2)

				opencv.imshow('WebCam', frame_temp)
				opencv.waitKey(timePerFrame)
#				opencv.waitKey(1)
				if key == "q" and Waiting == False:
					break
				del frame_temp
			else:
				opencv.imshow('WebCam', frame)
				opencv.waitKey(timePerFrame)
#				opencv.waitKey(1)
				if key == "q" and Waiting == False:
					break
		except:
			pass

		time_counter = time_counter + timePerFrame

		Bolean = Comm.Iprobe(source = _MASTER_RANK_, tag = _MASTER_RANK_)
		if Bolean: 
			Comm.Recv([Buf_2, Buf_2.size, MPI.INT], source = _MASTER_RANK_, tag = 0)
			
			if Buf_2[0] == -1:
				KeepGoing[0] = -1
				break
			elif Buf_2[0] == -2:
				key = "p"
			elif Buf_2[0] >=0:
				TimeToSave = Buf_2[0]*1e-3
				print TimeToSave
				

		if time_counter > TimeToSave and Waiting == False or (key == "p" and Waiting == False):
			key = "a"
			if len(faces) == 0:
				print "no face available"
				continue
			time_counter = 0

#		if key == "p" and Waiting == False:
			Waiting = True
#			time_counter = time_counter - TimeToSave
			KeepFrame = frame

			buf = frame.reshape(ArraySize_color, order = "C")
			Comm.Send(buf, dest = _PLOT_RANK_, tag = 0)

			buf = frame_BW.reshape(ArraySize_gray, order = "C")
			Comm.Send(KeepGoing, dest = _CAMWORKER_RANK_, tag = 0)
			Comm.Send(buf, dest = _CAMWORKER_RANK_, tag = 0)

		elif Waiting == True:
			Bolean = Comm.Iprobe(source = _CAMWORKER_RANK_, tag = 0)
			if Bolean:
				Comm.Recv([BoundingBox, 4, MPI.INT], source = _CAMWORKER_RANK_, tag = 0)
				Waiting = False
#				print "bounding Box", BoundingBox
#				opencv.rectangle(KeepFrame, (BoundingBox[0], BoundingBox[1]), (BoundingBox[2]+BoundingBox[0], BoundingBox[3]+BoundingBox[1]), (0, 211, 255), 5)
#				opencv.imshow("pic", KeepFrame)

	KeepGoing[0] = -1
	Comm.Send(KeepGoing, dest = _CAMWORKER_RANK_, tag = 0)
	WebCam.release()
	opencv.destroyWindow("WebCam")

elif rank == _CAMWORKER_RANK_: ##rekognition

	s3_connection = boto3.resource('s3')
	bucketName = "videojpg"
	bucket = s3_connection.Bucket(bucketName)
	client = boto3.client("rekognition")
	image_binary = io.BytesIO()
	Picture = Image.new("RGB", (2, 2))
	shape = np.empty(2, dtype=np.int)
	Comm.Recv(shape, source = _MAINCAM_RANK_, tag = 0)
	ArraySize = int(np.prod(shape))
	EmotionArray = np.zeros(7)

	buf = np.zeros(ArraySize, dtype=np.uint8)

	while True:
#		Comm.Recv(KeepGoing, source = _MAINCAM_RANK_, tag = 0)
		FCN.ProbeRecv(Comm, KeepGoing, KeepGoing.size, MPI.INT, _MAINCAM_RANK_, 0, 0.1)
		if KeepGoing[0] == -1:
			break

		Comm.Recv(buf, source = _MAINCAM_RANK_, tag = 0)
		buf = buf.reshape(shape, order="C")

		image = Image.fromarray(buf)

		stream = io.BytesIO()
		if 'exif' in image.info:
			exif=image.info['exif']
			image.save(stream,format=image.format, exif=exif)
		else:
			image.save(stream, format="jpeg")#
		image_binary = stream.getvalue()
		stream.close()

#		del image
#		image = Image.open(io.BytesIO(image_binary))
#		image.show()

		try:
#			result = FCN.GenerateDummyResults(shape)
#			result = GenerateDummyResults(shape)
			result = client.detect_faces( Image={"Bytes":image_binary} , Attributes=['ALL', 'DEFAULT'] )
	#		with open("data.json", 'w') as outfile:
	#			json.dump(result, outfile)
	#		FCN.PrintEmotions(result)
			Emotions = FCN.GetEmotions(result)
#			Emotions = GetEmotions(result)

			FCN.SaveRekognition(Database, "Emotions", result)

#			Command = "INSERT INTO Emotions (TIME,"
#			temp_Command = "VALUES("+str(int(time.time()))+","
#			for entry in Emotions[:len(Emotions)]:
#				Command      =      Command +     entry[u'Type']        +","
#				temp_Command = temp_Command + str(entry[u'Confidence']) +","
#			Command      =      Command +     Emotions[-1][u'Type']       +") "
#			temp_Command = temp_Command + str(Emotions[-1][u'Confidence']) +");"
#			Command = Command + temp_Command
#			Database.Exec(Command)
			Database.Commit()

			BoundingBox = FCN.GetRectangle(result, shape)
#			BoundingBox = GetRectangle(result, shape)
		except:
			BoundingBox.fill(0)
			print "error, but dont worry! Be :)"
			print sys.exc_info()
			pass

		Comm.Send([BoundingBox[:4], 4, MPI.INT], dest = _MAINCAM_RANK_, tag = 0)
#		Comm.Send([BoundingBox[:4], 4, MPI.INT], dest = _PLOT_RANK_, tag = 1)
		Comm.Send([BoundingBox, BoundingBox.size, MPI.INT], dest = _PLOT_RANK_, tag = 1)


elif rank == _MASTER_RANK_:

	root = tkinter.Tk()
	Gui = _Gui_.SimpleGui(root, _MAINCAM_RANK_, _PLOT_RANK_, Comm)
	root.mainloop()

#	Buf_0 = np.empty(3, dtype='i') 
#	while True:
#		Input = FCN.GetInput()
##		Input = GetInput()
#		if Input == "q":
#			Buf_0[0] = -1
#			Comm.Send([Buf_0, Buf_0.size, MPI.INT], dest = _MAINCAM_RANK_, tag = 0)
#			Comm.Send([Buf_0, Buf_0.size, MPI.INT], dest = _PLOT_RANK_, tag = 0)
#			break
#		elif Input == "p":
#			Buf_0[0] = -2
#		elif Input == "time":
#			while True:
#				print "type time in sec"
#				Input = FCN.GetInput()
#				if FCN.IsFloat(Input):
#					Buf_0[0] = int(Input)
#					break
#		elif Input == "h":
#			Buf_0[0] = 2
#			Comm.Send([Buf_0, Buf_0.size, MPI.INT], dest = _PLOT_RANK_, tag = 0)
#			continue

#		else:
#			print "Wut?"
#			continue

#		Comm.Send([Buf_0, Buf_0.size, MPI.INT], dest = _MAINCAM_RANK_, tag = 0)
##	Comm.Send([Buf_0, Buf_0.size, MPI.INT], dest = _MAINCAM_RANK_, tag = 0)

elif rank == _PLOT_RANK_:

	status = MPI.Status()

	Buf_0 = np.empty(3, dtype='i')
	shape = np.empty(3, dtype='i')
	BoundingBox = np.empty(4*100, dtype='i')

	Comm.Recv(shape, source=_MAINCAM_RANK_, tag = 0)

	ArraySize = np.prod(shape)
	frame = np.empty(ArraySize, dtype=np.uint8) 
	frame.fill(122)

	Command = "SELECT * FROM Emotions ORDER BY TIME ASC;"
	Data = np.asarray(Database.Exec(Command))
#	xx = np.arange(1, Data.shape[1] + 1)

	plt.ion()

	fig= plt.figure(num = 1, figsize=(7.0, 10.))
	Axes_list = []
	fig_xwidth = 0.8
	fig_ywidth = 0.3
	Axes_list.append(fig.add_axes([.1, .1, fig_xwidth, fig_ywidth]))
	Axes_list.append(fig.add_axes([.1, .4, fig_xwidth, fig_ywidth]))
	Axes_list.append(fig.add_axes([.1, .7, fig_xwidth, fig_ywidth]))

	xx = np.arange(1, 10)
	my_xticks = [' ', 'Angry','Calm','Confused','Disgusted', 'Happy', 'Sad', 'Surprised']

	Command = "SELECT ROWID,Time,Angry,Calm,Confused,Disgusted,Happy,Sad,Surprised FROM Emotions ORDER BY TIME ASC;"
	Data = np.asarray(Database.Exec(Command))
	if Data.size == 0:
		Data = np.zeros((2,9))
	CumEmotions = np.zeros( Data.shape[1])
	for index in np.arange(2, CumEmotions.size):
		CumEmotions[index] = np.sum(Data[:,index])/Data.shape[0]

	BarContainer = Axes_list[0].bar(xx[2:], CumEmotions[2:], color=np.random.rand(3), width = .8, align='center')
	Axes_list[0].axis([xx[2]-1, xx[-1]+1, 0, 100])
	Axes_list[0].set_xticklabels(my_xticks, rotation = 45)
#	fig.show()

	pieColors = ["red", "cyan", "orange", "forestgreen", "yellow", "blue", "magenta"]

	Axes_list[2].imshow(frame.reshape(shape, order = "C"))

	while True:
		Received1 = FCN.ProbeRecv(Comm, Buf_0, Buf_0.size, MPI.INT,_MASTER_RANK_, 0, .1, timeOut= 1)

		flag = Comm.Iprobe(source = _MAINCAM_RANK_, tag = 0)
		if flag == True:
			[p.remove() for p in reversed(Axes_list[2].patches)]
			Comm.Recv(frame, source = _MAINCAM_RANK_, tag = 0)
			Axes_list[2].imshow(opencv.cvtColor(frame.reshape(shape, order = "C"), opencv.COLOR_BGR2RGB))

			BoundingBox.fill(0)
			FCN.ProbeRecvStatus(Comm, BoundingBox, MPI.INT, _CAMWORKER_RANK_, 1, .1, status)
			Bbox = np.reshape(BoundingBox[:status.Get_elements(MPI.INT)], (status.Get_elements(MPI.INT)/4, 4) )
			for box in Bbox:
				rect = patches.Rectangle(box[0:2], box[2], box[3], linewidth = 2, facecolor="None", edgecolor="yellow")
				Axes_list[2].add_patch(rect)

		if Buf_0[0] == -1:
			break

		Command = "SELECT ROWID,Time,Angry,Calm,Confused,Disgusted,Happy,Sad,Surprised FROM Emotions ORDER BY TIME ASC;"
		Data = np.asarray(Database.Exec(Command))
		if Data.size == 0:
#			print "zero size"
			continue

		CumEmotions.fill(0.0)
		for index in np.arange(2, CumEmotions.size):
			CumEmotions[index] = np.sum(Data[:,index])/Data.shape[0]
		
		for rect, h in zip(BarContainer.patches, CumEmotions[2:]):
			rect.set_height(h)
#		Axes_list[0].set_xticklabels(my_xticks, rotation = 45)
		Axes_list[1].clear()
		Axes_list[1].pie(Data[-1,2:], labels=my_xticks[1:], colors=pieColors)

		fig.gca().relim() # recompute the data limits
#		fig.gca().autoscale_view() # automatic axis scaling
		fig.canvas.flush_events()
	plt.close()
	
sys.stderr.write("exit "+str(rank)+"\n")
