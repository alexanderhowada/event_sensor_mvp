import numpy as np
import cv2 as opencv
import time
from mpi4py import MPI

Comm = MPI.COMM_WORLD	
rank = Comm.Get_rank()

if rank == 0:
	while True:
		
		try:
			Input = raw_input()
			if Input == "q":
				break
			else:
				print Input
			
		except:
			continue
			time.sleep(0.5)
			print "stuff"


elif rank == 1:
	WebCam = opencv.VideoCapture(0)

	TimeToSave = 1e300
	framesPerSecond = WebCam.get(opencv.CAP_PROP_FPS)
	timePerFrame = int(1000./framesPerSecond)

	while True:
		key = opencv.waitKey(timePerFrame)
		if key == ord("q"):
			break

		Success, frame = WebCam.read(0)
		opencv.imshow('WebCam', frame)

	WebCam.release()
	opencv.destroyAllWindows()



