import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#img = cv2.imread('peopleEmotions.jpeg')

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("../../../Documents/casaToffo3.mp4")
start_frame_number = 5
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number) 
#while(True):
counter = 0
while( cap.isOpened() ):
        # Capture frame-by-frame
        ret, frame = cap.read()

	counter = (counter+ 1)%start_frame_number
	if counter != 0:
		continue

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.10, 5)
	for (x,y,w,h) in faces:
	    img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
#	    roi_gray = frame[y:y+h, x:x+w]
#	    roi_color = frame[y:y+h, x:x+w]
	#    eyes = eye_cascade.detectMultiScale(roi_gray)
	#    for (ex,ey,ew,eh) in eyes:
	#        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

	cv2.imshow("Image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cv2.destroyAllWindows()
