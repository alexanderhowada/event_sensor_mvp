import boto3
import botocore
import cv2 as opencv

class S3Object:

	s3_connection = boto3.resource('s3')
	bucketName = "videojpg"
	bucket = s3_connection.Bucket(bucketName)
	client = boto3.client("rekognition")

	WebCam = opencv.VideoCapture(-1)
	TimeToSave = 1000
	framesPerSecond = WebCam.get(opencv.CAP_PROP_FPS)
	timePerFrame = int(1000./framesPerSecond)

	def Cam(self):
		print "asdf"
		while True:
			key = opencv.waitKey(self.timePerFrame)
			if key == ord('q'):
				break
		
			Success, frame = self.WebCam.read()
			opencv.line(frame, (128,128), (256, 256), (0, 211, 255), 5)
			opencv.imshow('WebCam', frame)
			

		opencv.destroyAllWindows()

	def GetJPEG(self, frame, name, quality):
		opencv.imwrite(name, frame, [int(opencv.IMWRITE_JPEG_QUALITY), quality])

	
