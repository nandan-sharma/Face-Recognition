import os 	#this handles the paths , file handling
import numpy as np 
from PIL import Image 		# this library helps in handling images
import cv2

recog = cv2.face.LBPHFaceRecognizer_create()
path = 'data set'

def grtImageswitchID(path):
	imagePaths = [os.path.join(path,f) for f in os.listdir(path)] #os.path is a function witch joins two different paths  and  os.listdir list all the path of dataset or files and attach that with with the files ....we get imagepath like  dataset/1_1.jpg
	faces = []
	IDs = []

	for imgpath in imagePaths:
		faceImg = Image.open(imgpath).convert('L')	# read and opens the imgae and convert that into grey scale or rgb and bgr and so on
		faceNp = np.array(faceImg,'uint8')# uint8 data type of an array
		ID = int(os.path.split(imgpath)[-1].split('_')[0]) #now this splits the add (dataset/1_1.jpg) first dive add by "/" and then by "_" and get the actul vale of dataset
		faces.append(faceNp)
		IDs.append(ID)
		cv2.imshow('training',faceNp)
		cv2.waitKey(10)
	return IDs, faces

IDs, faces = grtImageswitchID(path)
recog.train(faces,np.array(IDs))
recog.write('trainingData.yml') 	#we can seve these traning data(already trainrd dataset) and can be used in another pc without going through the traning process ...can also be applied on sklearn modules....google it
cv2.destroyAllWindows()
