import cv2
import numpy as np 

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #this perticular file contain the peocess of detection facial features and hence detecting the face

sampleNum = 1

uid = input('enter user id')

cam = cv2.VideoCapture(0)

while(True):
	ret,img = cam.read()	#ret ois used to find if the camera is providing the frames or not.....we can ignore this with "_"
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = facedetect.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in faces:
		sampleNum+=1
		cv2.imwrite('data set/'+str(uid)+'_'+str(sampleNum)+'.jpg',gray[y:y+h,x:x+w])
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
		cv2.waitKey(100)	#there is a gap of 100 miliseconds between every frame caputured
	cv2.imshow('face',img)
	cv2.waitKey(1)
	if(sampleNum>200):
		break
cam.release()
cam.destroyAllWindows()