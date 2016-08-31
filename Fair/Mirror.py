import cv2  
import numpy as np
import tan
#im=cv2.imread('1.jpg')
#shape=im.shape
#print(im.shape)
#newIm=im.copy()
#m=shape[0]//2
#newIm[m:,:]=im[m-1::-1,:]
#cv2.imshow('IM',newIm)
#cv2.waitKey()
cap=cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    shape=frame.shape
    newIm=frame.copy()
    m=shape[1]//2
    newIm[:,m-1::]=im[:,m::-1]
    cv2.imshow('IM',newIm)
    cv2.waitKey()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
