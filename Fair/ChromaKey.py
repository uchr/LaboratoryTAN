#cv2.imshow('Independent entertainment of indi developers',newim)
#cv2.waitKey()
import cv2
import math
im4=cv2.imread('3.png')
cap=cv2.VideoCapture('4.mp4')
ret, framep=cap.read()
framep=cv2.medianBlur(framep,7)  
shape = framep.shape
while cap.isOpened():
    ret, framec = cap.read() 
    framec=cv2.medianBlur(framec,7)  
    mask=framec-framep    
    for i in range(mask.shape[0]):
        for s in range(mask.shape[1]):
            if math.sqrt(mask[i,s,0]**2+mask[i,s,1]**2+mask[i,s,2]**2) <= 100:
                mask[i,s]=im4[i,s]
            else:
                mask[i,s]=framec[i,s]
    cv2.imshow('Independent entertainment of indi developers',mask)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break