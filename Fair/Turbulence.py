import numpy as np
import cv2
import math
import random
cam=cv2.VideoCapture(0)
ret, frame = cam.read()
count=0
settings = open('settings.txt', 'r')
frameNumber=int(settings.read())
settings.close()
while True:
    ret, frame = cam.read()
    shape = frame.shape

    def dist(x0,y0,x1,y1):
        n0=(x1-x0)**2
        n1=(y1-y0)**2
        n=math.sqrt(n1+n0)
        return n
    #im=cv2.VideoCapture(0)
    newim=frame.copy()
    count+=1
    if count % 2 == 1:
        r=130
        x=250
        y=280
        for i in range(x-r-1,shape[0]):
            for j in range (y-r-1,shape[1]):
                if dist(i,j,x,y)<=r:
                    i0=i-x
                    j0=j-y
                    a=(1-dist(i,j,x,y)/r)*0.8*math.pi
                    ms=math.sin(a)
                    mc=math.cos(a)
                    x0=x+(i0*mc)-(j0*ms)
                    y0=y+(i0*ms)+(j0*mc)
                    x0=int(x0)
                    y0=int(y0)
                    if ((x0<=shape[0])and(y0<=shape[1])and(x0>=0)and(y0>=0)):
                        newim[i][j]=frame[x0][y0]
    if count % 2 == 0:
        r=130
        x=250
        y=215
        x1=250
        y1=360
        for i in range(x-r-1,shape[0]):
            for j in range (y-r-1,shape[1]):
                if dist(i,j,x,y)<=r:
                    i0=i-x
                    j0=j-y
                    a=(1-dist(i,j,x,y)/r)*0.4*math.pi
                    ms=math.sin(a)
                    mc=math.cos(a)
                    x0=x+(i0*mc)-(j0*ms)
                    y0=y+(i0*ms)+(j0*mc)
                    x0=int(x0)
                    y0=int(y0)
                    if ((x0<=shape[0])and(y0<=shape[1])and(x0>=0)and(y0>=0)):
                        newim[i][j]=frame[x0][y0]
        for i in range(x1-r-1,shape[0]):
            for j in range (y1-r-1,shape[1]):
                if dist(i,j,x1,y1)<=r:
                    i0=i-x1
                    j0=j-y1
                    a=(1-dist(i,j,x1,y1)/r)*-0.4*math.pi
                    ms=math.sin(a)
                    mc=math.cos(a)
                    x0=x1+(i0*mc)-(j0*ms)
                    y0=y1+(i0*ms)+(j0*mc)
                    x0=int(x0)
                    y0=int(y0)
                    if ((x0<=shape[0])and(y0<=shape[1])and(x0>=0)and(y0>=0)):
                        newim[i][j]=frame[x0][y0]
    cv2.imshow('TAH',newim)
    settings = open('settings.txt', 'w')
    frameNumber+=1
    settings.write(str(frameNumber))
    settings.close()

    cv2.imwrite("result/"+str(frameNumber)+".png",newim)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break