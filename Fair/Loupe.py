import math
import cv2
import numpy

settings = open('settings.txt','r')
framenumber=int(settings.read())
settings.close()
im = cv2.imread ('face.jpg')

def dist (x0, y0, x, y):
    result = math.sqrt((x - x0)**2 + (y - y0)**2)
    return result

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
#print(frame)
while True:
    #print('123')
    ret, frame = cap.read()
    shape = frame.shape
    x = 240
    y = 320
    r = 200
    p = frame.copy()
    for i in range (shape[0]):
        for j in range (shape[1]):
            d = dist(x, y, i, j)
            if d <= r:
                a = (d / r)
                if a < 0.01:
                    a = 1
                ax = int((i-x) * a + x)
                ay = int((j-y) * a + y)
                if ax < shape[0] and  ax  >= 0 and ay < shape[1] and ay >= 0:
                    p[i,j] = frame[ax, ay]
    #print('321')

    p2 = frame.copy()
    for i in range (shape[0]):
        for j in range (shape[1]):
            d = dist(x, y, i, j)
            if d <= r:
                a = (d / r)**2
                if a < 0.01:
                    a = 1
                ax = int((i-x) / a + x)
                ay = int((j-y) / a + y)
                if ax < shape[0] and  ax  >= 0 and ay < shape[1] and ay >= 0:
                    p2[i,j] = frame[ax, ay]
    
    cv2.imshow('TAHr',p)
    cv2.imshow('TAHl',p2)
    settings=open('settings.txt','w')
    framenumber+=1
    settings.write(str(framenumber))
    settings.close
    cv2.imwrite("results/"+str(framenumber)+".png",p)
    cv2.imwrite("results/narko"+str(framenumber)+".png",p2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#foo (im)