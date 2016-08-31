import numpy as np
import cv2
import tan
from random import randint, uniform, choice
triangles=[]
k=3
im0,shape=tan.loadImageLab("Tanya.jpg")
NUMIND=100
NUMTR=100
NUMGEN = 10000

def getIndividual(k):
    triangles=[]
    for i in range(k):
        oneTriangle = []
        for j in range(0,6):
            oneTriangle.append(randint(-50,shape[j%2]+50))
        oneTriangle.append(randint(0,255))
        oneTriangle.append(randint(0,255))
        oneTriangle.append(randint(0,255))
        oneTriangle.append(randint(100,255))
        
        triangles.append(tuple(oneTriangle))
    return triangles
def crossover(ind1,ind2):
    i=randint(0,len(ind2)-3)
    x=ind1[:i]+ind2[i:]
    return x
def mutate(ind):
    x=randint(0,len(ind)-1)
    t=list(ind[x])
    v=randint(0,9)
    if v<6:
        t[v]=randint(-50,shape[v%2]+50)
        ind[x]=tuple(t)
    elif v>5:
        t[v]=randint(100,255)
        ind[x]=tuple(t)
    elif v==9:
        x1=randint(0,len(ind)-1)
        x2=randint(0,len(ind)-1)
        ind[x1],ind[x2]=ind[x2],ind[x1]
    return ind
def fitness(ind):
    gagaga=tan.polygons2lab(ind, 3, shape, 255)
    x1=np.mean(np.square(im0-gagaga))
    return x1

A=[]
for i in range(NUMIND):
    A.append(getIndividual(NUMTR))

for i in range(NUMGEN) :
    A.sort(key=fitness)
    print(i, fitness(A[0]),'    ', fitness(A[len(A)-1]))
    B=A[:10]
    del A[:10]
    C=A[:10]
    for j in range(NUMIND):
        C.append(crossover(choice(A),choice(B)))
    for j in range(NUMIND):
        if uniform(0.0,1.0)<0.75:
            C[j] = mutate(C[j])
    A=C
    if i%20==0 or i == NUMGEN - 1:
        tan.saveImageRGB("images/"+str(i)+".png", tan.polygons2rgbAA(A[0], 3, shape, 255))

plgnImage = tan.polygons2lab(A[0], 3, shape, 255)

cv2.imshow("Polygons", plgnImage)
cv2.waitKey()



