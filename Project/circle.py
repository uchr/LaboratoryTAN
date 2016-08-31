import tan
import cv2
import random
import numpy as np
import threading

NUMIND = 150
NUM = 150
NUMGEN = 10000
img,SHAPE=tan.loadImageLab("dwarf2.jpg")

def fitness (ind):
    plgnImage=tan.circles2lab(ind, SHAPE, 255)
    return np.mean (np.square(img-plgnImage))

def getCircles(NUM):
    Circles=[]
    for u in range (NUM):
        x=random.randint(0,SHAPE[0])
        y=random.randint(0,SHAPE[1])
        r=random.randint(8, 50)
        R=random.randint(0,255)
        G=random.randint(0,255)
        B=random.randint(0,255)
        A=random.randint(100,255)
        Circles.append((x,y,r,R,G,B,A))
    return Circles

def getPopulation(NUMIND):
    population = []
    for i in range(NUMIND):
        population.append(getCircles(NUM))
    return population

def mutatecircles(ind):
    l = random.randint(0, len(ind) - 1)
    t=list(ind[l])    
    k=random.randint(0,7)
    if k < 2:
        v=random.randint(0,1)
        t[k]=random.randint(0,SHAPE[k % 2])
    elif k == 2:
         t[k]=random.randint(8, 50)
    elif k > 2 and k < 6:
         t[k]=random.randint(0,255)
    elif k == 6:
         t[k]=random.randint(100,255)
    ind[l]=tuple(t)
 
    if k == 7:
        l1 = random.randint(0, len(ind) - 1)
        ind[l], ind[l1] = ind[l1], ind[l]
    return ind

def crossCirc(ind1,ind2):
    n=len(ind1)
    m=random.randint(1,n-1)
    c=ind1[0:m]+ind2[m:n]
    return c

population=getPopulation(NUMIND)

k=10
for i in range(NUMGEN):
   population.sort(key=fitness)
    if i % 20 == 0:
        tan.saveImageRGB('images/' + str(i) + '.png', tan.circles2rgb(population[0],SHAPE, 255))
    print(str(i), fitness(population[0]))
    elite=population[:k]
    newpopulation=population[:k]
    for a in range (NUMIND-k):
        child=crossCirc(random.choice(elite), random.choice(population[k:]))
        newpopulation.append(child)
    for b in range (NUMIND):
        d = random.uniform (0.0,1.0)
        if d < 0.75:
            newpopulation[b]=mutatecircles(newpopulation[b])
    population=newpopulation

population.sort(key=fitness)
plgnImage=tan.circles2rgb(population[0],SHAPE, 255)