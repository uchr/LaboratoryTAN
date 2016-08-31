import numpy as np
import random
import cv2
import tan

img,SHAPE = tan.loadImageLab('РАЭs.jpg')

NUMTR = 150
NUMIND = 120
NUMGEN = 10000

def getIndividual(NUMTR, shape):
    points=[]
    for c in range(NUMTR):
        n1=random.randint(0,shape[0])
        m1=random.randint(0,shape[1])
        r=random.randint(0,255)
        g=random.randint(0,255)
        b=random.randint(0,255)
        points.append ((n1,m1,r,g,b))
    return points

def mutate(ind):
    k=random.randint(0,len(ind)-1)
    c=random.randint(0,1)
    if c==0:
        t=list(ind[k])
        d=random.randint(0,1)
        t[d]=random.randint(0,SHAPE[d % 2])
        ind[k]=tuple(t)
    else:
        t=list(ind[k])
        d=random.randint(2,4)
        t[d]=random.randint(0,255)
        ind[k]=tuple(t)
    return ind

def crossover(ind1,ind2):
    m=random.randint(1,len(ind1)-1)
    c=ind1[:m]+ind2[m:]
    return c

def getPopulation(NUMIND):
    population=[]
    for i in range(NUMIND):
        population.append(getIndividual(NUMTR,SHAPE))
    return population

def fitness(ind):
    plgnImage = tan.points2lab(ind, SHAPE)
    return np.mean(np.square(plgnImage-img))

population = getPopulation(NUMIND)
k=10
for i in range(NUMGEN):
    population.sort(key=fitness)

    print('f_' + str(i) +' =', fitness(population[0]))
    if i % 20 == 0 or i == NUMGEN - 1:
        tan.saveImageRGB("images/" + str(i) + ".png", tan.points2rgbAA(population[0], SHAPE))

    elite=population[:k]
    newPopulation=population[:k]
    for l in range(NUMIND-k):
        child = crossover(random.choice(elite),random.choice(population[k:]))
        newPopulation.append(child)

    for p in range(NUMIND):
        z=random.uniform(0,1.0)
        if z <= 0.5:
            newPopulation[p]=mutate(newPopulation[p])
    population=newPopulation

population.sort(key=fitness)
plgnImage = tan.points2rgbAA(population[0], SHAPE)