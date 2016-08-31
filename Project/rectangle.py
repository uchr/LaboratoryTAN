import tan
import numpy as np
import cv2
import random
cs=[]
NUM4=100
NUMIND=100
NUMGEN=7000
im1,shape=tan.loadImageLab("001.jpg")

def fitness4(a):
    plgnImage3=tan.polygons2lab(a,4,shape,255)
    return np.mean(np.square(im1-plgnImage3))
    
def crossover4(a,b):
    i=random.randint(0,NUM4)
    c=b[:i]+a[i:]
    return c

def genIndividual4(NUM4):  
    a=[]
    for c in range(NUM4):
        t = []       
        x1=random.randint(0,shape[0])
        y1=random.randint(0,shape[1])
        x2=random.randint(0,shape[0])
        y2=random.randint(0,shape[1])
        t.append(x1) #0
        t.append(y1) #1
        t.append(x1) #2
        t.append(y2) #3
        t.append(x2) #4
        t.append(y2) #5
        t.append(x2) #6
        t.append(y1) #7
        for k in range(4):
            t.append(random.randint(0,255))        
        a.append(tuple(t))
    return a

def mutate4(a):
    n=random.randint(0,NUM4-1)
    t=list(a[n]) 
    k=random.randint(0,8)
    if k==0:
        t[1]=random.randint(0,shape[1])
        t[7]=t[1]
    elif k==1:
        t[0]=random.randint(0,shape[0])
        t[2]=t[0]
    elif k==2:
        t[3]=random.randint(0,shape[1])
        t[5]=t[3]
    elif k==3:
        t[4]=random.randint(0,shape[0])
        t[6]=t[4]
    elif k>3 and k<8:
        t[k+4]=random.randint(0,255)
    elif k==8:
        l=random.randint(0,len(a)-1)
        j=random.randint(0,len(a)-1)
        a[l],a[j]=a[j],a[l]
    a[n]=tuple(t)
    return a

#d=genIndividual4(NUM4)
#plgnImagei=tan.polygons2rgbAA(d,4,shape,255)
#cv2.imshow("G4", plgnImagei)
#plgnImagef=tan.polygons2rgbAA(mutate4(d),4,shape,255)
#cv2.imshow("G5", plgnImagef)
#cv2.waitKey()


def getPopulation(NUMIND):
    a=[]
    for c in range(NUMIND):
        a.append(genIndividual4(NUM4)) # NUM4
    return a

population=getPopulation(NUMIND)

for c in range(NUMGEN): 
    population.sort(key=fitness4)
    #print(c,fitness(population[0]),fitness(population[len(population)-1]))
    nice=population[:10]
    del population[:10]
    populationNew=[]
    for k in range(NUMIND):
        populationNew.append(crossover4(random.choice(nice),random.choice(population)))              
        if random.uniform(0.0,1.0)<=0.5:
             populationNew[k]=mutate4(populationNew[k])    
    population=populationNew

    if c % 10 == 0 or c == (NUMGEN - 1):
        population.sort(key = fitness4)
        tan.saveImageRGB('images/' + str(c) + '.png', tan.polygons2rgb(population[0],4,shape,255))
        print(str(c), fitness4(population[0]))

population.sort(key = fitness4)
plgnImage1=tan.polygons2rgbAA(population[0],4,shape,255)
cv2.imshow("Genetic", plgnImage1)
cv2.waitKey()