import tan
import cv2
import random as rd
import numpy as np
import math
##############################################################

pictureLab,SHAPE  = tan.loadImageLab("yoj1.png")
print(SHAPE)
NUMTR = 200
NUMIND = 100

##########################################

####################
# От 30 треугольников фон случаен
def get_individual(NUMTR,SHAPE):
    tr = []
    for i in range(NUMTR): 
        t = []
        r = rd.randint(5,50)
        x = rd.randint(10, SHAPE[0] - 10)
        y = rd.randint(10, SHAPE[1] - 10)
        a0 = rd.uniform(0.0, 2.0 * math.pi / 3.0)
        a1 = rd.uniform(2.0 * math.pi / 3.0, 4 * math.pi / 3.0)
        a2 = rd.uniform(4.0 * math.pi / 3.0, 2.0 * math.pi)         
        t.append(int(r * math.cos(a0)) + x) #0
        t.append(int(r * math.sin(a0)) + y) #1
        t.append(int(r * math.cos(a1)) + x) #2
        t.append(int(r * math.sin(a1)) + y) #3
        t.append(int(r * math.cos(a2)) + x) #4
        t.append(int(r * math.sin(a2)) + y) #5
        for j in range (4): #6, 7, 8, 9
            t.append(rd.randint(0,255))
        tr.append(tuple(t))
    return tr

def get_population(NUMIND,SHAPE,NUMTR):
    p = []
    for i in range (NUMIND):
        p.append(get_individual(NUMTR,SHAPE))
    return p

def mutate_color(ind,NUMTR,SHAPE):
    n = rd.randint(0,NUMTR-1)
    t = list(ind[n])
    m = rd.randint(6,9)
    t[m] = rd.randint(0,255)
    ind[n] = tuple(t) 
    return ind

def mutate_rotate(ind,NUMTR,SHAPE):
    n = rd.randint(0,NUMTR-1)
    t = list(ind[n])
    mx = (t[0] + t[2] + t[4]) // 3
    my = (t[1] + t[3] + t[5]) // 3
    a = rd.uniform(0.0, 2.0 * math.pi)
    ntx = (t[0] - mx) * math.cos(a) - (t[1] - my) * math.sin(a) + mx
    nty = (t[0] - mx) * math.sin(a) + (t[1] - my) * math.cos(a) + my
    t[0], t[1] = ntx, nty
    ntx = (t[2] - mx) * math.cos(a) - (t[3] - my) * math.sin(a) + mx
    nty = (t[2] - mx) * math.sin(a) + (t[3] - my) * math.cos(a) + my 
    t[2], t[3] = ntx, nty
    ntx = (t[4] - mx) * math.cos(a) - (t[5] - my) * math.sin(a) + mx
    nty = (t[4] - mx) * math.sin(a) + (t[5] - my) * math.cos(a) + my     
    t[4], t[5] = ntx, nty
    ind[n] = tuple(t)
    return ind

def mutate_place(ind,NUMTR,SHAPE):
    n = rd.randint(0,NUMTR-1)
    t = list(ind[n])
    mx = (t[0] + t[2] + t[4]) // 3
    my = (t[1] + t[3] + t[5]) // 3
    new_mx = rd.randint(- 10, SHAPE[0] + 10)
    new_my = rd.randint(- 10, SHAPE[1] + 10)
    t[0] = t[0] - mx + new_mx
    t[1] = t[1] - my + new_my
    t[2] = t[2] - mx + new_mx
    t[3] = t[3] - my + new_my
    t[4] = t[4] - mx + new_mx
    t[5] = t[5] - my + new_my 

    ind[n] = tuple(t) 
    return ind

def mutate_layer_swap(ind,NUMTR,SHAPE):
    n1 = rd.randint(0,NUMTR-1)
    n2 = rd.randint(0,NUMTR-1)
    ind[n1], ind[n2] = ind[n2], ind[n1]
    return ind

def crossover(ind1, ind2, NUMTR, SHAPE):
    a = rd.randint(0,NUMTR - 1)
    b = rd.randint(a,NUMTR - 1)
    child = ind1[:]
    child[a:b] = ind2[a:b]
    return child

def fitness(ind):
    a = tan.polygons2lab(ind, 3, SHAPE, 255)
    return np.mean(np.square(a - pictureLab))

def evolushion(n,NUMIND,SHAPE,NUMTR):
    population = get_population(NUMIND,SHAPE,NUMTR)
    tan.saveImageRGB("im/" + "00" + ".png", tan.polygons2rgb(population[0], 3, SHAPE, 255))

    for i in range (n):
        population.sort(key = fitness)
        print(str(i) + ' ' + str(fitness(population[0])))
        a = population[:10]
        if i % 10 == 0:
            tan.saveImageRGB("im/" + str(i) + ".png", tan.polygons2rgbAA(population[0], 3, SHAPE, 255))
        
        newPopulation = []
        for j in range(NUMIND):
            t = crossover(rd.choice(population),rd.choice(a),NUMTR,SHAPE)
            r = rd.uniform(0.0,1.0)
            if r <= 0.15:
                t = mutate_color(t,NUMTR,SHAPE)
            r = rd.uniform(0.0,1.0)
            if r <= 0.15:
                t = mutate_place(t,NUMTR,SHAPE)
            r = rd.uniform(0.0,1.0)
            if r <= 0.15:
                t = mutate_rotate(t,NUMTR,SHAPE) 
            r = rd.uniform(0.0,1.0)
            if r <= 0.15:
                t = mutate_layer_swap(t,NUMTR,SHAPE)                 
            newPopulation.append(t)
        population = newPopulation

    population.sort(key = fitness)
    tan.saveImageRGB("im/" + "result" + ".png", tan.polygons2rgbAA(population[0], 3, SHAPE, 255))
    return population[0]


#############################################################

x = evolushion(9000,NUMIND,SHAPE,NUMTR)

image = tan.polygons2rgb(x, 3, SHAPE, 255)
cv2.imshow("RESULT", image)
cv2.waitKey()