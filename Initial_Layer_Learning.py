import numpy as np
import scipy.io
import pandas as pd
import math

def initial_layer_learning(PP, P_save, C, layer, InputWeight1, Learning_rate):
    PP1 = PP.real
    P = P_save
    H = None
    PP = None
    P_save = None

    a = (np.eye(len(P)) / C + (P @ (P.conj().T)))
    b = (P @ (PP1.conj().T))
    InputWeight1_temp = np.linalg.solve(a, b)

    #rc = InputWeight1_temp.shape
    #r = rc[0]
    #c = rc[1]

    #randomindex_r = (np.floor(np.random.rand(1,math.floor(r*1)) * r)).astype(int)  
    #randomindex_c = (np.floor(np.random.rand(1,math.floor(c*1)) * c)).astype(int)

    ########################## Improved complexity with roll. Try to avoid for and find a function if possible
    #########Observe mid_layer_learning. Same concept

    #for x in range (r):
    #    xindex = int(randomindex_r[0][x])
    #    for y in range (c):
    #        yindex = int(randomindex_c[0][y])
    #        InputWeight1_temp[xindex][yindex] = 0

    #for x in range (r):          
    #    InputWeight1_temp[randomindex_r, randomindex_c] = 0
    #    randomindex_r = np.roll(randomindex_r, 1)

    InputWeight1 = InputWeight1 + Learning_rate*InputWeight1_temp.conj().T
    InputWeight1 = InputWeight1.conj().T
    return layer, P, InputWeight1