import numpy as np
import scipy.io
import pandas as pd
import math

def mid_layer_learning(PP, YYM_Layer2_P, C, layer, InputWeight, Learning_rate):
    PP1 = PP.real
    P = YYM_Layer2_P

    for i in range(1,3):
        H = None
        PP = None
        if (i==1):
            a = (np.eye(len(P)) / C + (P @ (P.conj().T)))
            b = (P @ (PP1.conj().T))
            InputWeight_temp = np.linalg.solve(a, b)

            #rc = InputWeight_temp.shape
            #r = rc[0]
            #c = rc[1]

            #randomindex_r = (np.floor(np.random.rand(1,math.floor(r*1)) * r)).astype(int)   
            #randomindex_c = (np.floor(np.random.rand(1,math.floor(c*1)) * c)).astype(int)

            #for x in range (r):
            #    xindex = int(randomindex_r[0][x])
            #    for y in range (c):
            #        yindex = int(randomindex_c[0][y])
            #        InputWeight_temp[xindex][yindex] = 0
            
            ########################## Improved complexity with roll. Try to avoid for and find a function if possible
            #for x in range (r):          
            #    InputWeight_temp[randomindex_r, randomindex_c] = 0
            #    randomindex_r = np.roll(randomindex_r, 1)
            
            InputWeight = InputWeight + Learning_rate*InputWeight_temp.conj().T
            InputWeight = InputWeight.conj().T

            tempH = ((Learning_rate*InputWeight_temp)@YYM_Layer2_P)
            PP1 = PP1 - tempH

            YYM_Layer2_P = None
            tempH = None
        else:
            InputWeight_temp = None

            a = PP1.conj().T
            eye = (np.eye(len(InputWeight)))
            InputWeight_conj = InputWeight.conj().T
            a_ling = eye/C+InputWeight @ InputWeight_conj
            ling_solve = (np.linalg.solve(a_ling , InputWeight)).conj().T
            PP = a @ ling_solve
            PP = PP.conj().T

        if (i == 1):
            yiminy = np.random.random()
        
    PP1 = None
    layer = layer - 1
         
    return layer, P, InputWeight, PP
