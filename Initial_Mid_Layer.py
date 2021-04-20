import numpy as np
import scipy.io
import pandas as pd

def initial_mid_layer(P, InputWeight, layer):
    YYM_H = InputWeight @ P
    P = None
    YYM_H[YYM_H < 0] = 0
    H = YYM_H
    #YYM_Layer2_P = H 
    layer = layer + 1

    #######Clearing multiple variables

    return YYM_H, layer, H