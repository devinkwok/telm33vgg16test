import numpy as np
import scipy.io
import pandas as pd
import math

def initial_layer_forward(P_save, InputWeight1, layer):
    YYM_H = InputWeight1.conj().T @ P_save
    YYM_H[YYM_H < 0] = 0
    tempH = YYM_H
    layer = layer + 1

    return YYM_H, tempH, layer