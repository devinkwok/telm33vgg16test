import numpy as np
import scipy.io
import pandas as pd
import math

def mid_layer_forward(tempH, YYM_H, InputWeight, layer):
    YYM_H = InputWeight.conj().T @ YYM_H
    YYM_H[YYM_H < 0] = 0
    tempH = YYM_H
    layer = layer + 1

    return YYM_H, tempH, layer