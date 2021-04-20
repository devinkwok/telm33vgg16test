import numpy as np
import scipy.io
import pandas as pd

def initial_layer(P, InputWeight1, NumberofTrainingData, layer):
    
    NumberofHiddenNeurons = InputWeight1.shape[0]

    BiasofHiddenNeurons1 = pd.DataFrame(np.random.uniform(0,1,NumberofHiddenNeurons)) 
    BiasofHiddenNeurons1 = pd.DataFrame(scipy.linalg.orth(BiasofHiddenNeurons1))
 
    BBP = BiasofHiddenNeurons1.values

    YYM_H = InputWeight1 @ P

    YYM_tempH = (YYM_H.conj().T - BBP.T).conj().T

    YYM_H = YYM_tempH

    P_save = P
    P = pd.DataFrame(YYM_H) 
    P = P.iloc[:,0:NumberofTrainingData]
    P = P.values
    
    YYM_Layer2_P = P

    layer = layer + 1

    return YYM_H, layer, P, YYM_Layer2_P, P_save