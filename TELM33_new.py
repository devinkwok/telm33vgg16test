import numpy as np
import Initial_Layer, Initial_Mid_Layer, Initial_Layer_Learning, Mid_Layer_Learning, Initial_Layer_Forward, Mid_Layer_Forward
import math

def telm33_new(train_data, train_label, C, InputWeight1, InputWeight2, InputWeight3, Learning_rate, learning_rate2):
    
    #Macro Defination
    YYM = InputWeight3.conj().T
    fdafe = 0

    #Load training dataset
    T = train_label
    P = train_data.conj().T
    OrgP = P
    train_data = None

    NumberofTrainingData = P.shape[1]

    #Calculate Weight and Biases

    for subnetwork in range(1):
        yym = 0
        layer = 1

        for yym_loop in range(1,3):
            if (yym_loop == 2):
                index_y = [1, 1, 1, 2, 2, 3, 3, 3]
            else:
                index_y = [1, 1, 1, 2, 2]

            for j in index_y:
                yym = yym + 1
                if (j==1):
                    count = 1
                else:
                    count = 1

                # Layers are in reverse as we are incrementing layer value by 1 in each sub case. 
                # So if it increments to 2 then it will go in j==2 next.

                #Layer 1

                if (layer == 1):
                    if (j == 1):
                        if (yym_loop > 1):
                            InputWeight1 = InputWeight11.conj().T

                        (YYM_H, layer, P, YYM_Layer1_P, P_save) = Initial_Layer.initial_layer(P, InputWeight1, NumberofTrainingData, layer)
                 
                    if (j == 2): 
                        H = None
                        OrgP = None
                        P = None
                        YYM_Layer1_P = None
                        Y2 = None
                        (layer, P, InputWeight11) = Initial_Layer_Learning.initial_layer_learning(PP, P_save, C, layer, InputWeight1, Learning_rate)
                     
                    
                    if (j == 3):
                         (YYM_H, H, layer) = Initial_Layer_Forward.initial_layer_forward(P_save, InputWeight11, layer)

                    H = YYM_H
 
                else:
                    #Layer 2

                    if (layer == 2):
                        for nxh in range(0,count):
                            if (j == 1):
                                if (yym_loop > 1):
                                    InputWeight2 = InputWeight22.conj().T 
                            
                                (YYM_H, layer, H) = Initial_Mid_Layer.initial_mid_layer(YYM_H, InputWeight2, layer)

                            if (j == 2):
                                Y4 = None
                                Y22 = None
                                YJX = None
                                E1 = None
                                #FT1 = None 
                                FYY = None
                                GXZ2 = None
                                P = None

                                (layer, P, InputWeight22, PP) = Mid_Layer_Learning.mid_layer_learning(PP, YYM_Layer1_P, C, layer, InputWeight2, Learning_rate)

                            if (j == 3):
                                (YYM_H, H, layer) = Mid_Layer_Forward.mid_layer_forward(H, YYM_H, InputWeight22, layer)


                    else:
                        #Layer 3

                        if (layer == 3):
                            P = H
                            H = None
                            E1 = T - np.matmul(YYM.conj().T,P)

                            for i in range (1, 3):
                                a = 0.0000001
                                Y2 = E1
                                
                                tempH = None

                                if (fdafe == 0):
                                    Y22 = Y2
                                else:
                                    Y22 = Y2

                                Y2 = Y22
                                Y4 = Y2.real

                                if (fdafe == 0):
                                    a = (np.eye(len(P)) / C + (P @ (P.conj().T)))
                                    b = (P @ (Y4.conj().T))
                                    YYM_temp = np.linalg.solve(a, b)

                                    YYM = YYM + 1 * YYM_temp

                                    YJX = (YYM.conj().T @ P).conj().T

                                else:
                                    a = Y4.conj().T
                                    eye = (np.eye(len(YYM)))
                                    YYM_conj = YYM.conj().T
                                    a_ling = eye/C+YYM @ YYM_conj
                                    ling_solve = (np.linalg.solve(a_ling , YYM)).conj().T
                                    PP = a @ ling_solve
                                    PP = PP.conj().T  
                                     
                                    YJX = (PP.conj().T) @ YYM

                                BB1 = Y4.shape
                                BB2 = sum(YJX - Y4.conj().T)
                                BB = BB2/BB1[1]
                                BB = BB[0]
                                Bias3 = BB

                                GXZ2 = (YJX.conj().T - BB.T)

                                FYY = GXZ2.conj().T
                                
                                vars()['FT1'+str(i)] = FYY.conj().T

                                E1 = T - vars()['FT1'+str(i)]

                                if (i == 1):
                                    fdafe = 1

                            fdafe = 0
                            layer = layer - 1
                            mse = np.mean(E1**2)
                            sqr_mse = math.sqrt(mse)

        D_YYM0 = YYM 

        DBB0 = BB
       
    return InputWeight11, InputWeight22, D_YYM0, Bias3