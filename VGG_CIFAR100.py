import numpy as np 
import pandas as pd 
import os
import random
import cv2
import math
from keras.utils import to_categorical

import keras
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Activation, Dropout, Flatten,Input

import TELM33_new

import time

start = time.time()

C=2

net = VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))

net.summary()


numClasses = 100

x = net.output
x = Flatten()(x)
x = Dense(4096, activation='relu',name='fc1')(x)
x = Dropout(0.5,name='dopout1')(x)
x = Dense(4096, activation='relu',name='fc2')(x)
x = Dropout(0.5,name='dropout2')(x)
x = Dense(numClasses, activation='softmax', name='fc3')(x)
net = Model(inputs=net.input, outputs=x, name='model')

net.summary()


from keras import optimizers

rate = [0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.05,0.5,0.5,0.5]
rate2 = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]

#sgd = optimizers.SGD(lr=0.001, decay=0.1)
#net.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('./data/cifar100/unpacked/train',
                                                    target_size=(224, 224),
                                                    batch_size=16,
                                                    class_mode='categorical',
                                                    shuffle=False)

train_label = []

for i in range(0,3125):
    train_label.extend(np.array(train_generator[i][1]))

train_label = np.asarray(train_label, dtype=None, order=None)

print(train_label.shape)


#validation_generator = test_datagen.flow_from_directory('../input/sun397/sun397splitdata/SUN397SplitData/test',
#                                                        target_size=(224, 224),
#                                                        batch_size=64,
#                                                        class_mode='categorical')


test_datagen = ImageDataGenerator(rescale=1./255)



test_generator = test_datagen.flow_from_directory('./data/cifar100/unpacked/test',
                                                        target_size=(224, 224),
                                                        batch_size=16,
                                                        class_mode='categorical',
                                                        shuffle=False)


from keras import optimizers

sgd = optimizers.SGD(lr=0.001)#, decay=0.1)
net.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
#net.compile(optimizer='sgd',loss = 'categorical_crossentropy', metrics=['acc','mse'])
#size = len(testDigitData_Images_Path)
#net.fit(trainData, train_label, epochs = 1, batch_size = 16)

for i in range(8):
    net.fit_generator(train_generator,
                      steps_per_epoch=3125,
                      epochs=5)
                 #,
                 #validation_data=validation_generator,
                 #validation_steps=124)
    score = net.evaluate_generator(test_generator, steps=16, verbose=1)
    net.save(os.path.join('outputs', 'checkpoint_vgg16_base_' + str(i)))
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    end = time.time()
    print(end-start)


for i in range(20):

    print(i)
    sgd = optimizers.SGD(lr=rate2[i])#, decay=0.1)
    net.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    #net.compile(optimizer='sgd',loss = 'categorical_crossentropy', metrics=['acc','mse'])
    #size = len(testDigitData_Images_Path)
    #net.fit(trainData, train_label, epochs = 1, batch_size = 16)
    
    net.fit_generator(train_generator,
                      steps_per_epoch=3125,
                      epochs=1)
                     #,
                     #validation_data=validation_generator,
                     #validation_steps=124)


    layer_name = 'block5_pool'
    intermediate_layer_model = Model(inputs=net.input, outputs=net.get_layer(layer_name).output)
    
    #intermediate_layer_model.summary()

    trainingFeatures = intermediate_layer_model.predict(train_generator)
    
    #print(trainingFeatures.shape)
    #print(trainingFeatures.size)

    InputWeight1 = (net.layers[20].get_weights()) 
    InputWeight2 = (net.layers[22].get_weights())
    InputWeight3 = (net.layers[24].get_weights())

    Learning_rate = rate2[i]
    trainingFeatures = (np.reshape(trainingFeatures, [7*7*512,trainingFeatures.shape[0]])).conj().T

    (InputWeight11, InputWeight22, YYM, BB) = TELM33_new.telm33_new(trainingFeatures, train_label.conj().T, C, InputWeight1[0].T, InputWeight2[0].T, InputWeight3[0].T, Learning_rate, Learning_rate)

    InputWeight1[0] = InputWeight11
    InputWeight2[0] = InputWeight22
    InputWeight3[0] = YYM

    net.layers[20].set_weights(InputWeight1)
    net.layers[22].set_weights(InputWeight2) 
    net.layers[24].set_weights(InputWeight3)

    #net.fit(trainData, train_label, epochs = 1, batch_size = 16)
    net.fit_generator(train_generator,
                      steps_per_epoch=3125,
                      epochs=1)#,
                     #validation_data=validation_generator,
                     #validation_steps=124)

    net.save(os.path.join('outputs', 'checkpoint_vgg16_telm33_' + str(i)))
    score = net.evaluate_generator(test_generator, steps=16, verbose=1)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    end = time.time()
    print(end-start)

    




    