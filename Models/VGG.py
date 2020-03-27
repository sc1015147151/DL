#coding=utf-8
from keras.models import Sequential  
from keras.layers import Dense,Flatten,Dropout  
from keras.layers.convolutional import Conv2D,MaxPooling2D  
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np 
from keras import *
from keras.models import Sequential  
from keras.layers import *
from keras.layers import Input
from keras.utils.np_utils import to_categorical  
from keras.preprocessing.image import img_to_array  
from keras.callbacks import ModelCheckpoint ,TensorBoard
try:
    from Models.utils import *
    from Models.Area_interp import *
except:
    from utils import *
    from Area_interp import *
from sklearn.preprocessing import LabelEncoder  
from PIL import Image  
import matplotlib.pyplot as plt  
import cv2
import random
import os
from tqdm import tqdm  
from keras import backend as K 
from keras.applications import vgg16
from keras.layers import Input

def VGG(
        input_shape=(256,256,4),
        n_labels=2,
        kernel=3,
        pool_size=(2, 2),
        output_mode="softmax"):
    # encoder
 

  
        model = Sequential()  
        model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(256,256,4),padding='same',activation='relu',kernel_initializer='uniform')) 
        model.add(BatchNormalization())  
        model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(BatchNormalization())  
        model.add(MaxPooling2D(pool_size=(2,2)))  
        model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(BatchNormalization())  
        model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(BatchNormalization())  
        model.add(MaxPooling2D(pool_size=(2,2)))  
        model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(BatchNormalization())  
        
        model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(BatchNormalization()) 
        model.add(MaxPooling2D(pool_size=(2,2)))  
        model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(BatchNormalization()) 
        model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(BatchNormalization()) 
        model.add(MaxPooling2D(pool_size=(2,2)))  
        model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(BatchNormalization()) 
        model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
        model.add(BatchNormalization()) 
        model.add(MaxPooling2D(pool_size=(2,2)))  
        model.add(Flatten())  
        model.add(Dense(4096,activation='relu')) 
        model.add(BatchNormalization())  
        model.add(Dropout(0.5))  
        model.add(Dense(4096,activation='relu'))  
        model.add(BatchNormalization()) 
        model.add(Dropout(0.5))  
        model.add(Dense(1000,activation='relu'))
        model.add(BatchNormalization())   
        model.add(Dense(2,activation='softmax'))  
        model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])  
        model.summary()  
                
        return model