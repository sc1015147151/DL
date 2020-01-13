#coding=utf-8
import keras
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

from Models.utils import *
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
def AlexNet(
        input_shape=(256,256,4),
        n_labels=2,
        kernel=3,
        pool_size=(2, 2),
        output_mode="softmax"):
    # encoder
        inputs = keras.Input(shape = [256, 256, 4])

# AlexNet
        model = Sequential()
#第一段
        model.add(Conv2D(filters=96, kernel_size=(11,11),
                         strides=(4,4), padding='valid',
                         input_shape=input_shape,
                         activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3,3), 
                       strides=(2,2), 
                       padding='valid'))
#第二段
        model.add(Conv2D(filters=256, kernel_size=(5,5), 
                         strides=(1,1), padding='same', 
                         activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3,3), 
                               strides=(2,2), 
                               padding='valid'))
#第三段
        model.add(Conv2D(filters=384, kernel_size=(3,3), 
                         strides=(1,1), padding='same', 
                         activation='relu'))
        model.add(Conv2D(filters=384, kernel_size=(3,3), 
                         strides=(1,1), padding='same', 
                         activation='relu'))
        model.add(Conv2D(filters=256, kernel_size=(3,3), 
                         strides=(1,1), padding='same', 
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(3,3), 
                               strides=(2,2), padding='valid'))
#第四段
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
 
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
 
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.5))
 
# Output Layer
        model.add(Dense(2))
        model.add(Activation('softmax'))
 
        model.compile(loss='categorical_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])




        model.summary()  
        return model