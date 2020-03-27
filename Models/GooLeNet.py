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

try:
    from Models.utils import *
except:
    from utils import *
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
from keras.models import Model
from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D
import numpy as np
seed = 7
np.random.seed(seed)
def Conv2d_BN(x, nb_filter, kernel_size, padding='same', strides=(1, 1), name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def Inception(x, nb_filter):
    branch1x1 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

    branch3x3 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branch3x3 = Conv2d_BN(branch3x3, nb_filter, (3, 3), padding='same', strides=(1, 1), name=None)

    branch5x5 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branch5x5 = Conv2d_BN(branch5x5, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

    branchpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branchpool = Conv2d_BN(branchpool, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

    x = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=3)

    return x

def GooLeNet(
        input_shape=(256,256,4),
        n_labels=2,
        kernel=3,
        pool_size=(2, 2),
        output_mode="softmax"):
    inpt = Input(shape=input_shape)

# padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))
    x = Conv2d_BN(inpt, 64, (7, 7), strides=(2, 2), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2d_BN(x, 192, (3, 3), strides=(1, 1), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Inception(x, 64)  # 256
    x = Inception(x, 120)  # 480
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Inception(x, 128)  # 512
    x = Inception(x, 128)
    x = Inception(x, 128)
    x = Inception(x, 132)  # 528
    x = Inception(x, 208)  # 832
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Inception(x, 208)
    x = Inception(x, 256)  # 1024
    x = AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same')(x)
    x = Dropout(0.4)(x)
    x=Flatten()(x)
    x = Dense(1000, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(inpt, x, name='inception')
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()

def GooLeNet2Inputs(
        input_shape=[(256,256,4),(128,128,3)],
        n_labels=2,
        kernel=3,
        pool_size=(2, 2),
        output_mode="softmax"):
    inptMain = Input(shape=(256,256,4), name='Input10')
    inptSide = Input(shape=(128,128,3), name='Input20')
# padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))
    x = Conv2d_BN(inptMain, 64, (7, 7), strides=(2, 2), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2d_BN(x, 192, (3, 3), strides=(1, 1), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Inception(x, 64)  # 256
    x = Inception(x, 120)  # 480
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Inception(x, 128)  # 512
    x = Inception(x, 128)
    x = Inception(x, 128)
    x = Inception(x, 132)  # 528
    x = Inception(x, 208)  # 832
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Inception(x, 208)
    x = Inception(x, 256)  # 1024
    x = AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same')(x)
    x = Dropout(0.4)(x)
    x=Flatten()(x)
    x = Dense(1000, activation='relu')(x)

    y = Conv2d_BN(inptSide, 64, (7, 7), strides=(2, 2), padding='same')
    y = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(y)
    y = Conv2d_BN(y, 192, (3, 3), strides=(1, 1), padding='same')
    y = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(y)
    y = Inception(y, 64)  # 256
    y = Inception(y, 120)  # 480
    y = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(y)
    y = Inception(y, 128)  # 512
    y = Inception(y, 128)
    y = Inception(y, 128)
    y = Inception(y, 132)  # 528
    y = Inception(y, 208)  # 832
    y = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(y)
    y = Inception(y, 208)
    y = Inception(y, 256)  # 1024
    y = AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same')(y)
    y = Dropout(0.4)(y)
    y =Flatten()(y)
    y = Dense(1000, activation='relu')(y)

    x = concatenate([x,y], axis=-1)
    x = Dense(2, activation='softmax')(x)
    model = Model([inptMain,inptSide], x, name='inception')
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()

    return model
if __name__ == '__main__':
    m1 = GooLeNet()
    m2 = GooLeNet2Inputs ()
    
    from keras.utils import plot_model
   # plot_model(m1, show_shapes=True, to_file='GooLeNet.png')
   
    plot_model(m2, show_shapes=True, to_file='GooLeNet2Inputs.png')
