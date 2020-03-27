from keras.layers import Input
#coding=utf-8
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
from sklearn.preprocessing import LabelEncoder  
from PIL import Image  
import matplotlib.pyplot as plt  
import cv2
import random
import os
from tqdm import tqdm  
from keras import backend as K 
from keras.applications import vgg16
try:
    from Models.utils import *
except:
    from utils import *


def FCN32(
          input_shape=(256,256,4),
          n_labels=2,
          kernel=3,
          pool_size=(2, 2),
          output_mode="softmax"):
    nClasses=n_labels
    input_height=input_shape[0]
    input_width=input_shape[1]
    img_input = Input(shape=(input_height, input_width, input_shape[2]))
    assert input_height % 32 == 0
    assert input_width % 32 == 0


    model = vgg16.VGG16(
        include_top=False,
        weights=None, input_tensor=img_input)
    assert isinstance(model, Model)

    o = Conv2D(
        filters=4096,
        kernel_size=(
            7,
            7),
        padding="same",
        activation="relu",
        name="fc6")(
            model.output)
    o = Dropout(rate=0.5)(o)
    o = Conv2D(
        filters=4096,
        kernel_size=(
            1,
            1),
        padding="same",
        activation="relu",
        name="fc7")(o)
    o = Dropout(rate=0.5)(o)

    o = Conv2D(filters=nClasses, kernel_size=(1, 1), padding="same", activation="relu", kernel_initializer="he_normal",
               name="score_fr")(o)

    o = Conv2DTranspose(filters=nClasses, kernel_size=(32, 32), strides=(32, 32), padding="valid", activation=None,
                        name="score2")(o)

    o = Reshape((nClasses,-1))(o)
    o = Permute((2,1))(o)
    o = Activation("softmax")(o)

    fcn = Model(inputs=img_input, outputs=o)
    # mymodel.summary()
    return fcn
#m = FCN32()
#m.summary()