from keras import *

import numpy as np
import gdal
import  cv2
import random
import tensorflow as tf
from keras.backend import *
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint ,TensorBoard
from keras.models import Model
from keras.models import Sequential 
from keras.utils.np_utils import to_categorical 
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation,Input  
from keras import backend as K
from keras.engine.topology import Layer
 
class Area_interp(Layer):
 
    def __init__(self, scale, **kwargs):
        self.scale = scale
        super(Area_interp, self).__init__(**kwargs)
 
    def build(self, input_shape):
        # 为该层创建一个可训练的权
        
        print ("build",input_shape)
        '''
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)'''
        super(Area_interp, self).build(input_shape)  # 一定要在最后调用它
        
    def call(self, x):

        return tf.image.resize_images(x, (int_shape(x)[1]*self.scale,int_shape(x)[2]*self.scale),3)
    def compute_output_shape(self, input_shape):

        print ("compute_output_shape" ,input_shape)
        #return (input_shape[])
        return (input_shape[0],input_shape[1]*self.scale,input_shape[2]*self.scale,input_shape[3])