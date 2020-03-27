#coding=utf-8
import matplotlib
import argparse
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

from Models.SegNet import *
from Models.FCN32 import *
from Models.utils import *
from sklearn.preprocessing import LabelEncoder  
from PIL import Image  
import matplotlib.pyplot as plt  
import cv2
import random
import os
from tqdm import tqdm  
from keras import backend as K 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import gdal
seed = 7  
np.random.seed(seed)  
# data for training  
from keras.applications import vgg16

def get_train_val(val_rate = 0.25):
    filepath='D:\Python\seg-data\data_MB/'
    train_url = []    
    train_set = []
    val_set  = []
    #for pic in os.listdir(filepath + 'img'):
    #    train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i]) 
        else:
            train_set.append(train_url[i])
    return np.array(train_set),np.array(val_set)


def generateData(batch_size,data=[],n_label=2):  
    #print 'generateData...'
    filepath='D:\Python\seg-data\data_MB/'
    while True:  
        train_data = []  
        train_label = []  
        batch = 0  
        for i in (range(len(data))): 
            url,last_name = data[i].split('.')
            batch += 1 
            img = np.load(filepath + 'img/' + url+'.'+last_name)
            train_data.append(img)  
            print(filepath + 'label/' + url+'.png') 
            label = cv2.imread(filepath + 'label/' + url+'.png', cv2.IMREAD_GRAYSCALE) 
            label = to_categorical((label).flatten(), num_classes=n_label)
            train_label.append(label)  
            if batch % batch_size==0: 
                #print 'get enough bacth!\n'
                train_data = np.array(train_data)  
                train_label = np.array(train_label)  
                yield (train_data,train_label)  
                train_data = []  
                train_label = []  
                batch = 0  

def train(key,EPOCHS = 10,BatchSize = 4,train_numb_per_epoch = 12,valid_rate = 0.25): 
    key=args['key']
    stride = int(args['stride'])
    EPOCHS = int(args['epochs'])
    BS = int(args['batchsize'])
    img_w = int(args['size']) 
    img_h = int(args['size'])
    train_numb_per_epoch=int(args['train_numb_per_epoch'])
    train_numb=train_numb_per_epoch*EPOCHS
    valid_numb = train_numb*valid_rate

    method = {
        "FCN32": FCN32,
        'SegNet': SegNet,
        'SegNet1': SegNet1,
        'SegNet2': SegNet2,
        'SegNet0': SegNet0}
    m = method[key](input_shape=(img_w,img_h,4))#指定图像大小
    m.compile(loss='categorical_crossentropy',optimizer="adadelta",metrics=['acc'])
    
    modelcheck = ModelCheckpoint('D:\Python\seg-data/model/%s-%s-%s_model.h5' % (key,img_w,stride),
#modelcheck = ModelCheckpoint('..\..\Python\seg-data/model/SegNet-'+time.strftime(f'%Y-%m-%d-%a-%H-%M-%S',time.localtime(time.time()))+'.h5',
                                 monitor='val_acc',
                                 save_best_only=True,
                                 mode='max')  
    tb=TensorBoard(log_dir='D:\Python\seg-data/log/%s-%s-%s_log/' % (key,img_w,stride))
    callableTF = [modelcheck,tb]   


    data,vdata=get_train_val()
    H = m.fit_generator(generator=generateData(BS,data),
                            steps_per_epoch=train_numb_per_epoch//BS,
                            epochs=EPOCHS,
                            verbose=0,
                            validation_data=generateData(BS,vdata),
                            validation_steps=train_numb_per_epoch//BS*valid_rate,
                            callbacks=callableTF,
                            max_q_size=1)

    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on %s-%s-%s Satellite Seg" % (key,img_w,stride))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("D:\Python\seg-data/model/%s-%s-%splot.png"% (key,img_w,stride))
def args_parse():
# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--key", required=False,help="key of train model ")
    ap.add_argument("-e", "--epochs", required=False,help="train epochs")
    ap.add_argument("-b", "--batchsize", required=False,help="train batchsize")
    ap.add_argument("-s", "--size", required=False,help="sub image size")
    ap.add_argument("-t", "--stride", required=False,help=" image stride")
    ap.add_argument("-n", "--train_numb_per_epoch", required=False,help="train_numb_per_epoch")

    args = vars(ap.parse_args())    
    return args

if __name__ == '__main__':

    args = args_parse()
    train(args)