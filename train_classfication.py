#coding=utf-8

#编码器分类模型训练
import matplotlib
import argparse
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np 
import pandas as pd
from scipy.interpolate import make_interp_spline as spline

from keras import *
from keras.models import Sequential  
from keras.layers import *
from keras.layers import Input
from keras.utils.np_utils import to_categorical  
from keras.preprocessing.image import img_to_array  
from keras.callbacks import ModelCheckpoint ,TensorBoard
from Models import *
from Models.DeeplabEncoder import *
from Models.VGG import *
from Models.Xception import *
from Models.ResNet import *
from Models.GooLeNet import *
from Models.AlexNet import *
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

def generateDataTF(batch_size,img_w,img_h,n_label,image_names=[],label_names=[]): 
    print ('开始随机生成训练数据中。。。\nImage-Data-Generating...\n')
    image_filepath ='D:\Python\seg-data\data_MB/'
    batch_num=0

    while True:   
        bs=batch_size
        
        dataset = gdal.Open(os.path.join(image_filepath,image_names[batch_num%len(image_names)]))
        im_width = dataset.RasterXSize #栅格矩阵的列数
        im_height = dataset.RasterYSize #栅格矩阵的行数
        #print(im_width ,im_height)
        label_data=cv2.imread(os.path.join(image_filepath,label_names[batch_num%len(image_names)]),cv2.IMREAD_GRAYSCALE)
        #yield(label_data.shape)
        train_data = []  
        train_label =  []  
        i=0
        while (bs-i)!=0:
            random_width = random.randint(0, im_width - img_w - 1)
            random_height = random.randint(0, im_height - img_h - 1)
            tif_roi=dataset.ReadAsArray(random_width,random_height,img_w,img_h)
            if (np.sum(tif_roi[0]==0)/(im_width*im_height))<0.5:
                data_roi=cv2.merge(tif_roi)  
                tmp_img = [np.max(label_data[random_height: random_height + img_h , random_width: random_width + img_w])]
                label_roi = to_categorical([np.max(label_data[random_height: random_height + img_h , random_width: random_width + img_w])],2).reshape(-1)

                train_data.append( data_roi)  
                train_label.append(label_roi)
                i=i+1
                
        yield(np.array(train_data),np.array(train_label))
        batch_num=batch_num+1

def train(key,EPOCHS = 10,BatchSize = 4,train_numb_per_epoch = 10*8,valid_rate = 0.2): 
    key=args['key']
    EPOCHS = int(args['epochs'])
    BS = int(args['batchsize'])
    img_w = int(args['size']) 
    img_h = int(args['size'])

    train_numb=train_numb_per_epoch*EPOCHS
    valid_numb = train_numb*valid_rate 

    method = {
        "Xception":Xception,
        "DeeplabEncoder":DeeplabEncoder,
        "ResNet34":ResNet34,
        "ResNet50":ResNet50,
        "GooLeNet": GooLeNet,
        "AlexNet": AlexNet,
        "VGG": VGG
        }
    m = method[key]()
    m.compile(loss='categorical_crossentropy',optimizer="adadelta",metrics=['acc'])
    
    modelcheck = ModelCheckpoint('D:\Python\seg-data/model/%s_model.h5' % key,
#modelcheck = ModelCheckpoint('..\..\Python\seg-data/model/SegNet-'+time.strftime(f'%Y-%m-%d-%a-%H-%M-%S',time.localtime(time.time()))+'.h5',
                                 monitor='val_acc',
                                 save_best_only=True,
                                 mode='max')  
    tb=TensorBoard(log_dir='D:\Python\seg-data/log/%s_log/' % key)
    callableTF = [modelcheck,tb]   




    print ("the number of train data is",train_numb,train_numb//BS)  
    print ("the number of val data is",valid_numb,valid_numb//BS)
    train10m_name=['201612_10M.tif','201704_10M.tif','2019A3_10M.tif','2019B3_10M.tif','201905_10M.tif']
    train20m_name=['201612_20M.tif','201704_20M.tif','2019A3_20M.tif','2019B3_20M.tif','201905_20M.tif']
    label10m_name=['test_10m_roi.png','test_10m_roi.png','test_10m_roi.png','test_10m_roi.png','test_10m_roi.png']
    label20m_name=['test_20m_roi.png','test_20m_roi.png','test_20m_roi.png','test_20m_roi.png','test_20m_roi.png']



    H = m.fit_generator(generator=generateDataTF(BS,img_w,img_h,2,train10m_name,label10m_name),
                        steps_per_epoch=train_numb_per_epoch,
                        epochs=EPOCHS,
                        verbose=0,
                        validation_data=generateDataTF(BS,img_w,img_h,2,train10m_name,label10m_name),
                        validation_steps=train_numb_per_epoch*valid_rate,
                        callbacks=callableTF,
                        max_q_size=1)  
    df_plot=pd.DataFrame([H.history["acc"],H.history["val_acc"],H.history["loss"],H.history["val_loss"]],index=["acc","val_acc","loss","val_loss"],columns=(range(EPOCHS)))
    df_plot.to_csv(r'D:\Python\seg-data/model/%s-plot.csv'% key)
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.xlim((0, EPOCHS))
    plt.ylim((0, 1))
    N_new = np.linspace(np.arange(0, EPOCHS).min(),np.arange(0, EPOCHS).max(),EPOCHS*100)
    train_acc_smooth = spline(np.arange(0, EPOCHS),H.history["acc"],N_new)
    val_acc_smooth = spline(np.arange(0, EPOCHS),H.history["val_acc"] ,N_new)
    plt.plot(N_new, train_acc_smooth, label="train_acc")
    plt.plot(N_new, val_acc_smooth, label="val_acc")
    plt.title(" Accuracy on %s Classfication" % key)
    plt.xlabel("Epoch ")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("D:\Python\seg-data/model/%s plot.png"% key)
def args_parse():
# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--key", required=False,help="key of train model ")
    ap.add_argument("-e", "--epochs", required=False,help="train epochs")
    ap.add_argument("-b", "--batchsize", required=False,help="train batchsize")
    ap.add_argument("-s", "--size", required=False,default=256,help="sub image size")
    args = vars(ap.parse_args())    
    return args

if __name__ == '__main__':

    args = args_parse()
    train(args)