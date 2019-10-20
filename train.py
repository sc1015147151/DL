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
from SegNet0 import *
from SegNet2 import *
from SegNet1 import *
from SegNet import *
from FCN32 import *
from UNET import *
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
    print ('gen-Sub-Image-Data...')
    image_filepath ='D:\Python\seg-data\data_MB/'
    batch_num=0
    while True:   
        bs=batch_size
        
        dataset = gdal.Open(image_filepath+image_names[batch_num%len(image_names)])
        im_width = dataset.RasterXSize #栅格矩阵的列数
        im_height = dataset.RasterYSize #栅格矩阵的行数
        #print(im_width ,im_height)
        label_data=cv2.imread(image_filepath+label_names[batch_num%len(image_names)],cv2.IMREAD_GRAYSCALE)
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
                label_roi = to_categorical((label_data[random_height: random_height + img_h , random_width: random_width + img_w]).flatten(), num_classes=n_label)
                train_data.append( data_roi)  
                train_label.append(label_roi)
                i=i+1
                #yield(random_width,img_w,random_height,img_h)
                #yield(np.array(data_roi).shape,np.array(label_roi).shape)    
        #yield(np.array(train_data).shape,np.array(train_label).shape)    
        yield(np.array(train_data),np.array(train_label))
        batch_num=batch_num+1
#image_names_set=['test.tif']
#label_names_set=['test_label.png']
#for i in(generateDataTF(8,256,256,2,image_names_set,label_names_set)):
#    print(i)
def train(key,EPOCHS = 10,BatchSize = 4,train_numb_per_epoch = 10*8,valid_rate = 0.2): 
    key=args['key']
    EPOCHS = int(args['epochs'])
    BS = int(args['batchsize'])
    img_w = int(args['size']) 
    img_h = int(args['size'])

    train_numb=train_numb_per_epoch*EPOCHS
    valid_numb = train_numb*valid_rate

    method = {
        "UNET": unet,
        "FCN32": FCN32,
        'SegNet': SegNet,
        'SegNet1': SegNet1,
        'SegNet2': SegNet2,
        'SegNet0': SegNet0}
    m = method[key]()#指定图像大小
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

    H = m.fit_generator(generator=generateDataTF(BS,img_w,img_h,2,['2016.tif','2017.tif','2019.tif'],['2016.png','2017.png','2019.png']),
                            steps_per_epoch=train_numb_per_epoch,
                            epochs=EPOCHS,
                            verbose=0,
                            validation_data=generateDataTF(BS,img_w,img_h,2,['2016.tif','2017.tif','2019.tif'],['2016.png','2017.png','2019.png']),
                            validation_steps=train_numb_per_epoch*valid_rate,
                            callbacks=callableTF,
                            max_q_size=1)  

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on %s Satellite Seg" % key)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
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