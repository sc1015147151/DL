#coding=utf-8
import matplotlib
import os
import argparse
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np 
from keras import *
from keras.models import Sequential  
from keras.layers import *
from keras.applications import vgg16
from keras.utils.np_utils import to_categorical  
from keras.preprocessing.image import img_to_array  
from keras.callbacks import ModelCheckpoint ,TensorBoard
from Models.SegNet import *
from Models.SegNet2In import *
from Models.FCN32 import *
from Models.UNET import *
from Models.utils import *
#from Models.GooLeNet2Inputs import *
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
# Stop GDAL printing both warnings and errors to STDERR
gdal.PushErrorHandler('CPLQuietErrorHandler')

# Make GDAL raise python exceptions for errors (warnings won't raise an exception)
# gdal.UseExceptions()
import warnings
warnings.filterwarnings("ignore")
seed = 7  
np.random.seed(seed)  
# data for training  
def bandScale(band):
    return band/(np.max(band))

def genDataTwo(batch_size,img_w,img_h,n_label,image_names=[],label_names=[],image_file_path = 'D:/Python/seg-data/data_MB'): 
    print ('开始随机生成训练数据中。。。\nImage-Data-Generating...')
    img_size_20m=int(img_h/2)
    image_name_10m=image_names[0]
    image_name_20m=image_names[1]
    label_name_10m=label_names[0]
    label_name_20m=label_names[1]
    image_filepath=image_file_path
    #image_filepath ='D:/Python/seg-data/data_MB/'
    batch_num=0
    while True:   
        bs=batch_size
        #print('\n tiff路径 \n',os.path.join(image_filepath,image_name_10m[batch_num%len(image_name_10m)]))
        dataset_10m = gdal.Open(os.path.join(image_filepath,image_name_10m[batch_num%len(image_name_10m)]))
        im_width_10m = dataset_10m.RasterXSize #栅格矩阵的列数
        im_height_10m = dataset_10m.RasterYSize #栅格矩阵的行数
        label_data_10m=cv2.imread(os.path.join(image_filepath,label_name_10m[batch_num%len(label_name_10m)]),cv2.IMREAD_GRAYSCALE)
        # print('\n png路径 \n',os.path.join(image_filepath,label_name_10m[batch_num%len(label_name_10m)]))
        # print(set(label_data_10m.flatten()))
        # plt.imshow(label_data_10m)
        # plt.show()
        train_data_10m = []  
        train_label_10m =  []  

        dataset_20m = gdal.Open(os.path.join(image_filepath,image_name_20m[batch_num%len(image_name_20m)]))
        im_width_20m = dataset_20m.RasterXSize #栅格矩阵的列数
        im_height_20m = dataset_20m.RasterYSize #栅格矩阵的行数
        label_data_20m=cv2.imread(os.path.join(image_filepath,label_name_20m[batch_num%len(label_name_20m)]),cv2.IMREAD_GRAYSCALE)
        #yield(label_data.shape)
        train_data_20m = []  
        train_label_20m =  []  
        #影像可截取点起点
        im_height=np.floor(np.min([im_height_20m,im_height_10m/2])- img_h/2 - 1).astype(int)
        im_width=np.floor(np.min([im_width_20m,im_width_10m/2])- img_w/2 - 1).astype(int)
        i=0
        while (bs-i)!=0:

            random_width = random.randint(0, np.floor(np.min([im_width_20m,im_width_10m/2]) - img_w/2 - 1))
            random_height = random.randint(0, np.min([im_height_20m,im_height_10m/2]) - img_h - 1)
            tif_roi_10=dataset_10m.ReadAsArray(random_width*2,random_height*2,img_size_20m*2,img_size_20m*2)
          
            tif_roi_20=dataset_20m.ReadAsArray(random_width,random_height,int(img_size_20m),int(img_size_20m))
       
            if (np.sum(tif_roi_10[0]==0)/(256*256))<0.25:
                #print(np.sum(tif_roi_10[0]==0)/(256*256))
                data_roi_10=cv2.merge((bandScale(tif_roi_10[0]),
				                       bandScale(tif_roi_10[1]),
                                       bandScale(tif_roi_10[2]),
				                       bandScale(tif_roi_10[3])))
                #data_roi_20=cv2.merge(tif_roi_20)
                data_roi_20=cv2.merge((bandScale(tif_roi_20[0]),
                                       bandScale(tif_roi_20[1]),
                                       bandScale(tif_roi_20[2])))
                label_roi_10 = to_categorical((label_data_10m[(random_height*2):(random_height*2 + img_size_20m*2),(random_width*2):(random_width*2+img_size_20m*2)]).flatten(), num_classes=n_label)
                #print("数据集制作中的标签",set((label_data_10m[(random_height*2):(random_height*2 + img_size_20m*2),(random_width*2):(random_width*2+img_size_20m*2)]).flatten()))
                label_roi_20 = to_categorical((label_data_20m[random_height: random_height +img_size_20m, random_width: random_width + img_size_20m]).flatten(), num_classes=n_label)
                train_data_10m.append( data_roi_10)  
                train_data_20m.append( data_roi_20) 
                train_label_10m.append(label_roi_10)
                train_label_20m.append(label_roi_20)
                i=i+1
                #yield(random_width,img_w,random_height,img_h)
                #yield(np.array(data_roi).shape,np.array(label_roi).shape)    
        #yield(np.array(train_data).shape,np.array(train_label).shape) 
  
        yield({"Input10" :train_data_10m,
               "Input20" :train_data_20m},np.array(train_label_10m))
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
        'SegNet2In_2': SegNet2In_2,
        'SegNet2In': SegNet2In}
    m = method[key]()#指定图像大小
    m.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    
    modelcheck = ModelCheckpoint('D:/Python/seg-data/model/%s_model.h5' % key,
#modelcheck = ModelCheckpoint('..\..\Python\seg-data/model/SegNet-'+time.strftime(f'%Y-%m-%d-%a-%H-%M-%S',time.localtime(time.time()))+'.h5',
                                 monitor='val_acc',
                                 save_best_only=False,
                                 mode='max')  
    tb=TensorBoard(log_dir='D:\Python\seg-data/log/%s_log/' % key)
    callableTF = [modelcheck,tb]   

    print ("the number of train data is",train_numb,train_numb//BS)  
    print ("the number of val data is",valid_numb,valid_numb//BS)

    H = m.fit_generator(generator=genDataTwo(BS,img_w,img_h,2,[['test_10m_roi.tif'],['test_20m_roi.tif']],[['test_10m_roi.png'],['test_20m_roi.png']]),
                            steps_per_epoch=train_numb_per_epoch,
                            epochs=EPOCHS,
                            verbose=2,
                            validation_data=genDataTwo(BS,img_w,img_h,2,[['test_10m_roi.tif'],['test_20m_roi.tif']],[['test_10m_roi.png'],['test_20m_roi.png']]),
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