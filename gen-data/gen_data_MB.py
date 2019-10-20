#coding=utf-8

import cv2
import random
import os
import numpy as np
from tqdm import tqdm
import gdal
size=256
img_w = size
img_h = size

image_sets = ['2017','2019','2019-3']
def tiff2array(tif_data):
    x_width  = tif_data.RasterXSize    #栅格矩阵的列数
    x_height = tif_data.RasterYSize
    array_data=tif_data.ReadAsArray(0,0,x_width,x_height)
    return array_data,x_width,x_height
def changeImgDataShape(originalShapeData):
    data=originalShapeData
    changedShapeData=np.append(np.append(np.append(data[3,:,:][:,:,np.newaxis],data[2,:,:][:,:,np.newaxis],axis=2),data[1,:,:][:,:,np.newaxis],axis=2),data[0,:,:][:,:,np.newaxis],axis=2)
    return changedShapeData
def data_augment(xb,yb):

    return xb,yb
def creat_dataset(image_num = 1500, mode = 'original'):
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 0
    for i in tqdm(range(len(image_sets))):
        count = 0
        data_tif=gdal.Open('D:\Python\seg-data/data_MB/' + image_sets[i]+'.tif')
        im_data,x_width,x_height=tiff2array(data_tif)
        #im_data=changeImgDataShape(im_data)
       #####################################
        label_img = cv2.imread('D:\Python\seg-data/data_MB/'+ image_sets[i]+'.png' ,cv2.IMREAD_GRAYSCALE)  
        while count < image_each:
            random_width = random.randint(0, x_width - img_w - 1)
            random_height = random.randint(0, x_height - img_h - 1)
            

            src_roi = im_data[:,random_height: random_height + img_h, random_width: random_width + img_w]
            label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
            
            if mode == 'augment':
                src_roi,label_roi = data_augment(src_roi,label_roi)
                
                src_roi_band0=src_roi[:,:,0]
                
            if (src_roi_band0[src_roi_band0==0].size<60000):
                visualize = np.zeros((size,size)).astype(np.uint8)
                visualize = label_roi*250
                black_num = 0
                for column in label_roi:
                    for pixel in  column:
                        if (pixel == 0):
                            black_num=black_num+1
                if (black_num<size*size*0.5):
                    cv2.imwrite(('D:\Python\seg-data\gen_sub_img1/labelV/%d.png' % g_count),visualize)
                    np.save('D:\Python\seg-data\gen_sub_img1/src/%d' % g_count,src_roi)
                    cv2.imwrite(('D:\Python\seg-data\gen_sub_img1/label/%d.png' % g_count),label_roi)
                    
                    
                    count += 1 
                    g_count += 1
if __name__=='__main__':  
    creat_dataset(1000,'augment')
