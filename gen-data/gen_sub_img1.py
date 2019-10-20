import gdal
import numpy as np
import cv2
from tqdm import tqdm
def gen_sub_img1(size=1024, stride=64):
    key=args['key']
    size = int(args['size'])
    stride = int(args['stride'])
    image_size=size
    IMG_SET=['2017','2019','2019-3']
    predir=r'D:\Python\seg-data\data_MB/'
    g_count = 0
    for n in tqdm(range(len(IMG_SET))):
        tif_img = gdal.Open(predir+IMG_SET[n]+'.tif')
        label_img=cv2.imread(predir+IMG_SET[n]+'.png',cv2.IMREAD_GRAYSCALE)
        tif_w = tif_img.RasterXSize #栅格矩阵的列数
        tif_h = tif_img.RasterYSize
        tif_data=tif_img.ReadAsArray(0,0,tif_w,tif_h)
        tif_d=tif_data.shape[0]
        tif_data=np.array(tif_data, dtype=float)
        image=cv2.merge(tif_data)
        h,w,_ = image.shape
        padding_h = ((h-stride)//(size-stride )+ 1) * (size-stride)+stride
        padding_w = ((w-stride)//(size-stride )+ 1) * (size-stride)+stride
        padding_img = np.zeros((padding_h,padding_w,_))
        padding_label= np.zeros((padding_h,padding_w))
        padding_img[0:h,0:w,:] = image[:,:,:]
        padding_label[0:h,0:w] = label_img[:,:]
        #b1,b2,b3,b4=cv2.split(padding_img) 
        #print(np.sum(b1==0))
        for i in range((padding_h-stride)//(size-stride)):
            for j in range((padding_w-stride)//(size-stride)):
                crop = padding_img[i*(size-stride):i*(size-stride)+image_size,j*(size-stride):j*(size-stride)+image_size,:]
                sub_label = padding_label[i*(size-stride):i*(size-stride)+image_size,j*(size-stride):j*(size-stride)+image_size]
                 
                if (np.sum(sub_label==0)!=size*size):
                    print(np.sum(sub_label!=0))
                    cv2.imwrite(('D:\Python\seg-data\gen_sub_img1/labelV/%d.png' % g_count),sub_label*255)
                    np.save('D:\Python\seg-data\gen_sub_img1/src/%d' % g_count,crop)
                    cv2.imwrite(('D:\Python\seg-data\gen_sub_img1/label/%d.png' % g_count),sub_label)
                    g_count=g_count+1    
def args_parse():
# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_a
    rgument("-s", "--size", required=False,help="sub image size")
    ap.add_argument("-t", "--stride", required=False,help=" image stride")
    args = vars(ap.parse_args())    
    return args

if __name__ == '__main__':
    args = args_parse()
    gen_sub_img1()