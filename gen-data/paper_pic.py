# coding: utf8
from __future__ import division
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.data import coffee


def local_binary_pattern(image, P, R):
    """Gray scale and rotation invariant LBP (Local Binary Patterns).
    LBP is an invariant descriptor that can be used for texture classification.
    """

    # local position of texture elements
    rr = - R * np.sin(2 * np.pi * np.arange(P, dtype=np.double) / P)
    cc = R * np.cos(2 * np.pi * np.arange(P, dtype=np.double) / P)
    rp = np.round(rr, 5)
    cp = np.round(cc, 5)

    # pre-allocate arrays for computation
    texture = np.zeros(P, dtype=np.double)
    signed_texture = np.zeros(P, dtype=np.int8)

    output_shape = (image.shape[0], image.shape[1])
    output = np.zeros(output_shape, dtype=np.double)

    rows = image.shape[0]
    cols = image.shape[1]


    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            for i in range(P):
                texture[i] = bilinear_interpolation(image, rows, cols,
                                                    r + rp[i], c + cp[i],
                                                    'C', 0)
            # signed / thresholded texture
            for i in range(P):
                if texture[i] - image[r, c] >= 0:
                    signed_texture[i] = 1
                else:
                    signed_texture[i] = 0

            # Compute the variance without passing from numpy.
            # Following the LBP paper, we're taking a biased estimate
            # of the variance (ddof=0)
            sum_ = 0.0
            var_ = 0.0
            for i in range(P):
                texture_i = texture[i]
                sum_ += texture_i
                var_ += texture_i * texture_i
            var_ = (var_ - (sum_ * sum_) / P) / P
            if var_ != 0:
                lbp = var_
            else:
                lbp = np.nan
            output[r, c] = lbp

    return np.asarray(output)

def  bilinear_interpolation(image,  rows,cols,  r,  c, mode,  cval):
    """Bilinear interpolation at a given position in the image.
    """
    minr = int(np.floor(r))
    minc = int(np.floor(c))
    maxr = int(np.ceil(r))
    maxc = int(np.ceil(c))
    dr = r - minr
    dc = c - minc
    top = (1 - dc) * get_pixel2d(image, rows, cols, minr, minc, mode, cval) \
          + dc * get_pixel2d(image, rows, cols, minr, maxc, mode, cval)
    bottom = (1 - dc) * get_pixel2d(image, rows, cols, maxr, minc, mode,
                                    cval) \
             + dc * get_pixel2d(image, rows, cols, maxr, maxc, mode, cval)
    return (1 - dr) * top + dr * bottom



def get_pixel2d(image, rows, cols,r, c, mode,cval):
    """Get a pixel from the image, taking wrapping mode into consideration.
    """
    if (r < 0) or (r >= rows) or (c < 0) or (c >= cols):
        return cval
    else:
        return image[r , c]

def circular_LBP(src, n_points, radius):
    height = src.shape[0]
    width = src.shape[1]
    dst = src.copy()
    src.astype(dtype=np.float32)
    dst.astype(dtype=np.float32)

    neighbours = np.zeros((1, n_points), dtype=np.uint8)
    lbp_value = np.zeros((1, n_points), dtype=np.uint8)
    for x in range(radius, width - radius - 1):
        for y in range(radius, height - radius - 1):
            lbp = 0.
            # 先计算共n_points个点对应的像素值，使用双线性插值法
            for n in range(n_points):
                theta = float(2 * np.pi * n) / n_points
                x_n = x + radius * np.cos(theta)
                y_n = y - radius * np.sin(theta)

                # 向下取整
                x1 = int(math.floor(x_n))
                y1 = int(math.floor(y_n))
                # 向上取整
                x2 = int(math.ceil(x_n))
                y2 = int(math.ceil(y_n))

                # 将坐标映射到0-1之间
                tx = np.abs(x - x1)
                ty = np.abs(y - y1)

                # 根据0-1之间的x，y的权重计算公式计算权重
                w1 = (1 - tx) * (1 - ty)
                w2 = tx * (1 - ty)
                w3 = (1 - tx) * ty
                w4 = tx * ty

                # 根据双线性插值公式计算第k个采样点的灰度值
                neighbour = src[y1, x1] * w1 + src[y2, x1] * w2 + src[y1, x2] * w3 + src[y2, x2] * w4

                neighbours[0, n] = neighbour

            center = src[y, x]

            for n in range(n_points):
                if neighbours[0, n] > center:
                    lbp_value[0, n] = 1
                else:
                    lbp_value[0, n] = 0

            for n in range(n_points):
                lbp += lbp_value[0, n] * 2**n

            # 转换到0-255的灰度空间，比如n_points=16位时结果会超出这个范围，对该结果归一化
            dst[y, x] = int(lbp / (2**n_points-1) * 255)

    return dst

if __name__ == "__main__":
    img = coffee()
    fig = plt.figure()
    f1 = fig.add_subplot(121)
    f1.imshow(img)
    f1.set_title("image")

    for colour_channel in (0, 1, 2):
        img[:, :, colour_channel] = local_binary_pattern(
            img[:, :, colour_channel], 8,1.0)
    print (img[:, :, colour_channel])
    print (circular_LBP(img[:, :, colour_channel], 8,1))

    f2 = fig.add_subplot(122)
    f2.imshow(img)
    f2.set_title("LBP")
    plt.imshow()

"""
[[ 12  10  12 ..., 198 227  46]
 [ 11   0   0 ...,   7   3   5]
 [ 12   0   0 ...,   5   2  14]
 ..., 
 [ 13 105  23 ...,  19   6 203]
 [222  23   7 ...,   7  14 165]
 [ 84  26  15 ..., 174 185 179]]
[[ 12  10  12 ..., 198 227  46]
 [ 11  54   7 ...,   6   3   5]
 [ 12 112  98 ..., 100   2  14]
 ..., 
 [ 13  36  20 ...,   0   6 203]
 [222  23   7 ...,   7  14 165]
 [ 84  26  15 ..., 174 185 179]]

"""