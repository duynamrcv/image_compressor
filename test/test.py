import numpy as np
import cv2
import matplotlib.pyplot as plt

# mat = np.load("hiuhiu.npy")
# np.savetxt('test.csv', mat, delimiter=',')


def get_dct_image(img):
    f = cv2.dct(img.astype(np.float32))
    return f

def get_idct_image(f):
    img = cv2.idct(f.astype(np.float32))
    return img

def get_core_gauss(f, percent=0.5):
    rows, cols = f.shape

    mask = np.zeros([rows, cols])
    for i in range(rows):
        for j in range(cols):
            mask[i,j] = np.sqrt(i**2 + j**2)

    distance = int(percent*(rows+cols)/2)
    mask = np.exp(-mask**2/(2*distance**2))

    f = f*mask

    nrows = int(percent*rows); ncols = int(percent*cols)
    f_new = f[:nrows, :ncols]
    
    return f_new, rows, cols

def get_core(f, percent=0.5):
    rows, cols = f.shape
    nrows = int(percent*rows); ncols = int(percent*cols)
    f_new = f[:nrows, :ncols]
    return f_new, rows, cols

def get_origin(f_new, rows, cols):
    f = np.zeros((rows, cols), np.float32)
    nrows, ncols = f_new.shape
    f[:nrows, :ncols] = f_new
    return f

img = cv2.imread("images/lena.png", 0)

# Compress
f = get_dct_image(img)
f_new, rows, cols = get_core_gauss(f, percent=0.3)
re_img = get_idct_image(f_new)

# Decompress
f_re = get_dct_image(re_img)
f_rec = get_origin(f_re, rows, cols)
rec = get_idct_image(f_rec)

plt.subplot(131); plt.imshow(img, cmap="gray")
plt.subplot(132); plt.imshow(re_img, cmap="gray")
plt.subplot(133); plt.imshow(rec, cmap="gray")
plt.show()