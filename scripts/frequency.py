import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_fft_image(img):
    f = np.fft.fft2(img)
    return f

def get_ifft_image(f):
    img = np.fft.ifft2(f)
    return abs(img)

def get_dct_image(img):
    f = cv2.dct(img.astype(np.float32))
    return f

def get_idct_image(f):
    img = cv2.idct(f.astype(np.float32))
    return np.uint8(img)

def gaussian_lpf(f, percent=0.5):
    rows, cols = f.shape

    mask = np.zeros([rows, cols])
    for i in range(rows):
        for j in range(cols):
            mask[i,j] = np.sqrt(i**2 + j**2)

    distance = int(percent*(rows+cols)/2)
    mask = np.exp(-mask**2/(2*distance**2))

    f = f*mask
    
    return f

def gaussian_hpf(f, percent=0.5):
    rows, cols = f.shape

    mask = np.zeros([rows, cols])
    for i in range(rows):
        for j in range(cols):
            mask[i,j] = np.sqrt(i**2 + j**2)

    distance = int(percent*(rows+cols)/2)
    mask = 1 - np.exp(-mask**2/(2*distance**2))

    f = f*mask
    
    return f

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

# img = cv2.imread("images/lena.png", 0)
# f = get_dct_image(img)
# f = gaussian_lpf(f, percent=0.3)
# new = get_idct_image(f)

# print(f)
# # cv2.imshow(f)
# plt.subplot(131); plt.imshow(img, cmap="gray")
# plt.subplot(132); plt.imshow(f, cmap="gray")
# plt.subplot(133); plt.imshow(new, cmap="gray")
# plt.show()

# f = np.load("hiuhiu.npy")
# new = get_idct_image(f)
# plt.imshow(new, cmap="gray")
# plt.show()