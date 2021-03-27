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
    return np.uint8(img)

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

f = get_dct_image(img)
f_new, rows, cols = get_core(f, percent=0.5)
re_img = get_idct_image(f_new)

plt.subplot(121); plt.imshow(img, cmap="gray")
plt.subplot(122); plt.imshow(re_img, cmap="gray")
plt.show()