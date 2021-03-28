import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_dct_image(img):
    f = cv2.dct(img.astype(np.float32))
    return f

def get_idct_image(f):
    img = cv2.idct(f.astype(np.float32))
    return img

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

def mse(ori, rec):
    rows, cols = ori.shape
    return 1/(rows*cols)*(np.sum((ori - rec)**2))

img = cv2.imread("rec_img.png", 0)
f = get_dct_image(img)
f_new = gaussian_hpf(f, percent=0.6)
edge = get_idct_image(f_new)


# print(edge)
# print("------")
# k=1
# edge = k*edge
# edge = cv2.Sobel(img, cv2.CV_16S,1,1)
k = 1; edge = k*edge

bias = np.random.normal(0, 0.01, size=edge.shape)
edge = edge + bias

com = np.load("error.npy")
er = com - edge
# print(np.all(er>-20))
print("MSE = " + str(mse(com, edge)))

plt.figure()
plt.subplot(221); plt.imshow(img, cmap='gray')
plt.title("Original")
plt.subplot(222); plt.imshow(edge, cmap='gray')
plt.title("Edge")
plt.subplot(223); plt.imshow(com, cmap='gray')
plt.title("Compensate Ideal")
plt.subplot(224); plt.imshow(er, cmap='gray')
plt.title("Error")

plt.show()