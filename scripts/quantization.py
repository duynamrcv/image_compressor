import cv2
import numpy as np
import matplotlib.pyplot as plt

def quantization(img, ratio=4):
    out = (img//ratio)
    mapping = img - out*ratio
    # np.save("data/mapping.npy", mapping)
    # np.save("data/quan.npy", out)
    return out

def dequantization(quantized, ratio=4):
    mapping = np.random.randint(0, ratio, quantized.shape)
    # mapping = np.load("data/mapping.npy")
    out = quantized*ratio + mapping
    return out

# mapping = np.load("data/mapping.npy")
# new_map = np.round(mapping).astype(np.int16)
# plt.subplot(121); plt.imshow(new_map, cmap="gray")
# plt.subplot(122); plt.hist(new_map.ravel(),256,[-255,256])
# plt.show()