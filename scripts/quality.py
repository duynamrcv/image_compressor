import cv2
import numpy as np
import matplotlib.pyplot as plt

def diff(ori, rec):
    return np.abs(ori-rec)

def mse(ori, rec):
    rows, cols = ori.shape
    return 1/(rows*cols)*(np.sum((ori - rec)**2))

def psnr(ori, rec):
    mse_value = mse(ori, rec)
    return 10*(np.log10(255**2/mse_value))