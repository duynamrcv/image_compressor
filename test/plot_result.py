import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("images/lena.png", 0)

bit = [128.125, 81.406, 32.125, 20.445]
psnr = [37.325, 36.135, 34.257, 31.368]

plt.subplot(121); plt.imshow(img, cmap='gray')
plt.title("Original")
plt.subplot(122); plt.plot(bit, psnr)
plt.title("PSNR")
plt.xlabel("bit size (KB)")
plt.ylabel("PSNR (dB)")
plt.show()