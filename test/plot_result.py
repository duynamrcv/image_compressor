import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("images/lena.png", 0)

bit = [12, 25, 50, 79, 100]
mse = [54.74, 25.40, 15.09, 12.61, 12.06]
psnr = [30.75, 33.91, 36.34, 37.13, 37.31]

plt.subplot(131); plt.imshow(img, cmap='gray')
plt.title("Original")
plt.subplot(132); plt.plot(bit, mse)
plt.title("MSE")
plt.xlabel("rate")
plt.ylabel("MSE")
plt.subplot(133); plt.plot(bit, psnr)
plt.title("PSNR")
plt.xlabel("rate")
plt.ylabel("PSNR")
plt.show()