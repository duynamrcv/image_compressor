import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_dct_image(img):
    f = cv2.dct(img.astype(np.float32))
    return f

def get_idct_image(f):
    img = cv2.idct(f.astype(np.float32))
    return img

def mse(ori, rec):
    rows, cols = ori.shape
    return 1/(rows*cols)*(np.sum((ori - rec)**2))

if __name__ == "__main__":
    # Origin
    img = cv2.imread("images/image.png", 0)
    # print(img)
    # print("---------------")

    # DCT
    f = get_dct_image(img)
    # print(f.astype(np.int16))
    # print("---------------")

    # Cutting
    block_1 = f[:256, :256]; block_2 = f[256:384, 256:384]

    # Rebuild
    f_new = np.zeros(img.shape)
    f_new[:256,:256] = block_1
    f_new[256:384, 256:384] = block_2
    # print(f_new.astype(np.int16))
    # print("---------------")

    # IDCT
    rec = get_idct_image(f_new)
    # print(rec)
    # print("---------------")

    # TYPE2
    f_news = np.zeros(img.shape)
    f_news[:256,:256] = block_1
    # print(f_news)
    # print("---------------")

    rec2 = get_idct_image(f_news)
    # print(rec2)
    # print("---------------")

    # Different
    print("Type 1: " + str(mse(img, rec)))
    print("Type 2: " + str(mse(img, rec2)))
    print("---------------")


    plt.subplot(321); plt.imshow(img, cmap="gray")
    plt.subplot(322); plt.imshow(f.astype(np.int16), cmap="gray")
    plt.subplot(323); plt.imshow(f_new.astype(np.int16), cmap="gray")
    plt.subplot(324); plt.imshow(rec, cmap="gray")
    plt.subplot(325); plt.imshow(f_news.astype(np.int16), cmap="gray")
    plt.subplot(326); plt.imshow(rec2, cmap="gray")
    plt.show()