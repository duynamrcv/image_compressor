import cv2
import numpy as np
import matplotlib.pyplot as plt

def encode_image(pixels, block_size=4):
    rows, cols = pixels.shape
    total = rows*cols

    pixels1d = np.array(pixels).flatten()   # flatten the image into 1d array
    frq = np.zeros(256, dtype=int)          # frq of all gray scale levels

    res = np.zeros(total//block_size, dtype=np.float64)  # final codes
    prob = np.zeros(256, dtype=np.float64)
    grayLvl = 256

    for i in pixels1d:
        frq[i] += 1

    for i in range(0, grayLvl):
        prob[i] = frq[i] / (total)
        if(i > 0): prob[i] += prob[i-1]        

    for i in range(0, total, block_size):
        l = 0.0; r = 1.0
        for j in range(i, i + block_size):
            oldLeft = l; oldRight = r
            # base  + (range) * prob[cur pixel]
            if pixels1d[j] != 0:
                l = oldLeft + (oldRight - oldLeft) * prob[pixels1d[j] - 1]
            r = oldLeft + (oldRight - oldLeft) * prob[pixels1d[j]]
        # result of the block is the average of (upper - lower)
        it = int(i/block_size)
        res[it] = (l + r)/2

    # # export encoded tags && pro
    # np.save('data/encoded_image', res)
    # np.save('data/probability', prob)
    # block_size_file = open('data/block_size_file.txt', "w")
    # block_size_file.write(str(block_size))     # write block size
    # block_size_file.write('\n' + str(rows))    # write row dimension
    # block_size_file.write('\n' + str(cols))    # write col dimension

    return res, prob, rows, cols

def decode_image(res, prob, block_size, rows, cols):
    # res = np.load(encoded_path)
    # prob = np.load(prob_path)
    # block_size_file = open(block_size_path, "r")
    
    grayLvl = 256
    # block_size = int(block_size_file.readline())
    # row = int(block_size_file.readline())
    # col = int(block_size_file.readline())
    total = rows* cols
    
    out = np.zeros(total)

    for i in range(0, total, block_size):
        l = 0.0; r = 1.0
        for j in range(i, i + block_size):  # loop over block size  = 16
            for k in range(0, grayLvl):
                tag = res[int(i / block_size)]
                if tag < l + (r - l) * prob[k]:  # if This interval cover me
                    oldLeft = l; oldRight = r
                    if k != 0:
                        l = oldLeft + (oldRight - oldLeft) * prob[k - 1]
                    r = oldLeft + (oldRight - oldLeft) * prob[k]
                    out[j] = k
                    break

    out = np.array(out).reshape((rows, cols))
    # print("Decoding: Done!")
    return out

# img = cv2.imread("images/image.png", 0)
# res, prob, rows, cols = encode_image(img, block_size=4)
# rec = decode_image(res, prob, 4, rows, cols)

# plt.subplot(121); plt.imshow(img, cmap='gray')
# plt.title("Original")
# plt.subplot(122); plt.imshow(rec, cmap='gray')
# plt.title("Reconstruction")
# plt.show()