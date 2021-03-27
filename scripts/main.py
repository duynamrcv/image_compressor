import cv2
import numpy as np
import matplotlib.pyplot as plt

import arithmetic as ari
import frequency as fre
import quantization as qtz
import quality as qlt

def compressor(img, percent=0.5, qtz_ratio=4, ari_block=4):
    # TRANSFORMATION
    # Using to frequency domain uing fft and use the gaussian lowpass filter to reduce the redundancy
    f_img = fre.get_dct_image(img)
    f_img, rows, cols = fre.get_core(f_img, percent=percent)

    np.save("hiuhiu.npy", f_img)

    size_file = open('data/size.txt', "w")
    size_file.write(str(rows))          # write row dimention
    size_file.write('\n' + str(cols))   # write col dimention
    print("Transformation: Done!")
    print(f_img.astype(np.uint8))
    print("----------------------")

    # QUANTIZATION
    quan_img = qtz.quantization(f_img, ratio=qtz_ratio)
    print("Quantization: Done!")
    print(quan_img)
    print("----------------------")

    # CODING
    # Using Arithmetic coding
    ari.encode_image(quan_img, block_size=ari_block)
    print("Encoding: Done!")
    
    res = np.load("data/encoded_image.npy")
    print(res)
    print("----------------------")
    print("COMPRESS: SUCCESSFUL!!!")
    return quan_img

def decompressor(qtz_ratio=4):
    # DECODING
    # Using Arithmetic decoding
    dec = ari.decode_image().astype(np.uint8)
    print("Decoding: Done!")
    print(dec)
    print("----------------------")

    # DE-QUANTIZATION
    deqtz_img = qtz.dequantization(dec, ratio=qtz_ratio)
    print("De-quantization: Done!")
    print(deqtz_img.astype(np.int16))
    # np.save('hiuhiu.npy', deqtz_img)
    print("----------------------")

    # INVERSE TRANSFORMATION
    size_file = open("data/size.txt", "r")
    rows = int(size_file.readline())
    cols = int(size_file.readline())
    f = fre.get_origin(deqtz_img, rows, cols)
    img = fre.get_idct_image(f)
    print("Inverse Tranformation: Done!")
    print(img)
    print("----------------------")

    print("DECOMPRESS: SUCCESSFUL!!!")
    return img

def enhancer(img, hp=0.5, gain=1):
    f = fre.get_dct_image(img)
    f_edge = fre.gaussian_hpf(f, percent=hp)

    edge = fre.get_idct_image(f_edge)
    out = img + edge*gain
    return out

def main():
    # Read image
    img = cv2.imread("images/lena.png", 0)
    print(img)
    print("----------------------")

    percent=0.5
    qtz_ratio=4
    ari_block=4

    # Compression
    compressor(img, percent=percent, qtz_ratio=qtz_ratio, ari_block=ari_block)
    
    # res = np.load("data/encoded_image.npy")
    
    # Reconstruction
    new = decompressor(qtz_ratio=qtz_ratio)
    # new = enhancer(new, hp=0.3, gain=1.5)

    print("The MSE value: " + str(qlt.mse(img, new)))
    print("The PSNR value: " + str(qlt.psnr(img, new)))

    # Show image
    plt.subplot(121); plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.subplot(122); plt.imshow(new, cmap='gray')
    plt.title("Reconstruction")
    plt.show()

if __name__ == "__main__":
    main()
