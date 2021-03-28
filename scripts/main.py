import cv2
import numpy as np
import matplotlib.pyplot as plt

import arithmetic as ari
import frequency as fre
import quantization as qtz
import quality as qlt

def compressor(img, percent=0.5, ratio=4, block_size=4):
    ### RESIZE USING DCT AND IDCT
    f = fre.get_dct_image(img)
    f_new, img_row, img_col = fre.get_core_gauss(f, percent=percent)
    re_img = fre.get_idct_image(f_new)
    print("Resize Image: Done!")

    ### QUANTIZATION
    quan = qtz.quantization(re_img, ratio=ratio).astype(np.uint8)
    print("Quantization: Done!")

    ### ENCODING
    res, prob, quan_row, quan_col = ari.encode_image(quan, block_size=block_size)
    print("Encoding: Done!")
    print("COMPRESS: SUCCESSFUL!!!")
    return res, prob, img_row, img_col, quan_row, quan_col

def decompressor(res, prob, img_row, img_col, quan_row, quan_col, ratio=4, block_size=4):
    ### DECODING
    quan = ari.decode_image(res, prob, block_size, quan_row, quan_col)
    print("Decoding: Done!")

    ### DEQUANTIZATION
    re_img = qtz.dequantization(quan, ratio=ratio)
    print("Dequantization: Done!")

    ### GET ORIGINAL SIZE USING DCT AND IDCT
    f_re = fre.get_dct_image(re_img)
    f_rec = fre.get_origin(f_re, img_row, img_col)
    rec = fre.get_idct_image(f_rec)
    print("Restore Origin Image: Done!")

    ### COMPENSATION
    # Edge detection
    f_com = fre.get_dct_image(rec)
    f_com = fre.gaussian_hpf(f_com, percent=0.5)
    edge = fre.get_idct_image(f_com)

    # Compensation
    bias = np.random.normal(0, 0.02, size=edge.shape)
    comp = edge 

    rec += comp
    print("Compensation: Done!")

    print("DECOMPRESS: SUCCESSFUL!!!")
    return rec.astype(np.uint8)


def main():
    # Read image
    img_path = "images/image.png"
    img = cv2.imread(img_path, 0)

    # Compression
    res, prob, img_row, img_col, quan_row, quan_col = compressor(img, percent=0.5, ratio=4, block_size=4)
    
    np.save("encode.npy", res)

    # Reconstruction
    rec = decompressor(res, prob, img_row, img_col, quan_row, quan_col, ratio=4, block_size=4)
    # new = enhancer(new, hp=0.3, gain=1.5)

    print("The MSE value: " + str(qlt.mse(img, rec)))
    print("The PSNR value: " + str(qlt.psnr(img, rec)))

    # Show image
    plt.subplot(121); plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.subplot(122); plt.imshow(rec, cmap='gray')
    plt.title("Reconstruction")
    # plt.subplot(223); plt.imshow(np.abs(img-rec), cmap="gray")
    # plt.title("Error")
    # plt.subplot(224); plt.hist(np.abs(img-rec).ravel(), 256, [0,256])
    # plt.title("ErrorHist")
    plt.show()


if __name__ == "__main__":
    main()
