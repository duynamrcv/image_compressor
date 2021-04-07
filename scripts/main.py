import os
import sys
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import * 

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
    max_ = np.max(quan); min_ = np.min(quan)
    print("Max: {}, Min: {}, Range: {}".format(max_, min_, max_ - min_))
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

class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Variable for compress and decompress
        self.img = None; self.rec = None        # Original image and Restruction Image
        self.time_decom = 0; self.time_com = 0  # Time compress and Time decompress
        self.percent = 0.4                      # Gaussian percent to filter
        self.ratio = 4                          # Quantization ratio
        self.block_size = 4                     # Arithmetic block size

        # Window
        self.setWindowTitle("Image procesing")
        self.setGeometry(0, 0, 1280, 720) 
        self.setStyleSheet("background-color: #4682B4;") 

        # Original Image 
        self.original = QLabel(self)
        self.original.setText('Original')
        self.original.setAlignment(Qt.AlignCenter)
        self.original.setFont(QFont('Proxima Nova', 15))
        self.original.setGeometry(100, 60, 256, 30) 
        self.original.setStyleSheet("color: #000000;")

        self.input_image = QLabel(self) 
        self.input_image.setGeometry(100, 115, 256, 256) 
        self.input_image.setAlignment(Qt.AlignCenter)
        self.input_image.setStyleSheet("background-color:  #D2B29B; ")

        # Reconstruction Image
        self.reconstruction = QLabel(self)
        self.reconstruction.setText('Reconstruction')
        self.reconstruction.setAlignment(Qt.AlignCenter)
        self.reconstruction.setFont(QFont('Proxima Nova', 15))
        self.reconstruction.setGeometry(414, 60, 256, 30) 
        self.reconstruction.setStyleSheet("color: #000000;")

        self.output_image = QLabel(self)
        self.output_image.setGeometry(414, 115, 256, 256) 
        self.output_image.setAlignment(Qt.AlignCenter)
        self.output_image.setStyleSheet("background-color:  #D2B29B;")

        # BUTTON - File
        self.file = QPushButton('File', self)
        self.file.setFont(QFont('Proxima Nova', 10))
        self.file.setStyleSheet("background-color: #C4C4C4;color: #000000; border-radius: 20px")
        self.file.resize(120,60)
        self.file.move(100, 400)   
        self.file.clicked.connect(self.browsefiles)

        # BUTTON - Compress
        self.compress = QPushButton('Compress', self)
        self.compress.setFont(QFont('Proxima Nova', 10))
        self.compress.setStyleSheet("background-color: #C4C4C4;color: #000000; border-radius: 20px")
        self.compress.resize(120,60)
        self.compress.move(250, 400)  
        self.compress.clicked.connect(self.compress_image)

        # BUTTON - Decompress
        self.decompress = QPushButton('Decompress', self)
        self.decompress.setFont(QFont('Proxima Nova', 10))
        self.decompress.setStyleSheet("background-color: #C4C4C4;color: #000000; border-radius: 20px")
        self.decompress.resize(120,60)
        self.decompress.move(400, 400) 
        self.decompress.clicked.connect(self.decompress_image)

        # BUTTON - Properties
        self.properties = QPushButton('Propeties', self)
        self.properties.setFont(QFont('Proxima Nova', 10))
        self.properties.setStyleSheet("background-color: #C4C4C4;color: #000000; border-radius: 20px")
        self.properties.resize(120,60)
        self.properties.move(550, 400) 
        self.properties.clicked.connect(self.properties_image)

        # Create the properties area
        self.label = QLabel(self) 
        self.label.setGeometry(780, 0, 500, 720) 
        self.label.setStyleSheet("background-color:  #C4C4C4; ")        

        # LABEL - Original volumn
        self.original_volume = QLabel(self)
        self.original_volume.setText('Original volume :')
        self.original_volume.setAlignment(Qt.AlignRight)
        self.original_volume.setFont(QFont('Proxima Nova', 10))
        self.original_volume.setGeometry(780, 110, 180, 40) 
        self.original_volume.setStyleSheet("color: #000000; background-color: #C4C4C4;")

        self.box1 = QLabel(self)
        self.box1.setAlignment(Qt.AlignCenter)
        self.box1.setFont(QFont('Proxima Nova', 10))
        self.box1.setGeometry(985, 100, 250, 40) 
        self.box1.setStyleSheet("color: #000000; background-color: #FFFFFF; border-radius: 10px;")

        # LABEL - Compress volumn
        self.compress_volumn = QLabel(self)
        self.compress_volumn.setText('Compress volume :')
        self.compress_volumn.setAlignment(Qt.AlignRight)
        self.compress_volumn.setFont(QFont('Proxima Nova', 10))
        self.compress_volumn.setGeometry(780, 160, 180, 40) 
        self.compress_volumn.setStyleSheet("color: #000000; background-color: #C4C4C4;")

        self.box2 = QLabel(self)
        self.box2.setAlignment(Qt.AlignCenter)
        self.box2.setFont(QFont('Proxima Nova', 10))
        self.box2.setGeometry(985, 150, 250, 40) 
        self.box2.setStyleSheet("color: #000000; background-color: #FFFFFF; border-radius: 10px;")

        # LABEL - Compress ratio
        self.compress_ratio = QLabel(self)
        self.compress_ratio.setText('Compress ratio :')
        self.compress_ratio.setAlignment(Qt.AlignRight)
        self.compress_ratio.setFont(QFont('Proxima Nova', 10))
        self.compress_ratio.setGeometry(780, 210, 180, 40) 
        self.compress_ratio.setStyleSheet("color: #000000; background-color: #C4C4C4;")

        self.box3 = QLabel(self)
        self.box3.setAlignment(Qt.AlignCenter)
        self.box3.setFont(QFont('Proxima Nova', 10))
        self.box3.setGeometry(985, 200, 250, 40) 
        self.box3.setStyleSheet("color: #000000; background-color: #FFFFFF; border-radius: 10px;")

        # LABEL - Compress time
        self.time = QLabel(self)
        self.time.setText('Compress time :')
        self.time.setAlignment(Qt.AlignRight)
        self.time.setFont(QFont('Proxima Nova', 10))
        self.time.setGeometry(780, 260, 180, 40) 
        self.time.setStyleSheet("color: #000000; background-color: #C4C4C4;")

        self.box4 = QLabel(self)
        self.box4.setAlignment(Qt.AlignCenter)
        self.box4.setFont(QFont('Proxima Nova', 10))
        self.box4.setGeometry(985, 250, 250, 40) 
        self.box4.setStyleSheet("color: #000000; background-color: #FFFFFF; border-radius: 10px;")

        # LABEL - MSE
        self.mse = QLabel(self)
        self.mse.setText('MSE :')
        self.mse.setAlignment(Qt.AlignRight)
        self.mse.setFont(QFont('Proxima Nova', 10))
        self.mse.setGeometry(780, 310, 180, 40) 
        self.mse.setStyleSheet("color: #000000; background-color: #C4C4C4;")

        self.box5 = QLabel(self)
        self.box5.setAlignment(Qt.AlignCenter)
        self.box5.setFont(QFont('Proxima Nova', 10))
        self.box5.setGeometry(985, 300, 250, 40) 
        self.box5.setStyleSheet("color: #000000; background-color: #FFFFFF; border-radius: 10px;")

        # LABEL - PSNR
        self.psnr = QLabel(self)
        self.psnr.setText('PSNR :')
        self.psnr.setAlignment(Qt.AlignRight)
        self.psnr.setFont(QFont('Proxima Nova', 10))
        self.psnr.setGeometry(780, 360, 180, 40) 
        self.psnr.setStyleSheet("color: #000000; background-color: #C4C4C4;")

        self.box6 = QLabel(self)
        self.box6.setAlignment(Qt.AlignCenter)
        self.box6.setFont(QFont('Proxima Nova', 10))
        self.box6.setGeometry(985, 350, 250, 40) 
        self.box6.setStyleSheet("color: #000000; background-color: #FFFFFF; border-radius: 10px;")


        # Create the input area
        self.label2 = QLabel(self) 
        self.label2.setGeometry(100, 480, 580, 160) 
        self.label2.setStyleSheet("background-color:  #C4C4C4; border-radius: 20px ")       

        # INPUT - Gaussian percent
        self.gaussian_percent = QLabel(self)
        self.gaussian_percent.setText('Gaussian Percent')
        self.gaussian_percent.setAlignment(Qt.AlignCenter)
        self.gaussian_percent.setFont(QFont('Proxima Nova', 10))
        self.gaussian_percent.setGeometry(130, 515, 160, 40) 
        self.gaussian_percent.setStyleSheet("color: #000000; background-color: #C4C4C4;")

        self.gau_box = QLineEdit(self)
        self.gau_box.setText("0.4")
        self.gau_box.setAlignment(Qt.AlignCenter)
        self.gau_box.setFont(QFont('Proxima Nova', 10))
        self.gau_box.move(130, 570) 
        self.gau_box.resize(160, 50)
        self.gau_box.setStyleSheet("color: #000000; background-color: #FFFFFF; border-radius: 10px;")

        # INPUT - Quantization ratio
        self.quantization_ratio = QLabel(self)
        self.quantization_ratio.setText('Quantization Ratio')
        self.quantization_ratio.setAlignment(Qt.AlignCenter)
        self.quantization_ratio.setFont(QFont('Proxima Nova', 10))
        self.quantization_ratio.setGeometry(310, 515, 160, 40) 
        self.quantization_ratio.setStyleSheet("color: #000000; background-color: #C4C4C4;")

        self.quan_box = QLineEdit(self)
        self.quan_box.setText("4")
        self.quan_box.setAlignment(Qt.AlignCenter)
        self.quan_box.setFont(QFont('Proxima Nova', 10))
        self.quan_box.setGeometry(310, 575, 160, 50) 
        self.quan_box.setStyleSheet("color: #000000; background-color: #FFFFFF; border-radius: 10px;")

        # INPUT - Arithmetic block size
        self.arithmetic = QLabel(self)
        self.arithmetic.setText('Arithmetic Block Size')
        self.arithmetic.setAlignment(Qt.AlignCenter)
        self.arithmetic.setFont(QFont('Proxima Nova', 10))
        self.arithmetic.setGeometry(490, 515, 160, 40) 
        self.arithmetic.setStyleSheet("color: #000000; background-color: #C4C4C4;")

        self.arth_box = QLineEdit(self)
        self.arth_box.setText("4")
        self.arth_box.setAlignment(Qt.AlignCenter)
        self.arth_box.setFont(QFont('Proxima Nova', 10))
        self.arth_box.setGeometry(490, 575, 160, 50) 
        self.arth_box.setStyleSheet("color: #000000; background-color: #FFFFFF; border-radius: 10px;")

        # BUTTON - Ok
        self.button = QPushButton("OK", self)
        self.button.setGeometry(365, 650, 70, 40) 
        self.button.setStyleSheet("color: #000000; background-color: #FFFFFF; border-radius: 10px;")
        self.button.clicked.connect(self.save_result)

        # LABEL - Authors
        self.arithmetic = QLabel(self)
        self.arithmetic.setText('Authors\nBui, Duy Nam\nDuong, Thi Thuy Ngan\nDo, Tuan Anh')
        self.arithmetic.setAlignment(Qt.AlignCenter)
        self.arithmetic.setFont(QFont('Proxima Nova', 12, QFont.Bold))
        self.arithmetic.setGeometry(855, 410, 360, 100) 
        self.arithmetic.setStyleSheet("color: #000000; background-color: #C4C4C4;")

        # LABEL - State
        # Create the properties area
        self.ter = QLabel(self) 
        self.ter.setAlignment(Qt.AlignCenter)
        self.ter.setFont(QFont('Proxima Nova', 15))
        self.ter.setGeometry(855, 530, 360, 160)
        self.ter.setStyleSheet("color: #FFFFFF; background-color:  #202020; border-radius: 10px") 

    def browsefiles(self):
        # print("clicked file")
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'This PC', 'Images (*.png *.jpg)')
        file_name = fname[0]
        # print(file_name)
        self.img = cv2.imread(file_name, 0)

        h, w = self.img.shape
        bytesPerLine = w
        convertToQtFormat = QImage(self.img.data, w, h, bytesPerLine, QImage.Format_Grayscale8)
        p = convertToQtFormat.scaled(256, 256, Qt.KeepAspectRatio)

        # pixmap = QPixmap(file_name)
        self.input_image.setPixmap(QPixmap.fromImage(p))
        
        self.ter.setText("LOAD IMAGE SUCCESSFULLY!")
    
    def save_result(self):
        self.percent = float(self.gau_box.text())
        self.ratio = int(self.quan_box.text())
        self.block_size = int(self.arth_box.text())
        print("Gauss: {}, Ratio: {}, Block size: {}".format(self.percent, self.ratio, self.block_size))

    def compress_image(self):
        # put your code here
        print("Clicked compress image")
        self.ter.setText("COMPRESSING ...")
        start = time.time()
        # Compression
        self.res, self.prob, self.img_row, self.img_col, self.quan_row, self.quan_col = compressor(self.img,
                                        percent=self.percent, ratio=self.ratio, block_size=self.block_size)
        end = time.time()
        self.time_com = end - start
        np.save("encode.npy", self.res)
        self.ter.setText("COMPRESS SUCCESSFULLY!")


    def decompress_image(self):
        # put your code here
        print("Clicked decompress image")
        self.ter.setText("DECOMPRESSING ...")

        start = time.time()
        self.rec = decompressor(self.res, self.prob, self.img_row, self.img_col, self.quan_row, self.quan_col,
                                ratio=self.ratio, block_size=self.block_size)
        end = time.time()
        self.time_decom = end - start
        self.ter.setText("DECOMPRESS SUCCESSFULLY!")
        
        # Show resctruction image
        bytesPerLine = self.img_col
        convertToQtFormat = QImage(self.img.data, self.img_col, self.img_row,
                                    bytesPerLine, QImage.Format_Grayscale8)
        p = convertToQtFormat.scaled(256, 256, Qt.KeepAspectRatio)
        self.output_image.setPixmap(QPixmap.fromImage(p))

    def properties_image(self):
        # put your code here
        print("clicked properties")
        self.ter.setText("DISPLAY PROPERTIES")
        # Original volumn
        v_ori = self.img_row*self.img_col/1024
        text_ori = "{:.3f} KB".format(v_ori)
        self.box1.setText(text_ori)

        # Compress volumn
        v_com = os.path.getsize('encode.npy')/1024
        text_com = "{:.3f} KB".format(v_com)
        self.box2.setText(text_com)

        # Compress ratio
        r = v_ori/v_com
        text_ratio = "{:.3f}".format(r)
        self.box3.setText(text_ratio)

        # Compress time
        text_time = "C: {:.3f}s - D: {:.3}s".format(self.time_com, self.time_decom)
        self.box4.setText(text_time)

        # MSE
        text_mse = "{:.3f}".format(qlt.mse(self.img, self.rec))
        self.box5.setText(text_mse)
        
        # PSNR
        text_psnr = "{:.3f}".format(qlt.psnr(self.img, self.rec))
        self.box6.setText(text_psnr)
        

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()
