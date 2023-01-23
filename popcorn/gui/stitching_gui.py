import threading
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import sys
from qtrangeslider import QRangeSlider
from PyQt6.QtCharts import *
import os
import sys
import getopt
import glob

import math
import numpy as np
import random
import shutil
import fabio
import multiprocessing

from pathlib import Path
path_root = str(Path(__file__).parents[1])
if path_root not in sys.path:
    sys.path.append(path_root)

from input_output import open_image, open_sequence, save_tif_image, save_edf_image
from sixteen_bit_converter import conversion_from_list_of_files, multi_threading_conversion
from stitching import stitch_multiple_folders_into_one
from main import look_for_min_max_val

"""
###############
NON FONCTIONNEL
###############
"""


class Stitching(QWidget):
    """
    class which contains the different elements needed for stitching
    """
    def __init__(self, father):
        """
        init and create button
        Args:
            father: QWigdet (in our case it's QMainWindow from GUI_popcorn
        """
        super().__init__()
        self.father = father
        self.layoutStitching = QGridLayout()  # Son layout en grille
        self.setLayout(self.layoutStitching)

        self.open_stitching = QPushButton("Open Stitching Window")
        self.open_stitching.clicked.connect(self.stitching_para)
        self.layoutStitching.addWidget(self.open_stitching, 0, 0)

        self.start_stitching = QPushButton("Start Stitching")
        self.start_stitching.clicked.connect(self.stitching_start)
        self.layoutStitching.addWidget(self.start_stitching, 1, 0)

        self.window = Stitching_window(self)

        self.preview_stitching = QPushButton("preview")
        self.preview_stitching.clicked.connect(self.window.previewResult)
        self.layoutStitching.addWidget(self.preview_stitching, 2, 0)

    def stitching_para(self):
        """
        hide then show the stitching window
        Returns: void

        """
        self.window.hide()
        self.window.show()

    def stitching_start(self):
        """
        Create a thread and run it
        Returns: void

        """
        thread = start_thread(self)
        thread.run()
    def stitching_start_exec(self):
        """
        read parameter from the gui and do stitching
        Returns:
            void
        """
        inputFolder = self.window.liste_folder
        mainOutputFolder = self.window.display_output_folder.toPlainText()

        radix = ''

        speckleDone = self.window.speckleDone_value.isChecked()
        sixteenBitConversion = self.window.sixteenBitConversion_value.isChecked()
        manualMinMax16bit = self.window.sixteenBitConversion_value.isChecked()
        if manualMinMax16bit:
            minIm16Bit = float(self.window.minIm16Bit_value.text())
            maxIm16Bit = float(self.window.maxIm16Bit_value.text())
        else:
            minIm16Bit = 0.0
            maxIm16Bit = 0.0
        multiThreading = self.window.multiThreading_value.isChecked()
        deltaZ = int(self.window.deltaz_value.text())
        flipUD = self.window.flip_value.isChecked()
        security_band = self.window.security_band_size_value.text()
        if security_band == "":
            security_band = 20
        else:
            security_band = int(security_band)
        # SPECKLE : Parsing all the reconstructed folders and putting them in a list

        reconstructedFolders = inputFolder.copy()
        reconstructedFolders.sort()
        print(reconstructedFolders)
        # MIN-MAX : if not manual, parsing all floors histograms to determine min and max
        if not manualMinMax16bit and sixteenBitConversion:
            minIm16Bit, maxIm16Bit = look_for_min_max_val(reconstructedFolders, 0.005)

        # PADDING : Checking if all images have the same size : Yes = we don't care, No = We pad (all image same size)
        imageWidthList = []
        for inputFolder in reconstructedFolders:
            # We pick a random image
            randomFilename = random.choice(glob.glob(inputFolder + '/*.tif*') + glob.glob(inputFolder + '/*.edf'))
            randomImage = open_image(randomFilename)
            imageWidthList.append(randomImage.shape[1])
        maxImageWidth = max(imageWidthList)

        # CONVERSION : opening all files before converting them into uint16 and saving them as .tif files
        listOf16bitFolder = []
        folderNb = 1
        for inputFolder in reconstructedFolders:
            print("Starting 16Bit conversion for folder ", str(folderNb) + "/" + str(len(reconstructedFolders)))
            folderNb += 1
            baseName = os.path.basename(inputFolder)
            imageFiles = glob.glob(inputFolder + '/*.tif*') + glob.glob(inputFolder + '/*.edf')
            outputFolder = mainOutputFolder + baseName + '/'

            listOf16bitFolder.append(outputFolder)

            if sixteenBitConversion:
                currentImageWidth = imageWidthList.pop(0)
                paddingSize = maxImageWidth - currentImageWidth
                if multiThreading:
                    numCores = multiprocessing.cpu_count()
                    print("Number of cores on cpu :", numCores)
                    pool = multiprocessing.Pool(processes=numCores)

                    if len(imageFiles) > numCores:
                        sizeOfSubList = math.ceil(len(imageFiles) / numCores)
                    else:
                        sizeOfSubList = numCores
                    subListsOfFiles = [imageFiles[k:k + sizeOfSubList] for k in
                                       range(0, len(imageFiles), sizeOfSubList)]
                    multiThreadingArgs = []
                    for subList in subListsOfFiles:
                        multiThreadingArgs.append([subList, outputFolder, minIm16Bit, maxIm16Bit, paddingSize])
                    pool.map(multi_threading_conversion, multiThreadingArgs)
                else:
                    conversion_from_list_of_files(imageFiles, outputFolder, minIm16Bit, maxIm16Bit, paddingSize)

        # outputRadixFolder = mainOutputFolder + '/' + radix + '_' + str("%.2f" % minIm16Bit) + '_' + str(
        #   "%.2f" % maxIm16Bit)
        print(reconstructedFolders,mainOutputFolder,deltaZ,sep=" | ")

        stitch_multiple_folders_into_one(reconstructedFolders, mainOutputFolder, deltaZ, look_for_best_slice=True,
                                         copy_mode=1, security_band_size=security_band, overlap_mode=0,
                                         band_average_size=0,
                                         flip=flipUD)
        im_np = open_sequence(mainOutputFolder)
        print(im_np.shape)
        mid_ind=int(im_np.shape[1]/2)
        im_np = im_np[:, mid_ind , :]
        mini = im_np.min()
        maxi = im_np.max()
        if mini != maxi:
            im_np = im_np - mini

            im_np = im_np * 65535 / (maxi - mini)

            im_np[im_np > 65535] = 65535

        im_np = im_np.astype("uint16")
        qimage = QImage(
            im_np.data,
            im_np.shape[1],
            im_np.shape[0],
            im_np.strides[0],
            QImage.Format.Format_Grayscale16,
        )
        qpixmap = QPixmap.fromImage(qimage)
        self.window.previewImage.setPixmap((qpixmap.scaled(800, 800)))
        print("end")


class Stitching_window(QWidget):
    """
    Class for the stitching window
    """
    def __init__(self, father):
        """
        init and create button
        Args:
            father: class Stitching
        """
        super().__init__()
        self.father = father
        self.layoutStitching = QGridLayout()  # Son layout en grille
        self.setLayout(self.layoutStitching)
        self.liste_folder = []

        self.load_folder = QPushButton("Load Folder")
        self.load_folder.clicked.connect(self.open_folder)
        self.layoutStitching.addWidget(self.load_folder, 0, 0)

        self.display_list_folder = QTextEdit()
        self.layoutStitching.addWidget(self.display_list_folder, 1, 0)

        self.output_folder = QPushButton("Output Folder")
        self.output_folder.clicked.connect(self.choose_output_folder)
        self.layoutStitching.addWidget(self.output_folder, 0, 1)

        self.display_output_folder = QTextEdit()
        self.layoutStitching.addWidget(self.display_output_folder, 1, 1)

        self.speckleDone_text = QLabel("speckleDone")
        self.layoutStitching.addWidget(self.speckleDone_text, 2, 0)

        self.speckleDone_value = QCheckBox()
        self.layoutStitching.addWidget(self.speckleDone_value, 2, 1)

        self.sixteenBitConversion_text = QLabel("sixteenBitConversion")
        self.layoutStitching.addWidget(self.sixteenBitConversion_text, 3, 0)

        self.sixteenBitConversion_value = QCheckBox()
        self.layoutStitching.addWidget(self.sixteenBitConversion_value, 3, 1)

        self.minIm16Bit_text = QLabel("minIm16Bit")
        self.layoutStitching.addWidget(self.minIm16Bit_text, 4, 0)

        self.minIm16Bit_value = QLineEdit("")
        self.layoutStitching.addWidget(self.minIm16Bit_value, 4, 1)

        self.maxIm16Bit_text = QLabel("maxIm16Bit")
        self.layoutStitching.addWidget(self.maxIm16Bit_text, 5, 0)

        self.maxIm16Bit_value = QLineEdit("")
        self.layoutStitching.addWidget(self.maxIm16Bit_value, 5, 1)

        self.sixteenBitConversion_value.stateChanged.connect(self.display_min_max)
        self.sixteenBitConversion_value.setChecked(True)

        self.multiThreading_text = QLabel("multiThreading")
        self.layoutStitching.addWidget(self.multiThreading_text, 6, 0)

        self.multiThreading_value = QCheckBox()
        self.layoutStitching.addWidget(self.multiThreading_value, 6, 1)

        self.deltaz_text = QLabel("delta_z")
        self.layoutStitching.addWidget(self.deltaz_text, 7, 0)

        self.deltaz_value = QLineEdit("")
        self.layoutStitching.addWidget(self.deltaz_value, 7, 1)

        self.flip_text = QLabel("flip")
        self.layoutStitching.addWidget(self.flip_text, 8, 0)

        self.flip_value = QCheckBox()
        self.layoutStitching.addWidget(self.flip_value, 8, 1)

        self.security_band_size_text = QLabel("security_band_size")
        self.layoutStitching.addWidget(self.security_band_size_text, 9, 0)

        self.security_band_size_value = QLineEdit("")
        self.layoutStitching.addWidget(self.security_band_size_value, 9, 1)

        self.reset_text = QPushButton("reset")
        self.reset_text.pressed.connect(self.reset)
        self.layoutStitching.addWidget(self.reset_text, 10, 0, 1, 2)

        self.start_text = QPushButton("start")
        self.start_text.clicked.connect(self.father.stitching_start)
        self.layoutStitching.addWidget(self.start_text, 11, 0, 1, 2)

        self.preview_text = QPushButton("Preview")
        self.preview_text.clicked.connect(self.previewResult)
        self.layoutStitching.addWidget(self.preview_text, 12, 0, 1, 2)

        self.previewImage = QLabel()
        self.layoutStitching.addWidget(self.previewImage, 0, 2, 12, 2)

        self.already_load=False
    def reset(self):
        """
        reset all parameter
        Returns:
            void
        """
        self.liste_folder = []
        self.display_list_folder.setText("")

        self.display_output_folder.setText("")

        self.speckleDone_value.setChecked(False)

        self.sixteenBitConversion_value.setChecked(True)

        self.minIm16Bit_value.setText("")

        self.maxIm16Bit_value.setText("")

        self.multiThreading_value.setChecked(False)

        self.deltaz_value.setText("")

        self.flip_value.setChecked(False)

        self.previewImage = QLabel()

    def open_folder(self):
        """
        choose a file and add it in self.liste_folder and self.display_list_folder
        Then change self.already_load to False
        Returns: void
        """
        path = QFileDialog.getExistingDirectory(self, str("Choose Exp Directory"))
        if path != "":
            self.liste_folder.append(path + '/')
        self.display_list_folder.setText("\n".join(self.liste_folder))
        self.already_load=False

    def choose_output_folder(self):
        """
        choose a file and add it in self.display_output_folder
        Returns: void
        """
        path = QFileDialog.getExistingDirectory(self, str("Choose Output Directory"))
        if path != "":
            self.display_output_folder.setText(path + '/')

    def display_min_max(self):
        """
        hide or show min/max depending of if sixteenBitConversion_value is checked or not
        Returns:

        """
        if self.sixteenBitConversion_value.isChecked():
            self.maxIm16Bit_text.show()
            self.minIm16Bit_text.show()
            self.maxIm16Bit_value.show()
            self.minIm16Bit_value.show()
        else:
            self.maxIm16Bit_text.hide()
            self.minIm16Bit_text.hide()
            self.maxIm16Bit_value.hide()
            self.minIm16Bit_value.hide()

    def previewResult(self):
        """
        Copy a slice of a file in the input on Image Stitching and run stitching. Then show the result on the window.
        If 2 previewResult are call with the same input copy are not do.

        Returns:
            void
        """

        list_file_preview = []
        path = os.getcwd()
        path = path + "/Image Stitching/"
        print("Le répertoire courant est : " + path)

        if not self.already_load:

            for namefile in glob.glob(path + "*"):
                if namefile != path:
                    shutil.rmtree(namefile)


            for i in range(len(self.liste_folder)):
                file=open_sequence(self.liste_folder[i])
                directory_name=self.liste_folder[i].split("/")[-2]
                file_name=directory_name.split("__")[0]
                mid_ind = int(file.shape[1] / 2)
                for j in range(file.shape[0]):
                    if not os.path.exists(path + "\\before\\"):
                        os.makedirs(path + "\\before\\")
                    if len(glob.glob(self.liste_folder[i] + "*tif*")) > 0:
                        save_tif_image(file[j][[mid_ind],:], path + "\\before\\" + directory_name + "\\" + file_name + "__" + str(i+1) + '{:04d}'.format(j))
                    if len(glob.glob(self.liste_folder[i] + "*edf*")) > 0:
                        save_edf_image(file[j][[mid_ind],:], path  + "/" + str(i) + "/" + str(j))

        deltaZ = int(self.deltaz_value.text())
        flipUD = self.flip_value.isChecked()
        try:
            shutil.rmtree(path + "after/")
        except:
            pass
        try:
            os.makedirs(path + "after/")
        except:
            pass
        input=glob.glob(path + "before/*")
        input.sort()
        print(input,path + "after/",deltaZ,sep=" | ")
        stitch_multiple_folders_into_one(input, path + "after/", deltaZ,
                                         look_for_best_slice=True,
                                         copy_mode=1, security_band_size=20, overlap_mode=0, band_average_size=0,
                                         flip=flipUD)
        im_np = open_sequence(path + "after/")
        print(im_np.shape)
        im_np = im_np[:, 0, :]
        mini = im_np.min()
        maxi = im_np.max()
        if mini != maxi:
            im_np = im_np - mini

            im_np = im_np * 65535 / (maxi - mini)

            im_np[im_np > 65535] = 65535

        im_np = im_np.astype("uint16")
        qimage = QImage(
            im_np.data,
            im_np.shape[1],
            im_np.shape[0],
            im_np.strides[0],
            QImage.Format.Format_Grayscale16,
        )
        qpixmap = QPixmap.fromImage(qimage)
        self.previewImage.setPixmap((qpixmap.scaled(800, 800)))
        print("end")
        self.already_load=True

class start_thread(QThread):
    """
    Thread that run stitching
    """
    def __init__(self,gui):
        super().__init__()
        self.gui=gui

    def run(self):
        self.gui.stitching_start_exec()
