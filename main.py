import os
import glob

import math
import numpy
import random

import multiprocessing

from popcornIO import openImage, myMkdir
from SixteenBitConverter import conversionFromListOfFiles, multiThreadingConversion
from Stitching import stitchFolders


if __name__ == "__main__" :
    mainFolder       = '/Users/embrun/TestStitching/'
    mainOutputFolder = '/Users/embrun/TestStitching/voltif/'
    radix            = '11mPropagation_23um_33kev_026_CE_CT_GW_13'

    speckleDone       = False
    manualMinMax16bit = False
    multiThreading    = True
    deltaZ            = 234

    # SPECKLE : Parsing all the reconstructed folders and putting them in a list
    if speckleDone:
        reconstructedFolders = glob.glob(mainFolder + '/volfloat/*' + radix + '*Propag_pag')
    else:
        reconstructedFolders = glob.glob(mainFolder+'/volfloat/*'+radix+'*pag')
    reconstructedFolders.sort()

    # MIN-MAX : if not manual, parsing all floors histograms to determine min and max
    if manualMinMax16bit:
        minIm16Bit = 0.
        maxIm16Bit = 1.
    else:
        minMaxList = []
        for inputFolder in reconstructedFolders:
            with open(glob.glob(inputFolder+'/histogram*')[0]) as f:
                for line in f:
                    if "MaxVal" in line:
                        text1, text2, maxVal, text3 = line.split()
                        maximum_value = float(maxVal)
                    if "MinVal" in line:
                        text1, text2, minVal, text3 = line.split()
                        minimum_value = float(minVal)
                        break
                minMaxList.append([minimum_value, maximum_value])
        minIm16Bit = min(x[0] for x in minMaxList)
        maxIm16Bit = max(x[1] for x in minMaxList)

    # PADDING : Checking if all images have the same size : Yes = we don't care, No = We pad (all image same size)
    imageWidthList = []
    for inputFolder in reconstructedFolders:
        randomFilename = random.choice(glob.glob(inputFolder+'/*.tif')+glob.glob(inputFolder+'/*.edf')) # We pick a random image
        randomImage = openImage(randomFilename)
        imageWidthList.append(randomImage.shape[1])
    maxImageWidth = max(imageWidthList)

    # CONVERSION : opening all files before converting them into uint16 and saving them as .tif files
    listOf16bitFolder=[]
    for inputFolder in reconstructedFolders:
        print("Converting folder :", inputFolder)

        baseName     = os.path.basename(inputFolder)
        imageFiles   = glob.glob(inputFolder+'/*.tif')+glob.glob(inputFolder+'/*.edf')
        outputFolder = mainOutputFolder+baseName+'/'

        listOf16bitFolder.append(outputFolder)
        myMkdir(outputFolder)

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
            subListsOfFiles = [imageFiles[k:k + sizeOfSubList] for k in range(0, len(imageFiles), sizeOfSubList)]
            multiThreadingArgs = []
            for subList in subListsOfFiles:
                multiThreadingArgs.append([subList, outputFolder, minIm16Bit, maxIm16Bit, paddingSize])
            pool.map(multiThreadingConversion, multiThreadingArgs)
        else:
            conversionFromListOfFiles(imageFiles, outputFolder, minIm16Bit, maxIm16Bit, paddingSize)

    outputRadixFolder = mainOutputFolder + '/' + radix + '_' + str("%.2f" % minIm16Bit) + '_' + str("%.2f" % maxIm16Bit)
    myMkdir(outputRadixFolder)

    stitchFolders(listOf16bitFolder, outputRadixFolder, deltaZ, lookForBestSlice=True, copyMode=1, securityBandSize=30, overlapMode=0,
                  bandAverageSize=0, flipUD=0)
