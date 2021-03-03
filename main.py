import os, sys, getopt
import glob

import math
import numpy as np
import random

import fabio
import multiprocessing

from popcornIO import myMkdir, openImage
from SixteenBitConverter import conversionFromListOfFiles, multiThreadingConversion
from Stitching import stitchFolders


def lookForMinMaxVal(listOfFolders, percentile):
    """
    looks for min and max value of all folders
    :param listOfFolders: list of all input folders
    :param percentile: percentile of pixel values we get rid of
    :return: min and max values
    """
    minMaxList = []
    for inputFolder in listOfFolders:
        histo = fabio.open(glob.glob(inputFolder+'/histogram*')[0])
        maxVal = float(histo.header["MaxVal"])
        minVal = float(histo.header["MinVal"])
        pas = (maxVal - minVal) / histo.data.size

        sum_histo = 0
        index = 1
        histo.data = histo.data / np.sum(histo.data)
        while sum_histo < percentile:
            sum_histo = np.sum(histo.data[0:index])
            index += 1
        finalMinVal = minVal + index * pas

        index = 2
        while sum_histo < percentile:
            sum_histo = np.sum(histo.data[-index:-1])
            index += 1

        finalMaxVal = maxVal - index * pas
        minMaxList.append([finalMinVal, finalMaxVal])

    return min(x[0] for x in minMaxList), max(x[1] for x in minMaxList)

def usage():
    print("""usage: python main.py [-i|-o|-r|-s|-c|-m|-M|-t|-z|-f]
            -i, --ifolder:    input folder
            -o, --ofolder:    output folder
            -r, --radix:      regular expression 
            -s, --speckle:    speckle?
            -c, --conversion: 16bit conversion?
            -m, --min:,       forced min value
            -M, --max::       forced max value
            -t, --threading:  multi threading?
            -z, --deltaz:     delta Z value (nb of slices)
            -f, --flip:       need to flip each floor [0: no, 1: yes] ?""")


if __name__ == "__main__" :
    inputFolder      = '/data/visitor/md1217/id17/'
    mainOutputFolder = '/data       /visitor/md1217/id17/voltif/'
    radix            = 'HA750_6um_42kev_SP_023_PM'

    speckleDone          = False
    sixteenBitConversion = True
    manualMinMax16bit    = False
    minIm16Bit           = -0.02
    maxIm16Bit           = 0.8
    multiThreading       = True
    deltaZ               = 234
    flipUD               = 0

    if sys.argv[1:]:
        try:
            opts, args = getopt.getopt(sys.argv[1:], "hi:o:r:s:c:m:M:t:z:f:", ["help", "ifolder=", "ofolder=", "radix=",
                                                                               "conversion=", "speckle=", "min=", "max=",
                                                                               "threading=", "deltaz=", "flip="])
        except getopt.GetoptError as err:
            print(err)
            usage()
            sys.exit(2)

        for opt, arg in opts:
            if opt in ("-h", "--help"):
                usage()
                sys.exit()
            elif opt in ("-i", "--ifolder"):
                inputFolder = arg
            elif opt in("-o", "--ofolder"):
                mainOutputFolder = arg
            elif opt in("-r", "--radix"):
                radix = arg
            elif opt in("-s", "--speckle"):
                speckleDone = arg
            elif opt in("-c", "--conversion"):
                sixteenBitConversion = arg
            elif opt in("-m", "--min"):
                minIm16Bit = arg
                manualMinMax16bit = True
            elif opt in("-M", "--max"):
                maxIm16Bit = arg
                manualMinMax16bit = True
            elif opt in("-t", "--threading"):
                multiThreading = arg
            elif opt in("-z", "--deltaz"):
                deltaZ = arg
            elif opt in("-f", "--flip"):
                flipUD = arg

    # SPECKLE : Parsing all the reconstructed folders and putting them in a list
    if speckleDone:
        reconstructedFolders = glob.glob(inputFolder + '/volfloat/*' + radix + '*Propag_pag')
    else:
        reconstructedFolders = glob.glob(inputFolder + '/volfloat/*' + radix + '*pag')
    reconstructedFolders.sort()

    # MIN-MAX : if not manual, parsing all floors histograms to determine min and max
    if not manualMinMax16bit:
        minIm16Bit, maxIm16Bit = lookForMinMaxVal(reconstructedFolders, 0.005)

    # PADDING : Checking if all images have the same size : Yes = we don't care, No = We pad (all image same size)
    imageWidthList = []
    for inputFolder in reconstructedFolders:
        randomFilename = random.choice(glob.glob(inputFolder+'/*.tif')+glob.glob(inputFolder+'/*.edf')) # We pick a random image
        randomImage = openImage(randomFilename)
        imageWidthList.append(randomImage.shape[1])
    maxImageWidth = max(imageWidthList)

    # CONVERSION : opening all files before converting them into uint16 and saving them as .tif files
    listOf16bitFolder = []
    folderNb = 1
    for inputFolder in reconstructedFolders:
        print("Starting 16Bit conversion for folder ", str(folderNb) + "/" + str(len(reconstructedFolders)))
        folderNb += 1
        baseName = os.path.basename(inputFolder)
        imageFiles = glob.glob(inputFolder + '/*.tif') + glob.glob(inputFolder + '/*.edf')
        outputFolder = mainOutputFolder + baseName + '/'

        listOf16bitFolder.append(outputFolder)
        myMkdir(outputFolder)
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
                subListsOfFiles = [imageFiles[k:k + sizeOfSubList] for k in range(0, len(imageFiles), sizeOfSubList)]
                multiThreadingArgs = []
                for subList in subListsOfFiles:
                    multiThreadingArgs.append([subList, outputFolder, minIm16Bit, maxIm16Bit, paddingSize])
                pool.map(multiThreadingConversion, multiThreadingArgs)
            else:
                conversionFromListOfFiles(imageFiles, outputFolder, minIm16Bit, maxIm16Bit, paddingSize)

    outputRadixFolder = mainOutputFolder + '/' + radix + '_' + str("%.2f" % minIm16Bit) + '_' + str(
        "%.2f" % maxIm16Bit)
    myMkdir(outputRadixFolder)

    stitchFolders(listOf16bitFolder, outputRadixFolder, deltaZ, lookForBestSlice=True, copyMode=1, securityBandSize=30, overlapMode=0,
                  bandAverageSize=0, flipUD=flipUD)