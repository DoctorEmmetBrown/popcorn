import os
import sys
import getopt
import glob

import math
import numpy as np
import random

import fabio
import multiprocessing

from popcorn.input_output import open_image
from popcorn.sixteen_bit_converter import conversion_from_list_of_files, multi_threading_conversion
from popcorn.stitching import stitch_multiple_folders_into_one


def look_for_min_max_val(list_of_folders, percentile):
    """looks for min and max value of all folders

    Args:
        list_of_folders (str): list of all input folders
        percentile (float):  percentile of pixel values we get rid of

    Returns:
        (float, float): min and max values
    """
    min_max_list = []
    # We parse every folder to retrieve min and max values of all floors
    for input_folder in list_of_folders:
        # We open the histogram characteristics file as a dictionary
        histogram_values = fabio.open(glob.glob(input_folder + '/histogram*')[0])
        # We retrieve histogram's min and max values
        max_val = float(histogram_values.header["MaxVal"])
        min_val = float(histogram_values.header["MinVal"])

        # The step corresponds to the histogram's full range divided by the number of bins
        step = (max_val - min_val) / histogram_values.data.size

        # We're ignoring all values between 0% and percentile%
        sum_histogram = 0
        index = 1
        histogram_values.data = histogram_values.data / np.sum(histogram_values.data)
        while sum_histogram < percentile:
            sum_histogram = np.sum(histogram_values.data[0:index])
            index += 1
        final_min_val = min_val + index * step

        # We're also ignoring all values between 100% and 100-percentile%
        index = 2
        while sum_histogram < percentile:
            sum_histogram = np.sum(histogram_values.data[-index:-1])
            index += 1

        final_max_val = max_val - index * step
        min_max_list.append([final_min_val, final_max_val])

    # We return the min and max value among all floors
    return min(x[0] for x in min_max_list), max(x[1] for x in min_max_list)


def usage():
    """prints -help

    Returns:
        None
    """
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


if __name__ == "__main__":
    inputFolder = '/data/visitor/md1217/id17/volfloat/'
    mainOutputFolder = '/data/visitor/md1217/id17/voltif/'
    radix = 'HA750_6um_42kev_SP_023_PM'

    speckleDone = False
    sixteenBitConversion = True
    manualMinMax16bit = False
    minIm16Bit = -0.02
    maxIm16Bit = 0.8
    multiThreading = True
    deltaZ = 234
    flipUD = False

    if sys.argv[1:]:
        try:
            opts, args = getopt.getopt(sys.argv[1:], "hi:o:r:s:c:m:M:t:z:f:", ["help", "ifolder=", "ofolder=", "radix=",
                                                                               "conversion=", "speckle=", "min=",
                                                                               "max=", "threading=", "deltaz=",
                                                                               "flip="])
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
            elif opt in ("-o", "--ofolder"):
                mainOutputFolder = arg
            elif opt in ("-r", "--radix"):
                radix = arg
            elif opt in ("-s", "--speckle"):
                speckleDone = arg
            elif opt in ("-c", "--conversion"):
                sixteenBitConversion = arg
            elif opt in ("-m", "--min"):
                minIm16Bit = arg
                manualMinMax16bit = True
            elif opt in ("-M", "--max"):
                maxIm16Bit = arg
                manualMinMax16bit = True
            elif opt in ("-t", "--threading"):
                multiThreading = arg
            elif opt in ("-z", "--deltaz"):
                deltaZ = arg
            elif opt in ("-f", "--flip"):
                flipUD = arg

    # SPECKLE : Parsing all the reconstructed folders and putting them in a list
    if speckleDone:
        reconstructedFolders = glob.glob(inputFolder + '/*' + radix + '*Propag_pag')
    else:
        reconstructedFolders = glob.glob(inputFolder + '/*' + radix + '*pag')
    reconstructedFolders.sort()

    # MIN-MAX : if not manual, parsing all floors histograms to determine min and max
    if not manualMinMax16bit and sixteenBitConversion:
        minIm16Bit, maxIm16Bit = look_for_min_max_val(reconstructedFolders, 0.005)

    # PADDING : Checking if all images have the same size : Yes = we don't care, No = We pad (all image same size)
    imageWidthList = []
    for inputFolder in reconstructedFolders:
        # We pick a random image
        randomFilename = random.choice(glob.glob(inputFolder+'/*.tif*')+glob.glob(inputFolder+'/*.edf'))
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
                subListsOfFiles = [imageFiles[k:k + sizeOfSubList] for k in range(0, len(imageFiles), sizeOfSubList)]
                multiThreadingArgs = []
                for subList in subListsOfFiles:
                    multiThreadingArgs.append([subList, outputFolder, minIm16Bit, maxIm16Bit, paddingSize])
                pool.map(multi_threading_conversion, multiThreadingArgs)
            else:
                conversion_from_list_of_files(imageFiles, outputFolder, minIm16Bit, maxIm16Bit, paddingSize)

    outputRadixFolder = mainOutputFolder + '/' + radix + '_' + str("%.2f" % minIm16Bit) + '_' + str(
        "%.2f" % maxIm16Bit)

    stitch_multiple_folders_into_one(listOf16bitFolder, outputRadixFolder, deltaZ, look_for_best_slice=True,
                                     copy_mode=1, security_band_size=30, overlap_mode=0, band_average_size=0,
                                     flip=flipUD)
