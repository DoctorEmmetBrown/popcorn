import os
import math
import numpy
from popcornIO import openImage, saveTiff16bit

def pad_with(vector, pad_width, iaxis, kwargs):
    """
    adds rows and columns to an image : its size depends on pad_width value, the value of pixels depends on padder value
    :param vector: self
    :param pad_width: nb of pixels to add
    :param iaxis: self
    :param kwargs: 'padder' = value of the added pixels
    :return: self
    """
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def paddingImage(image, paddingSize):
    """
    determines the distribution of pixels to add on top/bottom/righ/left calls the pad_with function
    :param image: input image
    :param paddingSize: nb of pixels to add (in both x and y directions)
    :return: padded image
    """
    if paddingSize%2 != 0:
        return numpy.pad(image, (int(paddingSize/2), math.ceil(paddingSize/2)), pad_with)
    else:
        return numpy.pad(image, int(paddingSize/2), pad_with)


def multiThreadingConversion(listOfArgs):
    """
    transforms a list of args into 5 args before calling the conversion function
    :param listOfArgs: list of args
    :return: None
    """
    conversionFromListOfFiles(listOfArgs[0], listOfArgs[1], listOfArgs[2], listOfArgs[3], listOfArgs[4])



def conversionFromListOfFiles(listOfFiles, outputFolder, minValue=0., maxValue=1., paddingSize=0):
    """
    opens the files from the input list of files, converts them in uint16 and saves them in output folder as .tif files
    :param listOfFiles: input list of files
    :param outputFolder: output folder
    :param minValue: minimum value in the image
    :param maxValue: maximum value in the image
    :return: None
    """
    for fileName in listOfFiles:
        baseName = os.path.basename(fileName).split(".")[0]
        data = openImage(fileName)
        if paddingSize != 0 :
            data = paddingImage(data, paddingSize)
        saveTiff16bit(data, outputFolder + '/'+baseName + ".tif", minValue, maxValue)
    print("16 bit conversion done.")

if __name__ == "__main__" :
    print("Hello")
    number = 0
    import fabio

