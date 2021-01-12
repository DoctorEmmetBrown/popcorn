import fabio
import fabio.edfimage as edf
import fabio.tifimage as tif
import os
import glob

# from PIL import Image
import numpy as np


def openImage(filename):
    filename = str(filename)
    im = fabio.open(filename)
    imarray = im.data
    return imarray


def getHeader(filename):
    im = fabio.open(filename)
    header = im.header
    return header


def saveTiff16bit(data, filename, minIm=0, maxIm=0, header=None):
    if (minIm == maxIm):
        minIm = np.amin(data)
        maxIm = np.amax(data)
    datatoStore = 65536 * (data - minIm) / (maxIm - minIm)
    datatoStore[datatoStore > 65635] = 65535
    datatoStore[datatoStore < 0] = 0
    datatoStore = np.asarray(datatoStore, np.uint16)

    if (header != None):
        tif.TifImage(data=datatoStore, header=header).write(filename)
    else:
        tif.TifImage(data=datatoStore).write(filename)


def openSeq(filenames):
    if len(filenames) > 0:
        data = openImage(str(filenames[0]))
        height, width = data.shape
        toReturn = np.zeros((len(filenames), height, width), dtype=np.float32)
        i = 0
        for file in filenames:
            data = openImage(str(file))
            toReturn[i, :, :] = data
            i += 1
        return toReturn
    raise Exception('spytlabIOError')


def makeDarkMean(Darkfiedls):
    nbslices, height, width = Darkfiedls.shape
    meanSlice = np.mean(Darkfiedls, axis=0)
    print('-----------------------  mean Dark calculation done ------------------------- ')
    OutputFileName = '/Users/helene/PycharmProjects/spytlab/meanDarkTest.edf'
    outputEdf = edf.EdfFile(OutputFileName, access='wb+')
    outputEdf.WriteImage({}, meanSlice)
    return meanSlice


def saveEdf(data, filename):
    #    print(filename)
    dataToStore = data.astype(np.float32)
    edf.EdfImage(data=dataToStore).write(filename)


def saveEdfInt(data, filename):
    #    print(filename)
    dataToStore = data.astype(np.int)
    edf.EdfImage(data=dataToStore).write(filename)


def save3D_Edf(data, filename):
    nbslices, height, width = data.shape
    for i in range(nbslices):
        textSlice = '%4.4d' % i
        dataToSave = data[i, :, :]
        filenameSlice = filename + textSlice + '.edf'
        saveEdf(dataToSave, filenameSlice)
