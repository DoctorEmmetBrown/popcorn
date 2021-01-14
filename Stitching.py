import numpy
import glob
#from scipy.signal import correlate
from popcornIO import openSeq,save3D_Edf






def lookForMaximumCorrelation(imageA,imageB):
    """
    Function to look for the maximum correlated slice between two different volumes
    Preparation for stitching 2 sets of images
    The computation is only performed with the slice in the middle of imageA on the entire imageB volume
    :param imageA:3D numpy array
    :param imageB:3D numpy array
    :return: the slice number with highest zero normalized cross correlation.

    """
    nbSlicesA,widthA,heightA=imageA.shape
    nbSlicesB, widthB, heightB = imageB.shape
    width=max(widthA,widthB)
    height=max(heightA,heightB)

    middleSlice=int(nbSlicesA/2)
    imageToMultiply = numpy.copy(imageA[middleSlice, :, :].squeeze())
    imageToMultiply = imageToMultiply - numpy.mean(imageToMultiply)

    tmpB=numpy.copy(imageB)
    stdMul = numpy.std(imageToMultiply)
    corr=numpy.zeros(nbSlicesB)
    for slice in range(0, nbSlicesB):
        tmpB[slice, :, :] = tmpB[slice, :, :] - numpy.mean(tmpB[slice, :, :])

    imMultiplied = imageToMultiply * tmpB

    for slice in range(0,nbSlicesB):
        #tmpB[slice,:,:]=tmpB[slice,:,:]-numpy.mean(tmpB[slice,:,:])
        stdB = numpy.std(tmpB[slice, :, :])
        sumMultiplication = numpy.sum(imMultiplied[slice, :, :])
        normcrosscorr = sumMultiplication / (stdMul * stdB)
        normcrosscorr /= (width * height)
        corr[slice]=normcrosscorr
        print(normcrosscorr)

    maxCorSlice=numpy.argmax(corr)
    print(maxCorSlice)

    return (maxCorSlice)


def lookForMaximumCorrelationBand(imageA,imageB,bandSize):
    """
    Function to look for the maximum correlated slice between two different volumes
    Preparation for stitching 2 sets of images
    The computation is performed for every slices in a band of bandSize centered around the imageA middle slice
    :param imageA:3D numpy array
    :param imageB:3D numpy array
    :param bandSize: Number of slice the zero normalized cross correlation is made on
    :return: the median value of all slices number with highest zero normalized cross correlation.
    """
    nbSlicesA,widthA,heightA=imageA.shape
    nbSlicesB, widthB, heightB = imageB.shape

    width=max(widthA,widthB)
    height=max(heightA,heightB)
    middleSlice = int(nbSlicesA / 2)
    tmpB = numpy.copy(imageB)

    #Preparation for normalized cross correlation
    for slice in range(0, nbSlicesB):
        tmpB[slice, :, :] = tmpB[slice, :, :] - numpy.mean(tmpB[slice, :, :])

    argMaxCoors=numpy.zeros(bandSize)
    for i in range(int(-bandSize/2),(int(bandSize/2))):
        imageToMultiply = numpy.copy(imageA[middleSlice+i, :, :].squeeze())
        imageToMultiply = imageToMultiply - numpy.mean(imageToMultiply)
        stdMul = numpy.std(imageToMultiply)
        corr=numpy.zeros(nbSlicesB)
        imMultiplied = imageToMultiply * tmpB

        for slice in range(0,nbSlicesB):
            stdB = numpy.std(tmpB[slice, :, :])
            sumMultiplication = numpy.sum(imMultiplied[slice, :, :])
            normcrosscorr = sumMultiplication / (stdMul * stdB)
            normcrosscorr /= (width * height)
            corr[slice]=normcrosscorr

        maxCorSlice=numpy.argmax(corr)-i
        argMaxCoors[i] = maxCorSlice
        print('maxCorSlice found for slice:'+str(i)+' is :'+str(maxCorSlice))
    medianValue=numpy.median(argMaxCoors)
    return (medianValue)







if __name__ == "__main__" :
    print("Hello")
    imageAFolder = '/Users/embrun/TestStitching/Image2'
    imageBFolder = '/Users/embrun/TestStitching/Image3'
    imageAFiles = glob.glob(imageAFolder + '/*.edf')
    imageBFiles = glob.glob(imageAFolder + '/*.edf')
    imageA = openSeq(imageAFiles)
    imageB = openSeq(imageBFiles)
    lookForMaximumCorrelation(imageA,imageB)
    lookForMaximumCorrelationBand(imageA,imageB,10)





