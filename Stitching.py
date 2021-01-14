import numpy
import glob
#from scipy.signal import correlate
from popcornIO import openSeq,save3D_Edf,saveTif
import os
import shutil



def stitchFolders(listOfFolders,outputFolderName,deltaZ,copyMode=0,securityBandSize=10,overlapMode=0,bandAverageSize=0,flipUD=0):
    """
    Function that stitches different folders into a unique one
    The first and last folders are treated differently (correlation with other folders is looked only on one side)
    Every other slices are moved or copied (depending on copy mode) in the ouptutFolder
    For the rest of folders the slices are either moved or copied (depending on copy mode)
    :param listOfFolders: a simple list of strings (expect to have images in each of those folders whatever the format)
    :param outputFolderName: a string with the entire path
    :param deltaZ: the supposed z discrete displacement in number of slices
    :param copyMode: 0 files are simply moved (no backup) 1 files are copied in the outputfoldername
    :param securityBandSize: the bandsize (int) in which we will look for the best matched slice between two z folders
    :param overlapMode: 0 just copy or move files 1 standard average in the bandAverageSize 2 weighted average
    :param bandAverageSize: If bandmode >0 size of the band for the (weighted) average
    :param flipUD: 0 alphabetic filenames in a folder is in the right order 1 need to reverse the alphabetic order
    :return:
    """
    print('Stitching ....')
    numberOfFolders=len(listOfFolders)
    cptFolder=0
    listOfFolders.sort()
    begToCopy=0
    for folderName in listOfFolders:
        print(folderName)
        listOfImageFilenames=glob.glob(folderName+'/*.tif')+glob.glob(folderName+'/*.edf')+glob.glob(folderName+'/*.png')
        if flipUD == 1:
            listOfImageFilenames.sort(reverse=True)
        else:
            listOfImageFilenames.sort()
        print(folderName)
        nbSliceInAFolder=len(listOfImageFilenames)
        middleSliceIndex = int(nbSliceInAFolder/2)
        if (cptFolder<numberOfFolders-1):
            listOfImageFilenamesUpperFolder = glob.glob(listOfFolders[cptFolder+1] + '/*.tif') + glob.glob(listOfFolders[cptFolder+1] + '/*.edf') + glob.glob(listOfFolders[cptFolder+1] + '/*.png')
            suposedSliceOfOverlapDown=middleSliceIndex + int(deltaZ/2)
            suposedSliceOfOverlapUp = middleSliceIndex - int(deltaZ / 2)
            if securityBandSize>0:
                imageDownFileNames=listOfImageFilenames[suposedSliceOfOverlapDown-int(securityBandSize):suposedSliceOfOverlapDown+int(securityBandSize)]
                imageUpFileNames=listOfImageFilenamesUpperFolder[suposedSliceOfOverlapUp-int(securityBandSize):suposedSliceOfOverlapUp+int(securityBandSize)]
                imageDown=openSeq(imageDownFileNames)
                imageUp=openSeq(imageUpFileNames)
                indexOfOverlap=lookForMaximumCorrelationBand(imageDown,imageUp,securityBandSize)
                diffIndex=securityBandSize-indexOfOverlap
                trueSliceOverlapIndex=suposedSliceOfOverlapDown-diffIndex
                if overlapMode == 0:
                    #elbourinos
                    listToCopy=listOfImageFilenames[begToCopy:trueSliceOverlapIndex]
                    for fileName in listToCopy:
                        outputFilename=outputFolderName+'/'+os.path.basename(fileName)
                        if copyMode == 0:
                            os.rename(fileName,outputFilename)
                        else:
                            shutil.copy2(fileName,outputFilename)
                    begToCopy = suposedSliceOfOverlapUp + diffIndex

                else:
                    listToCopy=listOfImageFilenames[begToCopy:trueSliceOverlapIndex-int(bandAverageSize/2)]
                    for fileName in listToCopy:
                        outputFilename = outputFolderName + '/' + os.path.basename(fileName)
                        if copyMode == 0:
                            os.rename(fileName, outputFilename)
                        else:
                            shutil.copy2(fileName, outputFilename)
                    filenamesDownToAverage=listOfImageFilenames[trueSliceOverlapIndex-int(bandAverageSize/2):trueSliceOverlapIndex+int(bandAverageSize/2)]
                    filenamesUpToAverage=listOfImageFilenamesUpperFolder[suposedSliceOfOverlapUp+diffIndex-int(bandAverageSize/2):suposedSliceOfOverlapUp+diffIndex+int(bandAverageSize/2)]
                    averagedImage=averageImagesFromFilenames(filenamesDownToAverage,filenamesUpToAverage)
                    listOfFakeNames=listOfImageFilenames[trueSliceOverlapIndex-int(bandAverageSize/2):trueSliceOverlapIndex+int(bandAverageSize/2)]
                    for filename in listOfFakeNames:
                        outputFilename=outputFolderName+os.path.basename(filename)
                        for i in range(0:bandAverageSize):
                            data=averagedImage[i,:,:].squeeze()
                            saveTif(data.astype(numpy.uint16),outputFilename)

                    begToCopy = suposedSliceOfOverlapUp + diffIndex+int(bandAverageSize/2)
        else :
            print('Last Folder')
            listToCopy=listOfImageFilenames[begToCopy:-1]
            for fileName in listToCopy:
                outputFilename = outputFolderName + '/' + os.path.basename(fileName)
                if copyMode == 0:
                    os.rename(fileName, outputFilename)
                else:
                    shutil.copy2(fileName, outputFilename)

        cptFolder+=1


def averageImagesFromFilenames(filenameDown,fileNameUp,mode=0):
    imageDown=openSeq(filenameDown)
    imageUp = openSeq(fileNameUp)

    return (imageDown+imageUp)/2


def lookForMaximumCorrelation(imageA,imageB):
    """
    Function to look for the maximum correlated slice between two different volumes
    Preparation for stitching 2 sets of images
    The computation is only performed with the slice in the middle of imageA on the entire imageB volume
    :param imageA:3D numpy array
    :param imageB:3D numpy array
    :return: the slice number with highest zero normalized cross correlation.

    """
    nbSlicesA, widthA, heightA=imageA.shape
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





