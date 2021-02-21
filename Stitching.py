import numpy
import glob
#from scipy.signal import correlate
from popcornIO import openSeq, save3D_Edf,saveTif
import os
import shutil
from skimage import filters




def stitchFolders(listOfFolders,outputFolderName,deltaZ,lookForBestSlice=True,copyMode=0,securityBandSize=10,overlapMode=0,bandAverageSize=0,flipUD=0):
    """
    Function that stitches different folders into a unique one
    The first and last folders are treated differently (correlation with other folders is looked only on one side)
    Every other slices are moved or copied (depending on copy mode) in the ouptutFolder
    For the rest of folders the slices are either moved or copied (depending on copy mode)
    :param listOfFolders: a simple list of strings (expect to have images in each of those folders whatever the format)
    :param outputFolderName: a string with the entire path
    :param deltaZ: the supposed z discrete displacement in number of slices
    :param lookForBestSlice: False : we don't look for best matched slice between folders, True : we do
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
        nbSliceInAFolder = len(listOfImageFilenames)
        if (cptFolder<numberOfFolders-1):
            listOfImageFilenamesUpperFolder = glob.glob(listOfFolders[cptFolder+1] + '/*.tif') + glob.glob(listOfFolders[cptFolder+1] + '/*.edf') + glob.glob(listOfFolders[cptFolder+1] + '/*.png')

            if flipUD == 1:
                listOfImageFilenamesUpperFolder.sort(reverse=True)
            else:
                listOfImageFilenamesUpperFolder.sort()

            suposedSliceOfOverlapDown = nbSliceInAFolder - int((nbSliceInAFolder - deltaZ)/2)
            print('suposedSliceOfOverlapDown'+str(suposedSliceOfOverlapDown))
            suposedSliceOfOverlapUp = int((nbSliceInAFolder - deltaZ)/2)
            print('suposedSliceOfOverlapUp' + str(suposedSliceOfOverlapUp))

            if securityBandSize>0:
                if lookForBestSlice:
                    imageDownFileNames=listOfImageFilenames[suposedSliceOfOverlapDown-int(securityBandSize):suposedSliceOfOverlapDown+int(securityBandSize)]
                    imageUpFileNames=listOfImageFilenamesUpperFolder[suposedSliceOfOverlapUp-int(securityBandSize):suposedSliceOfOverlapUp+int(securityBandSize)]
                    #print('Band : ['+str(suposedSliceOfOverlapDown-int(securityBandSize))+','+str(suposedSliceOfOverlapDown+int(securityBandSize)))

                    imageDown=openSeq(imageDownFileNames)
                    imageUp=openSeq(imageUpFileNames)

                    indexOfOverlap = int(lookForMaximumCorrelationBand(imageDown, imageUp, 10, True))

                    diffIndex=securityBandSize - indexOfOverlap
                else:
                    diffIndex = 0
                trueSliceOverlapIndex=suposedSliceOfOverlapDown + diffIndex


                if overlapMode == 0:
                    #elbourinos

                    listToCopy = listOfImageFilenames[begToCopy:trueSliceOverlapIndex]
                    for sliceNb in range(0, len(listToCopy)):
                        if flipUD == 1:
                            outputFilename=outputFolderName+'/'+os.path.basename(listToCopy[-(sliceNb + 1)])
                        else:
                            outputFilename=outputFolderName+'/'+os.path.basename(listToCopy[sliceNb])

                        if copyMode == 0:
                            os.rename(listToCopy[sliceNb],outputFilename)
                        else:
                            shutil.copy2(listToCopy[sliceNb],outputFilename)
                    begToCopy = suposedSliceOfOverlapUp

                else:
                    listOfImageFilenames[begToCopy:trueSliceOverlapIndex-int(bandAverageSize/2)]

                    for fileNb in range(0, len(listToCopy)):
                        if flipUD == 1:
                            outputFilename = outputFolderName + '/' + os.path.basename(listToCopy[-(fileNb + 1)])
                        else:
                            outputFilename = outputFolderName + '/' + os.path.basename(listToCopy[fileNb])
                        if copyMode == 0:
                            os.rename(listToCopy[fileNb], outputFilename)
                        else:
                            shutil.copy2(listToCopy[fileNb], outputFilename)

                    filenamesDownToAverage=listOfImageFilenames[trueSliceOverlapIndex-int(bandAverageSize/2):trueSliceOverlapIndex+int(bandAverageSize/2)]

                    filenamesUpToAverage=listOfImageFilenamesUpperFolder[suposedSliceOfOverlapUp+diffIndex-int(bandAverageSize/2):suposedSliceOfOverlapUp+diffIndex+int(bandAverageSize/2)]
                    averagedImage=averageImagesFromFilenames(filenamesDownToAverage,filenamesUpToAverage)
                    listOfFakeNames=listOfImageFilenames[trueSliceOverlapIndex-int(bandAverageSize/2):trueSliceOverlapIndex+int(bandAverageSize/2)]

                    for filename in listOfFakeNames:
                        outputFilename=outputFolderName+os.path.basename(filename)
                        for i in range(0,bandAverageSize) :
                            data=averagedImage[i,:,:].squeeze()
                            saveTif(data.astype(numpy.uint16),outputFilename)

                    begToCopy = suposedSliceOfOverlapUp + diffIndex+int(bandAverageSize/2)
        else :
            print('Last Folder')

            listToCopy = listOfImageFilenames[begToCopy:-1]
            for fileNb in range(0, len(listToCopy)):
                if flipUD == 1:
                    outputFilename = outputFolderName + '/' + os.path.basename(listToCopy[-(fileNb + 1)])
                else:
                    outputFilename = outputFolderName + '/' + os.path.basename(listToCopy[fileNb])

                if copyMode == 0:
                    os.rename(listToCopy[fileNb], outputFilename)
                else:
                    shutil.copy2(listToCopy[fileNb], outputFilename)


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
    nbSlicesA, widthA, heightA = imageA.shape
    nbSlicesB, widthB, heightB = imageB.shape
    width = max(widthA, widthB)
    height = max(heightA, heightB)

    middleSlice = int(nbSlicesA/2)
    imageToMultiply = numpy.copy(imageA[middleSlice, :, :].squeeze())
    imageToMultiply = imageToMultiply - numpy.mean(imageToMultiply)

    tmpB = numpy.copy(imageB)
    stdMul = numpy.std(imageToMultiply)
    corr = numpy.zeros(nbSlicesB)

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
        #print("slice", slice, "cross-corr :", normcrosscorr)

    maxCorSlice=numpy.argmax(corr)
    print("best slice", maxCorSlice, "cross-corr :", corr[maxCorSlice])

    return (maxCorSlice)


def lookForMaximumCorrelationBand(imageA,imageB,bandSize,segmented=False):
    """
    Function to look for the maximum correlated slice between two different volumes
    Preparation for stitching 2 sets of images
    The computation is performed for every slices in a band of bandSize centered around the imageA middle slice
    :param imageA:3D numpy array
    :param imageB:3D numpy array
    :param bandSize: Number of slice the zero normalized cross correlation is made on
    :param segmented: True otsu thesholding is performed before correlation
    :return: the median value of all slices number with highest zero normalized cross correlation.
    """
    nbSlices, width, height = imageA.shape

    middleSlice = int(nbSlices / 2)

    tmpA = numpy.copy(imageA)
    tmpB = numpy.copy(imageB)

    if segmented:
        thresh = filters.threshold_otsu(tmpA[tmpA > 0.15 * 65535])
        mask   = tmpA > thresh
        tmpA = mask * tmpA
        tmpB = mask * tmpB

    #Preparation for normalized cross correlation
    for slice in range(0, nbSlices):
        tmpBSlice = tmpB[slice, :, :]
        if segmented:
            tmpB[slice, :, :] = mask[slice, :, :] * (tmpBSlice - numpy.mean(tmpBSlice[tmpBSlice > 0.0]))
        else:
            tmpB[slice, :, :] = tmpB[slice, :, :] - numpy.mean(tmpB[slice, :, :])

    argMaxCoors = numpy.zeros(bandSize)
    for i in range(int(-bandSize/2),(int(bandSize/2))):
        imageToMultiply = tmpA[middleSlice+i, :, :].squeeze()
        if segmented:
            imageToMultiply = mask[middleSlice+i, :, :] * (imageToMultiply - numpy.mean(imageToMultiply[imageToMultiply > 0.0]))
        else:
            imageToMultiply = imageToMultiply - numpy.mean(imageToMultiply)
        stdMul = numpy.std(imageToMultiply)
        corr = numpy.zeros(nbSlices)
        imMultiplied = imageToMultiply * tmpB

        for slice in range(0,nbSlices):
            stdB = numpy.std(tmpB[slice, :, :])
            sumMultiplication = numpy.sum(imMultiplied[slice, :, :])

            normcrosscorr = sumMultiplication / (stdMul * stdB)
            normcrosscorr /= (width * height)
            corr[slice]=normcrosscorr
            #print("slice", slice, "cross-corr :", normcrosscorr)

        maxCorSlice = numpy.argmax(corr)-i
        #print('maxCorSlice found for slice:'+str(i)+' is :'+str(maxCorSlice))
        argMaxCoors[i + int(bandSize/2)] = maxCorSlice
        print("For imageA slice", middleSlice + i, "best fit is imageB slice", numpy.argmax(corr))
    print(argMaxCoors)
    medianValue = numpy.median(argMaxCoors)
    return (medianValue)


if __name__ == "__main__" :
    print("Hello")
    imageAFolder = 'C:\\Users\\ctavakol\\Desktop\\test_for_popcorn\\voltif\\test1pag\\'
    imageBFolder = 'C:\\Users\\ctavakol\\Desktop\\test_for_popcorn\\voltif\\test2pag\\'
    imageAFiles = glob.glob(imageAFolder + '/*.tif')
    imageBFiles = glob.glob(imageAFolder + '/*.tif')
    imageA = openSeq(imageAFiles)
    imageB = openSeq(imageBFiles)
    #lookForMaximumCorrelation(imageA,imageB)
    #lookForMaximumCorrelationBand(imageA,imageB,10)