import os
import glob
from SixteenBitConverter import conversionFromListOfFiles
from Stitching import stitchFolders

def myMkdir(folderPath):
    if not os.path.isdir(folderPath):
        os.mkdir(folderPath)



if __name__ == "__main__" :
    #mainFolder='/data/visitor/md1217/id17/'
    mainFolder = "C:\\Users\\ctavakol\\Desktop\\test_for_popcorn\\"
    mainOutputFolder = "C:\\Users\\ctavakol\\Desktop\\test_for_popcorn\\voltif\\"


    radix='11mPropagation_23um_33kev_026_CE_CT_GW_13'
    speckleDone=False

    minIm16Bit=0.
    maxIm16Bit = 1.
    deltaZ=234


    if speckleDone:
        reconstructedFolders = glob.glob(mainFolder + '/volfloat/*' + radix + '*Propag_pag')
    else:
        reconstructedFolders = glob.glob(mainFolder+'/volfloat/*'+radix+'*pag')

    listOf16bitFolder=[]
    reconstructedFolders.sort()
    for inputFolder in reconstructedFolders:
        print(inputFolder)
        baseName=os.path.basename(inputFolder)
        print(baseName)
        imageFiles=glob.glob(inputFolder+'/*.tif')+glob.glob(inputFolder+'/*.edf')
        outputFolder=mainOutputFolder+baseName+'/'
        listOf16bitFolder.append(outputFolder)
        myMkdir(outputFolder)
        #conversionFromListOfFiles(imageFiles,outputFolder,minIm16Bit,maxIm16Bit)

    outpuRadixFolder=mainOutputFolder+'/'+radix+'_'+str(minIm16Bit)+'_'+str(maxIm16Bit)
    myMkdir(outpuRadixFolder)
    stitchFolders(listOf16bitFolder, outpuRadixFolder, deltaZ, copyMode=1, securityBandSize=30, overlapMode=0,bandAverageSize=0, flipUD=0)




