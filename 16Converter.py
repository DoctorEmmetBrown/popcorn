import os
from popcornIO import openImage, saveTiff16bit

def conversionFromListOfFiles(listOfFiles, outputFolder, minValue, maxValue):
    print("16 bit conversion started.")
    for fileName in listOfFiles:
        baseName = os.path.basename(fileName).split(".")[0]
        data = openImage(fileName)
        saveTiff16bit(data, outputFolder + baseName + ".tif", minValue, maxValue)
    print("16 bit conversion done.")

if __name__ == "__main__" :
    print("Hello")
