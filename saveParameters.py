#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 13:27:44 2020

@author: quenot
"""
import numpy as np
from matplotlib import pyplot as plt
import xlrd
from xlwt import Workbook, Formula
import os.path
from os import path
from xlutils.copy import copy

def saveParameters(expParam):
    xlsPath=expParam['outputFolder']+'.xls'
    xlsFile=Workbook()
    xlsSheet=xlsFile.add_sheet(expParam['expID'])
    
    xlsSheet.write_merge(0,0,0,3,expParam['studiedCase']+' - '+expParam['expID'])
    xlsSheet.write_merge(1,1,0,3,expParam['Comment'])
    i=2
    
    xlsSheet.write(2,0,"Output folder path")
    xlsSheet.write(2,1,expParam['outputFolder'])
    i+=1
    xlsSheet.write(3,0,"Data path")
    xlsSheet.write(3,1,expParam['expFolder'])
    i+=1
    xlsSheet.write(4,0,"Energy (keV)")
    xlsSheet.write(4,1,expParam['energy'])
    i+=1
    xlsSheet.write(5,0,"Pixel size (m)")
    xlsSheet.write(5,1,expParam['pixel'])
    i+=1
    xlsSheet.write(6,0,"Distance sample to detector (m)")
    xlsSheet.write(6,1,expParam['distOD'])
    i+=1
    xlsSheet.write(7,0,"Distance source to sample (m)")
    xlsSheet.write(7,1,expParam['distSO'])
    i+=1
    xlsSheet.write(8,0,"Delta")
    xlsSheet.write(8,1,expParam['delta'])
    i+=1
    xlsSheet.write(9,0,"Beta")
    xlsSheet.write(9,1,expParam['beta'])
    i+=1
    xlsSheet.write(10,0,"Source size (m)")
    xlsSheet.write(10,1,expParam['sourceSize'])
    i+=1
    xlsSheet.write(11,0,"Crop begginning x (pix)")
    xlsSheet.write(11,1,expParam['cropDebX'])
    i+=1
    xlsSheet.write(12,0,"Crop begginning y (pix)")
    xlsSheet.write(12,1,expParam['cropDebY'])
    i+=1
    xlsSheet.write(13,0,"Crop end x (pix)")
    xlsSheet.write(13,1,expParam['cropEndX'])
    i+=1
    xlsSheet.write(14,0,"Crop end y (pix)")
    xlsSheet.write(14,1,expParam['cropEndY'])
    i+=1
    xlsSheet.write(16,0,"Max shift (pix)")
    xlsSheet.write(16,1,expParam['umpaMaxShift'])
    i+=1
    xlsSheet.write(17,0,"Padding size (pix)")
    xlsSheet.write(17,1,expParam['padding'])
    i+=1
    xlsSheet.write(18,0,"Padding type")
    xlsSheet.write(18,1,expParam['padType'])
    i+=1
    xlsSheet.write(19,0,"LCS median filter (pix)")
    xlsSheet.write(19,1,expParam['LCS_median_filter'])
    i+=1
    xlsSheet.write(20,0,"LCS median filter (pix)")
    xlsSheet.write(20,1,expParam['LCS_gaussian_filter'])
    i+=1
    xlsSheet.write(21,0,"Number of points")
    xlsSheet.write(21,1,expParam['nbOfPoint'])
    i+=1
    xlsSheet.write(22,0,"PSF detector")
    xlsSheet.write(22,1,expParam['detectorPSF'])
    i+=1
    xlsSheet.write(23,0,"Deconvolution")
    xlsSheet.write(23,1,expParam['Deconvolution'])
    i+=1
    xlsSheet.write(24,0,"Deconvolution algo")
    xlsSheet.write(24,1,expParam['DeconvType'])
    i+=1
    xlsSheet.write(25,0,"processing time LCSv2")
    xlsSheet.write(25,1,expParam['processingtimeLCSv2'])
    i+=1
    xlsSheet.write(26,0,"processing Time PavlovDirDF")
    xlsSheet.write(26,1,expParam['processingtimePavlovDirDF'])
    i+=1
    xlsSheet.write(27,0,"PavlovDirDF median filter parameter")
    xlsSheet.write(27,1,expParam['PavlovDirDF_MedianFilter'])
    i+=1
    xlsSheet.write(28,0,"processing Time UMPA")
    xlsSheet.write(28,1,expParam['processingTimeUMPA'])
    i+=1
    xlsSheet.write(29,0,"Absorption_correction_sigma")
    xlsSheet.write(29,1,expParam['Absorption_correction_sigma'])
            
    print("Finished fiiling xls file just saving now")
    xlsFile.save(xlsPath)


def fillExcelFile(xlsFile,xlsSheet, xlsPath, qualityDict, membraneDict,displacementRetrievalMethods,retrievedImage,phaseRetrievalMethods):
#    xlsSheet=xlsFile.sheet_by_name(str(membraneID))
    line=4+membraneDict["membraneNumber"]
    
    xlsSheet.write(line,0,membraneDict["membraneNumber"])
    xlsSheet.write(line,1,membraneDict["currentParamValue"])
    xlsSheet.write(line,2,qualityDict["RawData"]["IrVisibility"])
    xlsSheet.write(line,3,qualityDict["RawData"]["IsVisibility"])
    xlsSheet.write(line,4,qualityDict["RawData"]["SubISNR"])
    xlsSheet.write(line,5,qualityDict["RawData"]["IpSNR"])
    xlsSheet.write(line,6,qualityDict["RawData"]["SubINiqe"])
    
    i=0
    j=0
    for method in displacementRetrievalMethods:
        j=0
        for image in retrievedImage:
            rowSNR=7+j+i
            xlsSheet.write(line,rowSNR,qualityDict["RetrievedDisplacements"][method][image]["SNR"])
            rowNiqe=7+15+j+i
            xlsSheet.write(line,rowNiqe,qualityDict["RetrievedDisplacements"][method][image]["Niqe"])    
            j+=3
        i+=1
    
    i=0
    for method in phaseRetrievalMethods:
        xlsSheet.write(line,37+i,qualityDict["RetrievedPhase"][method]["SNR"])
        xlsSheet.write(line,39+i,qualityDict["RetrievedPhase"][method]["Niqe"])
        i+=1
        
    print("Finished fiiling xls file just saving now")
    xlsFile.save(xlsPath)
    
def createExcelFile(xlsPath, qualityDict, membraneDict):
    membraneID=membraneDict["membraneID"]
    if path.exists(xlsPath):
        xlsFile=xlrd.open_workbook(xlsPath)
        xlsSheet=xlsFile.sheet_by_name(str(membraneID))
    else:
        xlsFile=Workbook()
        xlsSheet=xlsFile.add_sheet(str(membraneID))
    
#    xlsFile=copy(xlsFile)
#    xlsSheet=xlsFile.get_sheet(0)
    #Formating
    xlsSheet.write(0,0,"Membrane ID:"+str(membraneID))
    xlsSheet.write(0,1,membraneID)
    xlsSheet.write(3,0,"MembraneNumber")
    xlsSheet.write(2,1,"Membrane param varying:")
    xlsSheet.write(3,1,membraneDict["varyingParameter"]+" ["+membraneDict["paramUnit"]+"]")
    xlsSheet.write_merge(0,0,2,6,"Raw data values")
    xlsSheet.write_merge(1,1,2,3,"Visibility")
    xlsSheet.write_merge(1,1,4,5,"SNR")
    xlsSheet.write(3,2,"reference image visibility")
    xlsSheet.write(3,3,"sample image visibility")
    xlsSheet.write(3,4,"sub-image SNR")
    xlsSheet.write(3,5,"propagation image SNR")
    xlsSheet.write(1,6,"Niqe")
    xlsSheet.write(3,6,"propagation image Niqe")
    
    xlsSheet.write_merge(0,0,7,36,"Displacement retrieval set")
    
    xlsSheet.write_merge(1,1,7,21,"SNR")
    
    xlsSheet.write_merge(2,2,7,9,"Dx")
    xlsSheet.write_merge(2,2,10,12,"Dy")
    xlsSheet.write_merge(2,2,13,15,"phi FC")
    xlsSheet.write_merge(2,2,16,18,"phi K")
    xlsSheet.write_merge(2,2,19,21,"phi LA")
    for i in range(5):
        xlsSheet.write(3,7+i*3,"OF")
        xlsSheet.write(3,7+i*3+1,"Geo")
        xlsSheet.write(3,7+i*3+2,"UMPA")
        
    xlsSheet.write_merge(1,1,22,36,"Niqe")
    xlsSheet.write_merge(2,2,22,24,"Dx")
    xlsSheet.write_merge(2,2,25,27,"Dy")
    xlsSheet.write_merge(2,2,28,30,"phi FC")
    xlsSheet.write_merge(2,2,31,33,"phi K")
    xlsSheet.write_merge(2,2,34,36,"phi LA")   
    for i in range(5):
        xlsSheet.write(3,22+i*3,"OF")
        xlsSheet.write(3,22+i*3+1,"Geo")
        xlsSheet.write(3,22+i*3+2,"UMPA") 
    
    xlsSheet.write_merge(0,0,37,40,"Phase retrieval set")
    xlsSheet.write_merge(1,1,37,38,"SNR")
    xlsSheet.write_merge(1,1,39,40,"Niqe")
    xlsSheet.write(3,37,"phi Pavlov")
    xlsSheet.write(3,38,"phi TIE")
    xlsSheet.write(3,39,"phi Pavlov")
    xlsSheet.write(3,40,"phi TIE")

    return xlsFile, xlsSheet