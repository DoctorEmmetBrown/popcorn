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

def saveParameters(expParam, processing_time, do):
    """Save all experiment and algorithm parameters
    

    Args:
        expParam (PhaseRetrievalClass): instance of the calss containing all parameters as atributes.
        processing_time (DICTIONNARY): contains the processing times of all phase retrieval methods.

    Returns:
        None.

    """
    xlsPath=expParam.output_folder +'.xls'
    xlsFile=Workbook()
    xlsSheet=xlsFile.add_sheet(expParam.expID )
    
    xlsSheet.write_merge(0,0,0,3,expParam.experiment_name +' - '+expParam.expID )
    # xlsSheet.write_merge(1,1,0,3,expParam.Comment )
    i=2
    
    xlsSheet.write(i,0,"Output folder path")
    xlsSheet.write(i,1,expParam.output_folder )
    i+=1
    xlsSheet.write(i,0,"Data path")
    xlsSheet.write(i,1,expParam.exp_folder )
    i+=2
    xlsSheet.write_merge(i,i,0,3,"Experiment parameters")
    i+=1
    xlsSheet.write(i,0,"Energy (keV)")
    xlsSheet.write(i,1,expParam.energy )
    i+=1
    xlsSheet.write(i,0,"Pixel size (m)")
    xlsSheet.write(i,1,expParam.pixel )
    i+=1
    xlsSheet.write(i,0,"Distance sample to detector (m)")
    xlsSheet.write(i,1,expParam.dist_object_detector )
    i+=1
    xlsSheet.write(i,0,"Distance source to sample (m)")
    xlsSheet.write(i,1,expParam.dist_source_object )
    i+=1
    xlsSheet.write(i,0,"Delta")
    xlsSheet.write(i,1,expParam.delta )
    i+=1
    xlsSheet.write(i,0,"Beta")
    xlsSheet.write(i,1,expParam.beta )
    i+=1
    xlsSheet.write(i,0,"Source size (m)")
    xlsSheet.write(i,1,expParam.source_size )
    i+=1
    xlsSheet.write(i,0,"Crop begginning x (pix)")
    xlsSheet.write(i,1,expParam.cropDebX )
    i+=1
    xlsSheet.write(i,0,"Crop begginning y (pix)")
    xlsSheet.write(i,1,expParam.cropDebY )
    i+=1
    xlsSheet.write(i,0,"Crop end x (pix)")
    xlsSheet.write(i,1,expParam.cropEndX )
    i+=1
    xlsSheet.write(i,0,"Crop end y (pix)")
    xlsSheet.write(i,1,expParam.cropEndY )
    i+=2
    xlsSheet.write_merge(i,i,0,3,"Algorithm parameters")
    i+=1
    xlsSheet.write(i,0,"Max shift (pix)")
    xlsSheet.write(i,1,expParam.max_shift )
    i+=1
    xlsSheet.write(i,0,"Padding size (pix)")
    xlsSheet.write(i,1,expParam.pad_size )
    i+=1
    xlsSheet.write(i,0,"Padding type")
    xlsSheet.write(i,1,expParam.pad_type )
    i+=1
    if do["LCS"]:
        xlsSheet.write(i,0,"LCS median filter (pix)")
        xlsSheet.write(i,1,expParam.LCS_median_filter )
        i+=1
    if do["XSVT"]:
        xlsSheet.write(i,0,"XSVT Nw")
        xlsSheet.write(i,1,expParam.XSVT_Nw )
        i+=1
        xlsSheet.write(i,0,"XSVT median filter")
        xlsSheet.write(i,1,expParam.XSVT_median_filter )
        i+=1
    if do["UMPA"]:
        xlsSheet.write(i,0,"UMPA Nw")
        xlsSheet.write(i,1,expParam.umpaNw )
        i+=1
    xlsSheet.write(i,0,"Number of points")
    xlsSheet.write(i,1,expParam.nb_of_point )
    i+=1
    xlsSheet.write(i,0,"PSF detector")
    xlsSheet.write(i,1,expParam.detector_PSF )
    i+=1
    xlsSheet.write(i,0,"Deconvolution")
    xlsSheet.write(i,1,expParam.deconvolution )
    i+=1
    xlsSheet.write(i,0,"Deconvolution algo")
    xlsSheet.write(i,1,expParam.deconvolution_type )
    i+=1
    xlsSheet.write(i,0,"Absorption correction sigma")
    xlsSheet.write(i,1,expParam.absorption_correction_sigma )
    i+=1
    xlsSheet.write(i,0,"Fourier space filter sigma")
    xlsSheet.write(i,1,expParam.sigma_regularization )
    
    i+=2
    xlsSheet.write_merge(i,i,0,3,"PROCESSING TIMES (s)")
    
    processing_time_list=list(processing_time)
    processing_time_values_list=list(processing_time.values())
    for j in range(len(processing_time_list)):
        i+=1
        xlsSheet.write(i,0,processing_time_list[j])
        xlsSheet.write(i,1,processing_time_values_list[j])
            
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