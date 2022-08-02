#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:05:44 2020

@author: quenot
"""
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import radon, rescale
import json
import os
from InputOutput.pagailleIO import saveEdf,openImage
import glob
from numba import jit

def generateContrastPhantom(dimX, dimY, pixsize, angle):
    parameters={}
    
    ##Everything in mm
    pixsize=pixsize/1000 #um to mm
    geometry=[]
    Slice=np.zeros(((13,dimY,dimY)))
    lines=np.zeros(dimY)
    projTot=np.zeros((dimX, dimY))
    smallTubesRadius=2 #mm
    tubesHeight=10 #mm
    supportRadius=15 #mm
    TubesCenters0=[[22,7],[16.4,4.7],[10,7],[6,12.5],[6,19.5],[10,25],[16.4,27],[22,25],[21,16],[11,16],[16,11],[16,21]]
    Nmat=12        
    TubesCenters=np.asarray(TubesCenters0, dtype=np.float64) #mm

    
    sliceTotMat=np.zeros((dimY, dimY))
    
    # Everything in pix
    smallTubesRadiuspix=smallTubesRadius/pixsize #pix
    smallTubesRadiuspixInt=int(np.floor(smallTubesRadiuspix)+2) #pixEntier
    supportRadiuspix=supportRadius/pixsize
    supportRadiuspixInt=int(np.floor(supportRadiuspix)+2)
    tubesHeightPix2=int(tubesHeight/2//(pixsize))
    
    origin=dimY/2-16/pixsize #pix
    
    if origin<0:
        raise ValueError("Image too small for the contrast phantom size")
    
    TubesCenters= TubesCenters/pixsize+origin #pix

    for imat in range(Nmat):
        sliceMat=np.zeros((dimY, dimY))
        centeri=TubesCenters[imat,0] #pix
        centerj=TubesCenters[imat,1]
        centeriInt=int(np.round(centeri)) #pixInt
        centerjInt=int(np.round(centerj))
        for ii in range(centeriInt-smallTubesRadiuspixInt,centeriInt+smallTubesRadiuspixInt):
            for jj in range(centerjInt-smallTubesRadiuspixInt,centerjInt+smallTubesRadiuspixInt):
                dist=((ii-centeri))**2+((jj-centerj))**2
                if dist<smallTubesRadiuspix**2:
                    sliceMat[ii,jj]=1
                    
        Slice[imat]=sliceMat
        sliceTotMat+=sliceMat*(imat+1)
        
    
    sliceMat=np.zeros((dimY, dimY))
    centeri=dimY/2 #pix
    centerj=dimY/2
    centeriInt=int(np.round(centeri)) #pixInt
    centerjInt=int(np.round(centerj))
    for ii in range(centeriInt-supportRadiuspixInt,int(np.round(27/pixsize+origin))):
        for jj in range(centerjInt-supportRadiuspixInt,centerjInt+supportRadiuspixInt):
            if sliceTotMat[ii,jj]==0:
                dist=((ii-centeri))**2+((jj-centerj))**2
                if dist<supportRadiuspix**2:
                    sliceMat[ii,jj]=1
                
    Slice[12]=sliceMat
    sliceTotMat+=sliceMat*(13)
    
    print('Slice Geometry')
    plt.figure()
    plt.imshow(sliceTotMat)
    plt.colorbar()
    plt.show()
    
    for imat in range(Nmat+1):
        
        projMat=np.zeros((dimX,dimY))
        Line=radon(Slice[imat],[angle])
        Nx,Ny=Line.shape
        Line=np.transpose(Line[int(np.round(Nx/2-dimY/2)):int(np.round(Nx/2+dimY/2))])
        
        if imat<12:
            projMat[dimX//2-tubesHeightPix2:dimX//2+tubesHeightPix2]=Line
        if imat==12:
            projMat[dimX//2:dimX//2+tubesHeightPix2]=Line
        
        projTot+=projMat*(imat+1)*pixsize*1e-3
        geometry.append(projMat*pixsize*1e-3)
    
    
    print('Proj Geometry')
    plt.figure()
    plt.imshow(geometry[12])
    plt.colorbar()
    plt.show()

    parameters['smallTubesRadius']=(smallTubesRadius, 'mm')
    parameters['supportRadius']=(supportRadius, 'mm')
    parameters['tubes centers']=(TubesCenters0, 'mm')
    
    return geometry, parameters
        
        
    
if __name__ == "__main__":
        # PARAMETERS
    number_of_positions=2
    imageMargins=0
    overSamp=2
    Nx=500 #Detector size
    Ny=1500
    dimX=int((Nx+2*imageMargins)*overSamp)
    dimY=int((Ny+2*imageMargins)*overSamp)
    dist_source_sample=0.5 #in m
    dist_sample_detector=0.5 #in m
    detector_pixel_size=50 #in um
    magnification=(dist_sample_detector+dist_source_sample)/dist_source_sample
    pixSize=detector_pixel_size/overSamp/magnification #in um

    angle=50 #point of view for thickness maps
    geometry2, param=generateContrastPhantom(dimX, dimY, pixSize, angle)
    geometry2=np.array(geometry2)

    dim=geometry2[0].shape
#    geometry=geometry.tolist()
    
    pixsizestr='%2.2d' % (pixSize*overSamp)
    filepath="ContrastPhantom/CP_pixsize"+pixsizestr.replace(".","p")+"um_"+str(dimX)+"x"+str(dimY)+"_angle"+str(angle)+'deg'
    
    if not os.path.exists(filepath):
        os.mkdir(filepath)
        
    for i in range(13):
        txtPoint = '%2.2d' % i
        saveEdf(geometry2[i], filepath+"/mat"+txtPoint+".tif")
