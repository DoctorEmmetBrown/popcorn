# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:46:27 2021.

@author: quenot
"""
import numpy as np
from popcorn.phase_retrieval import frankoChellappa  as fc
from scipy.ndimage.filters import gaussian_filter, median_filter
from scipy.ndimage import laplace
from matplotlib import pyplot as plt
from skimage import color, data, restoration
from popcorn.phase_retrieval.phaseIntegration import kottler, LarkinAnissonSheppard


def LCS_DF(experiment):
    """Calculates the displacement images from sample and reference images using the LCS system
    

    Args:
        experiment (PhaseRetrievalClass): class with all parameters as attributes.

    Returns:
        Dx (NUMPY ARRAY): the displacements along x axis.
        Dy (NUMPY ARRAY): the displacements along y axis.
        absoprtion (NUMPY ARRAY): the absorption.

    """

    Nz, Nx, Ny=experiment.reference_images.shape
    LHS=np.ones(((experiment.nb_of_point, Nx, Ny)))
    RHS=np.ones((((experiment.nb_of_point,4, Nx, Ny))))
    solution=np.ones(((4, Nx, Ny)))

    #Prepare system matrices
    for i in range(experiment.nb_of_point):
        #Right handSide
        gX_IrIr,gY_IrIr=np.gradient(experiment.reference_images[i])
        lapIr=laplace(experiment.reference_images[i])
        RHS[i]=[experiment.sample_images[i],gX_IrIr, gY_IrIr, -lapIr]
        LHS[i]=experiment.reference_images[i]

    #Solving system for each pixel 
    for i in range(Nx):
        for j in range(Ny):
            a=RHS[:,:,i,j]
            b=LHS[:,i,j]
            Q,R = np.linalg.qr(a) # qr decomposition of A
            Qb = np.dot(Q.T,b) # computing Q^T*b (project b onto the range of A)
            
            if R[2,2]==0 or R[1,1]==0 or R[0,0]==0 or R[3,3]==0:
                temp=[1,0,0,0]
            else:
                temp = np.linalg.solve(R,Qb) # solving R*x = Q^T*b
            solution[:,i,j]=temp
        
    absoprtion=1/solution[0]
    Dx=solution[1]
    Dy=solution[2]
    DeltaDeff=solution[3]
    
    #Bit of post-processing
    #Limiting displacement to a threshold
    displacementLimit=experiment.max_shift
    Dx[Dx<-displacementLimit]=-displacementLimit
    Dx[Dx>displacementLimit]=displacementLimit
    Dy[Dy<-displacementLimit]=-displacementLimit
    Dy[Dy>displacementLimit]=displacementLimit
    #Trying different filters
    if experiment.LCS_median_filter !=0:
        Dx=median_filter(Dx,size=experiment.LCS_median_filter)
        Dy=median_filter(Dy,size=experiment.LCS_median_filter)
    DeltaDeff[DeltaDeff<0]=0
    DeltaDeff[DeltaDeff>1]=0
    DeltaDeff[DeltaDeff>np.std(DeltaDeff)*5]=0
    return Dx, Dy, absoprtion, DeltaDeff


def processProjectionLCS_DF(experiment):
    """launches calculation of displacement maps and phase images from LCS, FC, LA and K.
    
    Args:
        experiment (PHASERETRIEVALCLASS): class of the experiment.

    Returns:
        dict | NUMPY ARRAY : contains all the calculated images.

    """
    experiment.nb_of_point, Nx, Ny= experiment.sample_images.shape
    
    dx, dy , absorption ,DeltaDeff=LCS_DF(experiment)

    # Compute the phase gradient from displacements (linear relationship)
    # magnification=(experiment['distSO']+experiment['distOD'])/experiment['distSO'] #Not sure I need to use this yet
    
    print("experiment pixel", experiment.pixel)
    print("distance object detector", experiment.dist_object_detector)
    print("k", experiment.getk())
    
    
    dphix=dx*(experiment.pixel/experiment.dist_object_detector)*experiment.getk()
    dphiy=dy*(experiment.pixel/experiment.dist_object_detector)*experiment.getk()
    
    padForIntegration=True
    padSize=1000
    if padForIntegration:
        dphix = np.pad(dphix, ((padSize, padSize), (padSize, padSize)),mode='reflect')  # voir is edge mieux que reflect
        dphiy = np.pad(dphiy, ((padSize, padSize), (padSize, padSize)),mode='reflect')  # voir is edge mieux que reflect
    
    # Compute the phase from phase gradients with 3 different methods (still trying to choose the best one)
    phiFC = fc.frankotchellappa(dphiy, dphix, True)*experiment.pixel
    phiK = kottler(dphiy, dphix)*experiment.pixel
    phiLA = LarkinAnissonSheppard(dphiy, dphix)*experiment.pixel
    
    if padSize > 0:
        phiFC = phiFC[padSize:padSize + Nx, padSize:padSize + Ny]
        phiK = phiK[padSize:padSize + Nx , padSize:padSize + Ny]
        phiLA = phiLA[padSize:padSize + Nx, padSize:padSize + Ny]

    return {'dx': dx, 'dy': dy, 'phiFC': phiFC.real, 'phiK': phiK.real,'phiLA': phiLA.real, 'absorption':absorption, 'DeltaDeff':DeltaDeff}
            
    

