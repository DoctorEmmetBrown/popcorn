# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:46:27 2021.

@author: quenot
"""
import os
import sys

sys.path.append(os.path.realpath('../..'))

from scipy.ndimage.filters import gaussian_laplace,sobel,median_filter,laplace
from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
from math import pi as pi
import numpy as np
import glob
from scipy.ndimage import fourier_shift



def kevToLambda(energyInKev):
    energy = energyInKev * 1e3
    waveLengthInNanometer = 1240. / energy

    return waveLengthInNanometer * 1e-9

def MISTI(experiment):
    nbImages, Nx, Ny=experiment.reference_images.shape
    beta=experiment.beta
    gamma_mat=experiment.delta/beta
    distSampDet=experiment.dist_object_detector
    pixSize=experiment.pixel
    k=experiment.getk()
    Lambda=1.2398/experiment.energy*1e-9
    
    LHS=np.ones(((nbImages, Nx, Ny)))
    RHS=np.ones((((nbImages,2, Nx, Ny))))
    solution=np.ones(((2, Nx, Ny)))
    
    #Prepare system matrices
    for i in range(nbImages):
        #Left hand Side
        IrIs=experiment.reference_images[i]-experiment.sample_images[i]
        
        #Right handSide
        FirstRHS=experiment.reference_images[i]/k
        lapIr=laplace(experiment.reference_images[i])
        
        RHS[i]=[FirstRHS,-lapIr]
        LHS[i]=IrIs/distSampDet
        
#    Solving system for each pixel 
    for i in range(Nx):
        for j in range(Ny):
            a=RHS[:,:,i,j]
            b=LHS[:,i,j]
            Q,R = np.linalg.qr(a) # qr decomposition of A
            Qb = np.dot(Q.T,b) # computing Q^T*b (project b onto the range of A)
            if R[1,1]==0 or R[0,0]==0:
                temp=[1,1]
            else:
                temp = np.linalg.solve(R,Qb) # solving R*x = Q^T*b
            solution[:,i,j]=temp
            
    lapPhi=solution[0]
    Deff=solution[1]
    
    #Median filter
    medFiltSize=experiment.MIST_median_filter
    if medFiltSize!=0:
        Deff=median_filter(Deff, medFiltSize)
    
    #Calculation of the thickness of the object
    u, v = np.meshgrid(np.arange(0, Nx), np.arange(0, Ny))
    u = (u - (Nx / 2))
    v = (v - (Ny / 2))
    u_m = u / (Nx * pixSize)
    v_m = v / (Ny * pixSize)
    uv_sqr = np.transpose(u_m ** 2 + v_m ** 2)  # ie (u2+v2)
    alpha=1
    uv_sqr[uv_sqr==0]=alpha
    
    sig_scale=experiment.sigma_regularization
    if sig_scale==0:
        beta=1
    else:
        dqx = 2 * pi / (Nx)
        dqy = 2 * pi / (Ny)
        Qx, Qy = np.meshgrid((np.arange(0, Ny) - np.floor(Ny / 2) - 1) * dqy, (np.arange(0, Nx) - np.floor(Nx / 2) - 1) * dqx) #frequency ranges of the images in fqcy space
    
        #building filters
        sigmaX = dqx / 1. * np.power(sig_scale,2)
        sigmaY = dqy / 1. * np.power(sig_scale,2)
        #sigmaX=sig_scale
        #sigmaY = sig_scale
    
        g = np.exp(-(((Qx)**2) / 2. / sigmaX + ((Qy)**2) / 2. / sigmaY))
        #g = np.exp(-(((np.power(Qx, 2)) / 2) / sigmaX + ((np.power(Qy, 2)) / 2) / sigmaY))
        beta = 1 - g;
    #Calculation of absorption image
    phi=k/distSampDet*ifft2(ifftshift(fftshift(fft2(lapPhi))/(-4*np.pi*uv_sqr)*beta)).real   
    
    return {'Deff': Deff, 'phi': phi}
    