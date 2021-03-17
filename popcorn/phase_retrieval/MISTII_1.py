# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:46:27 2021.

@author: quenot
"""
from pagailleIO import saveEdf,openImage,openSeq,save3D_Edf
from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
import numpy as np
from matplotlib import cm
import matplotlib as mpl
from scipy.ndimage.filters import gaussian_filter, median_filter
from scipy.sparse.linalg import lsmr
from matplotlib import pyplot as plt
import colorsys

def MISTII_1(experiment):
    """
    Calculates the tensors of the dark field and the thickness of a phase object from the acquisitions
    """
        
    Nz, Nx, Ny=experiment.reference_images.shape
    beta=experiment.beta
    gamma_mat=experiment.delta/beta
    distSampDet=experiment.dist_object_detector
    pixSize=experiment.pixel
    Lambda=1.2398/experiment.energy*1e-9
    k=experiment.getk()
    
    LHS=np.ones(((experiment.nb_of_point, Nx, Ny)))
    RHS=np.ones((((experiment.nb_of_point,4, Nx, Ny))))
    FirstTermRHS=np.ones((Nx,Ny))
    solution=np.ones(((4, Nx, Ny)))
    
    #Prepare system matrices
    for i in range(experiment.nb_of_point):
        #Left hand Side
        IsIr=experiment.sample_images[i]/experiment.reference_images[i]
        
        #Right handSide
        gX_IrIr,gY_IrIr=np.gradient(experiment.reference_images[i],pixSize )
        gXX_IrIr,gYX_IrIr=np.gradient(gX_IrIr,pixSize)
        gXY_IrIr,gYY_IrIr=np.gradient(gY_IrIr,pixSize)
        
        gXX_IrIr=gXX_IrIr/experiment.reference_images[i]
        gXY_IrIr=gXY_IrIr/experiment.reference_images[i]
        gYX_IrIr=gYX_IrIr/experiment.reference_images[i]
        gYY_IrIr=gYY_IrIr/experiment.reference_images[i]
        
        RHS[i]=[FirstTermRHS,gXX_IrIr, gYY_IrIr,gXY_IrIr]
        LHS[i]=1-IsIr
        
#    Solving system for each pixel 
    for i in range(Nx):
        for j in range(Ny):
            a=RHS[:,:,i,j]
            b=LHS[:,i,j]
            Q,R = np.linalg.qr(a) # qr decomposition of A
            if R[2,2]==0 or R[1,1]==0 or R[0,0]==0 or R[3,3]==0:
                temp=[1,1,1,1]
            else:
                Qb = np.dot(Q.T,b) # computing Q^T*b (project b onto the range of A)
                temp = np.linalg.solve(R,Qb) # solving R*x = Q^T*b
                solution[:,i,j]=temp
            
    G1=solution[0]
    G2=solution[1]
    G3=solution[2]
    G4=solution[3]
    
    sig_scale=experiment.sigma_regularization
    if sig_scale==0:
        beta=1
    else:
        dqx = 2 * np.pi / (Nx)
        dqy = 2 * np.pi / (Ny)
        Qx, Qy = np.meshgrid((np.arange(0, Ny) - np.floor(Ny / 2) - 1) * dqy, (np.arange(0, Nx) - np.floor(Nx / 2) - 1) * dqx) #frequency ranges of the images in fqcy space
    
        #building filters
        sigmaX = dqx / 1. * np.power(sig_scale,2)
        sigmaY = dqy / 1. * np.power(sig_scale,2)
        #sigmaX=sig_scale
        #sigmaY = sig_scale
    
        g = np.exp(-(((Qx)**2) / 2. / sigmaX + ((Qy)**2) / 2. / sigmaY))
        #g = np.exp(-(((np.power(Qx, 2)) / 2) / sigmaX + ((np.power(Qy, 2)) / 2) / sigmaY))
        beta = 1 - g;
    
    #Calculation of the thickness of the object
    u, v = np.meshgrid(np.arange(0, Nx), np.arange(0, Ny))
    u = (u - (Nx / 2))
    v = (v - (Ny / 2))
    u_m = u / (Nx * pixSize)
    v_m = v / (Ny * pixSize)
    uv_sqr = np.transpose(u_m ** 2 + v_m ** 2)  # ie (u2+v2)
    alpha=1
    uv_sqr[uv_sqr==0]=alpha
    
    #Calculation of absorption image
    phi=k/distSampDet*ifft2(ifftshift(fftshift(fft2(G1))*beta/(-4*np.pi*uv_sqr))).real                                                                                                                                                                                                                           

    Deff_xx=-G2
    Deff_yy=-G3
    Deff_xy=-G4
    
    return phi, Deff_xx,Deff_yy,Deff_xy

def processProjectionMISTII_1(experiment):
    """
    This function calls PavlovDirDF to compute the tensors of the directional dark field and the phase of the sample
    The function should also convert the tensor into a coloured image
    """
    Nx, Ny=experiment.sample_images[0].shape
    #Calculate directional darl field
    phi, Deff_xx,Deff_yy,Deff_xy=MISTII_1(experiment)
    
    #Post processing tests
    #Median filter
    medFiltSize=experiment.MIST_median_filter
    if medFiltSize!=0:
        phi=median_filter(phi, medFiltSize)
        Deff_xx=median_filter(Deff_xx, medFiltSize)
        Deff_yy=median_filter(Deff_yy, medFiltSize)
        Deff_xy=median_filter(Deff_xy, medFiltSize)
    
    #Normalization of the result and restrict to thresholds
    a1=np.mean(np.mean([Deff_xx,Deff_yy,Deff_xy]))
    b1=np.mean(np.std([Deff_xx,Deff_yy,Deff_xy]))
    Deff_xx=((Deff_xx-a1)/(3*b1))
    Deff_yy=((Deff_yy-a1)/(3*b1))
    Deff_xy=((Deff_xy-a1)/(3*b1))
    print(b1)
    Deff_xx[Deff_xx>1]=1
    Deff_yy[Deff_yy>1]=1
    Deff_xy[Deff_xy>1]=1
    Deff_xx[Deff_xx<0]=0
    Deff_yy[Deff_yy<0]=0
    Deff_xy[Deff_xy<0]=0
    
    #Trying to create a coloured image from tensor (method probably wrong for now)
    colouredImage=np.zeros((( Nx, Ny,3)))
    colouredImage[:,:,0]=abs(Deff_xx)
    colouredImage[:,:,1]=abs(Deff_yy)
    colouredImage[:,:,2]=abs(Deff_xy)
    colouredImage[colouredImage>1]=1
    
    #colouredImage=mpl.colors.hsv_to_rgb(colouredImage)

    return {'phi': phi, 'Deff_xx': Deff_xx, 'Deff_yy': Deff_yy, 'Deff_xy': Deff_xy, 'ColoredDeff': colouredImage}

