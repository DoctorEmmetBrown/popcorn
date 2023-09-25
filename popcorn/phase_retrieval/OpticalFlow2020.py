# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:46:27 2021.

@author: quenot

Modified by Fardin on 25th Sep 2023: 
                    Corrected frequencies in the Fourier filter
                    Removed for loop on the number of images
                    Added new gradient integration methods
"""

from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
from numpy.fft import fftfreq as fftfreq
import numpy as np
from math import pi as pi
import frankoChellappa  as fc
from scipy.ndimage import gaussian_filter
from phase_integration import fourier_integration, ls_integration

def derivativesByOpticalflow(intensityImage,derivative,sig_scale=0):

    # Optical flow (OF) algorithm as described in Paganin et al., 2018.
    # A gaussian shaped high pass filter is implemented, to reduce the low frequency artefacts introduced by
    # the OF algorithm. The gaussian filter should be asymmetric, but here the same width (cut-off frequency) is used for X and Y.
    # The width is controlled by two parameters: dqx (frequency sampling) and sig_scale (selected by the user).
    # An asymmetric gaussian filter would require one sig_scale per dimension, to better control the cut-off frequencies: dqy
    # is in fact not always equal to dqx.


    epsilon=np.finfo(float).eps
    Nim, Ny, Nx = derivative.shape #Image size
    dImX=np.zeros((Nim,Ny,Nx),dtype='complex')
    dImY=np.zeros((Nim,Ny,Nx),dtype='complex')


    # fourier transform of the derivative
    ftdI = fft2(derivative)
    # calculate frequency sampling
    dqx = 2 * pi / (Nx)
    dqy = 2 * pi / (Ny)

    kx = 2*np.pi*fftfreq(Nx) 
    ky = 2*np.pi*fftfreq(Ny) 

    Qx, Qy = np.meshgrid(kx,ky)
    Qx2,Qy2 = np.meshgrid(kx**2,ky**2)

    #building filter in frequency space
    if(sig_scale!=0):
        sigmaX = 2 * (dqx*sig_scale)**2 
        sigmaY = sigmaX
        #sigmaY = 2 * (dqy*sig_scale)**2      
        g = np.exp(-((Qx2) / sigmaX + (Qy2) / sigmaY))
        beta = 1 - g
    else:
        beta = 1 

    # fourier filters
    ftfiltX = (beta * Qx / (Qx2 + Qy2 + epsilon))
    ftfiltX[0,0]=0
    ftfiltY = (beta * Qy/ (Qx2 + Qy2 + epsilon))
    ftfiltY[0,0]=0
    
    # output calculation
    dImX = (1j / intensityImage) * ifft2(np.array([ftfiltX,]*Nim) * ftdI) #Displacement field
    dImY = (1j / intensityImage) * ifft2(np.array([ftfiltY,]*Nim) * ftdI)
    dX=np.median(dImX.real, axis=0)
    dY=np.median(dImY.real, axis=0)

    return dX, dY


def processProjectionOpticalFlow2020(experiment):
    if experiment.absorption_correction_sigma!=0:
        absMask=gaussian_filter(np.median(experiment.sample_images,axis=0),experiment.absorption_correction_sigma)/gaussian_filter(np.median(experiment.reference_images,axis=0),experiment.absorption_correction_sigma)
    else:
        absMask=1
    subImage=experiment.sample_images/absMask-experiment.reference_images
    dx, dy = derivativesByOpticalflow(experiment.reference_images, subImage, sig_scale=experiment.sigma_regularization)
    #The displacement is given in pixel units. It is multiplied by the physical pixel size when it is converted to phase gradient
    dphix=dx*(experiment.pixel/experiment.dist_object_detector)*experiment.getk()
    dphiy=dy*(experiment.pixel/experiment.dist_object_detector)*experiment.getk()    #For the integration, the effective (magnified) pixel size is used. This is the sampling of the gradient.
    magnificationFactor = (experiment.dist_object_detector + experiment.dist_source_object) / experiment.dist_source_object
    gradientSampling = experiment.pixel / magnificationFactor
    phiFC = fc.frankotchellappa(dphix, dphiy, True)*gradientSampling
    phiK = fourier_integration.fourier_solver(dphix, dphiy, gradientSampling, gradientSampling, solver='kottler')
    #phiLS = ls_integration.least_squares(dphix, dphiy, gradientSampling, gradientSampling, model='southwell')

    return {'dx': dx, 'dy': dy, 'phiFC': phiFC.real, 'phiK': phiK} #, 'phiLS': phiLS}

