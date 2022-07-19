# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:46:27 2021.

@author: quenot
"""

import glob
from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
from numpy.fft import fftfreq as fftfreq
import numpy as np
from math import pi as pi
from math import floor as floor
from popcorn.phase_retrieval import frankoChellappa  as fc
from scipy.ndimage import gaussian_filter

def derivativesByOpticalflow(intensityImage,derivative,pixsize=1,sig_scale=0):

    epsilon=np.finfo(float).eps
    Nim, Nx, Ny = derivative.shape #Image size
    dImX=np.zeros(((Nim,Nx,Ny)))
    dImY=np.zeros(((Nim,Nx,Ny)))
    
    for i in range(Nim):
        # fourier transfomm of the derivative and shift low frequencies to the centre
        ftdI = fftshift(fft2(derivative[i])) #Fourier transform of the derivative
        # calculate frequencies
        dqx = 2 * pi / (Nx)
        dqy = 2 * pi / (Ny)
    
        Qx, Qy = np.meshgrid((np.arange(0, Ny) - floor(Ny / 2) - 1) * dqy, (np.arange(0, Nx) - floor(Nx / 2) - 1) * dqx) #frequency ranges of the images in fqcy space
    
        #building filters
        sigmaX = dqx / 1. * np.power(sig_scale,2)
        sigmaY = dqy / 1. * np.power(sig_scale,2)
        #sigmaX=sig_scale
        #sigmaY = sig_scale
    
        g = np.exp(-(((Qx)**2) / 2. / sigmaX + ((Qy)**2) / 2. / sigmaY))
        #g = np.exp(-(((np.power(Qx, 2)) / 2) / sigmaX + ((np.power(Qy, 2)) / 2) / sigmaY))
        beta = 1 - g;
    
        # fourier filters
        ftfiltX = (1j * Qx / ((Qx**2 + Qy**2))*beta)
        ftfiltX[np.isnan(ftfiltX)] = 0
        ftfiltX[ftfiltX==0]=epsilon
    
        ftfiltY = (1j* Qy/ ((Qx**2 + Qy**2))*beta)
        ftfiltY[np.isnan(ftfiltY)] = 0
        ftfiltX[ftfiltY==0] = epsilon
    
        # output calculation
        dImX[i] = 1. / intensityImage[i] * ifft2(ifftshift(ftfiltX * ftdI)) #Displacement field
        dImY[i] = 1. / intensityImage[i] * ifft2(ifftshift(ftfiltY * ftdI))
    
    dX=np.median(dImX.real, axis=0)
    dY=np.median(dImY.real, axis=0)

    return dX, dY



def kottler(dX,dY):
    print('kottler')
    i = complex(0, 1)
    Nx, Ny = dX.shape
    dqx = 2 * pi / (Nx)
    dqy = 2 * pi / (Ny)
    Qx, Qy = np.meshgrid((np.arange(0, Ny) - floor(Ny / 2) - 1) * dqy, (np.arange(0, Nx) - floor(Nx / 2) - 1) * dqx)

    polarAngle = np.arctan2(Qx, Qy)
    ftphi = fftshift(fft2(dX + i * dY))*np.exp(i*polarAngle)
    ftphi[np.isnan(ftphi)] = 0
    phi3 = ifft2(ifftshift(ftphi))
    return phi3.real



def LarkinAnissonSheppard(dx,dy):
    Nx, Ny = dx.shape
    i = complex(0, 1)
    G= dx + i*dy
    # fourier transfomm of the G function
    fourrierOfG = fftshift(fft2(G))


    dqx = 2 * pi / (Nx)
    dqy = 2 * pi / (Ny)
    Qx, Qy = np.meshgrid((np.arange(0, Ny) - floor(Ny / 2) - 1) * dqy, (np.arange(0, Nx) - floor(Nx / 2) - 1) * dqx)

    ftfilt = 1 / (i * Qx - Qy)
    ftfilt[np.isnan(ftfilt)] = 0.
    phi=ifft2(ifftshift(ftfilt*fourrierOfG))
    phi=phi.real
    return phi


def processProjectionOpticalFlow2020(experiment):
    if experiment.absorption_correction_sigma!=0:
        absMask=gaussian_filter(np.median(experiment.sample_images),experiment.absorption_correction_sigma)/gaussian_filter(np.median(experiment.reference_images),experiment.absorption_correction_sigma)
    else:
        absMask=1
    subImage=experiment.sample_images/absMask-experiment.reference_images
    dx, dy = derivativesByOpticalflow(experiment.reference_images, subImage,pixsize=experiment.pixel, sig_scale=experiment.sigma_regularization)
    dphix=dx*(experiment.pixel/experiment.dist_object_detector)
    dphiy=dy*(experiment.pixel/experiment.dist_object_detector)
    phiFC = fc.frankotchellappa(dphix, dphiy, False)
    phiK = kottler(dphix, dphiy)
    phiLA = LarkinAnissonSheppard(dphix, dphiy)

    return {'dx': dx, 'dy': dy, 'phiFC': phiFC.real, 'phiK': phiK.real,'phiLA': phiLA.real}

