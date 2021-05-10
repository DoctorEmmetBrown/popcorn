# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:46:27 2021.

@author: quenot
"""
from pagailleIO import saveEdf,openImage,openSeq,save3D_Edf
import glob
from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
from numpy.fft import fftfreq as fftfreq
import numpy as np
from math import pi as pi
from math import floor as floor
import frankoChellappa  as fc
from scipy.ndimage import gaussian_filter

def derivativesByOpticalflow(intensityImage,derivative,sig_scale=0):

    epsilon=np.finfo(float).eps
    Nx, Ny = derivative.shape #Image size
    dImX=np.zeros((Nx,Ny))
    dImY=np.zeros((Nx,Ny))
    
    # fourier transfomm of the derivative and shift low frequencies to the centre
    ftdI = fftshift(fft2(derivative)) #Fourier transform of the derivative
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
    dImX = 1. / intensityImage * ifft2(ifftshift(ftfiltX * ftdI)) #Displacement field
    dImY = 1. / intensityImage * ifft2(ifftshift(ftfiltY * ftdI))
    
    dX=np.median(dImX.real, axis=0)
    dY=np.median(dImY.real, axis=0)

    return dX, dY



def processProjectionOpticalFlow2020(image1, image2, sigma):
    
    
    Difference=image1-image2
    
    dx, dy = derivativesByOpticalflow(image2, Difference, sig_scale=sigma)
    

    return dx, dy

if __name__ == "__main__":

    image1=np.ones((20,20))#Open image 1
    image2=np.ones((20,20))*2#image 2
    sigma=0# Filtrage basses frequences
    
    Dx, Dy=processProjectionOpticalFlow2020(image1, image2, sigma)

