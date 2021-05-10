# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:46:27 2021.

@author: quenot
"""
from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
from math import pi as pi
import numpy as np
from scipy.ndimage import gaussian_filter


def kevToLambda(energyInKev):
    """Calculation of the wavelength in keV from the wavelength
    

    Args:
        energyInKev (float): energy in keV.

    Returns:
        float: wavelength in m.

    """
    energy = energyInKev * 1e3
    waveLengthInNanometer = 1240. / energy
    return waveLengthInNanometer * 1e-9

def tie_Pavlovetal2020(experiment):
    """Calculates sample thickness from the experiment
    
    Note:
        Pavlov, K. M., Li, H. (Thomas), Paganin, D. M., Berujon, S., Rougé-Labriet, H., & Brun, E. (2020). Single-Shot X-Ray Speckle-Based Imaging of a Single-Material Object. Physical Review Applied, 13(5), 054023.

    Args:
        experiment (Phase Retrieval class): contains images and all parameters.

    Returns:
        img_thickness (Numpy array): calculated thickness
    """
    
    
    lambda_energy = kevToLambda(experiment.energy)
    pix_size = kevToLambda(experiment.pixel)
    delta = experiment.delta
    beta = experiment.beta

    absMask=gaussian_filter(np.median(experiment.sample_images),experiment.absorption_correction_sigma)/gaussian_filter(np.median(experiment.reference_images),experiment.absorption_correction_sigma)

    waveNumber = (2 * pi) / lambda_energy
    mu = 2 * waveNumber * beta
    magnificationFactor = (experiment.dist_object_detector + experiment.dist_source_object) / experiment.dist_source_object
    pix_size=pix_size*magnificationFactor
    sigmaSource = experiment.source_size
    gamma = delta / beta

    Is_divided_by_Ir = np.true_divide(experiment.sample_images*absMask, experiment.reference_images)
    #experiment.sample_images_divided_by_Ir = np.true_divide(experiment.sample_images , Ir)
    if experiment.reference_images.ndim>2:
        Is_divided_by_Ir=np.median(Is_divided_by_Ir,axis=0)

    numerator = 1 - Is_divided_by_Ir


    fftNumerator = fftshift(fft2(numerator))
    Nx, Ny = fftNumerator.shape
    u, v = np.meshgrid(np.arange(0, Nx), np.arange(0, Ny))
    u = (u - (Nx / 2))
    v = (v - (Ny / 2))

    u_m=  u / (Nx * pix_size)
    v_m = v / (Ny * pix_size)
    uv_sqr=  np.transpose(u_m ** 2 + v_m ** 2)  # ie (u2+v2)

    # without taking care of source size
    denominator = 1 + pi * gamma * experiment.dist_object_detector * lambda_energy * uv_sqr

    # Beltran et al method to deblur with source
    #denominator = 1 + pi * (gamma * experiment['distOD'] - waveNumber * sigmaSource * sigmaSource) * lambda_energy * uv_sqr

#    denominator *= magnificationFactor
    tmp = fftNumerator / denominator

    # Low pass filter
    # building filters
    dqx = 2 * pi / (Nx)
    dqy = 2 * pi / (Ny)
    sig_scale=experiment.sigma_regularization
    if sig_scale==0:
        lff=1
    else:
        sigmaX = dqx / 1. * np.power(sig_scale, 2)
        sigmaY = dqy / 1. * np.power(sig_scale, 2)
        # sigmaX=sig_scale
        # sigmaY = sig_scale
    
        g = np.exp(-(((Nx) ** 2) / 2. / sigmaX + ((Ny) ** 2) / 2. / sigmaY))
        beta = 1 - g;

        #sigma_x = ((1/ (Nx * pix_size)) * scale) ** 2
        #sigma_y = ((1/ (Ny * pix_size)) * scale) ** 2
        #f = (1. - np.exp(-(u_m ** 2 / (2. * sigma_x) + v_m ** 2 / (2. * sigma_y))))  # ie f(x,y)
        lff = np.transpose(beta)  # ie LFF

    # Application of the Low pass filter
    tmp = lff * tmp

    # inverse fourier transform
    tmpThickness = ifft2(ifftshift(tmp))  # F-1
    img_thickness = np.real(tmpThickness)
    # Diision by mu
    img_thickness = img_thickness / mu
    # multiplication to be in micron
    img_thickness = img_thickness * 1e6

    return img_thickness



