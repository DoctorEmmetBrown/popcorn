import os
import sys

from pagailleIO import openImage, saveEdf
from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
from math import pi as pi
import numpy as np
from math import floor as floor

import glob
# from NoiseTracking.OpticalFlow import pavlovThread
import sys


def kevToLambda(energyInKev):
    energy = energyInKev * 1e3
    waveLengthInNanometer = 1240. / energy
    return waveLengthInNanometer * 1e-9

def tie_Pavlovetal2020(Is,Ir,absMask,expParam):
    lambda_energy = kevToLambda(expParam.energy)
    pix_size = kevToLambda(expParam.pixel)
    delta = expParam.delta
    beta = expParam.beta

    waveNumber = (2 * pi) / lambda_energy
    mu = 2 * waveNumber * beta
    magnificationFactor = (expParam.dist_object_detector + expParam.dist_sample_object) / expParam.dist_sample_object
    pix_size=pix_size*magnificationFactor
    sigmaSource = expParam.source_size
    gamma = delta / beta

    is_divided_by_Ir = np.true_divide(Is*absMask, Ir)
    #is_divided_by_Ir = np.true_divide(Is , Ir)
    if Ir.ndim>2:
        is_divided_by_Ir=np.median(is_divided_by_Ir,axis=0)

    numerator = 1 - is_divided_by_Ir


    fftNumerator = fftshift(fft2(numerator))
    Nx, Ny = fftNumerator.shape
    u, v = np.meshgrid(np.arange(0, Nx), np.arange(0, Ny))
    u = (u - (Nx / 2))
    v = (v - (Ny / 2))

    u_m=  u / (Nx * pix_size)
    v_m = v / (Ny * pix_size)
    uv_sqr=  np.transpose(u_m ** 2 + v_m ** 2)  # ie (u2+v2)

    # without taking care of source size
    denominator = 1 + pi * gamma * expParam.dist_object_detector * lambda_energy * uv_sqr

    # Beltran et al method to deblur with source
    #denominator = 1 + pi * (gamma * expParam['distOD'] - waveNumber * sigmaSource * sigmaSource) * lambda_energy * uv_sqr

#    denominator *= magnificationFactor
    tmp = fftNumerator / denominator

    # Low pass filter
    # building filters
    dqx = 2 * pi / (Nx)
    dqy = 2 * pi / (Ny)
    sig_scale=expParam.sigma_regularization
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
    # Division by mu
    img_thickness = img_thickness / mu
    # multiplication to be in micron
    img_thickness = img_thickness * 1e6

    return img_thickness



