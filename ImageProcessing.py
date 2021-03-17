#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:27:17 2021

@author: quenot
"""
from skimage import restoration
import numpy as np

def deconvolve(image, sigma, deconv_type):
    """Deconvolution of the acquisitions to correct detector's PSF using unsupervised wiener with a given sigma."""
    Nblur=int(np.floor(sigma*6))
    x,y= np.meshgrid(np.arange(0, Nblur), np.arange(0, Nblur))
    x = (x - ((Nblur-1) / 2))
    y = (y - ((Nblur-1) / 2))
    blur=np.exp(-(x**2+y**2)/sigma**2/2)
    blur=blur/np.sum(blur)
    MAX=np.max(image)
    # Restore image using unsupervised_wiener algorithm
    if deconv_type=='unsupervised_wiener':
        restored_image, _ = restoration.unsupervised_wiener(image/MAX, blur, clip=False)
    # Restore image using Richardson-Lucy algorithm
    elif deconv_type=='richardson_lucy':
        restored_image = restoration.richardson_lucy(image/MAX, blur, iterations=10, clip=False)
    else:
        raise Exception('Filter not found')
    restored_image=restored_image*MAX
    return restored_image