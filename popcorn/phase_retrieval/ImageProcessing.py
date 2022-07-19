#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:27:17 2021

@author: quenot
"""
from skimage import restoration
from popcorn.phase_retrieval.pagailleIO import saveEdf, openSeq, openImage
import numpy as np
import glob

def deconvolve(image, sigma, deconv_type):
    """Deconvolution of the acquisitions to correct detector's PSF using unsupervised wiener with a given sigma.
    
    Args:
        image (Numpy array): images to deconvolve.
        sigma (float): standard deviaition of the blurring.
        deconv_type (string): 'unsupervised_wiener' or 'richardson_lucy'.

    Raises:
        Exception: 'Filter not found'.

    Returns:
        restored_image (Numpy array): image after deconvolution.

    """
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



if __name__ == "__main__":
    
    FilesPath='/Users/quenot/Data/SIMAP/MoucheSimapAout2017/'
    refFolder=FilesPath+"ref"
    sampleFolder=FilesPath+"sample"
    refDestination='/Users/quenot/Data/SIMAP/MoucheSimapAout2017/Deconvolved/ref/'
    sampleDestination='/Users/quenot/Data/SIMAP/MoucheSimapAout2017/Deconvolved/sample/'
    deconvolutionType='unsupervised_wiener'
    sigma=1.2
    
    
    refImages = glob.glob(refFolder + '/*.tif') + glob.glob(refFolder + '/*.tiff') + glob.glob(refFolder + '/*.edf')
    refImages.sort()
    sampImages = glob.glob(sampleFolder + '/*.tif') + glob.glob(sampleFolder + '/*.tiff') + glob.glob(
        sampleFolder + '/*.edf')
    sampImages.sort()
    print("Nb of points limited to ", len(refImages))
    Ir = openSeq(refImages)
    Is = openSeq(sampImages)
    for i in range(len(refImages)):
        IrDec=deconvolve(Ir[i], sigma, deconvolutionType)
        IsDec=deconvolve(Is[i], sigma, deconvolutionType)
        txtPoint = '%2.2d' % i
        saveEdf(IrDec, refDestination + 'ref_'+txtPoint+'.edf')
        saveEdf(IsDec, sampleDestination + 'sample_'+txtPoint+'.edf')

    
    
    