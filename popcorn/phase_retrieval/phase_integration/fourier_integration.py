'''
Implementation of the Antisymmetric Derivative Integration algorithm described in:

Bon P., S. Monneret, B. Wattellier, Noniterative boundary-artifact-free wavefront reconstruction from its derivatives,
Applied Optics, 2012

Main Function: fourier()

@Author: Luca Fardin
@Date: 20/02/2023
'''

import numpy as np

def antisym(gx,gy):
    #Antisymmetrization of the gradient matrices as described in Bon et al, 2015
    #
    # Input:  gx, gy : gradient along x (h) and y (v) respectively
    # Output: antisym_gx, antisy_gy  : antisymmetric gradient matrices 

    antisym_gx = np.block([[gx,-gx[:,::-1]],[gx[::-1,:],-gx[::-1,::-1]]])
    antisym_gy = np.block([[gy,gy[:,::-1]],[-gy[::-1,:],-gy[::-1,::-1]]])
    return antisym_gx, antisym_gy

def mirrored(gx,gy):
    #Symmetrization of the gradient matrices which introduces low frequency artefacts
    mid_gx = np.block([[gx[::-1,::-1],gx[::-1,::]],[gx[::,::-1],gx]])
    mid_gy = np.block([[gy[::-1,::-1],gy[::-1,::]],[gy[::,::-1],gy]])
    return mid_gx, mid_gy

def kottler(gx,gy):
    #Implementation of Kottler et al, Opt Express 2007
    #
    # Input:  gx, gy : gradient along x (h) and y (v) respectively
    # Output: phase  : real part of the reconstructed ifft phase 

    #print("Kottler Solver")

    Gx = np.fft.fft2(gx)
    Gy = np.fft.fft2(gy)

    fx=np.fft.fftfreq(np.shape(gx)[1])#,px)
    fy=np.fft.fftfreq(np.shape(gx)[0])#,py)
    ffx,ffy = np.meshgrid(fx,fy)

    f_num = Gx + 1j * Gy
    f_den = 1j*2*np.pi*(ffx + 1j * ffy) + np.finfo(float).eps
    #Set zero frequency to zero
    f_phase = f_num / f_den
    f_phase[0,0] = 0
    phase = np.fft.ifft2(f_phase)

    return np.real(phase)


def frankot_chellappa(gx,gy):
    #Implementation of Frankot et Chellappa, IEEE 1988
    #
    # Input:  gx, gy : gradient along x (h) and y (v) respectively
    # Output: phase  : real part of the reconstructed ifft phase 

    Gx = np.fft.fft2(gx)
    Gy = np.fft.fft2(gy)

    fx=np.fft.fftfreq(np.shape(gx)[1])
    fy=np.fft.fftfreq(np.shape(gx)[0])
    ffx,ffy = np.meshgrid(fx,fy)
    ffx2,ffy2 = np.meshgrid(fx**2,fy**2)

    f_num = -1j* (ffx * Gx + ffy * Gy)
    f_den = 2*np.pi*(ffx2 +  ffy2)+np.finfo(float).eps
    #Set zero frequency to zero
    f_phase = f_num / f_den
    f_phase[0,0] = 0
    phase = np.fft.ifft2(f_phase)

    return np.real(phase)


def fourier_solver(gx,gy,px,py,solver='kottler',padding=True):
    
    # This is an implementation of the Antisymmetric Derivative Integration algorithm
    # The algorithm creates a symmetric phase, thus turning the  Descrete Fourier Transfrom into a Discrete Cosine Transform
    # A common Fourier solver (Frankot_Chellappa or Kottler) can then be applied
    # Input:  gx, gy : gradient along x (h) and y (v) respectively
    #         px, py : pixel size
    #         solver : Fourier Solver 'kottler','frankot_chellappa'
    # Output: phase  :Reconstructed phase image 


    print('Cosine transform based on Bon et al. 2015')
    print('The solver {} was chosen'.format(solver))

    #1st step: Normalize the gradient vectors based on the pixel size
    gxn=gx*px
    gyn=gy*py

    #Antisymmetrization
    if padding:
        gxn, gyn =antisym(gxn,gyn)
    #Solution

    #globals() retrieves the function required by the user and calls it
    solver_func = globals()[solver]
    phase_ext = solver_func(gxn,gyn)
    size_x, size_y = np.shape(gx)
    phase = phase_ext[:size_x,:size_y]
    return phase
