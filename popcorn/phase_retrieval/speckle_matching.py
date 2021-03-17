# -*- coding: utf-8 -*-
"""
Speckle matching.

Author: Pierre Thibault
Date: July 2015
"""

import numpy as np
from scipy import signal as sig
import frankoChellappa  as fc
from OpticalFlow2020 import kottler, LarkinAnissonSheppard


def processProjectionUMPA(experiment):

    nbImages, Nx, Ny= experiment.sample_images.shape
    
    result = match_speckles(experiment.sample_images, experiment.reference_images, Nw=experiment.umpaNw, step=1, max_shift=experiment.max_shift, df=False)
    dx=-result['dx']
    dy=-result['dy']
    thickness=result['T']
    df=result['df']
    f=result['f']
    dphix=dx*(experiment.pixel/experiment.dist_object_detector)*experiment.getk()
    dphiy=dy*(experiment.pixel/experiment.dist_object_detector)*experiment.getk()
    
    padForIntegration=True
    padSize=300
    if padForIntegration:
        dphix = np.pad(dphix, ((padSize, padSize), (padSize, padSize)),mode='reflect')  # voir is edge mieux que reflect
        dphiy = np.pad(dphiy, ((padSize, padSize), (padSize, padSize)),mode='reflect')  # voir is edge mieux que reflect

    #Compute the phase from phase gradients with 3 different methods (still trying to choose the best one)
    phiFC = fc.frankotchellappa(dphiy, dphix, True)*experiment.pixel
    phiK = kottler(dphiy, dphix)*experiment.pixel
    phiLA = LarkinAnissonSheppard(dphiy, dphix)*experiment.pixel
    
    if padSize > 0:
        phiFC = phiFC[padSize:padSize + Nx, padSize:padSize + Ny]
        phiK = phiK[padSize:padSize + Nx , padSize:padSize + Ny]
        phiLA = phiLA[padSize:padSize + Nx, padSize:padSize + Ny]

    return {'dx': dx, 'dy': dy, 'phiFC': phiFC.real, 'phiK': phiK.real,'phiLA': phiLA.real, 'thickness':thickness, 'df':df, 'f':f}


def match_speckles(Isample, Iref, Nw, step=1, max_shift=1, df=True, printout=True):
    """
    Compare speckle images with sample (Isample) and w/o sample
    (Iref) using a given window.
    max_shift can be set to the number of pixels for an "acceptable"
    speckle displacement.

    :param Isample: A list  of measurements, with the sample aligned but speckles shifted
    :param Iref: A list of empty speckle measurements with the same displacement as Isample.
    :param Nw: 2*Nw + 1 is the width of the window.
    :param step: perform the analysis on every other _step_ pixels in both directions (default 1)
    :param max_shift: Do not allow shifts larger than this number of pixels (default 4)
    :param df: Compute dark field (default True)

    Return T, dx, dy, df, f
    """

    Ish = Isample[0].shape

    # Create the window
    w = np.multiply.outer(np.hamming(2*Nw+1), np.hamming(2*Nw+1))
    w /= w.sum()

    NR = len(Isample)

    S2 = sum(I**2 for I in Isample)
    R2 = sum(I**2 for I in Iref)
    if df:
        S1 = sum(I for I in Isample)
        R1 = sum(I for I in Iref)
        Im = R1.mean()/NR

    L1 = cc(S2, w)
    L3 = cc(R2, w)
    if df:
        L2 = Im * Im * NR
        L4 = Im * cc(S1, w)
        L6 = Im * cc(R1, w)
    # (We need a loop for L5)

    # 2*Ns + 1 is the width of the window explored to find the best fit.
    Ns = max_shift

    ROIx = np.arange(Ns+Nw, Ish[0]-Ns-Nw-1, step)
    ROIy = np.arange(Ns+Nw, Ish[1]-Ns-Nw-1, step)

    # The final images will have this size
    sh = (len(ROIx), len(ROIy))
    tx = np.zeros(sh)
    ty = np.zeros(sh)
    tr = np.zeros(sh)
    do = np.zeros(sh)
    MD = np.zeros(sh)

    # Loop through all positions
    for xi, i in enumerate(ROIx):
        if printout:
            print ('line %d, %d/%d' % (i, xi, sh[0]))
        for xj, j in enumerate(ROIy):
            # Define local values of L1, L2, ...
            t1 = L1[i, j]
            t3 = L3[(i-Ns):(i+Ns+1), (j-Ns):(j+Ns+1)]
            if df:
                t2 = L2
                t4 = L4[i, j]
                t6 = L6[(i-Ns):(i+Ns+1), (j-Ns):(j+Ns+1)]
            else:
                t2 = 0.
                t4 = 0.
                t6 = 0.

            # Now we can compute t5 (local L5)
            t5 = np.zeros((2*Ns+1, 2*Ns+1))
            for k in range(NR):
                t5 += cc(Iref[k][(i-Nw-Ns):(i+Nw+Ns+1), (j-Nw-Ns):(j+Nw+Ns+1)],
                         w * Isample[k][(i-Nw):(i+Nw+1), (j-Nw):(j+Nw+1)], mode='valid')

            # Compute K and beta
            if df:
                K = (t2*t5 - t4*t6)/(t2*t3 - t6**2)
                beta = (t3*t4 - t5*t6)/(t2*t3 - t6**2)
            else:
                K = t5/t3
                beta = 0.

            # Compute v and a
            a = beta + K
            v = K/a

            # Construct D
            D = t1 + (beta**2)*t2 + (K**2)*t3 - 2*beta*t4 - 2*K*t5 + 2*beta*K*t6

            # Find subpixel optimum for tx an ty
            sx, sy = sub_pix_min(D)

            # We should re-evaluate the other values with sub-pixel precision but here we just round
            # We also need to clip because "sub_pix_min" can return the position of the minimum outside of the bounds...
            isy = np.clip(int(np.round(sy)), 0, 2*Ns)
            isx = np.clip(int(np.round(sx)), 0, 2*Ns)

            # store everything
            ty[xi, xj] = sy - Ns
            tx[xi, xj] = sx - Ns
            tr[xi, xj] = a[isy, isx]
            do[xi, xj] = v[isy, isx]
            MD[xi, xj] = D[isy, isx]

    return {'T': tr, 'dx': tx, 'dy': ty, 'df': do, 'f': MD}


def cc(A, B, mode='same'):
    """
    A fast cross-correlation based on scipy.signal.fftconvolve.

    :param A: The reference image
    :param B: The template image to match
    :param mode: one of 'same' (default), 'full' or 'valid' (see help for fftconvolve for more info)
    :return: The cross-correlation of A and B.
    """
    return sig.fftconvolve(A, B[::-1, ::-1], mode=mode)


def quad_fit(a):
    """\
    (c, x0, H) = quad_fit(A)
    Fits a parabola (or paraboloid) to A and returns the
    parameters (c, x0, H) such that

    a ~ c + (x-x0)' * H * (x-x0)

    where x is in pixel units. c is the value at the fitted optimum, x0 is
    the position of the optimum, and H is the hessian matrix (curvature in 1D).
    """

    sh = a.shape

    i0, i1 = np.indices(sh)
    i0f = i0.flatten()
    i1f = i1.flatten()
    af = a.flatten()

    # Model = p(1) + p(2) x + p(3) y + p(4) x^2 + p(5) y^2 + p(6) xy
    #       = c + (x-x0)' h (x-x0)
    A = np.vstack([np.ones_like(i0f), i0f, i1f, i0f**2, i1f**2, i0f*i1f]).T
    r = np.linalg.lstsq(A, af)
    p = r[0]
    x0 = - (np.matrix([[2*p[3], p[5]], [p[5], 2*p[4]]]).I * np.matrix([p[1], p[2]]).T).A1
    c = p[0] + .5*(p[1]*x0[0] + p[2]*x0[1])
    h = np.matrix([[p[3], .5*p[5]], [.5*p[5], p[4]]])
    return c, x0, h


def quad_max(a):
    """\
    (c, x0) = quad_max(a)

    Fits a parabola (or paraboloid) to A and returns the
    maximum value c of the fitted function, along with its
    position x0 (in pixel units).
    All entries are None upon failure. Failure occurs if :
    * A has a positive curvature (it then has a minimum, not a maximum).
    * A has a saddle point
    * the hessian of the fit is singular, that is A is (nearly) flat.
    """

    c, x0, h = quad_fit(a)

    failed = False
    if a.ndim == 1:
        if h > 0:
            print('Warning: positive curvature!')
            failed = True
    else:
        if h[0, 0] > 0:
            print('Warning: positive curvature along first axis!')
            failed = True
        elif h[1, 1] > 0:
            print('Warning: positive curvature along second axis!')
            failed = True
        elif np.linalg.det(h) < 0:
            print('Warning: the provided data fits to a saddle!')
            failed = True

    if failed:
        c = None
    return c, x0


def pshift(a, ctr):
    """\
    Shift an array so that ctr becomes the origin.
    """
    sh  = np.array(a.shape)
    out = np.zeros_like(a)

    ctri = np.floor(ctr).astype(int)
    ctrx = np.empty((2, a.ndim))
    ctrx[1,:] = ctr - ctri     # second weight factor
    ctrx[0,:] = 1 - ctrx[1,:]  # first  weight factor

    # walk through all combinations of 0 and 1 on a length of a.ndim:
    #   0 is the shift with shift index floor(ctr[d]) for a dimension d
    #   1 the one for floor(ctr[d]) + 1
    comb_num = 2**a.ndim
    for comb_i in range(comb_num):
        comb = np.asarray(tuple(("{0:0" + str(a.ndim) + "b}").format(comb_i)), dtype=int)

        # add the weighted contribution for the shift corresponding to this combination
        cc = ctri + comb
        out += np.roll( np.roll(a, -cc[1], axis=1), -cc[0], axis=0) * ctrx[comb,range(a.ndim)].prod()

    return out


def sub_pix_min(a, width=1):
    """
    Find the position of the minimum in 2D array a with subpixel precision (using a paraboloid fit).
    :param a:
    :param width: 2*width+1 is the size of the window to apply the fit.
    :return:
    """

    sh = a.shape

    # Find the global minimum
    cmin = np.array(np.unravel_index(a.argmin(), sh))

    # Move away from edges
    if cmin[0] < width:
        cmin[0] = width
    elif cmin[0]+width >= sh[0]:
        cmin[0] = sh[0] - width - 1
    if cmin[1] < width:
        cmin[1] = width
    elif cmin[1]+width >= sh[1]:
        cmin[1] = sh[1] - width - 1

    # Sub-pixel minimum position.
    mindist, r = quad_max(-np.real(a[(cmin[0]-width):(cmin[0]+width+1), (cmin[1]-width):(cmin[1]+width+1)]))
    r -= (width - cmin)

    return r


if __name__ == "__main__":
    import numpy as np
    from scipy import ndimage as ndi
    import scipy

    def free_nf(w, l, z, pixsize=1.):
        """\
        Free-space propagation (near field) of the wavefield of a distance z.
        l is the wavelength. 
        """
        if w.ndim != 2:
            raise RunTimeError("A 2-dimensional wave front 'w' was expected")

        sh = w.shape

        # Convert to pixel units.
        z = z / pixsize
        l = l / pixsize

        # Evaluate if aliasing could be a problem
        if min(sh)/np.sqrt(2.) < z*l:
            print ("Warning: z > N/(sqrt(2)*lamda) = %.6g: this calculation could fail." % (min(sh)/(l*np.sqrt(2.))) )
            print ("(consider padding your array, or try a far field method)" ) 

        q2 = np.sum((np.fft.ifftshift(np.indices(sh).astype(float) - np.reshape(np.array(sh)//2,(len(sh),) + len(sh)*(1,)), range(1,len(sh)+1)) * np.array([1./sh[0], 1./sh[1]]).reshape((2,1,1)))**2, axis=0)

        return np.fft.ifftn(np.fft.fftn(w) * np.exp(2j * np.pi * (z / l) * (np.sqrt(1 - q2*l**2) - 1) ) )
    
    # Simulation of a sphere
    sh = (512, 512)
    ssize = 2.    # rough speckle size
    sphere_radius = 150
    lam = .5e-10  # wavelength
    z = 5e-2      # propagation distance
    psize = 1e-6  # pixel size

    # Simulate speckle pattern
    speckle = ndi.gaussian_filter(np.random.normal(size=sh), ssize) +\
              1j * ndi.gaussian_filter(np.random.normal(size=sh), ssize)
    xx, yy = np.indices(sh)
    sphere = np.real(scipy.sqrt(sphere_radius**2 - (xx-256.)**2 - (yy-256.)**2))
    sample = np.exp(-15*np.pi*2j*sphere/sphere_radius)

    # Measurement positions
    #pos = np.array( [(0., 0.)] + [(np.round(15.*cos(pi*j/3)), np.round(15.*sin(pi*j/3))) for j in range(6)] )
    pos = 4*np.indices((5, 5)).reshape((2, -1)).T

    # Simulate the measurements
    measurements = np.array([abs(free_nf(sample*pshift(speckle, p), lam, z, psize))**2 for p in pos])
    reference = abs(free_nf(speckle, lam, z, psize))**2
    sref = [pshift(reference, p) for p in pos]

    result = match_speckles(measurements, sref, Nw=1, step=2)

