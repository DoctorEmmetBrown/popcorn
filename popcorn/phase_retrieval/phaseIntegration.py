from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
import numpy as np
from math import pi as pi
from math import floor as floor



def kottler(dX,dY):
    """ Image integration from 2D gradient for tomographic reconstruction
    Note:
        Kottler, C., David, C., Pfeiffer, F., & Bunk, O. (2007). A two-directional approach for grating based differential phase contrast imaging using hard x-rays. 

    Args:
        dX (NUMPY ARRAY): Gradient allong x.
        dY (NUMPY ARRAY): Gradient allong y.

    Returns:
        NUMPY ARRAY: integrated image.

    """
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
    """Image integration from 2D gradient
    
    Note:
        Arnison, M. R., Larkin, K. G., Sheppard, C. J. R., Smith, N. I., & Cogswell, C. J. (2004). Linear phase imaging using differential interference contrast microscopy. Journal of Microscopy, 214(1), 7–12. 

    Args:
        dx (Numpy array): Gradient allong x.
        dy (Numpy array): Gradient allong y.

    Returns:
        phi (Numpy array): integrated image.

    """
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

