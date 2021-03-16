#!/usr/bin/env python
# -*- coding: utf-8 -*-
# #########################################################################
# Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################


"""

Surface from gradient data
-------------------------------------------------

In many x-ray imaging techniques we obtain the differential phase of the
wavefront in two directions. This is the same as to say that we measured
the gradient of the phase. Therefore to obtain the phase we need a method to
integrate the differential data. Matematically we want to obtain the surface
:math:`s(x,y)` from the (experimental) differential curves
:math:`s_x = s_x(x,y) = \\frac{\\partial s (x, y)}{\\partial x}` and
:math:`s_y = s_y(x,y) = \\frac{\\partial s (x, y)}{\\partial y}`.


This is not as straight forward as it looks. The main reason is that, to be
able to calculate the function from its gradient, the partial derivatives must
be integrable. A function is integrable when the two cross partial
derivative of the function are equal, that is

    .. math::
        \\frac{\\partial^2 s (x, y)}{\\partial x \\partial y} =
        \\frac{\\partial^2 s (x, y)}{\\partial y \\partial x}



However, due to experimental errors and noises, there are no
guarantees that the data is integrable (very likely they are not).


To obtain a signal/image from differential information is a broader topic
with application in others topic of science. Few methods have been
developed in different context, in special
computer vision where this problem is refered as "Surface Reconstruction
from Gradient Fields". For consistense, we will (try to) stick to this same
term.

These methods try to find the best signal :math:`s(x,y)` that best describes
the differential curves. For this reason, it is advised to use some
kind of check for the integration, for instance by calculating the gradient
from the result and comparing with the original gradient. It is better if
this is done in the current library. See for instance the use of
:py:func:`wavepy.error_integration` in the function
:py:func:`wavepy.surdace_from_grad.frankotchellappa` below.

It is the goal for this library to add few different methods, since it is
clear that different methods have different strenghts and weakness
(precision, processing time, memory requirements, etc).


References
----------

:cite:`Frankot88`, :cite:`Agrawal06`, :cite:`Harker08`, :cite:`Sevcenco15`,
:cite:`Harker15`, :cite:`Huang15`.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
#import matplotlib.pyplot as plt
#import wavepy.utils as wpu

__authors__ = "Walan Grizolli"
__copyright__ = "Copyright (c) 2016-2017, Argonne National Laboratory"
__version__ = "0.1.0"
__docformat__ = "restructuredtext en"
__all__ = ['frankotchellappa', 'error_integration']


def frankotchellappa(del_f_del_x, del_f_del_y, reflec_pad=True):
    """

    The simplest method is the so-called Frankot-Chelappa method. The idea
    behind this method is to search (calculate) an integrable gradient field
    that best fits the data. Luckly, Frankot Chelappa were able in they article
    to find a simple single (non interective) equation for that. We are
    even luckier since this equation makes use of FFT, which is very
    computationally efficient.

    Considering a signal :math:`s(x,y)`
    with differential signal given by
    :math:`s_x = s_x(x,y) = \\frac{\\partial s (x, y)}{\\partial x}` and
    :math:`s_y = s_y(x,y) = \\frac{\\partial s (x, y)}{\\partial y}`. The
    Fourier Transform of :math:`s_x` and :math:`s_y` are given by

    .. math::
            \\mathcal{F} \\left [ s_x \\right ] =
            \\mathcal{F} \\left [ s_x \\right ] (f_x, f_y) =
            \\mathcal{F} \\left [ \\frac{\\partial s (x, y)}
            {\\partial x} \\right ](f_x, f_y), \\quad
            \\mathcal{F} \\left [ s_y \\right ] =
            \\mathcal{F} \\left [ s_y \\right ] (f_x, f_y) =
            \\mathcal{F} \\left [ \\frac{\\partial s (x, y)}
            {\\partial y} \\right ](f_x, f_y).



    Finally, Frankot-Chellappa method is base in solving the following
    equation:


    .. math::
            \\mathcal{F} \\left [ s \\right ] = \\frac{-i f_x \\mathcal{F}
            \\left [ s_x \\right ] - i f_y \\mathcal{F} \\left [ s_y
            \\right ]}{2 \\pi (f_x^2 + f_y^2 )}

    where

    .. math::
            \\mathcal{F} \\left [ s \\right ] = \\mathcal{F} \\left[
            s(x, y) \\right ] (f_x, f_y)

    is the Fourier Transform of :math:`s`.

    To avoid the singularity in the denominator, it is added
    :py:func:`numpy.finfo(float).eps`, the smallest float number in the
    machine.

    Keep in mind that Frankot-Chelappa is not the best method. More advanced
    alghorithms are available, where it is used more complex math and
    interactive methods. Unfortunatelly, these algorothims are only available
    for MATLAB.


    References
    ----------

        :cite:`Frankot88`. The padding we use here is :cite:`Huang15`.




    Parameters
    ----------

    del_f_del_x, del_f_del_y : ndarrays
        2 dimensional gradient data

    reflec_pad: bool
       This flag pad the gradient field in order to obtain a 2-dimensional
       reflected function. See more in the Notes below.

    Returns
    -------
    ndarray
        Integrated data, as provided by the Frankt-Chellappa Algorithm. Note
        that the result are complex numbers. See below


    Notes
    -----


    * Padding

        Frankt-Chellappa makes intensive use of the Discrete Fourier
        Transform (DFT), and due to the periodicity property of the DFT, the
        result of the integration will also be periodic (even though we
        only get one period of the answer). This property can result in a
        discontinuity at the edges, and Frankt-Chellappa method is badly
        affected by discontinuity,

        In this sense the idea of this padding is that by reflecting the
        function at the edges we avoid discontinuity. This was inspired by
        the code of the function
        `DfGBox
        <https://www.mathworks.com/matlabcentral/fileexchange/45269-dfgbox>`_,
        available in the MATLAB File Exchange website.

        Note that, since we only have the gradient data, we need to consider
        how a reflection at the edges will affect the partial derivatives. We
        show it below without proof (but it is easy to see).

        First lets consider the data for the :math:`x` direction derivative
        :math:`\\Delta_x = \\dfrac{\\partial f}{\\partial x}` consisting of a
        2D array of size :math:`N \\times M`. The padded matrix
        will be given by:

        .. math::
            \\left[
            \\begin{matrix}
              \\Delta_x(x, y) & -\\Delta_x(N-x, y) \\\\
              \\Delta_x(x, M-y) & -\\Delta_x(N-x, M-y)
            \\end{matrix}
            \\right]

        and for the for the y direction derivative
        :math:`\\Delta_y = \\dfrac{\\partial f}{\\partial y}` we have

        .. math::
            \\left[
            \\begin{matrix}
              \\Delta_y(x, y) & \\Delta_y(N-x, y) \\\\
              -\\Delta_y(x, M-y) & -\\Delta_y(N-x, M-y)
            \\end{matrix}
            \\right]

        Note that this padding increases the number of points from
        :math:`N \\times M` to :math:`2M \\times 2N`. However, **the function
        only returns the** :math:`N \\times M` **result**, since the
        other parts are only a repetion of the result. In other words,
        the padding is done only internally.


    * Results are Complex Numbers

        Again due to the use of DFT's, the results are complex numbers.
        In principle an ideal gradient field of real numbers results
        a real-only result. This "imaginary noise" is observerd
        even with theoretical functions, which leads to the conclusion
        that it is due to a numerical noise. It is left to the user
        to decide what to do with noise, for instance to use the
        modulus or the real part of the result. But it
        is recomended to use the real part.



    """

    from numpy.fft import fft2, ifft2, fftfreq

    if reflec_pad:
        del_f_del_x, del_f_del_y = _reflec_pad_grad_fields(del_f_del_x,
                                                           del_f_del_y)

    NN, MM = del_f_del_x.shape
    wx, wy = np.meshgrid(fftfreq(MM) * 2 * np.pi,
                         fftfreq(NN) * 2 * np.pi, indexing='xy')
    # by using fftfreq there is no need to use fftshift

    numerator = -1j * wx * fft2(del_f_del_x) - 1j * wy * fft2(del_f_del_y)

    denominator = (wx) ** 2 + (wy) ** 2 + np.finfo(float).eps

    res = ifft2(numerator / denominator)
    res -= np.mean(np.real(res))


    if reflec_pad:
        return _one_forth_of_array(res)
    else:
        return res



def _reflec_pad_grad_fields(del_func_x, del_func_y):
    """

    This fucntion pad the gradient field in order to obtain a 2-dimensional
    reflected function. The idea is that, by having an reflected function,
    we avoid discontinuity at the edges.


    This was inspired by the code of the function DfGBox, available in the
    MATLAB File Exchange website:
    https://www.mathworks.com/matlabcentral/fileexchange/45269-dfgbox

    """

    del_func_x_c1 = np.concatenate((del_func_x,
                                    del_func_x[::-1, :]), axis=0)

    del_func_x_c2 = np.concatenate((-del_func_x[:, ::-1],
                                    -del_func_x[::-1, ::-1]), axis=0)

    del_func_x = np.concatenate((del_func_x_c1, del_func_x_c2), axis=1)

    del_func_y_c1 = np.concatenate((del_func_y,
                                    -del_func_y[::-1, :]), axis=0)

    del_func_y_c2 = np.concatenate((del_func_y[:, ::-1],
                                    -del_func_y[::-1, ::-1]), axis=0)

    del_func_y = np.concatenate((del_func_y_c1, del_func_y_c2), axis=1)

    return del_func_x, del_func_y


def _one_forth_of_array(array):
    """
    Undo for the function
    :py:func:`wavepy:surface_from_grad:_reflec_pad_grad_fields`

    """

    array, _ = np.array_split(array, 2, axis=0)
    return np.array_split(array, 2, axis=1)[0]


def _grad(func):

    del_func_2d_x = np.diff(func, axis=1)
    del_func_2d_x = np.pad(del_func_2d_x, ((0, 0), (1, 0)), 'edge')

    del_func_2d_y = np.diff(func, axis=0)
    del_func_2d_y = np.pad(del_func_2d_y, ((1, 0), (0, 0)), 'edge')

    return del_func_2d_x, del_func_2d_y


def error_integration(del_f_del_x, del_f_del_y, func,
                      pixelsize, errors=False,
                      shifthalfpixel=False, plot_flag=True):

    func = np.real(func)

    if shifthalfpixel:
        func = wpu.shift_subpixel_2d(func, 2)

    xx, yy = wpu.realcoordmatrix(func.shape[1], pixelsize[1],
                                 func.shape[0], pixelsize[0])
    midleX = xx.shape[0] // 2
    midleY = xx.shape[1] // 2

    grad_x, grad_y = _grad(func)

    grad_x -= np.mean(grad_x)
    grad_y -= np.mean(grad_y)
    del_f_del_x -= np.mean(del_f_del_x)
    del_f_del_y -= np.mean(del_f_del_y)

    amp_x = np.max(del_f_del_x) - np.min(del_f_del_x)
    amp_y = np.max(del_f_del_y) - np.min(del_f_del_y)

    error_x = np.abs(grad_x - del_f_del_x)/amp_x*100
    error_y = np.abs(grad_y - del_f_del_y)/amp_y*100

    if plot_flag:
        plt.figure(figsize=(14, 10))

        ax1 = plt.subplot(221)
        ax1.ticklabel_format(style='sci', axis='both', scilimits=(0, 1))
        ax1.plot(xx[midleX, :], del_f_del_x[midleX, :], '-kx',
                 markersize=10, label='dx data')
        ax1.plot(xx[midleX, :], grad_x[midleX, :], '-r+',
                 markersize=10, label='dx reconstructed')
        ax1.legend()

        ax2 = plt.subplot(223, sharex=ax1)
        ax2.plot(xx[midleX, :],
                 error_x[midleX, :], '-g.', label='error x')
        plt.title(r'$\mu$ = {:.2g}'.format(np.mean(error_x[midleX, :])))
        ax2.legend()

        ax3 = plt.subplot(222, sharex=ax1, sharey=ax1)
        ax3.plot(yy[:, midleY], del_f_del_y[:, midleY], '-kx',
                 markersize=10, label='dy data')
        ax3.plot(yy[:, midleY], grad_y[:, midleY], '-r+',
                 markersize=10, label='dy reconstructed')
        ax3.legend()

        ax4 = plt.subplot(224, sharex=ax1, sharey=ax2)
        ax4.plot(yy[:, midleY],
                 error_y[:, midleY], '-g.', label='error y')
        plt.title(r'$\mu$ = {:.2g}'.format(np.mean(error_y[:, midleY])))
        ax4.legend()

        plt.suptitle('Error integration', fontsize=22)
        plt.show(block=False)

    if errors:
        return error_x, error_y
