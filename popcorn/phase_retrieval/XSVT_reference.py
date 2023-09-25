import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from itertools import product, chain
from scipy.ndimage import map_coordinates
from functools import partial
import frankoChellappa as fc
from phase_integration import fourier_integration, ls_integration
from numba import jit


def processProjection_rXSVT(experiment):
    """
    Calls start_tracking() which manages the calling of speckle_vector_tracking().
    Applies median filter if required and performs integration of displacement images.

    :param experiment: gets all the information related to the desired experiment.

    Returns:
        Diff_x: displacement in x (horizontal)(in terms of pixels)
        Diff_y: displacement in y (vertical)(in terms of pixels)
        Transmission: the transmission image (I/I0)
        Darkfield: the darkfield image
        phiFC: Frankot-Chelappa integrated phase image
        phiK: Kottler integrated phase image
        phiLS: Least Squares integrated phase image
    """
    nb_images, px_rows, px_cols = experiment.sample_images.shape
    diff_x, diff_y, transmission, darkfield = start_tracking(experiment.sample_images, experiment.reference_images, max_shift=experiment.max_shift, window=1+2*experiment.XSVT_Nw)

    # Convert the displacement (pixels) to phase gradient (m-1)
    dphix = diff_x * experiment.getk() * (experiment.pixel / experiment.dist_object_detector)
    dphiy = diff_y * experiment.getk() * (experiment.pixel / experiment.dist_object_detector)

    # Compute the phase from phase gradients with 3 different methods (still trying to choose the best one)
    # The sampling step for the gradient is the magnified pixel size
    magnificationFactor = (experiment.dist_object_detector + experiment.dist_source_object) / experiment.dist_source_object
    gradientSampling = experiment.pixel / magnificationFactor    
    phiFC = fc.frankotchellappa(dphix, dphiy, True).real * gradientSampling
    phiK = fourier_integration.fourier_solver(dphix, dphiy, gradientSampling, gradientSampling, solver='kottler')
    #phiLS = ls_integration.least_squares(dphix, dphiy, gradientSampling, gradientSampling, model='southwell')

    return {"dx": diff_x, "dy": diff_y, "Absorption": transmission, "Deff": darkfield, 'phiFC': phiFC, 'phiK': phiK} #, 'phiLS': phiLS}#


def start_tracking(Itarget, Ireference, max_shift, window):
    """
    Compare speckle images with sample (Isample) and w/o sample
    (Iref) pixel by pixel.
    Find maximum correlation using Pearson's correlation coefficient and produce maps of local displacement.
    The displacement is given in the coordinates of the reference image.
    max_shift can be set to the number of pixels for an "acceptable".
    speckle displacement.
    Option to use multiprocessing on all available cores.

    Note: Darkfield was not tested! To be checked

    :param Isample: A list  of measurements, with the sample aligned but speckles shifted
    :param Iref: A list of empty speckle measurements with the same displacement as Isample.
    :param max_shift: Do not allow shifts larger than this number of pixels
    :param window: window to consider when calculating the correlation

    Returns dx, dy
    """

    print("Speckle vector tracking started")

    nb_images, px_rows, px_cols = Ireference.shape

    i = range(0, px_rows)
    j = range(0, px_cols)

    multiprocessing = True

    # pm = number of pixels in window surrounding the central pixel in each direction (up, down, left, right)
    pm = int((window - 1) / 2) if window >= 1 else 0

    # Pad Ir with max_shift+pm pixels, pad Is with pm pixels
    # This is to ensure that dx and dy have the same dimensions as original images
    paddedItarget = np.array([np.pad(Itarget[im, :, :], max_shift + pm, 'edge') for im in range(0, nb_images)])
    paddedIreference = np.array([np.pad(Ireference[im, :, :], pm, 'edge') for im in range(0, nb_images)])

    # Multiprocessing will use all available cores
    # speckle_vector_tracking() is dispatched to cores as they become available until end of loop
    if multiprocessing:
        print("Multiprocessing on: " + str(mp.cpu_count()) + " cores")
        paramlist = list(product(i, j))
        pool = mp.Pool(mp.cpu_count())
        # Need to create partial function because multiprocessing.map only accepts one input parameter
        pfunc = partial(speckle_vector_tracking, paddedItarget, paddedIreference, max_shift, window)
        result = pool.map(pfunc, paramlist)
        dx = list(chain(*result))[0::2]
        dy = list(chain(*result))[1::2]
        #tr = list(chain(*result))[2::4]
        #df = list(chain(*result))[3::4]
        pool.close()
    # If multiprocessing not used, simple for-loop is used
    else:
        print("Multiprocessing off")
        dx = []
        dy = []
        #tr = []
        #df = []

        for a, b in product(i, j):
            results = speckle_vector_tracking(paddedItarget, paddedIreference, max_shift, window, [a, b])
            dx.append(results[0])
            dy.append(results[1])
            #tr.append(results[2])
            #df.append(results[3])

    dx = -1*np.array(dx).reshape(px_rows, px_cols)
    dy = -1*np.array(dy).reshape(px_rows, px_cols)

    tr,df = calc_tr_df(Itarget,Ireference,dy,dx)

    #tr = np.array(tr).reshape(px_rows, px_cols)
    #df = np.array(df).reshape(px_rows, px_cols)

    print("End of speckle vector tracking")

    return dx, dy, tr, df


def speckle_vector_tracking(target_image, fixed_image, shift, w, params):
    """
    Compare speckle images with sample (Isample) and w/o sample
    (Iref) pixel by pixel.
    Find maximum correlation using Pearson's correlation coefficient and produce maps of local displacement.
    max_shift can be set to the number of pixels for an "acceptable"
    speckle displacement.

    :param fixed_image: A list  of measurements, with the sample aligned but speckles shifted
    :param padded_ref_image: A list of empty speckle measurements with the same displacement as Isample, padded
    on each side with number of pixels = shift so that resulting image is of the same size as input images
    :param shift: Number of pixels to consider when comparing fixed_image with padded_ref_image
    :param params: row, column
    :param w: window

    Returns diff_x, diff_y
    """

    i = params[0]
    j = params[1]

    nb, rows, cols = target_image.shape
    roi = 2 * shift + 1
    pm = int((w - 1) / 2) if w > 1 else 0

    # if j == 0:
    #     try:
    #         process = mp.current_process()
    #         print("Process ID: " + str(process.name) + "; Row: " + str(i))
    #     except:
    #         print("Row: " + str(i))

    # Sub-matrix of Isample intensity values with size window**2
    roi_sample = fixed_image[:, i:i+w, j:j+w]
    # Sub-matrix of Iref intensity values with size (window+2*shift)**2
    roi_ref = np.array(target_image[:, i:i+2*pm+roi, j:j+2*pm+roi])

    # Determine the correlation between v_sample and each value in v_ref
    pearson_map = compute_covariance(roi_ref,roi_sample,pm,w)

    # Fit a polynomial surface to pearson_map and find the maximum correlation peak
    # To avoid instabilities, the fit is performed only around the maximum, on a 3x3 ROI.
    # The fine-tuning is limited to one pixel. Larger values imply a failure of the fit. 
    maxcorr = np.unravel_index(np.argmax(pearson_map, axis=None), pearson_map.shape)
    roixmin = define_roi(maxcorr[1],shift)
    roiymin = define_roi(maxcorr[0],shift)
    cropped_pearson = pearson_map[roiymin:roiymin+3,roixmin:roixmin+3]
    
    fit_params = polyfit2d(cropped_pearson)
    dy_fit, dx_fit = find_max(fit_params) - (maxcorr-np.array([roiymin,roixmin]))
    dy_fit, dx_fit = np.minimum([dy_fit, dx_fit],[0.55,0.55])
    dy_fit, dx_fit = np.maximum([dy_fit, dx_fit],[-0.55,-0.55])
    diffy = dy_fit  + maxcorr[0]
    diffx = dx_fit  + maxcorr[1]

    # Give the shift in terms of displacement (in terms of pixels) of v_sample relative to v_ref
    diff_x = ((pearson_map.shape[0]-1)/2. - diffx)
    diff_y = ((pearson_map.shape[0]-1)/2. - diffy)

    return diff_x, diff_y #, transn, dark


@jit(nopython=True)
def compute_covariance(roi_ref,roi_sample,pm,w):
    pearson_map = np.zeros((roi_ref.shape[1] - 2*pm, roi_ref.shape[2] - 2*pm))
    for l in range(roi_ref.shape[1] - 2*pm):
        for m in range(roi_ref.shape[2] - 2*pm):
            if np.std(roi_ref[:, l:l + w, m:m + w]) == 0 or np.std(roi_sample) == 0:
                pearson_map[l][m] = 0.
            else:
                pearson_map[l][m] = nc(roi_sample, roi_ref[:, l:l + w, m:m + w])
    return pearson_map

@jit(nopython=True)
def nc(x, y):
    """
    Calculate the Pearson correlation of two matrices, x and y. The Pearson correlation
    coefficient lies between -1 and 1, where r = 1 means that x and y are the same,
    r = -1 corresponds to inverse correlation and p = 0 means that there is no correlation
    between x and y.

    :param x: Matrix of intensity values corresponding to the window (i-(Nw-1)/2:i+(Nw-1)/2, j-(Nw-1)/2:j+(Nw-1)/2) of Isample
    :param y: Matrix of intensity values corresponding to the window (i-(Nw-1)/2+m:i+(Nw-1)/2+m, j-(Nw-1)/2+l:j+(Nw-1)/2+l) of Iref

    Returns r
    """
    xv = x - np.mean(x)
    yv = y - np.mean(y)

    r = np.sum(xv * yv) / np.sqrt(np.sum(xv**2) * np.sum(yv**2))
    return r


def polyfit2d(pmap):
    """
    Fit a 2nd order polynomial surface (paraboloid) to the map of Pearson's correlation
    coefficients (pmap) and return a list (a) containing the fit parameters.

    Model: C(i, j) = a0*i*i + a1*j*j + a2*i*j + a3*i + a4*j + a5

    where (i, j) are the rows and columns in pmap.

    :param pmap: 2D array containing Pearson's correlation coefficients.

    Returns a
    """

    i, j = np.indices(pmap.shape)
    I = i.flatten()
    J = j.flatten()
    mf = pmap.flatten()

    M = np.array([I ** 2, J ** 2, I * J, I, J, I * 0 + 1]).T
    result = np.linalg.lstsq(M, mf,rcond=None)
    a = result[0]

    return a


def find_max(a):
    """
    Find the coordinates (i0, j0) in terms of pixels of the maximum correlation peak from
    the polynomial surface fit parameters.

    :param a: List of polynomial surface fit parameters returned by polyfit2d()

    Returns i0, j0
    """

    #If the quadratic terms are zero, the fit won't find a stationary point.
    #We set the maximum to (1,1) to introduce no correction to argmax.
    denominator = (4*a[0]*a[1] - a[2]**2)
    if(denominator==0):
        i0 = j0 = 1
    else:    
        j0 = ((a[2]*a[3]) - (2*a[0]*a[4])) / denominator
        i0 = ((a[2] * a[4]) - (2 * a[1] * a[3])) / denominator

    return i0, j0

def calc_transmission(vs, vr):
    """
    Calculate the transmission image.

    :param vs: vector of sample intensity values
    :param vr: vector of reference intensity values (shifted corresponding to max correlation)

    Returns np.mean(vs) / np.mean(vr) => transmission image
    """

    return np.mean(vs) / np.mean(vr)


def calc_df(tr, vs, vr):
    """
    Calculate the darkfield image.

    :param tr: vector of transmission values
    :param vs: vector of sample intensity values
    :param vr: vector of reference intensity values (shifted corresponding to max correlation)

    Returns (1/tr) * np.std(vs) / np.std(vr) => darkfield image
    """

    return (1/tr) * np.std(vs) / np.std(vr)



def calc_tr_df(Itarget,Ireference,dy,dx):

    nb, rows, cols = Itarget.shape
    gpx,gpy = np.meshgrid(np.arange(cols),np.arange(rows))
    coordinates = np.array([gpy+dy,gpx+dx])
    Itarget_shifted = [map_coordinates(Itarget[i],coordinates) for i in np.arange(nb)]

    tr = np.sum(Itarget_shifted,axis=0)/np.sum(Ireference,axis=0)
    df = (1/tr) *np.std(Itarget_shifted,axis=0) / np.std(Ireference,axis=0)

    return tr,df

def define_roi(index,shift):

    if index==0:
        ind_min=0
    elif index == (2*shift):
        ind_min = 2*(shift-1)
    else:
        ind_min = index-1
    return ind_min

def plot_pmap():
    return
