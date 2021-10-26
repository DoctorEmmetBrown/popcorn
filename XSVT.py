import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from itertools import product, chain
from scipy.interpolate import interp2d
from functools import partial
from scipy.ndimage.filters import median_filter
import frankoChellappa as fc
from OpticalFlow2020 import kottler, LarkinAnissonSheppard


def processProjectionXSVT(experiment):
    """
    Calls start_tracking() which manages the calling of speckle_vector_tracking().
    Applies median filter if required and performs integration of displacement images.

    :param experiment: gets all the information related to the desired experiment.

    Returns:
        Diff_x: displacement in x (in terms of pixels)
        Diff_y: displacement in y (in terms of pixels)
        Transmission: the transmission image (I/I0)
        Darkfield: the darkfield image
        DPhi_x: displacement in x (radians)
        DPhy_y: displacement in y (radians)
        phiFC: Frankot-Chelappa integrated phase image
        phiK: Kottler integrated phase image
        phiLA: Larkin-Anisson-Sheppard integrated phase image
    """

    nb_images, px_rows, px_cols = experiment.sample_images.shape
    diff_x, diff_y, transmission, darkfield = start_tracking(experiment.sample_images, experiment.reference_images, max_shift=experiment.max_shift, window=experiment.XSVT_Nw)

    if experiment.XSVT_median_filter != 0:
        diff_x = median_filter(diff_x, size=experiment.XSVT_median_filter)
        diff_y = median_filter(diff_y, size=experiment.XSVT_median_filter)

    dphix = diff_x * experiment.getk() * (experiment.pixel / experiment.dist_object_detector)
    dphiy = diff_y * experiment.getk() * (experiment.pixel / experiment.dist_object_detector)

    padForIntegration = True
    padSize = 300
    if padForIntegration:
        dphix = np.pad(dphix, ((padSize, padSize), (padSize, padSize)), mode='reflect')  # voir is edge mieux que reflect
        dphiy = np.pad(dphiy, ((padSize, padSize), (padSize, padSize)), mode='reflect')  # voir is edge mieux que reflect

    # Compute the phase from phase gradients with 3 different methods (still trying to choose the best one)
    phiFC = fc.frankotchellappa(dphiy, dphix, True) * experiment.pixel
    phiK = kottler(dphiy, dphix) * experiment.pixel
    phiLA = LarkinAnissonSheppard(dphiy, dphix) * experiment.pixel

    if padSize > 0:
        phiFC = phiFC[padSize:padSize + px_rows, padSize:padSize + px_cols]
        phiK = phiK[padSize:padSize + px_rows, padSize:padSize + px_cols]
        phiLA = phiLA[padSize:padSize + px_rows, padSize:padSize + px_cols]

    return {"Diff_x": diff_x, "Diff_y": diff_y, "Transmission": transmission, "Darkfield": darkfield, "DPhi_x": dphix, "DPhi_y": dphiy, 'phiFC': phiFC.real, 'phiK': phiK.real, 'phiLA': phiLA.real}


def start_tracking(Isample, Iref, max_shift, window):
    """
    Compare speckle images with sample (Isample) and w/o sample
    (Iref) pixel by pixel.
    Find maximum correlation using Pearson's correlation coefficient and produce maps of local displacement.
    max_shift can be set to the number of pixels for an "acceptable"
    speckle displacement.
    Option to use multiprocessing on all available cores.

    :param Isample: A list  of measurements, with the sample aligned but speckles shifted
    :param Iref: A list of empty speckle measurements with the same displacement as Isample.
    :param max_shift: Do not allow shifts larger than this number of pixels
    :param window: window to consider when calculating the correlation

    Returns dx, dy
    """

    print("Speckle vector tracking started")

    nb_images, px_rows, px_cols = Iref.shape

    i = range(0, px_rows)
    j = range(0, px_cols)

    multiprocessing = True

    # pm = number of pixels in window surrounding the central pixel in each direction (up, down, left, right)
    pm = int((window - 1) / 2) if window >= 1 else 0

    # Pad Ir with max_shift+pm pixels, pad Is with pm pixels
    # This is to ensure that dx and dy have the same dimensions as original images
    paddedIref = np.array([np.pad(Iref[im, :, :], max_shift + pm, 'edge') for im in range(0, nb_images)])
    paddedIsample = np.array([np.pad(Isample[im, :, :], pm, 'edge') for im in range(0, nb_images)])

    # Multiprocessing will use all available cores
    # speckle_vector_tracking() is dispatched to cores as they become available until end of loop
    if multiprocessing:
        print("Multiprocessing on: " + str(mp.cpu_count()) + " cores")
        paramlist = list(product(i, j))
        pool = mp.Pool(mp.cpu_count())
        # Need to create partial function because multiprocessing.map only accepts one input parameter
        pfunc = partial(speckle_vector_tracking, paddedIsample, paddedIref, max_shift, window)
        result = pool.map(pfunc, paramlist)
        dx = list(chain(*result))[0::4]
        dy = list(chain(*result))[1::4]
        tr = list(chain(*result))[2::4]
        df = list(chain(*result))[3::4]
        pool.close()
    # If multiprocessing not used, simple for-loop is used
    else:
        print("Multiprocessing off")
        dx = []
        dy = []
        tr = []
        df = []

        for a, b in product(i, j):
            results = speckle_vector_tracking(paddedIsample, paddedIref, max_shift, window, [a, b])
            dx.append(results[0])
            dy.append(results[1])
            tr.append(results[2])
            df.append(results[3])

    dx = np.array(dx).reshape(px_rows, px_cols)
    dy = np.array(dy).reshape(px_rows, px_cols)
    tr = np.array(tr).reshape(px_rows, px_cols)
    df = np.array(df).reshape(px_rows, px_cols)

    print("End of speckle vector tracking")

    return dx, dy, tr, df


def speckle_vector_tracking(sample_image, padded_ref_image, shift, w, params):
    """
    Compare speckle images with sample (Isample) and w/o sample
    (Iref) pixel by pixel.
    Find maximum correlation using Pearson's correlation coefficient and produce maps of local displacement.
    max_shift can be set to the number of pixels for an "acceptable"
    speckle displacement.

    :param sample_image: A list  of measurements, with the sample aligned but speckles shifted
    :param padded_ref_image: A list of empty speckle measurements with the same displacement as Isample, padded
    on each side with number of pixels = shift so that resulting image is of the same size as input images
    :param shift: Number of pixels to consider when comparing sample_image with padded_ref_image
    :param params: row, column
    :param w: window

    Returns diff_x, diff_y
    """

    i = params[0]
    j = params[1]

    nb, rows, cols = padded_ref_image.shape
    roi = 2 * shift + 1
    pm = int((w - 1) / 2) if w > 1 else 0

    if j == 0:
        try:
            process = mp.current_process()
            print("Process ID: " + str(process.name) + "; Row: " + str(i))
        except:
            print("Row: " + str(i))

    v_sample = sample_image[:, i+pm, j+pm]
    # Sub-matrix of Isample intensity values with size window**2
    roi_sample = sample_image[:, i:i+w, j:j+w]
    # Sub-matrix of Iref intensity values with size (window+2*shift)**2
    roi_ref = np.array(padded_ref_image[:, i:i+2*pm+roi, j:j+2*pm+roi])

    # Determine the correlation between v_sample and each value in v_ref
    pearson_map = np.zeros((roi_ref.shape[1] - 2*pm, roi_ref.shape[2] - 2*pm))
    for l, m in product(range(roi_ref.shape[1] - 2*pm), range(roi_ref.shape[2] - 2*pm)):
        if np.std(roi_ref[:, l:l + w, m:m + w]) == 0 or np.std(roi_sample) == 0:
            pearson_map[l][m] = 0.
        else:
            pearson_map[l][m] = nc(roi_sample, roi_ref[:, l:l + w, m:m + w])

    # Fit a polynomial surface to pearson_map and find the maximum correlation peak
    fit_params = polyfit2d(pearson_map)
    diffy, diffx = find_max(fit_params)

    # Give the shift in terms of displacement (in terms of pixels) of v_sample relative to v_ref
    diff_x = ((pearson_map.shape[0]-1)/2. - diffx)
    diff_y = ((pearson_map.shape[0]-1)/2. - diffy)

    plot = False

    if plot and i > 0 and j > 0:
        a = fit_params
        interp_points = 10

        i0 = j0 = np.linspace(0, pearson_map.shape[0] - 1, pearson_map.shape[0])
        is0 = js0 = np.linspace(0, pearson_map.shape[0] - 1, interp_points * (pearson_map.shape[0] - 1) + 1)

        print(i0, is0)

        imesh, jmesh = np.meshgrid(i0, j0)
        iss, jss = np.meshgrid(is0, js0)

        IS = iss.flatten()
        JS = jss.flatten()

        fit = a[0] * IS ** 2 + a[1] * JS ** 2 + a[2] * IS * JS + a[3] * IS + a[4] * JS + a[5]
        fit = fit.reshape((len(is0), len(js0)))

        imesh = [(i0[-1]/2 - v) for v in imesh]
        jmesh = [(j0[-1]/2 - v) for v in jmesh]
        iss = [(is0[-1]/2 - v) for v in iss]
        jss = [(js0[-1]/2 - v) for v in jss]

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(imesh, jmesh, pearson_map, zorder=2)
        surf = ax.plot_surface(iss, jss, fit, cmap='plasma', vmin=0.9, vmax=1, zorder=-10, alpha = 0.8)
        ax.scatter(diff_x, diff_y, np.amax(fit), color='k', zorder=3)
        ax.set_zlim(0, 1.2)
        ax.set_xlabel(r'$\Delta$ x')
        ax.set_ylabel(r'$\Delta$ y')
        ax.set_zlabel('p')
        plt.colorbar(surf)
        plt.show()

    # We need to interpolate roi_ref in order to obtain Iref vector corresponding to max correlation
    r = c = np.linspace(0, 2*shift+2*pm, roi+2*pm)
    px2subpx = [interp2d(c, r, roi_ref[n, :, :]) for n in range(nb)]

    v_ref_shifted = [f(diffx, diffy) for f in px2subpx]
    transn = calc_transmission(v_sample, v_ref_shifted)
    dark = calc_df(transn, v_sample, v_ref_shifted)

    return diff_x, diff_y, transn, dark


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

    r = (np.inner(xv.ravel(), yv.ravel()) / np.sqrt(np.outer(np.sum(xv**2), np.sum(yv**2))))
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
    result = np.linalg.lstsq(M, mf)
    a = result[0]

    return a


def find_max(a):
    """
    Find the coordinates (i0, j0) in terms of pixels of the maximum correlation peak from
    the polynomial surface fit parameters

    :param a: List of polynomial surface fit parameters returned by polyfit2d()

    Returns i0, j0
    """

    i0 = ((a[2]*a[3]) - (2*a[0]*a[4])) / (4*a[0]*a[1] - a[2]**2)
    j0 = ((a[2] * a[4]) - (2 * a[1] * a[3])) / (4 * a[0] * a[1] - a[2] ** 2)

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


def plot_pmap():
    return
