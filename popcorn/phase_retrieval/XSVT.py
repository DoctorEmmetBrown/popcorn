import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, chain
from scipy.interpolate import interp2d
from functools import partial
from scipy.ndimage.filters import median_filter
import frankoChellappa as fc
from OpticalFlow2020 import kottler, LarkinAnissonSheppard


def processProjectionXSVT(experiment):

    nb_images, px_rows, px_cols = experiment.sample_images.shape
    diff_x, diff_y, transmission, darkfield = start_tracking(experiment.sample_images, experiment.reference_images, max_shift=experiment.max_shift)

    if experiment.LCS_median_filter !=0:
        diff_x = median_filter(diff_x, size=experiment.LCS_median_filter)
        diff_y = median_filter(diff_y, size=experiment.LCS_median_filter)

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

    return {"Diff_x" : diff_x, "Diff_y" : diff_y, "Transmission" : transmission, "Darkfield" : darkfield, "DPhi_x" : dphix, "DPhi_y" : dphiy, 'phiFC': phiFC.real, 'phiK': phiK.real,'phiLA': phiLA.real}


def start_tracking(Isample, Iref, max_shift):
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

    Returns dx, dy
    """

    print("Speckle vector tracking started")

    nb_images, px_rows, px_cols = Iref.shape
    paddedIref = np.array([np.pad(Iref[im, :, :], max_shift, 'edge') for im in range(0, nb_images)])

    i = range(0, px_rows)
    j = range(0, px_cols)

    multiprocessing = True

    if multiprocessing:
        print("Multiprocessing on")
        paramlist = list(product(i, j))
        pool = mp.Pool(mp.cpu_count())
        # Need to create partial function because multiprocessing.map only accepts one input parameter
        func = partial(speckle_vector_tracking, Isample, paddedIref, max_shift)
        result = pool.map(func, paramlist)
        dx = list(chain(*result))[0::4]
        dy = list(chain(*result))[1::4]
        tr = list(chain(*result))[2::4]
        df = list(chain(*result))[3::4]
        pool.close()
    else:
        dx = []
        dy = []
        tr = []
        df = []

        for a, b in product(i, j):
            results = speckle_vector_tracking(Isample, paddedIref, max_shift, [a, b])
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


def speckle_vector_tracking(sample_image, padded_ref_image, shift, params):
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
    :params: Size of sample_image

    Returns diff_x, diff_y
    """

    i = params[0]
    j = params[1]

    if j == 0:
        print(i)

    # Vector of Isample intensity values for pixel (i, j)
    v_sample = sample_image[:, i, j]

    # 2D array of Iref intensity values for pixel (i + l, j + m) where -shift <= l, m <= shift
    v_ref = np.array([padded_ref_image[k, i + l, j + m] for k in range(padded_ref_image.shape[0]) for l in range(2 * shift +1) for m in range(2 * shift +1)])
    v_ref = v_ref.reshape((padded_ref_image.shape[0], 2 * shift + 1, 2 * shift + 1))

    # To have more points for the polynomial surface fit, we interpolate v_ref at every half-pixel
    r = c = np.linspace(0, 2*shift, 2*shift+1)
    rs = cs = np.linspace(r[0], r[-1], 2 * len(r))
    px2subpx = [interp2d(c, r, v_ref[n, :, :]) for n in range(v_ref.shape[0])]
    v_ref_s = [f(cs, rs) for f in px2subpx]
    v_ref = np.array(v_ref_s)

    # Determine the correlation between v_sample and each value in v_ref
    pearson_map = np.zeros((v_ref.shape[1], v_ref.shape[2]))
    for l, m in product(range(v_ref.shape[1]), range(v_ref.shape[2])):
        if np.std(v_ref[:, l, m]) == 0 or np.std(v_sample) == 0:
            pearson_map[l][m] = 0.
        else:
            pearson_map[l][m] = pearson_correlation(v_sample, v_ref[:, l, m])

    # Fit a polynomial surface to pearson_map and find the maximum correlation peak
    fit_params = polyfit2d(pearson_map)
    diffy, diffx = find_max(fit_params)

    # Give the shift in terms of displacement (in terms of pixels) of v_sample relative to v_ref
    diff_x = ((len(cs)-1)/2. - diffx) * 0.5
    diff_y = ((len(rs)-1)/2. - diffy) * 0.5

    plot = False

    if plot:
        a = fit_params
        interp_points = 10

        i0 = j0 = np.linspace(0, pearson_map.shape[0] - 1, pearson_map.shape[0])
        is0 = js0 = np.linspace(0, pearson_map.shape[0] - 1, interp_points * (pearson_map.shape[0] - 1) + 1)

        iss, jss = np.meshgrid(is0, js0)

        IS = iss.flatten()
        JS = jss.flatten()

        fit = a[0] * IS ** 2 + a[1] * JS ** 2 + a[2] * IS * JS + a[3] * IS + a[4] * JS + a[5]
        fit = fit.reshape((len(is0), len(js0)))

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(i0, j0, pearson_map)
        ax.scatter(diffx, diffy, 1)
        ax.plot_surface(iss, jss, fit, cmap='plasma')
        ax.set_zlim(0, np.max(pearson_map))
        plt.show()

    v_ref_shifted = [f(diffx, diffy) for f in px2subpx]
    transn = calc_transmission(v_sample, v_ref_shifted)
    dark = calc_df(transn, v_sample, v_ref_shifted)

    return diff_x, diff_y, transn, dark


def pearson_correlation(x, y):
    """
    Calculate the Pearson correlation of two vectors, x and y. The Pearson correlation
    coefficient lies between -1 and 1, where p = 1 means that x and y are the same,
    p = -1 corresponds to inverse correlation and p = 0 means that there is no correlation
    between x and y.

    Note: this function is faster than scipy.stats.pearsonr due to the vectorised matrix multiplication
    (numpy.matmul).

    :param x: Vector of intensity values corresponding pixel to (i, j) of Isample
    :param y: Vector of intensity values corresponding to pixel (i + m, j + l) of Iref

    Returns p
    """

    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)

    # bound the values to -1 to 1 in the event of precision issues
    if np.sqrt(np.outer(xvss, yvss)) == 0:
        p = 0.
    else:
        result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
        p = np.maximum(np.minimum(result, 1.0), -1.0)
    return p


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
    return np.mean(vs) / np.mean(vr)


def calc_df(tr, vs, vr):
    return (1/tr) * np.std(vs) / np.std(vr)
