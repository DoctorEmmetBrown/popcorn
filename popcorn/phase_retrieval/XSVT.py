import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from itertools import product, chain
from scipy.stats import pearsonr
from scipy.interpolate import interp2d
from functools import partial


def processProjectionXSVT(experiment):

    result = speckle_vector_tracking(experiment.sample_images, experiment.reference_images, max_shift=experiment.max_shift)

    return


def speckle_vector_tracking(Isample, Iref, max_shift):
    """
    Compare speckle images with sample (Isample) and w/o sample
    (Iref) pixel by pixel.
    Find maximum correlation using Pearson's correlation coefficient and produce maps of local displacement.
    max_shift can be set to the number of pixels for an "acceptable"
    speckle displacement.

    :param Isample: A list  of measurements, with the sample aligned but speckles shifted
    :param Iref: A list of empty speckle measurements with the same displacement as Isample.
    :param max_shift: Do not allow shifts larger than this number of pixels

    Returns dx, dy
    """

    print("Speckle vector tracking in process")

    nb_images, px_x, px_y = Iref.shape
    paddedIref = np.array([np.pad(Iref[im, :, :], max_shift, 'constant') for im in range(0, nb_images)])

    #px_x = px_y = 5

    i = range(0, px_x)
    j = range(0, px_y)

    paramlist = list(product(i, j))
    pool = mp.Pool(mp.cpu_count())

    # for i, j in product(range(px_x), range(px_y)):
    #     print(i)
    #     v_sample = Isample[:, i, j]
    #
    #     for l, m in product(range(2*max_shift+1), range(2*max_shift+1)):
    #         v_ref = paddedIref[:, i+l, j+m]
    #
    #         pearson_map[l][m] = pearson_correlation(v_sample, v_ref)
    #
    #     dx_px, dy_px = [max_shift-idx for idx in np.unravel_index(pearson_map.argmax(), pearson_map.shape)]
    #
    #     dx[i, j] = dx_px
    #     dy[i, j] = dy_px

    func = partial(calc_dx_dy, Isample, paddedIref, max_shift)
    result = pool.map(func, paramlist)
    #result = pool.starmap(calc_dx_dy, [(Isample, paddedIref, max_shift, x) for x in paramlist])

    dx_px = list(chain(*result))[0::2]
    dy_px = list(chain(*result))[1::2]

    dx = np.array(dx_px).reshape(px_x, px_y)
    dy = np.array(dy_px).reshape(px_y, px_x)

    pool.close()

    plt.imshow(dx, cmap='gray')
    plt.colorbar()
    plt.show()

    return "something"


def calc_dx_dy(sample_image, padded_ref_image, shift, params):
    i = params[0]
    j = params[1]

    if j == 0:
        print(i)

    pearson_map = np.zeros((2 * shift + 1, 2 * shift + 1))
    v_sample = sample_image[:, i, j]

    pixel = np.linspace(0, 2 * shift, 2 * shift + 1)
    subpixel = np.linspace(0, 2 * shift, 20 * shift + 1)  # 10th of a pixel resolution

    for l, m in product(range(2 * shift +1), range(2 * shift + 1)):
        v_ref = padded_ref_image[:, i + l, j + m]
        pearson_map[l][m] = pearson_correlation(v_sample, v_ref)

    diff_x, diff_y = [shift - idx for idx in np.unravel_index(pearson_map.argmax(), pearson_map.shape)]

    return diff_x, diff_y


def pearson_correlation(sample_vector, ref_vector):
    pcoeff = 0 if np.isnan(abs(pearsonr(sample_vector, ref_vector)[0])) else abs(pearsonr(sample_vector, ref_vector)[0])

    return pcoeff
