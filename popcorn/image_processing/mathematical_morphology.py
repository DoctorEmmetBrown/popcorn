# -- IPSDK Library --
import PyIPSDK
import PyIPSDK.IPSDKIPLMorphology as Morpho
import PyIPSDK.IPSDKIPLBinarization as Bin

import numpy as np


def dilate(image, radius):
    """
    binary dilatation of an image using IPSDK library
    :param image: input image (binary)
    :param radius: radius of the structure element
    :return: dilated image
    """
    image_ipsdk = PyIPSDK.fromArray(image)
    image_ipsdk = Bin.lightThresholdImg(image_ipsdk, 1)

    morpho_mask = PyIPSDK.sphericalSEXYZInfo(radius)
    image_ipsdk = Morpho.dilate3dImg(image_ipsdk, morpho_mask)

    return np.copy(image_ipsdk.array)
