import numpy as np

from skimage.transform import resize


def conversion_from_uint16_to_float32(image, min_value, max_value):
    """Converts 16 bit uint into 32 bit float using min max parameters (0 -> min, 65535 -> max)

    Args:
        image (numpy.ndarray): input 16bit image
        min_value (float):     min float value
        max_value (float):     max float value

    Returns:
        (numpy.ndarray): converted image (float32)
    """
    image = image.astype(np.float32)
    return (image/65535) * (max_value - min_value) - min_value


def conversion_from_float32_to_uint16(image, min_value, max_value):
    """Converts 32 bit float into 16 bit uint using min max parameters (min -> 0, max -> 65535)

    Args:
        image (numpy.ndarray): input 32bit image
        min_value (float):     min float value
        max_value (float):     max float value

    Returns:
        (numpy.ndarray): converted image (uint16)
    """
    image = (image - min_value) / (max_value - min_value) * 65535
    return image.astype(np.uint16)


def normalize_image(image):
    """normalizes an image (from the [image.min;image.max] to [0;1])

    Args:
        image (numpy.ndarray): input image

    Returns:
        (numpy.ndarray): normalized image
    """
    max_image = np.amax(image)
    min_image = np.amin(image)
    image = np.asarray(image, np.float32)

    image = (image - min_image)/(max_image - min_image)
    return image


def normalize_image_min_max(image, min_image, max_image):
    """normalizes an image (from [min_image;max_image] to [0;1])

    Args:
        image (numpy.ndarray): input image
        min_image (float):     input image manual min value (all values below will be set to 0)
        max_image (float):     input image manual max value (all values above will be set to 1)

    Returns:
        (numpy.ndarray): normalized image
    """
    image = np.asarray(image, np.float32)

    image = (image - min_image)/(max_image - min_image)
    return image


def bin_resize(image, bin_factor):
    """resizes the image depending on a bin_factor

    Args:
        image (numpy.ndarray): input image
        bin_factor (int): binning factor

    Returns:
        (numpy.ndarray): binned image
    """
    nb_slices, width, height = image.shape
    if bin_factor > 0:
        nb_slices = int(nb_slices/bin_factor)
        width = int(width/bin_factor)
        height = int(height/bin_factor)
        dim = (nb_slices, width, height)
        return resize(image, dim, preserve_range=True)
    else:
        raise Exception('bin_factor must be strictly positive')
