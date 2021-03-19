import numpy as np

from skimage.transform import resize


def conversion_uint16_to_float32(image, min_value, max_value):
    """Converts 16 bit int into 32 bit float using min max parameters (0 -> min, 65535 -> max)

    Args:
        image (numpy.ndarray): input 16bit image
        min_value (float):     min float value
        max_value (float):     max float value

    Returns:
        (numpy.ndarray) converted image (float32)

    """
    image = image.astype(np.float32)
    return (image/65535) * (max_value - min_value) + min_value


def normalize_image(image):
    """normalizing an image (from the image [min;max] to [0;1])

    Args:
        image (numpy.ndarray): input image

    Returns:
        (numpy.ndarray) normalized image
    """
    max_image = np.amax(image)
    min_image = np.amin(image)
    image = np.asarray(image, np.float32)

    image = (image - min_image)/(max_image - min_image)
    return image


def normalize_image_min_max(image, min_image, max_image):
    """normalizing an image (from [minImage;maxImage] to [0;1])

    Args:
        image (numpy.ndarray): input image
        min_image (float):     min value
        max_image (float):     max value

    Returns:
        (numpy.ndarray) normalized image
    """
    image = np.asarray(image, np.float32)

    image = (image - min_image)/(max_image - min_image)
    return image


def bin_resize(image, bin_factor):
    """resizing the image depending on a bin_factor

    Args:
        image (numpy.ndarray): input image
        bin_factor (int):      binning factor

    Returns:
        (numpy.ndarray) binned image
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
