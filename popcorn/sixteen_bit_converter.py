import os
from math import ceil
import numpy as np
from popcorn.input_output import open_image, save_tif_image
from resampling import conversion_from_float32_to_uint16


def padding_image(image, pad_size):
    """determines the distribution of pixels to add on top/bottom/right/left calls the pad_with function

    Args:
        image (numpy.ndarray): input image
        pad_size (int):     nb of pixels to add (in both x and y directions)

    Returns:
        (numpy.ndarray): padded image
    """
    if pad_size % 2 != 0:
        return np.pad(image,
                      pad_width=((int(pad_size / 2), ceil(pad_size / 2)),
                                 (int(pad_size / 2), ceil(pad_size / 2))),
                      mode='constant')
    else:
        return np.pad(image,
                      pad_width=((int(pad_size / 2), int(pad_size / 2)),
                                 (int(pad_size / 2), int(pad_size / 2))),
                      mode='constant')


def multi_threading_conversion(list_of_args):
    """
    transforms a list of args into 5 args before calling conversion function
    :param list_of_args: list of args
    :return: None
    """
    conversion_from_list_of_files(list_of_args[0], list_of_args[1], list_of_args[2], list_of_args[3], list_of_args[4])


def conversion_from_list_of_files(list_of_files, output_folder, min_value=0., max_value=1., padding_size=0):
    """opens files from the input list of files, converts them in uint16 and saves them in output folder as .tif files

    Args:
        list_of_files (): input list of files
        output_folder (): output folder path
        min_value ():     minimum value (any value below will be set to 0)
        max_value ():     maximum value (any value above will be set to 65535)
        padding_size ():  image padding size

    Returns:
        None
    """
    for file_name in list_of_files:
        base_name = os.path.basename(file_name).split(".")[0]
        data = open_image(file_name)
        if padding_size != 0:
            data = padding_image(data, padding_size)
        converted_data = conversion_from_float32_to_uint16(data, min_value, max_value)
        save_tif_image(converted_data, output_folder + '/' + base_name + ".tif", bit=16)
