import os
import glob

import fabio
import fabio.edfimage as edf
import fabio.tifimage as tif

import numpy as np
import imageio


def create_directory(path):
    """creates a directory at the specified path

    Args:
        path (str): complete path

    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)


def create_list_of_files(folder_name, extension):
    """creating a list of files with a corresponding extension in an input folder

    Args:
        folder_name (str): folder name
        extension (str):   extension of the target files

    Returns:
        (str): the list of files sorted
    """
    if not os.path.exists(folder_name):
        raise Exception('Error: Given path does not exist.')
    list_of_files = glob.glob(folder_name + '/*' + extension)
    list_of_files.sort()
    if len(list_of_files) == 0:
        raise Exception('Error: No file corresponds to the given extension: .' + extension)

    return list_of_files


def get_header(filename):
    """retrieves the header of an image

    Args:
        filename (str): file name

    Returns:
        (str): header
    """
    im = fabio.open(filename)
    return im.header


def open_image(filename):
    """opens a 2D image

    Args:
        filename (str): file name

    Returns:
        (numpy.ndarray): 2D image
    """
    filename = str(filename)
    im = fabio.open(filename)
    return im.data


def open_sequence(filenames):
    """opens a sequence of images

    Args:
        filenames (str): file names

    Returns:
        (numpy.ndarray): sequence of 2D images
    """
    if len(filenames) == 0:
        raise Exception('Error: no file corresponds to the given path/extension')
    if len(filenames) > 0:
        data = open_image(str(filenames[0]))
        height, width = data.shape
        to_return = np.zeros((len(filenames), height, width), dtype=np.float32)
        for i, file in enumerate(filenames):
            data = open_image(str(file))
            to_return[i, :, :] = data
        return to_return


def save_edf_image(image, filename):
    """saves an image to .edf format (32 bit)

    Args:
        image (numpy.ndarray): 2D image
        filename (str): filename (complete path)

    Returns:
        None
    """
    create_directory(remove_filename_in_path(filename))

    data_to_store = image.astype(np.float32)
    edf.EdfImage(data=data_to_store).write(filename)


def save_edf_sequence(image, path):
    """ saves a sequence of images to .edf format (32 bit)

    Args:
        image (numpy.ndarray): 3D sequence of images
        path (str): complete path and regular expression of file names

    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(image.shape[0]):
        save_edf_image(image[i, :, :], path + '{:04d}'.format(i) + '.edf')


def save_edf_sequence_and_crop(image, bounding_box, path):
    """crops a sequence of images and saves it to .edf format (32 bit)

    Args:
        image (numpy.ndarray):        3D sequence of images
        bounding_box (numpy.ndarray): shape to crop into
        path (str):                   complete path and regular expression of file names

    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)

    cropped_image = image[bounding_box[4]:bounding_box[5] + 1,
                          bounding_box[2]:bounding_box[3] + 1,
                          bounding_box[0]:bounding_box[1] + 1]

    for i in range(cropped_image.shape[0]):
        slice_to_save = cropped_image[i, :, :]
        save_edf_image(slice_to_save, path + '{:04d}'.format(i) + '.edf')


def save_tif_image(image, filename, bit=32, rgb=False, header=None):
    """saves an image to .tif format (either int16 or float32)

    Args:
        image (numpy.ndarray): 2D image
        filename (str):        file name
        bit (int):             16: int16, 32: float32
        rgb (bool):            rgb format
        header (str):          header

    Returns:
        None
    """
    create_directory(remove_filename_in_path(filename))
    if rgb:
        imageio.imwrite(filename + '.tif', image)
    elif header:
        if bit == 32:
            tif.TifImage(data=image.astype(np.float32), header=header).write(filename + '.tif')
        else:
            tif.TifImage(data=image.astype(np.uint16), header=header).write(filename + '.tif')
    elif bit == 32:
        tif.TifImage(data=image.astype(np.float32)).write(filename + '.tif')
    else:
        tif.TifImage(data=image.astype(np.uint16)).write(filename + '.tif')


def save_tif_sequence(image, path, bit=32, header=None):
    """saves a sequence of images to .tif format (either int16 or float32)

    Args:
        image (numpy.ndarray): 2D image
        path (str):            complete path + regular expression of file names
        bit (int):             16: int16, 32: float32
        header (str):          header

    Returns:
        None
    """
    for i in range(image.shape[0]):
        image_path = path + '{:04d}'.format(i)
        save_tif_image(image[i, :, :], image_path, bit, header)


def save_tif_sequence_and_crop(image, bounding_box, path, bit=32, header=None):
    """crops a sequence of images and saves it to .tif format (either int16 or float32)

    Args:
        image (numpy.ndarray):        3D sequence of images
        bounding_box (numpy.ndarray): shape to crop into
        path (str):                   complete path and regular expression of file names
        bit (int):                    16: int16, 32: float32
        header (str):                 header

    Returns:
        None
    """
    cropped_image = image[bounding_box[4]:bounding_box[5] + 1,
                          bounding_box[2]:bounding_box[3] + 1,
                          bounding_box[0]:bounding_box[1] + 1]

    for i in range(cropped_image.shape[0]):
        cropped_slice = cropped_image[i, :, :]
        image_path = path + '{:04d}'.format(i) + '.tif'
        save_tif_image(cropped_slice, image_path, bit, header)


def remove_filename_in_path(path):
    """remove the file name from a path

    Args:
        path (str): complete path

    Returns:
        complete path without the file name
    """
    splitter = "\\" if len(path.split("\\")) > 1 else "/"
    path_list = path.split(splitter)[:-1]
    return "".join(elt + splitter for elt in path_list)


def remove_last_folder_in_path(path):
    """remove the last folder from a path

    Args:
        path (str): complete path

    Returns:
        complete path without the last folder
    """
    splitter = "\\" if len(path.split("\\")) > 1 else "/"
    path_list = path.split(splitter)[:-2]
    return "".join(elt + splitter for elt in path_list)


def make_dark_mean(dark_fields):
    """TODO

    Args:
        dark_fields (TODO): TODO

    Returns:
        TODO
    """
    mean_slice = np.mean(dark_fields, axis=0)
    print('-----------------------  mean Dark calculation done ------------------------- ')
    output_filename = '/Users/helene/PycharmProjects/spytlab/meanDarkTest.edf'
    output_edf = edf.EdfFile(output_filename, access='wb+')
    output_edf.WriteImage({}, mean_slice)

    return mean_slice


if __name__ == '__main__':
    total_path = "/data/visitor/test/ok.tif"
    print(remove_filename_in_path(total_path))
    print(remove_last_folder_in_path(total_path))
    total_path = "\\data\\visitor\\test\\ok.tif"
    print(remove_filename_in_path(total_path))
    print(remove_last_folder_in_path(total_path))
