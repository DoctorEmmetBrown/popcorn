import os
import glob

import fabio
import fabio.edfimage as edf
import fabio.tifimage as tif

import numpy as np
import imageio

from popcorn.resampling import bin_resize


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
        if "tif" in extension.lower():
            extension = "edf"
            list_of_files = glob.glob(folder_name + '/*' + extension)
            list_of_files.sort()
        elif "edf" in extension.lower():
            extension = "tif"
            list_of_files = glob.glob(folder_name + '/*' + extension)
            list_of_files.sort()
        else:
            raise Exception('Error: No file corresponds to the given extension: .' + extension)

    if len(list_of_files) == 0:
        raise Exception('Error: No file corresponds to the following extensions: .tif and .edf')
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
    if "edf" in filename:
        im = fabio.open(filename)
        return im.data
    elif "tif" in filename or "tiff" in filename:
        im = imageio.imread(filename)
        return im


def open_sequence(filenames_or_input_folder, extension="tif"):
    """opens a sequence of images

    Args:
        filenames_or_input_folder (str): file names
        extension (str):                 files extension

    Returns:
        (numpy.ndarray): sequence of 2D images
    """
    # If the given arg is empty, we raise an error
    if len(filenames_or_input_folder) == 0:
        raise Exception('Error: no file corresponds to the given path/extension')
    # We check if the given filenames is a regular expression of input files:
    if type(filenames_or_input_folder) != list:
        # We try opening .extension files
        list_of_files = create_list_of_files(filenames_or_input_folder, extension)
    else:
        list_of_files = filenames_or_input_folder
    # If the created list_of_files is empty
    if len(list_of_files) == 0:
        raise Exception('Error: no file corresponds to the given path/extension')

    # Next line is computed iff given regex/list of files correspond to existing files that can be opened
    if len(list_of_files) > 0:
        reference_image = open_image(str(list_of_files[0]))
        height, width = reference_image.shape
        # We create an empty image sequence
        sequence = np.zeros((len(list_of_files), height, width), dtype=np.float32)
        # We fill the created empty sequence
        for i, file in enumerate(list_of_files):
            image = open_image(str(file))
            sequence[i, :, :] = image
        return sequence


def open_cropped_image(filename, min_max_y_x_list):
    """

    Args:
        filename (str):          file name
        min_max_y_x_list (list): list of [min,max] for y, x

    Returns:
        (numpy.ndarray): cropped image
    """

    ref_image = open_image(filename)
    # In case of negative coordinates, we get the reverse position (max - val)
    for coors_nb, coordinates in enumerate(min_max_y_x_list):
        for coor_nb, coordinate in enumerate(coordinates):
            if coordinate < 0:
                min_max_y_x_list[coors_nb][coor_nb] += ref_image.shape[coors_nb]

    image = ref_image[min_max_y_x_list[0][0]: min_max_y_x_list[0][1] + 1,
                      min_max_y_x_list[1][0]: min_max_y_x_list[1][1] + 1]

    return image


def open_cropped_sequence(filenames_or_input_folder, min_max_z_y_x_list):
    """opens a sequence of images and returns a cropped version. Major default : opens the full image

    Args:
        filenames_or_input_folder (str): file names
        min_max_z_y_x_list (list):       list of [min,max] for z, y, x

    Returns:
        (numpy.ndarray): sequence of 2D images
    """
    # If the given arg is empty, we raise an error
    if len(filenames_or_input_folder) == 0:
        raise Exception('Error: no file corresponds to the given path/extension')
    # We check if the given filenames is a regular expression of input files:
    if type(filenames_or_input_folder) != list:
        # We try opening either .tif files
        list_of_files = create_list_of_files(filenames_or_input_folder, "tif")
        # or .edf files
        if len(list_of_files) == 0:
            list_of_files = create_list_of_files(filenames_or_input_folder, "edf")
    else:
        list_of_files = filenames_or_input_folder
    # If the created list_of_files is empty
    if len(list_of_files) == 0:
        raise Exception('Error: no file corresponds to the given path/extension')

    if len(min_max_z_y_x_list) < 3:
        raise Exception('Error: please specify 3 dimensions crop information for min_max_z_y_x_list parameter')
    # Next line is computed iff given regex/list of files correspond to existing files that can be opened
    if len(list_of_files) > 0:
        ref_image = open_image(str(list_of_files[0]))
        # In case of negative coordinates, we get the reverse position (max - val)
        for coors_nb, coordinates in enumerate(min_max_z_y_x_list):
            for coor_nb, coordinate in enumerate(coordinates):
                if coordinate < 0:
                    if coors_nb > 0:
                        min_max_z_y_x_list[coors_nb][coor_nb] += ref_image.shape[coors_nb - 1]
                    else:
                        min_max_z_y_x_list[coors_nb][coor_nb] += len(list_of_files)
        nb_of_files = min_max_z_y_x_list[0][1] - min_max_z_y_x_list[0][0] + 1
        height = min_max_z_y_x_list[1][1] - min_max_z_y_x_list[1][0] + 1
        width = min_max_z_y_x_list[2][1] - min_max_z_y_x_list[2][0] + 1
        # We create an empty image sequence
        sequence = np.zeros((nb_of_files, height, width), dtype=np.float32)
        # We fill the created empty sequence
        for i, file in enumerate(list_of_files):
            if min_max_z_y_x_list[0][0] <= i <= min_max_z_y_x_list[0][1]:
                sequence[i - min_max_z_y_x_list[0][0], :, :] = open_cropped_image(file, [min_max_z_y_x_list[1],
                                                                                         min_max_z_y_x_list[2]])
        return sequence


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
        save_tif_image(image[i, :, :], image_path, bit, header=header)


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
        image_path = path + '{:04d}'.format(i)
        save_tif_image(cropped_slice, image_path, bit, header=header)


def open_bin_and_save(input_folder, output_folder, bin_factor=2, input_image_type="tif"):
    """Open input folder images, bins and saves them in given output folder

    Args:
        input_folder (str):                                  input folder
        output_folder (str):                                 output folder
        bin_factor (int):                                    bin factor (usually 2)
        input_image_type (str):                              type of input images (tif or edf)

    Returns:
        None
    """

    image_filenames = create_list_of_files(input_folder, input_image_type)

    for index in range(len(image_filenames) // bin_factor):
        image_to_bin = open_sequence(image_filenames[:bin_factor])
        del image_filenames[:bin_factor]
        binned_image = bin_resize(image_to_bin, bin_factor)
        save_tif_image(binned_image[0], output_folder + '{:04d}'.format(index))


def open_crop_and_save(input_folder, output_path, min_max_list, input_image_type="tif"):
    """opens files - crops them - saves them as tiff images in given output_path

    Args:
        input_folder (str):             input folder
        output_path (str):              output path
        min_max_list (list[list[int]]): list of 4 int, [[X-min, X-max], [Y-min, Y-max]]
        input_image_type (str):         tif or edf file

    Returns:
        None
    """

    above_map_files = create_list_of_files(input_folder, input_image_type)
    for image_nb, image_file in enumerate(above_map_files):
        image = open_image(image_file)
        cropped_image = image[min_max_list[1][0]:min_max_list[1][1], min_max_list[0][0]:min_max_list[0][1]]
        save_tif_image(cropped_image, output_path + '{:04d}'.format(image_nb))


def open_crop_bin_and_save(input_folder, output_folder, min_max_list, bin_factor=2, input_image_type="tif"):
    """Open input folder images, crops/bins and saves them in given output folder

    Args:
        input_folder (str):                                  input folder
        output_folder (str):                                 output folder
        min_max_list (list[list[int, int], list[int, int]]): cropping dimensions
        bin_factor (int):                                    bin factor (usually 2)
        input_image_type (str):                              type of input images (tif or edf)

    Returns:
        None
    """

    image_filenames = create_list_of_files(input_folder, input_image_type)

    for index in range(len(image_filenames) // bin_factor):
        image_to_bin = open_sequence(image_filenames[:bin_factor])
        cropped_image = image_to_bin[:, min_max_list[1][0]:min_max_list[1][1], min_max_list[0][0]:min_max_list[0][1]]
        del image_filenames[:bin_factor]
        cropped_image = bin_resize(cropped_image, bin_factor)
        save_tif_image(cropped_image[0], output_folder + '{:04d}'.format(index))


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
