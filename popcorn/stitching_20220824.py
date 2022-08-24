import os
import sys
import gc
import glob
import shutil
import time
import tkinter as tk
from tkinter import filedialog, simpledialog
import PySimpleGUI as sg

from skimage import filters
import numpy as np

from popcorn.input_output import open_image, open_sequence, save_tif_image, open_cropped_sequence, save_tif_sequence, \
    open_cropped_image, create_list_of_files
from popcorn.spectral_imaging.registration import registration_computation, apply_itk_transformation


def stitch_multiple_folders_into_one(list_of_folders, output_folder, delta_z, look_for_best_slice=True, copy_mode=0,
                                     security_band_size=10, overlap_mode=0, band_average_size=0, flip=False):
    """Function that stitches different folders into a unique one.

    Notes:
        1. First and last folders are treated differently (correlation with other folders is looked only on one side).
        2. Every other slices are moved or copied (depending on copy mode) in the output_folder.
        3. For the rest of folders, the slices are either moved or copied (depending on copy mode).

    Args:
        list_of_folders (list[str]): list of input folders (expects images in each of those folders whatever the format)
        output_folder (str):         complete output total_path
        delta_z (int):               supposed z discrete displacement (number of slices)
        look_for_best_slice (bool):  False: we don't look for best matched slice between folders, True : we do
        copy_mode (int):             0: files are moved (no backup), 1: files are copied in the output_folder
        security_band_size (int):    nb of slices above and below delta_z used for stitching computation
        overlap_mode (int):          0: copy/move files, 1: standard average in band_average_size, 2: weighted average
        band_average_size (int):     If overlap_mode > 0: size of the band for the (weighted) average
        flip (bool):                 True: alphabetic filenames in a folder False: Reverse alphabetic order
    """
    list_of_folders.sort()
    number_of_folders = len(list_of_folders)
    folder_nb = 0
    bottom_overlap_index = 0
    top_overlap_index = 0

    # Parsing all input folders
    for folder_name in list_of_folders:
        print("Stitching step ", str(folder_nb) + "/", str(len(list_of_folders)))

        # We retrieve the list of filenames in the very first folder
        bottom_image_filenames = glob.glob(folder_name + '/*.tif') + glob.glob(folder_name + '/*.edf') \
                                 + glob.glob(folder_name + '/*.png')

        if flip:
            bottom_image_filenames.sort(reverse=True)
        else:
            bottom_image_filenames.sort()

        nb_slices = len(bottom_image_filenames)

        # We compute stitching on all folders (N folders implies N-1 stitching computations)
        if folder_nb < number_of_folders - 1:
            # We retrieve the list of filenames for the folder next to the current one
            top_image_filenames = glob.glob(list_of_folders[folder_nb + 1] + '/*.tif') \
                                  + glob.glob(list_of_folders[folder_nb + 1] + '/*.edf') \
                                  + glob.glob(list_of_folders[folder_nb + 1] + '/*.png')
            if flip:
                top_image_filenames.sort(reverse=True)
            else:
                top_image_filenames.sort()

            # We use delta_z to determine the theoretical overlapping slice index
            supposed_bottom_overlap_slice = nb_slices - int((nb_slices - delta_z) / 2)
            supposed_top_overlap_slice = int((nb_slices - delta_z) / 2)

            # We're computing stitching on a band (not only one image)
            if security_band_size > 0:
                # If we don't trust delta_z value
                if look_for_best_slice:
                    # We only keep the filenames of the bands used for stitching computation
                    bottom_band_filenames = \
                        bottom_image_filenames[supposed_bottom_overlap_slice - int(security_band_size):
                                               supposed_bottom_overlap_slice + int(security_band_size)]
                    top_band_filenames = \
                        top_image_filenames[supposed_top_overlap_slice - int(security_band_size):
                                            supposed_top_overlap_slice + int(security_band_size)]

                    # We load the corresponding bands
                    bottom_band_image = open_sequence(bottom_band_filenames, imtype=np.uint16)
                    top_band_image = open_sequence(top_band_filenames, imtype=np.uint16)

                    # Stitching computation. Returns the overlapping slices index between given bands
                    overlap_index = int(look_for_maximum_correlation_band(bottom_band_image, top_band_image, 10, True))

                    # We compute the difference between theoretical overlap index and real overlap index
                    overlap_index_difference = security_band_size - overlap_index
                # If we trust delta_z value, we set the difference between theory and practice to 0
                else:
                    overlap_index_difference = 0

                # We compute for overlap index for the current folder
                bottom_overlap_index = supposed_bottom_overlap_slice + overlap_index_difference

                # List of filenames from current folder we need to copy
                list_to_copy = bottom_image_filenames[top_overlap_index:bottom_overlap_index]

                # If we do not average images
                if overlap_mode == 0:
                    for slice_index in range(0, len(list_to_copy)):
                        # If the filenames are in reverse order
                        if flip == 1:
                            output_filename = output_folder + '/' + os.path.basename(list_to_copy[-(slice_index + 1)])
                        else:
                            output_filename = output_folder + '/' + os.path.basename(list_to_copy[slice_index])
                        # We either copy or move files depending on copy_mode
                        if copy_mode == 0:
                            os.rename(list_to_copy[slice_index], output_filename)
                        else:
                            shutil.copy2(list_to_copy[slice_index], output_filename)

                    # In case of no average, the overlapping index in the next folder is the supposed one
                    top_overlap_index = supposed_top_overlap_slice
                else:
                    for slice_index in range(0, len(list_to_copy)):
                        # If the filenames are in reverse order
                        if flip:
                            output_filename = output_folder + '/' + os.path.basename(list_to_copy[-(slice_index + 1)])
                        else:
                            output_filename = output_folder + '/' + os.path.basename(list_to_copy[slice_index])

                        # We either copy or move files depending on copy_mode
                        if copy_mode == 0:
                            os.rename(list_to_copy[slice_index], output_filename)
                        else:
                            shutil.copy2(list_to_copy[slice_index], output_filename)

                    # We retrieve the filenames used for averaging
                    bottom_average_filenames = \
                        bottom_image_filenames[bottom_overlap_index - int(band_average_size / 2):
                                               bottom_overlap_index + int(band_average_size / 2)]

                    top_average_filenames = \
                        top_image_filenames[supposed_top_overlap_slice +
                                            overlap_index_difference - int(band_average_size / 2):
                                            supposed_top_overlap_slice +
                                            overlap_index_difference + int(band_average_size / 2)]
                    # We compute the average between the two images depending on
                    averaged_image = average_images_from_filenames(bottom_average_filenames, top_average_filenames,
                                                                   overlap_mode)
                    # We save the averaged images
                    list_of_new_filenames = bottom_image_filenames[bottom_overlap_index - int(band_average_size / 2):
                                                                   bottom_overlap_index + int(band_average_size / 2)]

                    for filename in list_of_new_filenames:
                        output_filename = output_folder + os.path.basename(filename)
                        for i in range(0, band_average_size):
                            slice_data = averaged_image[i, :, :].squeeze()
                            save_tif_image(slice_data.astype(np.uint16), output_filename, bit=16)

                    # In case of no average, the overlapping index in the next folder is
                    # the supposed one + half of average band
                    top_overlap_index = supposed_top_overlap_slice + overlap_index_difference + int(
                        band_average_size / 2)

            # If the security_band_size is not > 0
            else:
                sys.exit("Please use a security_band_size > 0")
        # Once we computed stitching on all folders, we copy the remaining files (from the last folder)
        else:
            list_to_copy = bottom_image_filenames[top_overlap_index:-1]
            for slice_index in range(0, len(list_to_copy)):
                # If the filenames are in reverse order
                if flip:
                    output_filename = output_folder + '/' + os.path.basename(list_to_copy[-(slice_index + 1)])
                else:
                    output_filename = output_folder + '/' + os.path.basename(list_to_copy[slice_index])

                # We either copy or move files depending on copy_mode
                if copy_mode == 0:
                    os.rename(list_to_copy[slice_index], output_filename)
                else:
                    shutil.copy2(list_to_copy[slice_index], output_filename)
        print(" > corresponding slices found: slice", bottom_overlap_index, "and slice", top_overlap_index)
        folder_nb += 1


def average_images_from_filenames(first_image_filenames, second_image_filenames, mode=1):
    """Averages two images

    Args:
        first_image_filenames (list[str]):  list of first image filenames
        second_image_filenames (list[str]): list of second image filenames
        mode (int):                         1: standard average, 2: weighted average TODO

    Returns:
        numpy.ndarray: averaged image
    """
    # Opens image
    first_image = open_sequence(first_image_filenames, imtype=np.uint16)
    second_image = open_sequence(second_image_filenames, imtype=np.uint16)

    # If standard average requested
    if mode == 1:
        return (first_image + second_image) / 2
    # If weighted average requested
    else:
        return (first_image + second_image) / 2


def look_for_maximum_correlation(first_image, second_image):
    """Looks for the maximum correlated slice between two images

    The computation is only performed with the slice in the middle of first image and on the entire second image

    Args:
        first_image (numpy.ndarray):  first image
        second_image (numpy.ndarray): second image

    Returns:
        int: the slice number with highest zero normalized cross correlation.
    """
    first_nb_slices, first_width, first_height = first_image.shape
    second_nb_slices, second_width, second_height = second_image.shape

    width = max(first_width, second_width)
    height = max(first_height, second_height)

    middle_slice = int(first_nb_slices / 2)

    # We compute what we need for normalized cross correlation (first image middle slice)
    first_image_middle_slice = np.copy(first_image[middle_slice, :, :].squeeze())
    first_image_middle_slice = first_image_middle_slice - np.mean(first_image_middle_slice)
    first_image_middle_slice_std = np.std(first_image_middle_slice)

    # We compute what we need for normalized cross correlation (second image)
    centered_second_image = np.copy(second_image)
    for slice_nb in range(second_nb_slices):
        centered_second_image[slice_nb, :, :] = centered_second_image[slice_nb, :, :] \
                                                - np.mean(centered_second_image[slice_nb, :, :])
    centered_images_multiplication_result = first_image_middle_slice * centered_second_image

    # We compute normalized cross-correlation between first image middle slice and all second image slices
    normalized_cross_correlations = np.zeros(second_nb_slices)
    for slice_nb in range(0, second_nb_slices):
        second_image_slice_std = np.std(centered_second_image[slice_nb, :, :])
        sum_of_multiplied_images = np.sum(centered_images_multiplication_result[slice_nb, :, :])
        normalized_cross_correlation = sum_of_multiplied_images / (
                first_image_middle_slice_std * second_image_slice_std)
        normalized_cross_correlation /= (width * height)
        normalized_cross_correlations[slice_nb] = normalized_cross_correlation  # array of normalized-cross correlations

    # The best candidate corresponds to the nb with max normalized cross-correlation
    best_corresponding_slice_nb = np.argmax(normalized_cross_correlations)

    return best_corresponding_slice_nb


def look_for_maximum_correlation_band(first_image, second_image, band_size, with_segmentation=True):
    """Looks for the maximum correlated slice between two images

    The computation is performed for every slices in a band of band_size centered around the first image middle slice

    Args:
        first_image (numpy.ndarray):  first image
        second_image (numpy.ndarray): second image
        band_size (int):              nb of slices above/below middle slice for computation
        with_segmentation (bool):     True: we perform thresholding, False: we don't

    Returns:
        int: the slice number with highest zero normalized cross correlation.
    """
    nb_slices, width, height = first_image.shape
    mask = np.zeros(first_image.shape, np.uint16)
    middle_slice_nb = int(nb_slices / 2)

    first_image_copy = np.copy(first_image)
    centered_second_image = np.copy(second_image)

    # If a thresholding is requested, we use Otsu thresholding on top 85% of the first image histogram
    if with_segmentation:
        thresh = filters.threshold_otsu(first_image_copy[first_image_copy > 0.15 * np.amax(first_image_copy)])
        if (first_image_copy[first_image_copy > thresh]).size / first_image_copy.size < 0.005:
            thresh = filters.threshold_otsu(first_image_copy)
        mask = first_image_copy > thresh
        first_image_copy = mask * first_image_copy
        centered_second_image = mask * centered_second_image

    # We compute what we need for normalized cross correlation (second image)
    for slice_nb in range(nb_slices):
        second_image_slice = centered_second_image[slice_nb, :, :]
        if with_segmentation:
            centered_second_image[slice_nb, :, :] = \
                mask[slice_nb, :, :] * (second_image_slice - np.mean(second_image_slice[second_image_slice > 0.0]))
        else:
            centered_second_image[slice_nb, :, :] = centered_second_image[slice_nb, :, :] \
                                                    - np.mean(centered_second_image[slice_nb, :, :])

    # We parse every slice of first_image[-band_size/2: band_size/2]
    best_slice_candidates = np.zeros(band_size)
    for i in range(int(-band_size / 2), (int(band_size / 2))):
        first_image_middle_slice = first_image_copy[middle_slice_nb + i, :, :].squeeze()
        # In case of thresholding, we use the computed mask on the current slice for computation
        if with_segmentation:
            first_image_middle_slice = \
                mask[middle_slice_nb + i, :, :] * \
                (first_image_middle_slice - np.mean(first_image_middle_slice[first_image_middle_slice > 0.0]))
        # In case of no thresholding, we don't use the mask for computation
        else:
            first_image_middle_slice = first_image_middle_slice - np.mean(first_image_middle_slice)
        first_image_middle_slice_std = np.std(first_image_middle_slice)
        normalized_cross_correlations = np.zeros(nb_slices)
        centered_images_multiplication_result = first_image_middle_slice * centered_second_image

        # We parse every slice of second image to compute normalized cross-correlations
        for slice_nb in range(nb_slices):
            centered_second_image_std = np.std(centered_second_image[slice_nb, :, :])
            sum_of_multiplied_images = np.sum(centered_images_multiplication_result[slice_nb, :, :])

            normalized_cross_correlation = \
                sum_of_multiplied_images / (first_image_middle_slice_std * centered_second_image_std)
            normalized_cross_correlation /= (width * height)
            normalized_cross_correlations[slice_nb] = normalized_cross_correlation  # arrays of normalized cross-corr

        # We store the best candidate for overlapping slice for each first image slice.
        best_corresponding_slice_nb = np.argmax(normalized_cross_correlations) - i
        best_slice_candidates[i + int(band_size / 2)] = best_corresponding_slice_nb

    # We finally retrieve the final best candidate (victory royale)
    computed_corresponding_slice_nb = np.median(best_slice_candidates)
    return computed_corresponding_slice_nb


def rearrange_folders_list(starting_position, number_of_lines, number_of_columns):
    """Sorts indices of multiple-tiles image based on starting position and size of the grid

    Args:
        starting_position (str): Position of first tile (either top-left, top-right, bottom-left or bottom-right)
        number_of_lines (int):   Number of lines in the final grid
        number_of_columns (int): Number of columns in the final grid

    Returns (list[int]): list of sorted indices

    """
    list_of_folders = list(range(0, number_of_lines * number_of_columns))
    for nb_line in range(number_of_lines):
        if "meader" in starting_position:
            if (nb_line + ("left" in starting_position) * 1) % 2 == 0:
                list_of_folders[nb_line * number_of_columns: nb_line * number_of_columns + number_of_columns] = \
                    list(reversed(
                        list_of_folders[nb_line * number_of_columns: nb_line * number_of_columns + number_of_columns]))

    if "bottom" in starting_position:
        new_list_of_folders = list(range(1, number_of_lines * number_of_columns + 1))
        for nb_line in range(number_of_lines):
            if nb_line * number_of_columns > 0:
                new_list_of_folders[nb_line * number_of_columns:
                                    nb_line * number_of_columns + number_of_columns] = \
                    list_of_folders[-(nb_line * number_of_columns + number_of_columns):
                                    -(nb_line * number_of_columns)]
            else:
                new_list_of_folders[nb_line * number_of_columns:
                                    nb_line * number_of_columns + number_of_columns] = \
                    list_of_folders[-(nb_line * number_of_columns + number_of_columns):]
    else:
        new_list_of_folders = list_of_folders
    return new_list_of_folders


def compute_two_tiles_registration(ref_image_input_folder, ref_image_coordinates, moving_image_input_folder,
                                   moving_image_coordinates):
    """ Computes and returns offset between overlap of two images

    Args:
        ref_image_input_folder (str):
        ref_image_coordinates (list[list[int]]):
        moving_image_input_folder (str):
        moving_image_coordinates (list[list[int]]):

    Returns:
        (Sitk.Transform): computed transformation

    """
    # We open the overlapping part of the ref image
    ref_image = open_cropped_sequence(glob.glob(ref_image_input_folder + "\\*"), ref_image_coordinates)

    # We compute the mask the registration will be based on (otsu threshold) -> faster registration
    threshold = filters.threshold_otsu(ref_image)
    ref_mask = np.copy(ref_image)
    ref_mask[ref_mask <= threshold] = 0
    ref_mask[ref_mask > threshold] = 1

    # We open the overlapping part of the moving image
    moving_image = open_cropped_sequence(glob.glob(moving_image_input_folder + "\\*"), moving_image_coordinates)
    moving_mask = np.copy(moving_image)

    # We compute the mask the registration will be based on (Otsu threshold) -> faster registration
    moving_mask[moving_mask <= threshold] = 0
    moving_mask[moving_mask > threshold] = 1

    # Registration computation (translation only)
    return registration_computation(moving_image=moving_image, ref_image=ref_image, ref_mask=ref_mask,
                                    moving_mask=moving_mask, transform_type="translation",
                                    metric="msq", verbose=False)


def multiple_tile_registration(input_folder, radix, starting_position="top-left-straight",
                               number_of_lines=4, number_of_columns=3,
                               supposed_overlap=120, integer_values_for_offset=False, verbose=False):
    """Stitches multiple 3D tiles altogether and saves the result in a "final image" folder

    Args:
        input_folder (str):               Input tiles folder
        radix (str):                      Regex of input images
        starting_position (str):          Position of first tile (either top-left, top-right, bottom-left or bottom-right)
        number_of_lines (int):            Number of lines in the final grid
        number_of_columns (int):          Number of columns in the final grid
        supposed_overlap (int):           Theoretical overlap between each images
        integer_values_for_offset (bool): Integer values for registration ? (to avoid interpolation)

    Returns (None):

    """
    if verbose:
        full_time_start = time.time()
        time_start = time.time()
    # Sorts indices of input tiles so that every tile is registered from left to right & from top to bottom.
    folders_indices = rearrange_folders_list(starting_position, number_of_lines, number_of_columns)

    # Get info if there are several channels
    list_of_channels = glob.glob(input_folder + "*")
    number_of_channels = len(list_of_channels)

    list_of_channels_folders = []
    for n in range(number_of_channels):
        # Listing input folders
        temp_parent = os.path.join(list_of_channels[n], radix + "*", "")
        list_of_channels_folders.append(glob.glob(temp_parent))
    # Path to the very first slide of the very first tile of the first channel (upper left corner)
    reference_image_path = glob.glob(list_of_channels_folders[0][0] + "*")[0]
    # Opening the reference image for reference height/width (should be 1920x1920 in HR)
    reference_image = open_image(reference_image_path)
    # Get the number of slices
    temp_list = glob.glob(list_of_channels_folders[0][0] + "*")
    nb_of_slices = len(temp_list)
    # Check that the number of slice is the same for all channels
    for n in range(number_of_channels):
        temp_list = glob.glob(list_of_channels_folders[n][0] + "*")
        if len(temp_list) != nb_of_slices:
            print("The number of slice in channel ", n, "is different from the first channel")
            exit()

    ref_height = reference_image.shape[0]
    ref_width = reference_image.shape[1]
    # Define the maximum value for displacement
    offset_max = (120, 120, 20)  # (60, 60, 10)  #(120, 120, 20)  ## (120, 60, 10)

    # Registration is computed on a subpart of each tile,
    # ref_image_coordinates correspond to the coordinates of the right sub-part of the left image
    # moving_image_coordinates correspond to the coordinates of the left sub-part of the right image
    ref_image_coordinates = [[0, nb_of_slices - 1],
                             [0, reference_image.shape[0] - 1],
                             [max(reference_image.shape[1] - supposed_overlap, 0), reference_image.shape[1] - 1]]

    moving_image_coordinates = [[0, nb_of_slices - 1],
                                [0, reference_image.shape[0] - 1],
                                [0, min(supposed_overlap, reference_image.shape[1]) - 1]]
    list_of_transformations = []
    if verbose:
        print("1. Introduction time:", (time.time() - time_start))

    time_start = time.time()
    #comment="""
    # 1. Registration computation, for each line of the final grid, we compute the offset between each neighbour tiles
    # Registration is computed on the first channel (channel 0)
    # Registered tile are saved in registered_tile__XX folders
    for nb_line in range(number_of_lines):
        for nb_col in range(number_of_columns - 1):  # We compute registration on number_of_columns - 1 pairs of images
            image_number = nb_line * number_of_columns + nb_col  # We keep track of the ref_image number we're working on

            print("-"*10)
            print("Stitching tile number", folders_indices[image_number],
                  "and tile number", folders_indices[image_number + 1])

            reg_start = time.time()
            transformation = compute_two_tiles_registration(
                list_of_channels_folders[0][folders_indices[image_number]],
                ref_image_coordinates,
                list_of_channels_folders[0][folders_indices[image_number + 1]],
                moving_image_coordinates)
            print("Registration time:", (time.time() - reg_start))
            # If we want to avoid interpolation -> integer offset
            if integer_values_for_offset:
                transformation.SetOffset((round(transformation.GetOffset()[0]),
                                          round(transformation.GetOffset()[1]),
                                          round(transformation.GetOffset()[2])))
            print("Offset :", transformation.GetParameters())
            offset_computed = transformation.GetParameters()
            if any(np.abs(x) > y for x, y in zip(offset_computed, offset_max)):
                new_offset_computed = list(offset_computed)
                for x in range(len(offset_computed)):
                    if np.abs(offset_computed[x]) > offset_max[x]:
                        new_offset_computed[x] = 0
                offset_computed = tuple(new_offset_computed)
                transformation.SetParameters(offset_computed)
                print("Final Offset (after correction):", transformation.GetParameters())
            list_of_transformations.append(transformation)

    # Application of the computed transformation on each channel
    for nb_line in range(number_of_lines):
        for nb_col in range(number_of_columns):
            image_number = nb_line * number_of_columns + nb_col
            # == Apply transformation in a larger (/X) image in order to keep pixels if overlap < expected
            for n in range(number_of_channels):
                gc.collect()
                tile = open_sequence(list_of_channels_folders[n][image_number] + "\\", imtype=np.uint16)
                if nb_col > 0:
                    range_trsf = np.arange(nb_line * (number_of_columns - 1),
                                           nb_line * (number_of_columns - 1) + nb_col)
                    for nb_transformation in range_trsf:
                        ext_x = np.ceil(list_of_transformations[nb_transformation].GetParameters()[0])
                        if ext_x < 0:
                            ext_tile = np.zeros((tile.shape[0], tile.shape[1],
                                                 np.absolute(supposed_overlap + ext_x).astype(np.uint16)),
                                                dtype=np.uint16)
                            extended_tile = np.append(tile, ext_tile,axis=2)
                            tile = extended_tile
                        tile = apply_itk_transformation(tile, list_of_transformations[nb_transformation])
                temp_folder = os.path.join(list_of_channels[n], "registered_tile_" + str(image_number), "")
                save_tif_sequence(tile, temp_folder, bit=16)
                print("Registered tile number", image_number, "channel", n, "saved !")

    if verbose:
        print("2. First Registrations time:", (time.time() - time_start))
        time_start = time.time()

    # 2. Concatenation of same line tiles, line are saved in combined_line_XX folders
    for nb_line in range(number_of_lines):
        for n in range(number_of_channels):
            list_of_list_of_tile_images = []
            list_of_len_tile = []
            # For this purpose, we need to list all input slices
            for nb_col in range(number_of_columns):
                idx = nb_line * number_of_columns + nb_col
                temp_folder = os.path.join(list_of_channels[n], "registered_tile_" + str(folders_indices[idx]), "")
                list_of_list_of_tile_images.append(create_list_of_files(temp_folder, "tif"))
                list_of_len_tile.append(len(list_of_list_of_tile_images[-1]))

            # We add each tile one after the other, slice by slice
            for nb_slice in range(min(list_of_len_tile)):
                x_position = 0
                full_width = ref_width * number_of_columns  ##- supposed_overlap * (number_of_columns - 1)
                empty_slice = np.zeros((ref_height, full_width), dtype=np.uint16)
                overlap_final_x = supposed_overlap
                for nb_col in range(number_of_columns):
                    image_number = nb_line * number_of_columns + nb_col
                    trsf_idx = nb_line * (number_of_columns-1) + nb_col - 1
                    #print("Registration of Tile #", image_number)

                    # The first tile doesn't need any registration but need to be cropped until the overlapping area
                    if nb_col == 0:
                        out_ref_slice = open_image(list_of_list_of_tile_images[nb_col][nb_slice])

                        # Adapt overlap transformation given the final overlap in X
                        #print("Transformation ids:", trsf_idx+1, "with Tile #", image_number+1)
                        overlap_final_x = np.ceil(
                            supposed_overlap
                            + list_of_transformations[trsf_idx+1].GetParameters()[0]).astype(np.int16)

                        #print("Line:", nb_line, "\nFinal overlap in X between col", image_number, "and col", image_number + 1, "in /X (pix):", overlap_final_x)
                        x_position = ref_width - overlap_final_x
                        empty_slice[:, 0:x_position] = out_ref_slice[:, 0:x_position]

                    # The next tile need to be registered/cropped
                    else:
                        # -- Manage overlapping part
                        out_ref_slice = open_image(list_of_list_of_tile_images[nb_col - 1][nb_slice])
                        out_mov_slice = open_image(list_of_list_of_tile_images[nb_col][nb_slice])

                        # Adapt overlap window given the final overlap in X
                        #print("Transformation ids:", trsf_idx, "with Tile #", image_number-1)
                        overlap_final_x = np.ceil(np.absolute(
                            supposed_overlap
                            + list_of_transformations[trsf_idx].GetParameters()[0])).astype(np.int16)
                        #print("Line:", nb_line, "\nFinal overlap in X between col", trsf_idx-1, "and col", trsf_idx, "in /X (pix):", overlap_final_x)
                        overlapped_ref_slice = np.zeros((ref_height, overlap_final_x), dtype=np.uint16)
                        overlapped_mov_slice = np.zeros((ref_height, overlap_final_x), dtype=np.uint16)

                        # Compute weight matrix for overlapping part along the width axis
                        comment = """
                        # version 1: linear
                        x_line = np.arange(overlap_final_x)
                        y_line = -1 / (x_line[-1] - x_line[0]) * (x_line - x_line[-1])
                        w_1d_W = y_line
                        w_1d_W = w_1d_W.astype('e')
                        w_2d_W = np.tile(w_1d_W, (ref_height, 1))
                        """

                        # version 2: 1 (1/3) - linear from 1 to 0 (1/3) - 0 (1/3)
                        mixing_size = overlap_final_x // 3
                        bef_size = (overlap_final_x - mixing_size)//2
                        aft_size = overlap_final_x - mixing_size - bef_size
                        x_line = np.arange(mixing_size) + mixing_size
                        y_line = -1 / (x_line[-1] - x_line[0]) * (x_line - x_line[-1])
                        w_1d_W = np.concatenate((np.ones(bef_size), y_line, np.zeros(aft_size)))
                        w_1d_W = w_1d_W.astype('e')
                        w_2d_W = np.tile(w_1d_W, (ref_height, 1))

                        # Then there is a need to handle the overlap between the 2 tiles
                        x_pos_ref_start = ref_width - overlap_final_x + (nb_col-1) * (supposed_overlap-overlap_final_x)
                        if x_pos_ref_start+overlap_final_x > out_ref_slice.shape[1]:
                            slice_to_copy = out_ref_slice[:, x_pos_ref_start:]
                            overlapped_ref_slice[:, :slice_to_copy.shape[1]] = slice_to_copy
                        else:
                            slice_to_copy = out_ref_slice[:, x_pos_ref_start:x_pos_ref_start+overlap_final_x]
                            overlapped_ref_slice = slice_to_copy
                        x_pos_mov_start = (nb_col) * (supposed_overlap-overlap_final_x)
                        overlapped_mov_slice = out_mov_slice[:, x_pos_mov_start:x_pos_mov_start+overlap_final_x]
                        # Apply weight to each overlapping parts
                        woverlapped_ref_slice = overlapped_ref_slice * w_2d_W
                        woverlapped_mov_slice = overlapped_mov_slice * (1 - w_2d_W)
                        # Compute the sum of the two weighted overlapping parts
                        overlapped_section = np.nansum(np.array([woverlapped_ref_slice, woverlapped_mov_slice]), axis=0)
                        # Copy the resulting overlap part to the final image
                        empty_slice[:, x_position:x_position + overlap_final_x] = overlapped_section
                        x_position += overlap_final_x

                        # -- Manage non-overlapping part = Copy non-overlapping part as it is
                        if nb_col == number_of_columns - 1:
                            x_pos_mov_start = (nb_col) * (supposed_overlap - overlap_final_x) + overlap_final_x
                            x_pos_mov_stop = x_pos_mov_start + (ref_width - overlap_final_x)
                            slice_to_copy = out_mov_slice[:, x_pos_mov_start:x_pos_mov_stop]
                            empty_slice[:, x_position:x_position + slice_to_copy.shape[1]] = slice_to_copy
                        else:
                            # Prepare for next column if there is
                            #print("Transformation ids:", trsf_idx+1, "with Tile #", image_number + 1)
                            next_overlap_final_x = np.ceil(np.absolute(
                                supposed_overlap
                                + list_of_transformations[trsf_idx+1].GetParameters()[0])).astype(np.int16)
                            #next_x_position = ref_width - next_overlap_final_x
                            x_pos_mov_start = (nb_col) * (supposed_overlap - overlap_final_x) + overlap_final_x
                            x_pos_mov_stop = x_pos_mov_start + (ref_width - overlap_final_x - next_overlap_final_x)

                            slice_to_copy = out_mov_slice[:, x_pos_mov_start:x_pos_mov_stop]
                            empty_slice[:, x_position:x_position + slice_to_copy.shape[1]] = slice_to_copy
                            x_position += slice_to_copy.shape[1]

                temp_folder = os.path.join(list_of_channels[n], "combined_line_" + str(nb_line), "")
                if verbose:
                    print("-> Saving slice number", nb_slice, "in folder", temp_folder)
                save_tif_image(empty_slice, temp_folder + '{:04d}'.format(nb_slice), bit=16)
                if verbose:
                    print("--> Slice number", nb_slice, "saved !")
                if 'empty_slice' in locals():
                    del empty_slice
                if 'overlapped_ref_slice' in locals():
                    del overlapped_ref_slice
                if 'overlapped_mov_slice' in locals():
                    del overlapped_mov_slice
                if 'overlapped_section_stack' in locals():
                    del overlapped_section_stack
                if 'woverlapped_ref_slice' in locals():
                    del woverlapped_ref_slice
                if 'woverlapped_mov_slice' in locals():
                    del woverlapped_mov_slice
                if 'w_1d_W' in locals():
                    del w_1d_W
                if 'w_2d_W' in locals():
                    del w_2d_W
                gc.collect()

    if verbose:
        print("2. Line saving time:", (time.time() - time_start))
        time_start = time.time()

    # 2bis. Crop Black pixel row of combine line
    # https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy

    def crop_image_outside_xyz(img, tol=0):
        # img is 2D or 3D image data
        # tol  is tolerance
        mask = img > tol
        slice_size, row_size, col_size = mask.shape
        mask0, mask1 = mask.any(0), mask.any(1)
        coords_yx = np.argwhere(mask0)
        # Bounding box of non-black pixels.
        r_start, c_start = coords_yx.min(axis=0)
        r_end, c_end = coords_yx.max(axis=0) + 1
        coords_zx = np.argwhere(mask1)
        # Bounding box of non-black pixels.
        s_start, _ = coords_zx.min(axis=0)
        s_end, _ = coords_zx.max(axis=0) + 1
        return s_start, s_end, r_start, r_end, c_start, c_end

    for nb_line in range(number_of_lines):
        # a. read in first channel 2DTIFF sequence
        in_img_path = os.path.join(list_of_channels[0], "combined_line_" + str(nb_line))
        sequence = open_sequence(in_img_path, imtype=np.uint16)
        # b. get row coordinates for cropping
        _, _, row_start, row_end, _, _ = crop_image_outside_xyz(sequence, tol=0)
        print("Kept rows in line ", nb_line, ": [", row_start, ",", row_end, "]")

        for n in range(number_of_channels):
            in_img_path = os.path.join(list_of_channels[n], "combined_line_" + str(nb_line))
            sequence = open_sequence(in_img_path, imtype=np.uint16)
            cropped_sequence = sequence[:, row_start:row_end, :]
            # c. save 2DTIFF sequence as "cropped combine_line"
            temp_folder = os.path.join(list_of_channels[n], "cropped_combined_line_" + str(nb_line), "")
            save_tif_sequence(cropped_sequence, temp_folder, bit=16)
            print("Cropped line number", nb_line, "channel", n, "saved !")
    #"""
    #comment="""
    # 3. Registration computation, we compute the offset between each neighboring lines
    # Registration is computed on the first channel (channel 0)
    list_of_transformations = []
    for nb_line in range(number_of_lines - 1):
        print("-" * 10)
        print("Stitching line number", nb_line, "and line number", nb_line + 1)

        # This time, the registration is vertical, not horizontal. The overlapping parts correspond to the
        # bottom/upper-end of each line
        reg_start = time.time()
        transformation = compute_two_tiles_registration(
            os.path.join(list_of_channels[0], "cropped_combined_line_" + str(nb_line)),
            [[0, -1], [-supposed_overlap, -1], [0, -1]],
            os.path.join(list_of_channels[0], "cropped_combined_line_" + str(nb_line + 1)),
            [[0, -1], [0, supposed_overlap], [0, -1]])
        print("Registration time:", (time.time() - reg_start))

        # If we want to avoid interpolation -> integer offset
        if integer_values_for_offset:
            transformation.SetOffset((round(transformation.GetOffset()[0]),
                                      round(transformation.GetOffset()[1]),
                                      round(transformation.GetOffset()[2])))
        print("Offset :", transformation.GetParameters())
        offset_computed = transformation.GetParameters()
        if any(np.abs(x) > y for x, y in zip(offset_computed, offset_max)):
            new_offset_computed = list(offset_computed)
            for x in range(len(offset_computed)):
                if np.abs(offset_computed[x]) > offset_max[x]:
                    new_offset_computed[x] = 0
            offset_computed = tuple(new_offset_computed)
            transformation.SetParameters(offset_computed)
            print("Final Offset (after correction):", transformation.GetParameters())
        list_of_transformations.append(transformation)

    # """
    # comment="""
    # 4. Registration of each line (Because of memory limitations)
    # Application of the registration on each channel
    # Registered line are saved in registered_line_XX folders (expect line 0 which has no transformation)
    for nb_line in range(number_of_lines - 1):
        for n in range(number_of_channels):
            gc.collect()
            line = open_sequence(os.path.join(list_of_channels[n], "cropped_combined_line_" + str(nb_line + 1), ""),
                                 imtype=np.uint16)
            for nb_transformation in range(nb_line + 1):
                line = apply_itk_transformation(line, list_of_transformations[nb_transformation])
            save_tif_sequence(line, os.path.join(list_of_channels[n], "registered_line_" + str(nb_line), ""), bit=16)
            print("Registered line number", nb_line, "saved !")
    if verbose:
        print("3. Second Registrations time:", (time.time() - time_start))
        time_start = time.time()

    #"""
    # comment="""
    # 5. Concatenation of every lines of the final grid, one channel at a time
    for n in range(number_of_channels):
        list_of_line_images = []
        list_of_len = []

        # For this purpose, we need to list all input slices
        for nb_line in range(number_of_lines):
            if nb_line == 0:
                temp_folder = os.path.join(list_of_channels[n], "cropped_combined_line_" + str(nb_line), "")
                list_of_line_images.append(create_list_of_files(temp_folder, "tif"))
            else:
                temp_folder = os.path.join(list_of_channels[n], "registered_line_" + str(nb_line - 1), "")
                list_of_line_images.append(create_list_of_files(temp_folder, "tif"))
            list_of_len.append(len(list_of_line_images[-1]))

        # We add each line one after the other, slice by slice
        for nb_slice in range(min(list_of_len)):
            y_position = 0
            full_width = ref_width * number_of_columns ##- supposed_overlap * (number_of_columns - 1)
            full_height = ref_height * number_of_lines ##- supposed_overlap * (number_of_lines - 1)
            empty_slice = np.zeros((full_height, full_width), dtype=np.uint16)
            overlap_final_y = supposed_overlap
            overlap_with_prev_line = 0
            overlap_with_next_line = 0
            # Concatenation
            for nb_line in range(number_of_lines):
                image_number = nb_line
                trsf_idx = nb_line - 1
                #print("Registration of Line #", image_number)

                # The first line doesn't need any registration but need to be cropped until the overlapping area
                if nb_line == 0:
                    out_ref_slice = open_image(list_of_line_images[nb_line][nb_slice])

                    # Adapt overlap transformation given the final overlap in Y
                    #("Transformation ids:", trsf_idx+1, "with Line #", image_number+1)
                    overlap_final_y = np.ceil(
                        supposed_overlap
                        + list_of_transformations[trsf_idx + 1].GetParameters()[1]).astype(np.int16)

                    y_position = ref_height - overlap_final_y
                    empty_slice[0:y_position, :] = out_ref_slice[0:y_position, :]
                    ##overlapped_ref_slice = out_slice[y_position:, :]

                # The next tile need to be registered/cropped
                else:
                    # -- Manage overlapping part
                    out_ref_slice = open_image(list_of_line_images[nb_line - 1][nb_slice])
                    out_mov_slice = open_image(list_of_line_images[nb_line][nb_slice])

                    # Adapt overlap window given the final overlap in Y
                    #print("Transformation ids:", trsf_idx, "with Line #", image_number-1)
                    overlap_final_y = np.ceil(np.absolute(
                        supposed_overlap
                        + list_of_transformations[trsf_idx].GetParameters()[1])).astype(np.int16)
                    # print("Line:", nb_line, "\nFinal overlap in X between col", trsf_idx-1, "and col", trsf_idx, "in /X (pix):", overlap_final_x)
                    overlapped_ref_slice = np.zeros((overlap_final_y, full_width), dtype=np.uint16)
                    overlapped_mov_slice = np.zeros((overlap_final_y, full_width), dtype=np.uint16)

                    # Compute weight matrix for overlapping part along the height axis
                    # version 2: 1 (1/3) - linear from 1 to 0 (1/3) - 0 (1/3)
                    mixing_size = overlap_final_y // 3
                    bef_size = (overlap_final_y - mixing_size) // 2
                    aft_size = overlap_final_y - mixing_size - bef_size
                    x_line = np.arange(mixing_size) + mixing_size
                    y_line = -1 / (x_line[-1] - x_line[0]) * (x_line - x_line[-1])
                    w_1d_H = np.concatenate((np.ones(bef_size), y_line, np.zeros(aft_size)))
                    w_1d_H = w_1d_H.astype('e')
                    w_1d_H = np.reshape(w_1d_H, (-1, 1))
                    w_2d_H = np.tile(w_1d_H, (1, full_width))

                    # Then there is a need to handle the overlap between the 2 lines
                    # pour line 1:
                    # y_pos_ref_start = ref_height - overlap_with_prev_line - overlap_with_next_line + supposed_overlap
                    # overlapped_ref_slice = out_ref_slice[y_pos_ref_start:y_pos_ref_start+overlap_final_y, :]
                    # pour line 2:
                    # y_pos_ref_start = ref_height - overlap_with_prev_line - diff (-16) + supposed_overlap
                    y_pos_ref_start = ref_height - overlap_final_y + (nb_line - 1) * (supposed_overlap - overlap_final_y)
                    overlapped_ref_slice = out_ref_slice[y_pos_ref_start:y_pos_ref_start+overlap_final_y, :]
                    y_pos_mov_start = (nb_line) * (supposed_overlap-overlap_final_y)
                    y_pos_mov_stop = (nb_line) * (supposed_overlap-overlap_final_y)+overlap_final_y

                    if y_pos_mov_start < 0:
                        slice_to_copy = out_mov_slice[:y_pos_mov_stop, :]
                        overlapped_mov_slice[(overlap_final_y - slice_to_copy.shape[0]):, :] = slice_to_copy
                        overlapped_mov_slice[:(overlap_final_y - slice_to_copy.shape[0]), :] = overlapped_ref_slice[:(overlap_final_y - slice_to_copy.shape[0]), :]
                    else:
                        slice_to_copy = out_mov_slice[y_pos_mov_start:y_pos_mov_stop, :]
                        overlapped_mov_slice = slice_to_copy

                    # Apply weight to each overlapping parts
                    woverlapped_ref_slice = overlapped_ref_slice * w_2d_H
                    woverlapped_mov_slice = overlapped_mov_slice * (1 - w_2d_H)
                    # Compute the sum of the two weighted overlapping parts
                    overlapped_section = np.nansum(np.array([woverlapped_ref_slice, woverlapped_mov_slice]), axis=0)
                    # Copy the resulting overlap part to the final image
                    empty_slice[y_position:y_position + overlap_final_y, :] = overlapped_section
                    y_position += overlap_final_y

                    # -- Manage non-overlapping part = Copy non-overlapping part as it is
                    if nb_line == number_of_lines - 1:
                        y_pos_mov_start = overlap_final_y + (nb_line) * (supposed_overlap - overlap_final_y)
                        y_pos_mov_stop = y_pos_mov_start + (ref_height - overlap_final_y)
                        slice_to_copy = out_mov_slice[y_pos_mov_start:y_pos_mov_stop, :]
                        empty_slice[y_position:y_position + slice_to_copy.shape[0], :] = slice_to_copy
                    else:
                        # Prepare for next line if there is
                        #print("Transformation ids:", trsf_idx+1, "with Line #", image_number + 1)
                        next_overlap_final_y = np.ceil(np.absolute(
                            supposed_overlap
                            + list_of_transformations[trsf_idx + 1].GetParameters()[1])).astype(np.int16)
                        #next_y_position = ref_height - next_overlap_final_y
                        diff_overlap = supposed_overlap - overlap_final_y
                        y_pos_mov_start = overlap_final_y + (nb_line) * diff_overlap
                        if diff_overlap < 0:
                            y_pos_mov_stop = ref_height + diff_overlap - next_overlap_final_y
                        else:
                            y_pos_mov_stop = y_pos_mov_start + (ref_height - overlap_final_y - next_overlap_final_y)
                        slice_to_copy = out_mov_slice[y_pos_mov_start:y_pos_mov_stop, :]
                        empty_slice[y_position:y_position + slice_to_copy.shape[0], :] = slice_to_copy
                        y_position += slice_to_copy.shape[0]
                    overlap_with_prev_line = overlap_final_y
                    overlap_with_next_line = next_overlap_final_y

            temp_folder = os.path.join(list_of_channels[n], "final_image", radix + '{:04d}'.format(nb_slice))
            if verbose:
                print("-> Saving slice number", nb_slice, "in folder", temp_folder)
            save_tif_image(empty_slice, temp_folder, bit=16)

            if 'empty_slice' in locals():
                del empty_slice
            if 'overlapped_ref_slice' in locals():
                del overlapped_ref_slice
            if 'overlapped_mov_slice' in locals():
                del overlapped_mov_slice
            if 'woverlapped_ref_slice' in locals():
                del woverlapped_ref_slice
            if 'woverlapped_mov_slice' in locals():
                del woverlapped_mov_slice
            if 'w_1d_H' in locals():
                del w_1d_H
            if 'w_2d_H' in locals():
                del w_2d_H
            gc.collect()

    if verbose:
        print("4. Second Concatenation time:", (time.time() - time_start))
        print("Total:", (time.time() - full_time_start))

    #"""
    # comment="""
    # 5bis. Crop Black pixel row of combine line
    # https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
    # a. read in first channel 2DTIFF sequence
    in_img_path = os.path.join(list_of_channels[0], "final_image")
    sequence = open_sequence(in_img_path, imtype=np.uint16)
    # b. get slice, row and columns coordinates for cropping
    slice_start, slice_end, row_start, row_end, col_start, col_end = crop_image_outside_xyz(sequence, tol=0)
    print("Kept in line ", nb_line, ", slices: [", slice_start, ",", slice_end, "], \n",
          "rows: [", row_start, ",", row_end, "], \n",
          "columns: [", col_start, ",", col_end, "]")

    for n in range(number_of_channels):
        in_img_path = os.path.join(list_of_channels[n], "final_image")
        sequence = open_sequence(in_img_path, imtype=np.uint16)
        cropped_sequence = sequence[slice_start:slice_end, row_start:row_end, col_start:col_end]
        # c. save 2DTIFF sequence as "cropped final image"
        temp_folder = os.path.join(list_of_channels[n], "cropped_final_image", "")
        save_tif_sequence(cropped_sequence, temp_folder, bit=16)
        print("Cropped final image saved !")

    # 6. Delete intermediate directories
    #list_of_directory_to_remove = glob.glob(os.path.join(input_folder, "*", "registered_*"))
    #list_of_directory_to_remove += glob.glob(os.path.join(input_folder, "*", "combined*"))
    #list_of_directory_to_remove += glob.glob(os.path.join(input_folder, "*", "cropped_combined*"))
    #list_of_directory_to_remove += glob.glob(os.path.join(input_folder, "*", "final*"))

    if 'list_of_directory_to_remove' in locals():
        for nb_dir in range(len(list_of_directory_to_remove)):
            print("dir to remove", list_of_directory_to_remove[nb_dir])
            shutil.rmtree(list_of_directory_to_remove[nb_dir])
    #"""

if __name__ == "__main__":
    # INPUT GUI
    sg.theme("DarkTeal6")
    layout = [
        [sg.T("Please enter the features for registration and merge")],
        [sg.Text("Select Input Directory to TIFF image: "),
         sg.InputText(key="-IODIR-", default_text="E:\\KIIR\\Marine Breuilly\\DATA_KIIR\\TIFF\\KIIR_1-22_zs"),
         sg.FolderBrowse(key="-IN-")],
        [sg.Text("Radix (ex \"kiir_1-22_zs_\"): "), sg.InputText(key="-RADIX-", default_text='kiir_1-22_zs_')],
        [sg.Text("Nb of columns: "), sg.In(size=(8, 1), key='-COL-', default_text='2')],
        [sg.Text("Nb of lines: "), sg.In(size=(8, 1), key='-ROW-', default_text='4')],
        [sg.Text("Nb of overlapping pixels: "), sg.In(size=(8, 1), key='-OVERLAP-', default_text='240')],
        [sg.Submit(button_text='Register'), sg.Cancel()]
    ]
    # Building Window
    window = sg.Window('Register and Merge Mosaic', layout, size=(800, 600))

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == "Cancel":
            break
            # exit()
        elif event == "Register":
            try:
                folder = os.path.join(values['-IODIR-'], "")
                subdir_short = values['-RADIX-']
                nb_of_columns = np.uint16(values['-COL-'])
                nb_of_lines = np.uint16(values['-ROW-'])
                pix_overlap = np.int16(values['-OVERLAP-'])
            except:
                output = 'Following arguments should be integers: nb of columns, lines or overlapping pixels'
                print(output)
                exit()

            print(values["-IN-"])
            window.close()

    st = time.time()
    multiple_tile_registration(folder, radix=subdir_short,
                               number_of_lines=nb_of_lines, number_of_columns=nb_of_columns,
                               supposed_overlap=pix_overlap, verbose=False)
    print("Total Time spent: ----%.2f (sec)----" % (time.time() - st))
