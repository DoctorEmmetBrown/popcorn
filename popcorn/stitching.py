import os
import sys
import gc
import glob
import shutil
import time

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


def multiple_tile_registration(input_folder, radix, starting_position="top-left", number_of_lines=4,
                               number_of_columns=3,
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

    # Listing input folders
    list_of_folders = glob.glob(input_folder + radix + "*")
    reference_image_path = glob.glob(list_of_folders[0] + "\\" + "*")[0]

    # Opening a reference image for reference height/width
    reference_image = open_image(reference_image_path)
    nb_of_slices = len(glob.glob(list_of_folders[0] + "\\" + "*"))
    ref_height = reference_image.shape[0]
    ref_width = reference_image.shape[1]
    offset_max = (60, 60, 20)

    # Registration is computed on a subpart of each tile,
    # ref_image_coordinates correspond to the coordinates of the right sub-part of the left image
    # moving_image_coordinates correspond to the coordinates of the left sub-part of the right image
    ref_image_coordinates = [[0, nb_of_slices - 1],
                             [0, reference_image.shape[0] - 1],
                             [max(reference_image.shape[1] - supposed_overlap, 0), reference_image.shape[1] - 1]]

    moving_image_coordinates = [[0, nb_of_slices - 1],
                                [0, reference_image.shape[0] - 1],
                                [0, min(supposed_overlap, reference_image.shape[1] - 1)]]
    list_of_transformations = []
    if verbose:
        print("1. Introduction time:", (time.time() - time_start))

    time_start = time.time()
    # 1. Registration computation, for each line of the final grid, we compute the offset between each neighboring tiles
    for nb_line in range(number_of_lines):
        for nb_col in range(number_of_columns - 1):  # We compute registration on number_of_columns -1 pairs of images
            image_number = nb_line * number_of_columns + nb_col  # We keep track of the ref_image number we're working on

            print("stitching tile number", folders_indices[image_number],
                  "and tile number", folders_indices[image_number + 1])

            reg_start = time.time()
            transformation = compute_two_tiles_registration(list_of_folders[folders_indices[image_number]],
                                                            ref_image_coordinates,
                                                            list_of_folders[folders_indices[image_number + 1]],
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

    for nb_line in range(number_of_lines):
        for nb_col in range(number_of_columns):
            idx = nb_line * number_of_columns + nb_col
            gc.collect()
            tile = open_sequence(list_of_folders[idx] + "\\", imtype=np.uint16)
            if nb_col > 0:
                range_trsf = np.arange(nb_line * (number_of_columns - 1), nb_line * (number_of_columns - 1) + nb_col)
                for nb_transformation in range_trsf:
                    tile = apply_itk_transformation(tile, list_of_transformations[nb_transformation])
            save_tif_sequence(tile, input_folder + "registered_tile_" + str(idx) + "\\", bit=16)
            print("Registered tile number", idx, "saved !")

    if verbose:
        print("2. First Registrations time:", (time.time() - time_start))
        time_start = time.time()

    # 2. Concatenation of same line tiles, line are saved in combined_line_XX folders
    for nb_line in range(number_of_lines):
        x_position = 0
        list_of_list_of_tile_images = []
        list_of_len_tile = []
        # We add each tile on after the other, slice by slice
        for nb_col in range(number_of_columns):
            idx = nb_line * number_of_columns + nb_col
            list_of_list_of_tile_images.append(
                create_list_of_files(input_folder + "registered_tile_" + str(folders_indices[idx]) + "\\", "tif"))
            list_of_len_tile.append(len(list_of_list_of_tile_images[-1]))

        for nb_slice in range(min(list_of_len_tile)):
            full_width = ref_width * number_of_columns - supposed_overlap * (number_of_columns - 1)
            empty_slice = np.zeros((ref_height, full_width), dtype=np.uint16)
            overlapped_ref_slice = np.zeros((ref_height, supposed_overlap), dtype=np.uint16)
            overlapped_mov_slice = np.zeros((ref_height, supposed_overlap), dtype=np.uint16)
            woverlapped_ref_slice = np.zeros((ref_height, supposed_overlap), dtype=np.uint16)
            woverlapped_mov_slice = np.zeros((ref_height, supposed_overlap), dtype=np.uint16)

            # Compute weight matrix for overlapping part along the width axis
            mixing_size = supposed_overlap - (2 * supposed_overlap // 3)
            x = np.arange(mixing_size) + mixing_size
            y = -1 / (x[-1] - x[0]) * (x - x[-1])
            w_1d_W = np.concatenate((np.ones(supposed_overlap // 3), y, np.zeros(supposed_overlap // 3)))
            w_1d_W = w_1d_W.astype('e')
            w_2d_W = np.tile(w_1d_W, (ref_height, 1))

            for nb_col in range(number_of_columns):
                out_slice = open_image(list_of_list_of_tile_images[nb_col][nb_slice])
                image_number = nb_line * number_of_columns + nb_col
                # The first tile doesn't need any registration
                if nb_col == 0:
                    x_position = ref_width - supposed_overlap
                    empty_slice[:, 0:x_position] = out_slice[:, 0:x_position]
                    overlapped_ref_slice = out_slice[:, x_position:]

                # The next tile need to be registered/cropped
                else:
                    # -- Manage overlapping part
                    # Apply weight to each overlapping parts
                    woverlapped_ref_slice = overlapped_ref_slice * w_2d_W
                    # After registration comes the cropping part (based on initial supposed overlap)
                    overlapped_mov_slice = out_slice[:, 0:supposed_overlap]
                    woverlapped_mov_slice = overlapped_mov_slice * (1 - w_2d_W)

                    # Compute the sum of the two weighted overlapping parts and copy it to the final image
                    overlapped_section = np.nansum(np.array([woverlapped_ref_slice, woverlapped_mov_slice]), axis=0)
                    # Copy the resulting overlap part
                    empty_slice[:, x_position:x_position + supposed_overlap] = overlapped_section
                    #
                    x_position += supposed_overlap
                    # -- Manage non-overlapping part
                    # Copy non-overlapping part as it is
                    slice_to_copy = out_slice[:, supposed_overlap:]
                    empty_slice[:, x_position:x_position + slice_to_copy.shape[1]] = slice_to_copy
                    # Prepare for next column if there is
                    x_position += slice_to_copy.shape[1] - supposed_overlap
                    overlapped_ref_slice = out_slice[:, out_slice.shape[1] - supposed_overlap:]
            if verbose:
                print("-> Saving slice number", nb_slice, "in folder", input_folder + "combined_line_" + str(nb_line) + "\\")
            save_tif_image(empty_slice,
                           input_folder + "combined_line_" + str(nb_line) + "\\" + '{:04d}'.format(nb_slice), bit=16)
            if verbose:
                print("--> Slice number", nb_line, "saved !")
            del empty_slice
            del overlapped_ref_slice, overlapped_mov_slice
            del woverlapped_ref_slice, woverlapped_mov_slice
            del w_1d_W, w_2d_W
            gc.collect()

    if verbose:
        print("2. Line saving time:", (time.time() - time_start))
        time_start = time.time()

    # 3. Registration computation, we compute the offset between each neighboring lines
    list_of_transformations = []
    for nb_line in range(number_of_lines - 1):
        # This time, the registration is vertical, not horizontal. The overlapping parts correspond to the
        # bottom/upper-end of each line
        transformation = compute_two_tiles_registration(input_folder + "combined_line_" + str(nb_line),
                                                        [[0, -1], [-supposed_overlap, -1], [0, -1]],
                                                        input_folder + "combined_line_" + str(nb_line + 1),
                                                        [[0, -1], [0, supposed_overlap], [0, -1]])
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

    # 4. Registration of each line (Because of memory limitations)
    for nb_line in range(number_of_lines - 1):
        gc.collect()
        line = open_sequence(input_folder + "combined_line_" + str(nb_line + 1) + "\\", imtype=np.uint16)
        for nb_transformation in range(nb_line + 1):
            line = apply_itk_transformation(line, list_of_transformations[nb_transformation])
        save_tif_sequence(line, input_folder + "registered_line_" + str(nb_line) + "\\", bit=16)
        print("Registered line number", nb_line, "saved !")

    if verbose :
        print("3. Second Registrations time:", (time.time() - time_start))
        time_start = time.time()
    # 5. Concatenation of every lines of the final grid
    list_of_line_images = []
    list_of_len = []

    # For this purpose, we need to list all input slices
    for nb_line in range(number_of_lines):
        if nb_line == 0:
            list_of_line_images.append(create_list_of_files(input_folder + "combined_line_" + str(nb_line) + "\\",
                                                            "tif"))
        else:
            list_of_line_images.append(create_list_of_files(input_folder + "registered_line_" + str(nb_line - 1) + "\\",
                                                            "tif"))
        list_of_len.append(len(list_of_line_images[-1]))

    for nb_slice in range(min(list_of_len)):
        y_position = 0
        full_width = ref_width * number_of_columns - supposed_overlap * (number_of_columns - 1)
        full_height = ref_height * number_of_lines - supposed_overlap * (number_of_lines - 1)
        empty_slice = np.zeros((full_height, full_width), dtype=np.uint16)
        overlapped_ref_slice = np.zeros((supposed_overlap, full_width), dtype=np.uint16)
        overlapped_mov_slice = np.zeros((supposed_overlap, full_width), dtype=np.uint16)

        # Compute weight matrix for overlapping part along the height axis
        mixing_size = supposed_overlap - (2 * supposed_overlap // 3)
        x = np.arange(mixing_size) + mixing_size
        y = -1 / (x[-1] - x[0]) * (x - x[-1])
        w_1d_H = np.concatenate((np.ones(supposed_overlap // 3), y, np.zeros(supposed_overlap // 3)))
        w_1d_H = w_1d_H.astype('e')
        w_1d_H = np.reshape(w_1d_H, (-1, 1))
        w_2d_H = np.tile(w_1d_H, (1, full_width))

        # Concatenation
        for nb_line in range(number_of_lines):
            out_slice = open_image(list_of_line_images[nb_line][nb_slice])
            # The first tile doesn't need any registration
            if nb_line == 0:
                y_position = out_slice.shape[0] - supposed_overlap
                empty_slice[0:y_position, :] = out_slice[0:y_position, :]
                overlapped_ref_slice = out_slice[y_position:, :]
            else:
                # -- Manage overlapping part
                # Apply weight to each overlapping parts
                # overlapped_ref_slice[overlapped_ref_slice == 0] = np.nan
                woverlapped_ref_slice = overlapped_ref_slice * w_2d_H

                overlapped_mov_slice = out_slice[0:supposed_overlap, :]
                woverlapped_mov_slice = overlapped_mov_slice * (1 - w_2d_H)
                # Compute the sum of the two weighted overlapping parts and copy it to the final image
                overlapped_section = np.nansum(np.array([woverlapped_ref_slice, woverlapped_mov_slice]), axis=0)
                # Copy the resulting overlap part
                empty_slice[y_position:y_position + supposed_overlap, :] = overlapped_section
                #
                y_position += supposed_overlap
                # -- Manage non-overlapping part
                # Copy non-overlapping part as it is
                slice_to_copy = out_slice[supposed_overlap:, :]
                empty_slice[y_position:y_position + slice_to_copy.shape[0], :] = slice_to_copy
                # Prepare for next column if there is
                y_position += slice_to_copy.shape[0] - supposed_overlap
                overlapped_ref_slice = out_slice[out_slice.shape[0] - supposed_overlap:, :]

        save_tif_image(empty_slice, input_folder + "final_image\\" + '{:04d}'.format(nb_slice), bit=16)

        del empty_slice
        del overlapped_ref_slice, overlapped_mov_slice
        del woverlapped_ref_slice, woverlapped_mov_slice
        del w_1d_H, w_2d_H
        gc.collect()

    if verbose:
        print("4. Second Concatenation time:", (time.time() - time_start))
        print("Total:", (time.time() - full_time_start))

    list_of_directory_to_remove = glob.glob(input_folder + "\\" + "registered_*")
    list_of_directory_to_remove += glob.glob(input_folder + "\\" + "combined_*")

    for nb_dir in range(len(list_of_directory_to_remove)):
        print("dir to remove", list_of_directory_to_remove[nb_dir])
        shutil.rmtree(list_of_directory_to_remove[nb_dir])


if __name__ == "__main__":
    folder = "C:\\Users\\Marine Breuilly\\Documents\\DATA\\Projet_COEUR\\1.26 ZS\\binned_tiles\\"
    st = time.time()
    multiple_tile_registration(folder,
                               radix="kiir_1-26_zS_",  # "kiir 1-26 zone saine.czi - kiir 1-26 zS ", #
                               number_of_lines=2,
                               number_of_columns=2,
                               supposed_overlap=120,
                               verbose=False)
    print("Total Time spent: ----%.2f (sec)----" % (time.time() - st))
