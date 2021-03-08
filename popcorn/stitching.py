import os
import sys
import glob
import shutil

from skimage import filters
import numpy as np

from popcorn.input_output import open_sequence, save_tif_image


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
                    bottom_band_image = open_sequence(bottom_band_filenames)
                    top_band_image = open_sequence(top_band_filenames)

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
                    top_overlap_index = supposed_top_overlap_slice + overlap_index_difference + int(band_average_size/2)

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
    first_image = open_sequence(first_image_filenames)
    second_image = open_sequence(second_image_filenames)

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
    for slice_nb in range(0, second_nb_slices):
        centered_second_image[slice_nb, :, :] = centered_second_image[slice_nb, :, :] \
                                                - np.mean(centered_second_image[slice_nb, :, :])
    centered_images_multiplication_result = first_image_middle_slice * centered_second_image

    # We compute normalized cross-correlation between first image middle slice and all second image slices
    normalized_cross_correlations = np.zeros(second_nb_slices)
    for slice_nb in range(0, second_nb_slices):
        second_image_slice_std = np.std(centered_second_image[slice_nb, :, :])
        sum_of_multiplied_images = np.sum(centered_images_multiplication_result[slice_nb, :, :])
        normalized_cross_correlation = sum_of_multiplied_images/(first_image_middle_slice_std * second_image_slice_std)
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
    mask = np.zeros(first_image.shape)
    middle_slice_nb = int(nb_slices / 2)

    first_image_copy = np.copy(first_image)
    centered_second_image = np.copy(second_image)

    # If a thresholding is requested, we use Otsu thresholding on top 85% of the first image histogram
    if with_segmentation:
        thresh = filters.threshold_otsu(first_image_copy[first_image_copy > 0.15 * 65535])
        mask = first_image_copy > thresh
        first_image_copy = mask * first_image_copy
        centered_second_image = mask * centered_second_image

    # We compute what we need for normalized cross correlation (second image)
    for slice_nb in range(0, nb_slices):
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
        for slice_nb in range(0, nb_slices):
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


if __name__ == "__main__":
    print("Hello")
    imageAFolder = '/data/visitor/md1237/id17/volfloat/blablabla_001'
    imageBFolder = '/data/visitor/md1237/id17/volfloat/blablabla_002'
    imageAFiles = glob.glob(imageAFolder + '/*.tif')
    imageBFiles = glob.glob(imageAFolder + '/*.tif')
    imageA = open_sequence(imageAFiles)
    imageB = open_sequence(imageBFiles)
