import sys

from pathlib import Path
path_root = str(Path(__file__).parents[1])
if path_root not in sys.path:
    sys.path.append(path_root)

import popcorn.input_output as in_out
import popcorn.resampling as resampling
import material_decomposition
import popcorn.spectral_imaging.registration as registration
import popcorn.image_processing.segmentation as segmentation

from popcorn.spectral_imaging.pipelines import skull_alignment_pipeline

import numpy as np
import math
import time

import SimpleITK as Sitk

from skimage.measure import label, regionprops
from skimage import img_as_ubyte

import PyIPSDK
import PyIPSDK.IPSDKIPLBinarization as Bin
import PyIPSDK.IPSDKIPLMorphology as Morpho
import PyIPSDK.IPSDKIPLAdvancedMorphology as AdvMorpho
import PyIPSDK.IPSDKIPLShapeAnalysis as ShapeAnalysis

def conversion_pipeline(image, bin_factor, min, max):
    if bin_factor > 1:
        image = resampling.bin_resize(image, bin_factor)
    return resampling.conversion_uint16_to_float32(image, min, max)

def retrieve_min_max_from_path(path):
    """looks for min and max float in path

    Args:
        path (str): folder path

    Returns:
        (float, float) retrieved min and max values
    """
    path = path.replace("\\", "")
    path = path.replace("/", "")
    return float(path.split("_")[-2]), float(path.split("_")[-1])


def segmentation_coloration(output_folder, concentration_map, segmentation_result, material):

    cells = np.copy(concentration_map)
    cells[segmentation_result == 0] = 0
    cells_part = np.copy(cells)

    concentration_map = resampling.normalize_image(concentration_map) / 1.5

    cells_part = resampling.normalize_image(cells_part)

    noise = np.copy(concentration_map)
    noise[cells_part > 0] = 0

    mask = np.copy(cells)
    mask[mask > 0] = 1

    cells_offset = np.copy(concentration_map)
    cells_offset[cells_part <= 0] = 0
    cells_offset[cells_part > 0] = 0.2

    rgb = [0.8, 0.8, 0] if material == "Au" else [0.8, 0.35, 0]
    for i in range(cells_part.shape[0]):
        red_slice = img_as_ubyte(noise[i, :, :] + rgb[0] * cells_part[i, :, :] + cells_offset[i, :, :])
        green_slice = img_as_ubyte(noise[i, :, :] + rgb[1] * cells_part[i, :, :] + cells_offset[i, :, :])
        blue_slice = img_as_ubyte(noise[i, :, :] + rgb[2] * cells_part[i, :, :] + cells_offset[i, :, :])

        rgb_cells = np.stack((red_slice, green_slice, blue_slice), axis=-1)
        in_out.save_tif_image(rgb_cells, output_folder + '{:04d}'.format(i), rgb=True)


class SpectralImagingExperiment:

    def __init__(self, name, path, sample_type, modality, materials, resolution, bin_factor=1):
        """constructor of class SpectralImagingExperiment

        Args:
            name (str):            sample radix
            path (str):            data path
            sample_type (str):     either "phantom", "rat brain" or "rat knee"
            modality (str):        modality used for acquisition
            materials (list[str]): list of kedge materials
            resolution (float):    image resolution
            bin_factor (int):      binning factor when analyzing
        """
        self.name = name
        self.path = path
        self.type = sample_type
        self.modality = modality
        self.materials = materials
        self.resolution = resolution * bin_factor
        self.bin_factor = bin_factor

    def conversion(self):
        """Converts images from uint16 to float32 using the sample's name min/max inb4 binning them using defined bin
        factor
        Returns:
            None
        """
        for material in self.materials:
            print("Conversion of :", material)
            # We retrieve float min and max from given path
            above_image_filenames = in_out.create_list_of_files(self.path + "*Above*" + material + "*", "tif")
            above_min, above_max = retrieve_min_max_from_path(in_out.remove_filename_in_path(above_image_filenames[0]))

            for index in range(len(above_image_filenames)//self.bin_factor):
                # [1/5] Opening
                image_to_bin = in_out.open_sequence(above_image_filenames[:self.bin_factor])
                # [2/5] Deleting opened files from previous list of files
                del above_image_filenames[:self.bin_factor]
                # [3/5] Binning
                binned_image = resampling.bin_resize(image_to_bin, self.bin_factor)
                # [4/5] Conversion
                converted_image = resampling.conversion_uint16_to_float32(binned_image, above_min, above_max)
                # [5/5] Saving
                in_out.save_tif_image(converted_image[0],
                                      self.path + material + "\\Above_Acquisition\\" + '{:04d}'.format(index))

            # We retrieve float min and max from given path
            below_image_filenames = in_out.create_list_of_files(self.path + "*Below*" + material + "*", "tif")
            below_min, below_max = retrieve_min_max_from_path(in_out.remove_filename_in_path(below_image_filenames[0]))

            for index in range(len(below_image_filenames)//self.bin_factor):
                # [1/5] Opening
                image_to_bin = in_out.open_sequence(below_image_filenames[:self.bin_factor])
                # [2/5] Deleting opened files from previous list of files
                del below_image_filenames[:self.bin_factor]
                # [3/5] Binning
                binned_image = resampling.bin_resize(image_to_bin, self.bin_factor)
                # [4/5] Conversion
                converted_image = resampling.conversion_uint16_to_float32(binned_image, below_min, below_max)
                # [5/5] Saving
                in_out.save_tif_image(converted_image[0],
                                      self.path + material + "\\Below_Acquisition\\" + '{:04d}'.format(index))

    def register_volumes(self):
        if self.type == "rat brain":
            for material in self.materials:
                print("--------", "Registering", material, "--------")
                above_filenames = in_out.create_list_of_files(self.path + material + "\\Above_Acquisition\\",
                                                              "tif")
                below_filenames = in_out.create_list_of_files(self.path + material + "\\Below_Acquisition\\",
                                                              "tif")
                above_image = in_out.open_sequence(above_filenames)
                below_image = in_out.open_sequence(below_filenames)

                # -- Threshold computation
                above_threshold_value = segmentation.find_threshold_value(material, "esrf")
                above_mask = above_image > above_threshold_value
                below_mask = below_image > above_threshold_value

                # -- Extracting skulls
                above_skull, above_skull_bbox = segmentation.extract_skull(above_mask)
                below_skull, below_skull_bbox = segmentation.extract_skull(below_mask)

                rotation_transform = registration.registration_computation(above_image,
                                                                           below_image,
                                                                           transform_type="rotation",
                                                                           metric="msq",
                                                                           moving_mask=above_skull,
                                                                           ref_mask=below_skull,
                                                                           verbose=True)

                # Registering the above image
                above_image = registration.apply_itk_transformation(above_image, rotation_transform, "linear")
                in_out.save_tif_sequence(above_image,
                                         self.path + material + "\\Above_Acquisition_Registered\\")
                skull_alignment_pipeline(self.path + material + "\\Above_Acquisition_Registered\\",
                                         self.path + material + "\\Below_Acquisition\\", material)
        elif self.type == "mouse knee":
            for material in self.materials:
                print("--------", "Registering", material, "--------")
                above_filenames = in_out.create_list_of_files(self.path + material + "\\Above_Acquisition_Binned\\",
                                                              "tif")
                below_filenames = in_out.create_list_of_files(self.path + material + "\\Below_Acquisition_Binned\\",
                                                              "tif")
                above_image = in_out.open_sequence(above_filenames)
                below_image = in_out.open_sequence(below_filenames)

                # -- Threshold computation
                above_threshold_value = segmentation.find_threshold_value(material, "above", "esrf")
                above_mask = np.copy(above_image)
                above_mask[above_mask > above_threshold_value] = 1
                above_mask[above_mask <= above_threshold_value] = 0
                below_threshold_value = segmentation.find_threshold_value(material, "below", "esrf")
                below_mask = np.copy(below_image)
                below_mask[below_mask > below_threshold_value] = 1
                below_mask[below_mask <= below_threshold_value] = 0
                translation_transform = registration.registration_computation(above_image,
                                                                              below_image,
                                                                              transform_type="translation",
                                                                              metric="msq",
                                                                              moving_mask=above_mask,
                                                                              ref_mask=below_mask,
                                                                              verbose=True)

                rotation_transform = registration.registration_computation(above_image,
                                                                           below_image,
                                                                           transform_type="rotation",
                                                                           metric="msq",
                                                                           moving_mask=above_mask,
                                                                           ref_mask=below_mask,
                                                                           verbose=True)

                image_to_register = in_out.open_sequence(self.path + material + "\\Above_Acquisition\\")

                # Registering the above image
                translation_parameters = translation_transform.GetParameters()
                translation_transform.SetParameters((translation_parameters[0]*2,
                                                     translation_parameters[1]*2,
                                                     translation_parameters[2]*2))

                image_to_register = registration.apply_itk_transformation(image_to_register,
                                                                          translation_transform,
                                                                          "linear")

                image_to_register_itk = Sitk.GetImageFromArray(image_to_register)
                actual_rotation = Sitk.CenteredTransformInitializer(image_to_register_itk,
                                                                    image_to_register_itk,
                                                                    Sitk.Euler3DTransform(),
                                                                    Sitk.CenteredTransformInitializerFilter.GEOMETRY)
                rotation_parameters = rotation_transform.GetParameters()
                actual_rotation.SetParameters((rotation_parameters[0],
                                               rotation_parameters[1],
                                               rotation_parameters[2],
                                               rotation_parameters[3] * 2,
                                               rotation_parameters[4] * 2,
                                               rotation_parameters[5] * 2))
                image_to_register = registration.apply_itk_transformation(image_to_register, actual_rotation, "linear")

                print("\r[1/2] Saving registered image :", end="", flush=True)
                in_out.save_tif_sequence(image_to_register,
                                         self.path + material + "\\Above_Acquisition_Registered\\")

                image_to_register = None
                below_image = in_out.open_sequence(self.path + material + "\\Below_Acquisition\\")
                below_image[below_image > below_threshold_value] = 1
                below_image[below_image <= below_threshold_value] = 0
                print("\r[2/2] Saving below mask :", end="", flush=True)
                in_out.save_tif_sequence(below_image,
                                         self.path + material + "\\Below_Mask\\")
                print("\n")
        elif self.type == "phantom":
            for material in self.materials:
                print("--------", "Registering", material, "--------")
                above_filenames = in_out.create_list_of_files(self.path + material + "\\Above_Acquisition\\",
                                                              "tif")
                below_filenames = in_out.create_list_of_files(self.path + material + "\\Below_Acquisition\\",
                                                              "tif")
                above_image = in_out.open_sequence(above_filenames)
                below_image = in_out.open_sequence(below_filenames)

                # -- Threshold computation
                above_mask = np.ones(above_image.shape)

                above_mask[above_image > 0.18] = 0
                above_mask[above_image < 0.14] = 0

                below_mask = np.ones(below_image.shape)

                below_mask[below_image > 0.17] = 0
                below_mask[below_image < 0.14] = 0

                rotation_transform = registration.registration_computation(above_image,
                                                                           below_image,
                                                                           transform_type="rotation",
                                                                           metric="msq",
                                                                           moving_mask=above_mask,
                                                                           ref_mask=below_mask,
                                                                           verbose=True)

                # Registering the above image
                above_image = registration.apply_itk_transformation(above_image, rotation_transform, "linear")
                in_out.save_tif_sequence(above_image,
                                         self.path + material + "\\Above_Acquisition_Registered\\")

    def manual_registration(self, slice_of_interest):
        """Function made for manual registration tests

        Args:
            slice_of_interest (int): around which slice we want to test our shit
        """
        for material in self.materials:
            if material == "Au":
                densities = np.array([19.3, 1.0])
                material_attenuations = np.array([[165.8642, 000.1827],
                                                  [040.7423, 000.1835]])
            else:
                densities = np.array([4.93, 1.0])
                material_attenuations = np.array([[170.9231, 000.3188],
                                                  [033.6370, 000.3307]])
            print("--------", "Registering", material, "--------")
            above_filenames = in_out.create_list_of_files(self.path + material + "\\Above_Acquisition\\",
                                                          "tif")
            belowsimage = in_out.open_sequence(
                in_out.create_list_of_files(self.path + material + "\\Below_Acquisition\\",
                                            "tif")[slice_of_interest - 5:slice_of_interest + 6])
            above_image = in_out.open_sequence(above_filenames)
            for z in range(20):
                above_image_itk = Sitk.GetImageFromArray(above_image)
                translation_transformation = Sitk.TranslationTransform(above_image_itk.GetDimension())
                translation_transformation.SetOffset((-0.14, z/10 - 1, 0))
                above_image_itk = Sitk.Resample(above_image_itk, translation_transformation, Sitk.sitkLinear, 0.0,
                                                above_image_itk.GetPixelIDValue())
                registered_image = Sitk.GetArrayFromImage(above_image_itk)
                images = np.stack((registered_image[slice_of_interest-5:slice_of_interest+6], belowsimage), axis=0)
                concentration_maps = \
                    material_decomposition.decomposition_equation_resolution(images, densities, material_attenuations,
                                                                             volume_fraction_hypothesis=False,
                                                                             verbose=False)
                in_out.save_tif_sequence(concentration_maps[0],
                                         self.path + material + "\\manual_registrationz_" + str(z/10 - 1) + "\\")

    def material_decomposition(self, registration_done=False):
        """material decomposition method

        Args:
            registration_done (bool): did we use registration ?

        Returns:
            None
        """
        if len(self.materials) == 1:
            material = self.materials[0]
            if self.type == "phantom":
                if registration_done:
                    above_filenames = \
                        in_out.create_list_of_files(self.path + self.materials[0] + "\\Above_Acquisition_Registered\\",
                                                    "tif")
                else:
                    above_filenames = \
                        in_out.create_list_of_files(self.path + self.materials[0] + "\\Above_Acquisition\\",
                                                    "tif")

                below_filenames = in_out.create_list_of_files(self.path + self.materials[0] + "\\Below_Acquisition\\",
                                                              "tif")
            elif self.type == "rat brain":
                above_filenames = \
                    in_out.create_list_of_files(self.path + self.materials[0] + "\\Aligned_Above_Acquisition\\",
                                                "tif")

                below_filenames = \
                    in_out.create_list_of_files(self.path + self.materials[0] + "\\Aligned_Below_Acquisition\\",
                                                "tif")
            else:
                above_filenames = \
                    in_out.create_list_of_files(self.path + self.materials[0] + "\\Above_Acquisition_Registered\\",
                                                "tif")

                below_filenames = in_out.create_list_of_files(self.path + self.materials[0] + "\\Below_Acquisition\\",
                                                              "tif")
            time_list = []
            for filename_index in range(min(len(above_filenames), len(below_filenames))):
                above_image = in_out.open_image(above_filenames[filename_index])
                below_image = in_out.open_image(below_filenames[filename_index])
                images = np.stack((above_image, below_image), axis=0)
                if material == "Au":
                    densities = np.array([19.3, 1.0])
                    material_attenuations = np.array([[165.8642, 000.1827],
                                                      [040.7423, 000.1835]])
                else:
                    densities = np.array([4.93, 1.0])
                    material_attenuations = np.array([[170.9231, 000.3188],
                                                      [033.6370, 000.3307]])
                concentration_maps = \
                    material_decomposition.decomposition_equation_resolution(images, densities, material_attenuations,
                                                                             volume_fraction_hypothesis=False,
                                                                             verbose=False)
                start_time = time.time()
                material_decomposition.loading_bar(filename_index, min(len(above_filenames), len(below_filenames)))
                in_out.save_tif_image(concentration_maps[0], self.path + material + "\\"
                                      + material + "2materials_decomposition\\" + '{:04d}'.format(filename_index))
                # in_out.save_tif_image(concentration_maps[1], self.path + material + "\\"
                #                       + "Bone_decomposition\\" + '{:04d}'.format(filename_index))
                # in_out.save_tif_image(concentration_maps[2], self.path + material + "\\"
                #                       + "Water_decomposition\\" + '{:04d}'.format(filename_index))
                time_list.append(time.time() - start_time)
            print("")
            print("Average time for decomposition  :", sum(time_list)/len(time_list), "s")
            print("Min time for decomposition      :", min(time_list), "s")
            print("Evolution of decomposition time :", (time_list[-1] - time_list[0])/len(time_list), "s")
        elif len(self.materials) == 2:
            for material in self.materials:
                if material == "I":
                    if self.type == "phantom":
                        if registration_done:
                            above_filenames = in_out.create_list_of_files(self.path + material +
                                                                          "\\Above_Acquisition_Registered\\",
                                                                          "tif")
                        else:
                            above_filenames = in_out.create_list_of_files(self.path + material +
                                                                          "\\Above_Acquisition\\",
                                                                          "tif")
                        below_filenames = in_out.create_list_of_files(
                            self.path + material + "\\Below_Acquisition\\",
                            "tif")
                    elif self.type == "rat brain":
                        above_filenames = in_out.create_list_of_files(
                            self.path + material + "\\Aligned_Above_Acquisition\\",
                            "tif")

                        below_filenames = in_out.create_list_of_files(
                            self.path + material + "\\Aligned_Below_Acquisition\\",
                            "tif")
                    else:
                        above_filenames = in_out.create_list_of_files(
                            self.path + material + "\\Above_Acquisition_Registered\\",
                            "tif")

                        below_filenames = in_out.create_list_of_files(
                            self.path + material + "\\Below_Acquisition\\",
                            "tif")

                    for filename_index in range(min(len(above_filenames), len(below_filenames))):
                        above_image = in_out.open_image(above_filenames[filename_index])
                        below_image = in_out.open_image(below_filenames[filename_index])
                        images = np.stack((above_image, below_image), axis=0)
                        if material == "Au":
                            densities = np.array([19.3, 4.93, 1.0])
                            material_attenuations = np.array([[165.8642, 016.3330, 000.1827],
                                                              [040.7423, 016.8852, 000.1835]])
                        else:
                            densities = np.array([4.93, 19.3, 1.0])
                            material_attenuations = np.array([[170.9231, 389.5878, 000.3188],
                                                              [033.6370, 421.9694, 000.3307]])
                        concentration_maps = \
                            material_decomposition.decomposition_equation_resolution(images, densities,
                                                                                     material_attenuations,
                                                                                     volume_fraction_hypothesis=True,
                                                                                     verbose=False)
                        material_decomposition.loading_bar(filename_index, min(len(above_filenames),
                                                                               len(below_filenames)))
                        concentration_maps[0][below_image == 0] = 0
                        concentration_maps[1][below_image == 0] = 0
                        concentration_maps[2][below_image == 0] = 0

                        in_out.save_tif_image(concentration_maps[0], self.path + material + "\\"
                                              + material + "_decomposition\\" + '{:04d}'.format(filename_index))
                        if material == "Au":
                            in_out.save_tif_image(concentration_maps[1], self.path + material + "\\"
                                                  + "I_decomposition\\" + '{:04d}'.format(filename_index))
                        elif material == "I":
                            in_out.save_tif_image(concentration_maps[1], self.path + material + "\\"
                                                  + "Au_decomposition\\" + '{:04d}'.format(filename_index))
                        in_out.save_tif_image(concentration_maps[2], self.path + material + "\\"
                                              + "Water_decomposition\\" + '{:04d}'.format(filename_index))

    def material_segmentation(self):
        if self.type == "phantom":
            # tube_slices = [54, 51, 52, 0, 0, 0, 100]
            tube_slices = [0, 0, 0, 0, 0, 0, 0]
            for material in self.materials:
                list_of_above_attenuation_files = in_out.create_list_of_files(self.path + material +
                                                                              "\\Above_Acquisition\\", "tif")

                first_file_index = int(len(list_of_above_attenuation_files) * 0.2)
                first_image = in_out.open_image(list_of_above_attenuation_files[first_file_index])
                second_file_index = int(len(list_of_above_attenuation_files) * 0.8)
                second_image = in_out.open_image(list_of_above_attenuation_files[second_file_index])
                print("indices :", first_file_index, second_file_index)
                if material == "Au":
                    first_mask = (first_image > 0.17)
                    second_mask = (second_image > 0.17)
                else:
                    first_mask = (first_image > 0.355)
                    second_mask = (second_image > 0.355)
                in_out.save_tif_image(second_image, self.path + material + "\\second_image")
                first_label_img = label(first_mask)
                first_regions = regionprops(first_label_img)
                second_label_img = label(second_mask)
                second_regions = regionprops(second_label_img)
                first_max_radius = 0
                first_centroids = []
                first_radii = []
                for region in first_regions:
                    if region.area > 1000:
                        radius = math.sqrt(region.area / 3.14159)
                        first_radii.append(radius)
                        if radius > first_max_radius:
                            first_max_radius = radius
                        first_centroids.append([int(region.centroid[0]), int(region.centroid[1])])

                second_max_radius = 0
                second_centroids = []
                second_radii = []
                for region in second_regions:
                    if region.area > 1000:
                        print("second ok")
                        radius = math.sqrt(region.area / 3.14159)
                        second_radii.append(radius)
                        if radius > second_max_radius:
                            second_max_radius = radius
                        second_centroids.append([int(region.centroid[0]), int(region.centroid[1])])

                if sum(first_radii)/len(first_radii) > sum(second_radii)/len(second_radii):
                    centroids = first_centroids
                    max_radius = first_max_radius
                else:
                    centroids = second_centroids
                    max_radius = second_max_radius
                diameter = 2 * int(1.25 * max_radius)
                list_of_coordinates = [
                    [
                        max(0, (centroid[0] - int(diameter / 2))),
                        min(
                            first_image.shape[0], (centroid[0] + int(diameter / 2))
                        ),
                        max(0, (centroid[1] - int(diameter / 2))),
                        min(
                            first_image.shape[0], (centroid[1] + int(diameter / 2))
                        ),
                    ]
                    for centroid in centroids
                ]

                print(str(len(centroids)) + " tubes found")
                material_list_of_files = in_out.create_list_of_files(self.path + material + "\\" + material
                                                                     + "_decomposition\\", "tif")

                for file_nb in range(len(material_list_of_files)):
                    print("cropping: " + str(int(file_nb / len(material_list_of_files) * 100)) + "%",
                          end="\r")
                    current_image = in_out.open_image(material_list_of_files[file_nb])
                    for tube_nb in range(len(centroids)):
                        cropped_image = current_image[list_of_coordinates[tube_nb][0]:list_of_coordinates[tube_nb][1],
                                                      list_of_coordinates[tube_nb][2]:list_of_coordinates[tube_nb][3]]
                        in_out.save_tif_image(cropped_image,  self.path + material + "\\" + material + "_tubes\\tube_"
                                              + str(tube_nb) + "\\" + '{:04d}'.format(file_nb) + ".tif")

                for tube_nb in range(len(centroids)):
                    tube_filenames = in_out.create_list_of_files(self.path + material + "\\" + material
                                                                 + "_tubes\\tube_" + str(tube_nb) + "\\", "tif")
                    tube = in_out.open_sequence(tube_filenames)

                    segmented_nanoparticles = np.copy(tube)
                    for slice_nb in range(segmented_nanoparticles.shape[0]):
                        slice_of_interest = segmented_nanoparticles[slice_nb, :, :]
                        if slice_nb < tube_slices[tube_nb]:
                            slice_of_interest[slice_of_interest < 100.0] = 0
                        else:
                            slice_of_interest[slice_of_interest < 0.5] = 0
                        mean_val = np.mean(slice_of_interest[slice_of_interest > 0.5])
                        # print(slice_nb, "mean val:", mean_val)
                        slice_of_interest[slice_of_interest < 0.35 * mean_val] = 0
                        # slice_of_interest[slice_of_interest > 5 * mean_val] = 0
                        segmented_nanoparticles[slice_nb, :, :] = slice_of_interest

                    if np.mean(segmented_nanoparticles * tube) > 0.2:
                        segmented_nanoparticles_ipsdk = PyIPSDK.fromArray(segmented_nanoparticles)
                        segmented_nanoparticles_ipsdk = Bin.lightThresholdImg(segmented_nanoparticles_ipsdk, 0.5)
                        eroding_sphere = PyIPSDK.sphericalSEXYZInfo(2)
                        segmented_nanoparticles_ipsdk = Morpho.erode3dImg(segmented_nanoparticles_ipsdk, eroding_sphere)
                        eroding_sphere = PyIPSDK.sphericalSEXYZInfo(4)
                        segmented_nanoparticles_ipsdk = Morpho.closing3dImg(segmented_nanoparticles_ipsdk, eroding_sphere)
                        segmented_nanoparticles_ipsdk = AdvMorpho.keepBigShape3dImg(segmented_nanoparticles_ipsdk, 1)
                        segmentation_coloration(self.path + material + "\\" + material + "_tubes\\tube_"
                                                      + str(tube_nb) + "_colored\\",
                                                tube,
                                                segmented_nanoparticles_ipsdk.array,
                                                material)
                        segmentation.segmented_cells_analysis(tube, segmented_nanoparticles_ipsdk.array, self.resolution,
                                                              filename=self.path + material + "\\" + material
                                                                       + "_tubes\\analysis.txt")
        elif self.type == "rat brain":
            for material in self.materials:
                threshold = 0.4 if material == "I" else None
                concentration_map_files = in_out.create_list_of_files(self.path + material + "\\" + material +
                                                                      "_decomposition\\", "tif")
                concentration_map = in_out.open_sequence(concentration_map_files)
                skull_files = in_out.create_list_of_files(self.path + material + "\\Aligned_Below_Skull\\", "tif")
                skull_mask = in_out.open_sequence(skull_files)
                print("output txt :", self.path + material + "\\" + material + "_quantification_results.txt")

                segmented_cells = segmentation.brain_nanoparticles_segmentation(concentration_map, skull_mask,
                                                                                self.resolution,
                                                                                filename=self.path + material + "\\"
                                                                                + material
                                                                                + "_quantification_results.txt",
                                                                                left_right=True,
                                                                                threshold=threshold)
                segmentation_coloration(self.path + material + "\\" + material + "_segmentation\\", concentration_map,
                                        segmented_cells, material)
                in_out.save_tif_sequence(segmented_cells, self.path + material + "\\nanoparticles_mask\\")

        elif self.type == "mouse knee":
            for material in self.materials:
                print("Segmenting", self.name)
                if material == "Au":
                    concentration_map_files = in_out.create_list_of_files(self.path + material + "\\" + material +
                                                                          "_decomposition\\", "tif")
                                                                          # "2materials_decomposition\\", "tif")
                    concentration_map = in_out.open_sequence(concentration_map_files)

                    mask_files = in_out.create_list_of_files(self.path + material + "\\Below_Mask\\", "tif")
                    mask = in_out.open_sequence(mask_files)

                    maskIPSDK = PyIPSDK.fromArray(mask)
                    dilating_sphere = PyIPSDK.sphericalSEXYZInfo(1)
                    maskIPSDK = Morpho.dilate3dImg(maskIPSDK, dilating_sphere)

                    concentration_map[maskIPSDK.array == 1] = 0

                    concentration_mapIPSDK = PyIPSDK.fromArray(concentration_map)
                    concentration_mapIPSDK = Bin.lightThresholdImg(concentration_mapIPSDK, 1)
                    concentration_mapIPSDK = AdvMorpho.keepBigShape3dImg(concentration_mapIPSDK, 1)
                    opening_sphere = PyIPSDK.sphericalSEXYZInfo(2)
                    concentration_mapIPSDK = Morpho.opening3dImg(concentration_mapIPSDK, opening_sphere)
                    concentration_mapIPSDK = Morpho.closing3dImg(concentration_mapIPSDK, opening_sphere)
                    in_out.save_tif_sequence(concentration_mapIPSDK.array, self.path + material + "\\Segmented_Cells\\")

                    segmentation.segmented_cells_analysis(concentration_map, concentration_mapIPSDK.array, 1.0,
                                                          self.resolution,
                                                          self.path + material + "\\" + material + "_quantification.txt")
                    segmentation_coloration(self.path + material + "\\" + material + "_segmentation\\",
                                            concentration_map,
                                            concentration_mapIPSDK.array, material)
                if material == "I":
                    seuil_detection_iode = .15

                    concentration_map_files = in_out.create_list_of_files(self.path + material + "\\" + material +
                                                                          "_decomposition\\", "tif")
                    mask_files = in_out.create_list_of_files(self.path + material + "\\Below_Mask\\", "tif")

                    concentration_map = in_out.open_sequence(concentration_map_files)

                    mask = in_out.open_sequence(mask_files)
                    maskIPSDK = PyIPSDK.fromArray(mask)
                    dilating_sphere = PyIPSDK.sphericalSEXYZInfo(1)
                    print("First bone dilatation")
                    maskIPSDK = Morpho.dilate3dImg(maskIPSDK, dilating_sphere)

                    concentration_map[maskIPSDK.array == 1] = 0
                    import time
                    dilating_sphere = PyIPSDK.sphericalSEXYZInfo(10)
                    print("Second bone dilatation")
                    start_time = time.time()
                    maskIPSDK = Morpho.dilate3dImg(maskIPSDK, dilating_sphere)
                    print("--- %s seconds ---" % (time.time() - start_time))
                    print("Third bone dilatation")
                    start_time = time.time()
                    maskIPSDK = Morpho.dilate3dImg(maskIPSDK, dilating_sphere)
                    print("--- %s seconds ---" % (time.time() - start_time))
                    print("Fourth bone dilatation")
                    start_time = time.time()
                    maskIPSDK = Morpho.dilate3dImg(maskIPSDK, dilating_sphere)
                    print("--- %s seconds ---" % (time.time() - start_time))
                    print("Fifth bone dilatation")
                    start_time = time.time()
                    maskIPSDK = Morpho.dilate3dImg(maskIPSDK, dilating_sphere)
                    print("--- %s seconds ---" % (time.time() - start_time))
                    print("Sixth bone dilatation")
                    start_time = time.time()
                    maskIPSDK = Morpho.dilate3dImg(maskIPSDK, dilating_sphere)
                    print("--- %s seconds ---" % (time.time() - start_time))
                    print("Seventh bone dilatation")
                    start_time = time.time()
                    maskIPSDK = Morpho.dilate3dImg(maskIPSDK, dilating_sphere)
                    print("--- %s seconds ---" % (time.time() - start_time))
                    print("Eighth bone dilatation")
                    start_time = time.time()
                    maskIPSDK = Morpho.dilate3dImg(maskIPSDK, dilating_sphere)
                    print("--- %s seconds ---" % (time.time() - start_time))
                    print("Ninth bone dilatation")
                    start_time = time.time()
                    maskIPSDK = Morpho.dilate3dImg(maskIPSDK, dilating_sphere)
                    print("--- %s seconds ---" % (time.time() - start_time))
                    print("Tenth bone dilatation")
                    start_time = time.time()
                    maskIPSDK = Morpho.dilate3dImg(maskIPSDK, dilating_sphere)
                    print("--- %s seconds ---" % (time.time() - start_time))

                    concentration_map[maskIPSDK.array == 0] = 0

                    print("End of dilatation")
                    concentration_mapIPSDK = PyIPSDK.fromArray(concentration_map)
                    concentration_mapIPSDK = Bin.lightThresholdImg(concentration_mapIPSDK, seuil_detection_iode)
                    stack_of_cellsIPSDK = AdvMorpho.keepBigShape3dImg(concentration_mapIPSDK, 1)
                    opening_sphere = PyIPSDK.sphericalSEXYZInfo(1)

                    stack_of_cellsIPSDK = ShapeAnalysis.shapeFiltering3dImg(stack_of_cellsIPSDK, "Volume3dMsr > 200")

                    final_segmentationIPSDK = Morpho.opening3dImg(stack_of_cellsIPSDK, opening_sphere)
                    final_segmentationIPSDK = AdvMorpho.keepBigShape3dImg(final_segmentationIPSDK, 1)

                    in_out.save_tif_sequence(final_segmentationIPSDK.array, self.path + material + "\\Segmented_Cells\\")
                    segmentation_coloration(self.path + material + "\\" + material + "_segmentation\\",
                                            concentration_map,
                                            final_segmentationIPSDK.array, material)


def bin_and_save(input_folder, output_folder, bin_factor):
    """opens edf files, bins them and saves them

    Args:
        input_folder (str):  input folder
        output_folder (str): output folder
        bin_factor (int):    binning factor

    Returns:
        None
    """
    image_filenames = in_out.create_list_of_files(input_folder, "edf")

    for index in range(len(image_filenames) // 2):
        image_to_bin = in_out.open_sequence(image_filenames[:bin_factor])
        del image_filenames[:bin_factor]
        binned_image = resampling.bin_resize(image_to_bin, bin_factor)
        in_out.save_tif_image(binned_image[0], output_folder + '{:04d}'.format(index))


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

    above_map_files = in_out.create_list_of_files(input_folder, input_image_type)
    for image_nb, image_file in enumerate(above_map_files):
        image = in_out.open_image(image_file)
        cropped_image = image[min_max_list[1][0]:min_max_list[1][1], min_max_list[0][0]:min_max_list[0][1]]
        in_out.save_tif_image(cropped_image, output_path + '{:04d}'.format(image_nb))


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

    image_filenames = in_out.create_list_of_files(input_folder, input_image_type)

    for index in range(len(image_filenames) // bin_factor):
        image_to_bin = in_out.open_sequence(image_filenames[:bin_factor])
        cropped_image = image_to_bin[:, min_max_list[1][0]:min_max_list[1][1], min_max_list[0][0]:min_max_list[0][1]]
        del image_filenames[:bin_factor]
        cropped_image = resampling.bin_resize(cropped_image, bin_factor)
        in_out.save_tif_image(cropped_image[0], output_folder + '{:04d}'.format(index))


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

    image_filenames = in_out.create_list_of_files(input_folder, input_image_type)

    for index in range(len(image_filenames) // bin_factor):
        image_to_bin = in_out.open_sequence(image_filenames[:bin_factor])
        del image_filenames[:bin_factor]
        binned_image = resampling.bin_resize(image_to_bin, bin_factor)
        in_out.save_tif_image(binned_image[0], output_folder + '{:04d}'.format(index))


def easy_registration(input_folder):
    """Made for direct registrations: will be removed in future versions

    Args:
        input_folder (str): input folder

    Returns:
        None
    """
    above_filenames = in_out.create_list_of_files(input_folder + "\\Cropped_Above_Acquisition\\",
                                                  "tif")
    below_filenames = in_out.create_list_of_files(input_folder + "\\Cropped_Below_Acquisition\\",
                                                  "tif")
    above_image = in_out.open_sequence(above_filenames)
    below_image = in_out.open_sequence(below_filenames)

    above_mask = np.copy(above_image)
    above_mask[above_mask > 0.28] = 1
    above_mask[above_mask <= 0.28] = 0

    below_mask = np.copy(below_image)
    below_mask[below_mask > 0.28] = 1
    below_mask[below_mask <= 0.28] = 0

    translation_transform = registration.registration_computation(above_image,
                                                                  below_image,
                                                                  transform_type="translation",
                                                                  metric="cc",
                                                                  moving_mask=above_mask,
                                                                  ref_mask=below_mask,
                                                                  verbose=True)

    rotation_transform = registration.registration_computation(above_image,
                                                               below_image,
                                                               transform_type="rotation",
                                                               metric="msq",
                                                               moving_mask=above_mask,
                                                               ref_mask=below_mask,
                                                               verbose=True)
    #
    # # Registering the above image
    above_image = registration.apply_itk_transformation(above_image, translation_transform, "linear")
    above_image = registration.apply_itk_transformation(above_image, rotation_transform, "linear")

    smol_translation = Sitk.TranslationTransform(3)
    smol_translation.SetOffset(([0.0, -0.2, 0.0]))
    # above_image = registration.apply_itk_transformation(above_image, smol_translation, "linear")
    smol_translation.SetOffset(([0.0, 0.0, -0.5]))
    above_image = registration.apply_itk_transformation(above_image, smol_translation, "linear")
    below_image = registration.apply_itk_transformation(below_image, smol_translation, "linear")

    in_out.save_tif_sequence(above_image, input_folder + "\\Cropped_Above_Acquisition_Registered\\")
    in_out.save_tif_sequence(below_image, input_folder + "\\Cropped_Below_Acquisition_Registered\\")


def easy_decomposition(input_folder, material):

    above_filenames = in_out.create_list_of_files(input_folder + material + "\\Cropped_Above_Acquisition_Registered\\", "tif")
    below_filenames = in_out.create_list_of_files(input_folder + material + "\\Cropped_Below_Acquisition_Registered\\", "tif")

    for filename_index in range(min(len(above_filenames), len(below_filenames))):

        above_image = in_out.open_image(above_filenames[filename_index])
        below_image = in_out.open_image(below_filenames[filename_index])
        images = np.stack((above_image, below_image), axis=0)
        if material == "Au":
            densities = np.array([19.3, 4.93, 1.0])
            material_attenuations = np.array([[165.8642, 016.3330, 000.1827],
                                              [040.7423, 016.8852, 000.1835]])
        else:
            densities = np.array([4.93, 19.3, 1.0])
            material_attenuations = np.array([[170.9231, 389.5878, 000.3188],
                                              [033.6370, 421.9694, 000.3307]])

        concentration_maps = material_decomposition.decomposition_equation_resolution(images, densities,
                                                                                      material_attenuations,
                                                                                      volume_fraction_hypothesis=True,
                                                                                      verbose=False)

        material_decomposition.loading_bar(filename_index, min(len(above_filenames), len(below_filenames)))
        concentration_maps[0][below_image == 0] = 0
        concentration_maps[1][below_image == 0] = 0
        concentration_maps[2][below_image == 0] = 0

        in_out.save_tif_image(concentration_maps[0], input_folder + material + "\\Cropped_"
                              + material + "_decomposition\\" + '{:04d}'.format(filename_index))
        if material == "Au":
            in_out.save_tif_image(concentration_maps[1], input_folder + material + "\\Cropped_"
                                  + "I_decomposition\\" + '{:04d}'.format(filename_index))
        elif material == "I":
            in_out.save_tif_image(concentration_maps[1], input_folder + material + "\\Cropped_"
                                  + "Au_decomposition\\" + '{:04d}'.format(filename_index))
        in_out.save_tif_image(concentration_maps[2], input_folder + material + "\\Cropped_"
                              + "Water_decomposition\\" + '{:04d}'.format(filename_index))


if __name__ == '__main__':

    folder = "E:\\Annee_2\\md1237\\Knees\\Knee3_WhiteDot\\"
    Knee3_WhiteDot = SpectralImagingExperiment("Knee3_WhiteDot",
                                                    folder, "mouse knee",
                                                    "esrf", ["Au", "I"], resolution=6.0, bin_factor=2)

    # Knee3_WhiteDot.material_decomposition()
    # folder = "C:\\Users\\ctavakol\\Desktop\\Experiences_Fevrier_2021\\ESRF_md1237\\Genoux\\Knee11\\"
    # Knee11 = SpectralImagingExperiment("Knee11",
    #                                     folder, "mouse knee",
    #                                     "esrf", ["Au", "I"], resolution=6.43, bin_factor=2)
    # Knee11.material_decomposition()
    # Knee11.material_decomposition()

    folder = "D:\\md1237\\Knees\\Knee11_WhiteDot\\"
    Knee11_WhiteDot = SpectralImagingExperiment("Knee11_WhiteDot",
                                            folder, "mouse knee",
                                            "esrf", ["Au"], resolution=6.0, bin_factor=2)

    # Knee11_WhiteDot.register_volumes()

    folder = "D:\\md1237\\Knees\\Knee12_WhiteDot\\"
    Knee12_WhiteDot = SpectralImagingExperiment("Knee12_WhiteDot",
                                            folder, "mouse knee",
                                            "esrf", ["Au"], resolution=6.0, bin_factor=2)

    # Knee12_WhiteDot.material_segmentation()


    folder = "E:\\Annee_2\\md1237\\Knees\\Knee13_WhiteDot\\"
    Knee13_WhiteDot = SpectralImagingExperiment("Knee13_WhiteDot",
                                                folder, "mouse knee",
                                                "esrf", ["Au"], resolution=6.0, bin_factor=2)

    # Knee13_WhiteDot.material_decomposition()


    folder = "E:\\Annee_2\\md1237\\Knees\\Knee14_WhiteDot\\"
    Knee14_WhiteDot = SpectralImagingExperiment("Knee14_WhiteDot",
                                                folder, "mouse knee",
                                                "esrf", ["Au"], resolution=6.0, bin_factor=2)

    # Knee14_WhiteDot.material_decomposition()


    folder = "E:\\Annee_2\\md1237\\Knees\\Knee16_WhiteDot\\"
    Knee16_WhiteDot = SpectralImagingExperiment("Knee16_WhiteDot",
                                                folder, "mouse knee",
                                                "esrf", ["Au"], resolution=6.0, bin_factor=2)

    # Knee16_WhiteDot.material_decomposition()
    knee_name = "Knee3"
    whitedot = False
    name = knee_name + whitedot * "_WhiteDot"
    folder = "D:\\Annee_2\\md1237\\Knees\\" + name + "\\"
    print(folder)
    Knee = SpectralImagingExperiment(knee_name,
                                     folder, "mouse knee",
                                     "esrf", ["Au"], resolution=6.0, bin_factor=2)
    Knee.register_volumes()
    # open_crop_bin_and_save(folder + name + "_Mtp_6um_AboveAu__001_pag\\",
    #                        folder + "Au\\Above_Acquisition\\",
    #                        [[250, 2310],[250, 2310]],
    #                        2,
    #                        "edf")
    # open_crop_bin_and_save(folder + name + "_Mtp_6um_BelowAu__001_pag\\",
    #                        folder + "Au\\Below_Acquisition\\",
    #                        [[250, 2310],[250, 2310]],
    #                        2,
    #                        "edf")
    # open_bin_and_save(folder + "Au\\Above_Acquisition\\",
    #                   folder + "Au\\Above_Acquisition_Binned\\",
    #                   2,
    #                   "tif")
    # open_bin_and_save(folder + "Au\\Below_Acquisition\\",
    #                   folder + "Au\\Below_Acquisition_Binned\\",
    #                   2,
    #                   "tif")
    #
    # open_crop_bin_and_save(folder + name + "_Mtp_6um_AboveI__001_pag\\",
    #                        folder + "I\\Above_Acquisition\\",
    #                        [[250, 2310],[250, 2310]],
    #                        2,
    #                        "edf")
    # open_crop_bin_and_save(folder + name + "_Mtp_6um_BelowI__001_pag\\",
    #                        folder + "I\\Below_Acquisition\\",
    #                        [[250, 2310],[250, 2310]],
    #                        2,
    #                        "edf")
    # open_bin_and_save(folder + "I\\Above_Acquisition\\",
    #                   folder + "I\\Above_Acquisition_Binned\\",
    #                   2,
    #                   "tif")
    # open_bin_and_save(folder + "I\\Below_Acquisition\\",
    #                   folder + "I\\Below_Acquisition_Binned\\",
    #                   2,
    #                   "tif")

    # Knee.material_decomposition()


    # folder = "D:\\md1237\\Knees\\Knee17\\"
    # Knee3_WhiteDot = SpectralImagingExperiment("Knee17",
    #                                             folder, "mouse knee",
    #                                             "esrf", ["Au", "I"], resolution=6.0, bin_factor=2)
    # Knee3_WhiteDot.register_volumes()

    # BiColor_B1toB9__ = SpectralImagingExperiment("BiColor_B1toB9__", "D:\\md1237\\BiColor_B1toB9__\\", "phantom",
    #                                             ["Au", "I"], 21.4, bin_factor=2)
    # BiColor_B1toB9__.material_decomposition()
    # open_crop_and_save("D:\\md1237\\BiColor_Cell_Pellet__\\Au\\", np.array([[146, 305], [412, 571]]))
    # easy_registration("D:\\md1237\\BiColor_Cell_Pellet__\\Au\\")
    # easy_decomposition("D:\\md1237\\BiColor_Cell_Pellet__\\", "Au")

    # pellet_bicolore = SpectralImagingExperiment("BiColor_Cell_Pellet__", "D:\\md1237\\BiColor_Cell_Pellet__\\", "phantom",
    #                                             ["Au", "I"], 21.4, bin_factor=2)
    # pellet_bicolore.()
    # pellet_bicolore.material_decomposition()

    folder = "C:\\Users\\ctavakol\\Desktop\\Experiences_Fevrier_2021\\ESRF_md1237\\Genoux\\KneeReal17_WhiteDot_Mtp_6um\\"
    KneeReal17_WhiteDot = SpectralImagingExperiment("KneeReal17_WhiteDot_Mtp_6um",
                                                    "C:\\Users\\ctavakol\\Desktop\\Experiences_Fevrier_2021\\" +
                                                    "ESRF_md1237\\Genoux\\KneeReal17_WhiteDot_Mtp_6um\\", "mouse knee",
                                                    "esrf", ["Au", "I"], resolution=6.0, bin_factor=2)
    # KneeReal17_WhiteDot.register_volumes()
    # KneeReal17_WhiteDot.material_segmentation()
    # KneeReal17_WhiteDot.material_decomposition()
