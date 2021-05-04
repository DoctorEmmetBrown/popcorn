import popcorn.input_output as in_out
import popcorn.image_processing.resampling as resampling
import material_decomposition
import popcorn.registration.registration as registration
import popcorn.registration.segmentation as segmentation

from popcorn.registration.pipelines import aligning_skull_pipeline

import numpy as np
import math

import SimpleITK as Sitk
from skimage import filters

from skimage.measure import label, regionprops
from skimage import img_as_ubyte

import PyIPSDK
import PyIPSDK.IPSDKIPLBinarization as Bin
import PyIPSDK.IPSDKIPLMorphology as Morpho
import PyIPSDK.IPSDKIPLAdvancedMorphology as AdvMorpho


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

    if material == "Au":
        rgb = [0.8, 0.8, 0]
    else:
        rgb = [0.8, 0.35, 0]

    for i in range(0, cells_part.shape[0]):
        red_slice = img_as_ubyte(noise[i, :, :] + rgb[0] * cells_part[i, :, :] + cells_offset[i, :, :])
        green_slice = img_as_ubyte(noise[i, :, :] + rgb[1] * cells_part[i, :, :] + cells_offset[i, :, :])
        blue_slice = img_as_ubyte(noise[i, :, :] + rgb[2] * cells_part[i, :, :] + cells_offset[i, :, :])

        rgb_cells = np.stack((red_slice, green_slice, blue_slice), axis=-1)
        in_out.save_tif_image(rgb_cells, output_folder + '{:04d}'.format(i), rgb=True)


class SpectralImagingExperiment:

    def __init__(self, name, path, sample_type, materials, resolution, bin_factor=1):
        """constructor

        Args:
            name (str):            sample radix
            path (str):            data path
            sample_type (str):     either "phantom", "rat brain" or "rat knee"
            materials (list[str]): list of kedge materials
        """
        self.name = name
        self.path = path
        self.type = sample_type
        self.materials = materials
        self.resolution = resolution * bin_factor
        self.bin_factor = bin_factor

    def conversion(self):
        for material in self.materials:
            print("material :", material)
            above_image_filenames = in_out.create_list_of_files(self.path + "*Above*" + material + "*", "tif")
            above_min, above_max = retrieve_min_max_from_path(in_out.remove_filename_in_path(above_image_filenames[0]))
            print("Above min -> ", above_min)
            print("Above max -> ", above_max)

            for index in range(0, len(above_image_filenames)//self.bin_factor):
                image_to_bin = in_out.open_sequence(above_image_filenames[:self.bin_factor])
                del above_image_filenames[:self.bin_factor]
                binned_image = resampling.bin_resize(image_to_bin, self.bin_factor)
                converted_image = resampling.conversion_uint16_to_float32(binned_image, above_min, above_max)
                in_out.save_tif_image(converted_image[0],
                                      self.path + material + "\\Above_Acquisition\\" + '{:04d}'.format(index))

            below_image_filenames = in_out.create_list_of_files(self.path + "*Below*" + material + "*", "tif")
            below_min, below_max = retrieve_min_max_from_path(in_out.remove_filename_in_path(below_image_filenames[0]))
            print("Below min -> ", below_min)
            print("Below max -> ", below_max)

            for index in range(0, len(below_image_filenames)//self.bin_factor):
                image_to_bin = in_out.open_sequence(below_image_filenames[:self.bin_factor])
                del below_image_filenames[:self.bin_factor]
                binned_image = resampling.bin_resize(image_to_bin, self.bin_factor)
                converted_image = resampling.conversion_uint16_to_float32(binned_image, below_min, below_max)
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
                threshold_value = segmentation.find_threshold_value(material)
                above_mask = above_image > threshold_value
                below_mask = below_image > threshold_value

                # -- Extracting skulls
                above_skull, above_skull_bbox = segmentation.extract_skull(above_mask)
                below_skull, below_skull_bbox = segmentation.extract_skull(below_mask)

                rotation_transform = registration.registration_computation(above_image,
                                                                           below_image,
                                                                           moving_mask = above_skull,
                                                                           reference_mask = below_skull,
                                                                           is_translation_needed=False,
                                                                           is_rotation_needed=True,
                                                                           verbose=True)

                # Registering the above image
                above_image = registration.apply_itk_transformation(above_image, rotation_transform, "linear")
                in_out.save_tif_sequence(above_image,
                                      self.path + material + "\\Above_Acquisition_Registered\\")
                aligning_skull_pipeline(self.path + material + "\\Above_Acquisition_Registered\\",
                                        self.path + material + "\\Below_Acquisition\\",
                                        material)
        elif self.type == "mouse legs":
            for material in self.materials:
                print("--------", "Registering", material, "--------")
                above_filenames = in_out.create_list_of_files(self.path + material + "\\Above_Acquisition\\",
                                                              "tif")
                below_filenames = in_out.create_list_of_files(self.path + material + "\\Below_Acquisition\\",
                                                              "tif")
                above_image = in_out.open_sequence(above_filenames)
                below_image = in_out.open_sequence(below_filenames)

                # -- Threshold computation
                threshold_value = segmentation.find_threshold_value(material)
                above_mask = np.copy(above_image)
                above_mask[above_mask > threshold_value] = 1
                above_mask[above_mask <= threshold_value] = 0
                below_mask = np.copy(below_image)
                below_mask[below_mask > threshold_value] = 1
                below_mask[below_mask <= threshold_value] = 0

                rotation_transform = registration.registration_computation(above_image,
                                                                           below_image,
                                                                           moving_mask=above_mask,
                                                                           reference_mask=below_mask,
                                                                           is_translation_needed=False,
                                                                           is_rotation_needed=True,
                                                                           verbose=True)

                # Registering the above image
                above_image = registration.apply_itk_transformation(above_image, rotation_transform, "linear")
                in_out.save_tif_sequence(above_image,
                                         self.path + material + "\\Above_Acquisition_Registered\\")
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
                threshold_value = segmentation.find_threshold_value(material)
                if material == "Au":
                    threshold_value = 0.2
                else:
                    threshold_value = 0.2

                above_mask = np.ones(above_image.shape)
                # above_mask[above_mask > threshold_value] = 0
                # above_mask[above_mask <= threshold_value] = 1

                above_mask[above_image > 0.18] = 0
                above_mask[above_image < 0.14] = 0

                # below_mask = np.copy(below_image)
                # below_mask[below_mask > threshold_value] = 0
                # below_mask[below_mask <= threshold_value] = 1

                below_mask = np.ones(below_image.shape)
                # above_mask[above_mask > threshold_value] = 0
                # above_mask[above_mask <= threshold_value] = 1

                below_mask[below_image > 0.17] = 0
                below_mask[below_image < 0.14] = 0

                rotation_transform = registration.registration_computation(above_image,
                                                                           below_image,
                                                                           moving_mask=above_mask,
                                                                           reference_mask=below_mask,
                                                                           is_translation_needed=False,
                                                                           is_rotation_needed=True,
                                                                           verbose=True)

                # Registering the above image
                above_image = registration.apply_itk_transformation(above_image, rotation_transform, "linear")
                in_out.save_tif_sequence(above_image,
                                         self.path + material + "\\Above_Acquisition_Registered\\")

    def manual_registration(self, slice_of_interest):
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
            belowsimage = in_out.open_sequence(in_out.create_list_of_files(self.path + material + "\\Below_Acquisition\\",
                                                          "tif")[slice_of_interest-5:slice_of_interest+6])
            above_image = in_out.open_sequence(above_filenames)
            for z in range(0, 20):
                above_image_itk = Sitk.GetImageFromArray(above_image)
                translation_transformation = Sitk.TranslationTransform(above_image_itk.GetDimension())
                translation_transformation.SetOffset((-0.14, z/10 - 1, 0))
                above_image_itk = Sitk.Resample(above_image_itk, translation_transformation, Sitk.sitkLinear, 0.0,
                                                above_image_itk.GetPixelIDValue())
                registered_image = Sitk.GetArrayFromImage(above_image_itk)
                images = np.stack((registered_image[slice_of_interest-5:slice_of_interest+6], belowsimage), axis=0)
                concentration_maps = material_decomposition.decomposition_equation_resolution(images, densities,
                                                                                              material_attenuations,
                                                                                              volume_fraction_hypothesis=False,
                                                                                              verbose=False)
                in_out.save_tif_sequence(concentration_maps[0], self.path + material + "\\manual_registrationz_" + str(z/10 - 1) + "\\")
            # for y in range(0, 10):
            #     above_image_itk = Sitk.GetImageFromArray(above_image)
            #     translation_transformation = Sitk.TranslationTransform(above_image_itk.GetDimension())
            #     translation_transformation.SetOffset((1 - y/5, 0, 0))
            #     above_image_itk = Sitk.Resample(above_image_itk, translation_transformation, Sitk.sitkLinear, 0.0,
            #                                     above_image_itk.GetPixelIDValue())
            #     registered_image = Sitk.GetArrayFromImage(above_image_itk)
            #     images = np.stack((registered_image[slice_of_interest], below_slice), axis=0)
            #     concentration_maps = material_decomposition.decomposition_equation_resolution(images, densities,
            #                                                                                   material_attenuations,
            #                                                                                   volume_fraction_hypothesis=False,
            #                                                                                   verbose=False)
            #     in_out.save_tif_image(concentration_maps[0], self.path + material + "\\manual_registration\\y_" + str(1 - y/5) + ".tif")

            # below_image = in_out.open_sequence(below_filenames)


            # rotation_transform = registration.registration_computation(above_image,
            #                                                            below_image,
            #                                                            moving_mask = above_skull,
            #                                                            reference_mask = below_skull,
            #                                                            is_translation_needed=False,
            #                                                            is_rotation_needed=True,
            #                                                            verbose=True)
    def material_decomposition(self, registration=False):
        if len(self.materials) == 1:
            material = self.materials[0]
            if self.type == "phantom":
                if registration:
                    above_filenames = in_out.create_list_of_files(self.path + self.materials[0] + "\\Above_Acquisition_Registered\\",
                                                                 "tif")
                else:
                    above_filenames = in_out.create_list_of_files(self.path + self.materials[0] + "\\Above_Acquisition\\",
                                                                 "tif")

                below_filenames = in_out.create_list_of_files(self.path + self.materials[0] + "\\Below_Acquisition\\",
                                                              "tif")
            elif self.type == "rat brain":
                above_filenames = in_out.create_list_of_files(self.path + self.materials[0] + "\\Aligned_Above_Acquisition\\",
                                                              "tif")

                below_filenames = in_out.create_list_of_files(self.path + self.materials[0] + "\\Aligned_Below_Acquisition\\",
                                                              "tif")
            elif self.type == "mouse legs":
                above_filenames = in_out.create_list_of_files(self.path + self.materials[0] + "\\Above_Acquisition_Registered\\",
                                                              "tif")

                below_filenames = in_out.create_list_of_files(self.path + self.materials[0] + "\\Below_Acquisition\\",
                                                              "tif")
            for filename_index in range(0, min(len(above_filenames), len(below_filenames))):
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
                concentration_maps = material_decomposition.decomposition_equation_resolution(images, densities,
                                                                                              material_attenuations,
                                                                                              volume_fraction_hypothesis=False,
                                                                                              verbose=False)

                material_decomposition.loading_bar(filename_index, min(len(above_filenames), len(below_filenames)))
                in_out.save_tif_image(concentration_maps[0], self.path + material + "\\"
                                      + material + "_decomposition\\" + '{:04d}'.format(filename_index))
                in_out.save_tif_image(concentration_maps[1], self.path + material + "\\"
                                      + "Water_decomposition\\" + '{:04d}'.format(filename_index))
        elif len(self.materials) == 2:
            for material in self.materials:
                if material == "I":
                    if self.type == "phantom":
                        if registration:
                            above_filenames = in_out.create_list_of_files(self.path + self.materials[0] + "\\Above_Acquisition_Registered\\",
                                                                         "tif")
                        else:
                            above_filenames = in_out.create_list_of_files(self.path + self.materials[0] + "\\Above_Acquisition\\",
                                                                         "tif")
                        below_filenames = in_out.create_list_of_files(
                            self.path + material + "\\Below_Acquisition\\",
                            "tif")
                    else:
                        above_filenames = in_out.create_list_of_files(
                            self.path + material + "\\Aligned_Above_Acquisition\\",
                            "tif")
                        below_filenames = in_out.create_list_of_files(
                            self.path + material + "\\Aligned_Below_Acquisition\\",
                            "tif")
                    for filename_index in range(0, min(len(above_filenames), len(below_filenames))):
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
                list_of_coordinates = []
                diameter = 2 * int(1.25 * max_radius)
                for centroid in centroids:
                    list_of_coordinates.append([max(0, (centroid[0] - int(diameter / 2))),
                                                min(first_image.shape[0], (centroid[0] + int(diameter / 2))),
                                                max(0, (centroid[1] - int(diameter / 2))),
                                                min(first_image.shape[0], (centroid[1] + int(diameter / 2)))])

                print(str(len(centroids)) + " tubes found")
                material_list_of_files = in_out.create_list_of_files(self.path + material + "\\" + material
                                                                      + "_decomposition\\", "tif")

                for file_nb in range(0, len(material_list_of_files)):
                    print("cropping: " + str(int(file_nb / len(material_list_of_files) * 100)) + "%",
                          end="\r")
                    current_image = in_out.open_image(material_list_of_files[file_nb])
                    for tube_nb in range(0, len(centroids)):
                        cropped_image = current_image[list_of_coordinates[tube_nb][0]:list_of_coordinates[tube_nb][1],
                                                      list_of_coordinates[tube_nb][2]:list_of_coordinates[tube_nb][3]]
                        in_out.save_tif_image(cropped_image,  self.path + material + "\\" + material + "_tubes\\tube_"
                                              + str(tube_nb) + "\\" + '{:04d}'.format(file_nb) + ".tif")
                f = open(self.path + material + "\\" + material + "_tubes\\analysis.txt" , mode='w', encoding='utf-8')
                for tube_nb in range(0, len(centroids)):
                    f.write("tube nb " + str(tube_nb) + " - position : " + str(centroids[tube_nb][0]) + " "
                            + str(centroids[tube_nb][1]) + "\n")
                    tube_filenames = in_out.create_list_of_files(self.path + material + "\\" + material + "_tubes\\tube_"
                                                  + str(tube_nb) + "\\", "tif")
                    tube = in_out.open_sequence(tube_filenames)

                    segmented_nanoparticles = np.copy(tube)
                    # tube_slices = [54, 51, 52, 0, 0, 0, 100]
                    tube_slices = [0, 0, 0, 0, 0, 0, 0]
                    for slice_nb in range(0, segmented_nanoparticles.shape[0]):
                        slice = segmented_nanoparticles[slice_nb, :, :]
                        if slice_nb < tube_slices[tube_nb]:
                            slice[slice < 100.0] = 0
                        else:
                            slice[slice < 0.5] = 0
                        mean_val = np.mean(slice[slice > 0.5])
                        # print(slice_nb, "mean val:", mean_val)
                        slice[slice < 0.35 * mean_val] = 0
                        # slice[slice > 5 * mean_val] = 0
                        segmented_nanoparticles[slice_nb, :, :] = slice

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
                                                              file=f)
                        f.write("\n")
                f.close()

        elif self.type == "rat brain":
            for material in self.materials:
                if material == "I":
                    threshold = 0.4
                else:
                    threshold = None
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

        elif self.type == "mouse legs":
            for material in self.materials:
                positions = [[294, 584],
                             [618, 1066],
                             [982, 678],
                             [604, 330]]
                concentration_map_files = in_out.create_list_of_files(self.path + material + "\\" + material +
                                                                      "_decomposition\\", "tif")
                concentration_map = in_out.open_sequence(concentration_map_files)
                nb = 0
                for position in positions:
                    subvolume = concentration_map[:,
                                position[0] - 180: position[0] + 180,
                                position[1] - 180: position[1] + 180]
                    print(position)
                    print(subvolume.shape)
                    submask = np.copy(subvolume)
                    submask[submask < 2] = 0
                    submask[submask >= 2] = 1

                    subvolume_ipsdk = PyIPSDK.fromArray(submask)
                    opening_sphere = PyIPSDK.sphericalSEXYZInfo(1)
                    subvolume_ipsdk = Morpho.opening3dImg(subvolume_ipsdk, opening_sphere)
                    segmentation.segmented_cells_analysis(subvolume, subvolume_ipsdk.array, 2, self.resolution,
                                                          self.path + material + "\\legs_" + str(nb) + "_" + material + "_results.txt")
                    segmentation_coloration(self.path + material + "\\legs_" + str(nb) + "_" + material + "_segmentation\\", subvolume,
                                            subvolume_ipsdk.array, material)
                    in_out.save_tif_sequence(subvolume, self.path + material + "\\legs_" + str(nb) + "_" + material + "_decomposition\\")
                    nb += 1



if __name__ == '__main__':


    # BiColor_B1toB9__ = SpectralImagingExperiment("BiColor_B1toB9__", "D:\\md1237\\BiColor_B1toB9__\\", "phantom",
    #                                             ["Au", "I"], 21.4, bin_factor=2)
    # BiColor_B1toB9__.material_segmentation()

    pellet_bicolore = SpectralImagingExperiment("BiColor_Cell_Pellet__", "D:\\md1237\\BiColor_Cell_Pellet__\\", "phantom",
                                               ["Au", "I"], 21.4, bin_factor=2)
    # pellet_bicolore.()
    pellet_bicolore.material_decomposition()

    # GammeAu_0_to_14__ = SpectralImagingExperiment("GammeAu_0_to_14__", "D:\\md1237\\GammeAu_0_to_14__\\", "phantom",
    #                                            ["Au"], 21.4, bin_factor=2)
    # GammeAu_0_to_14__.material_decomposition()

    # GammeI_0_to_10__ = SpectralImagingExperiment("GammeI_0_to_10__", "D:\\md1237\\GammeI_0_to_10__\\", "phantom",
    #                                             ["I"], 21.4, bin_factor=2)
    # GammeI_0_to_10__.material_decomposition()
    #
    # GammeI_HI__ = SpectralImagingExperiment("GammeI_HI__", "D:\\md1237\\GammeI_HI__\\", "phantom",
    #                                            ["I"], 21.4, bin_factor=2)
    # GammeI_HI__.material_segmentation()
    #
    # GammeI_MS_301020_ = SpectralImagingExperiment("GammeI_MS_301020_", "D:\\md1237\\GammeI_MS_301020_\\", "phantom",
    #                                            ["I"], 21.4, bin_factor=2)
    # GammeI_MS_301020_.material_segmentation()

    # GammeI_PBS_301020_ = SpectralImagingExperiment("GammeI_PBS_301020_", "D:\\md1237\\GammeI_PBS_301020_\\", "phantom",
    #                                            ["I"], 21.4, bin_factor=2)
    # GammeI_PBS_301020_.material_segmentation()

    # MonoColor_Cell_Pellet__ = SpectralImagingExperiment("MonoColor_Cell_Pellet__", "D:\\md1237\\MonoColor_Cell_Pellet__\\", "phantom",
    #                                            ["Au"], 21.4, bin_factor=2)
    # MonoColor_Cell_Pellet__.conversion()

    # R1392_01__ = SpectralImagingExperiment("R1392_01__", "D:\\md1237\\R1392_01\\", "rat brain",
    #                                             ["Au", "I"], 21.4, bin_factor=2)
    # R1392_01__.material_segmentation()
    #
    # R1392_02__ = SpectralImagingExperiment("R1392_02__", "D:\\md1237\\R1392_02\\", "rat brain",
    #                                             ["Au", "I"], 21.4, bin_factor=2)
    # R1392_02__.material_segmentation()
    #
    # R1392_03__ = SpectralImagingExperiment("R1392_03__", "D:\\md1237\\R1392_03\\", "rat brain",
    #                                             ["Au", "I"], 21.4, bin_factor=2)
    # R1392_03__.material_segmentation()
    #
    # R1392_04__ = SpectralImagingExperiment("R1392_04__", "D:\\md1237\\R1392_04\\", "rat brain",
    #                                             ["Au", "I"], 21.4, bin_factor=2)
    # R1392_04__.material_segmentation()

    # Gamme_Cellules_13_14_16_11__ = SpectralImagingExperiment("Gamme_Cellules_13_14_16_11__", "D:\\md1237\\Gamme_Cellules_13_14_16_11__\\", "mouse legs",
    #                                            ["Au"], 21.4, bin_factor=2)
    # Gamme_Cellules_13_14_16_11__.material_segmentation()

    # Gamme_Cellules_13_14_16_11_whitedot_ = SpectralImagingExperiment("Gamme_Cellules_13_14_16_11_whitedot_", "D:\\md1237\\Gamme_Cellules_13_14_16_11_whitedot_\\", "mouse legs",
    #                                            ["Au"], 21.4, bin_factor=2)
    # Gamme_Cellules_13_14_16_11_whitedot_.material_segmentation()
