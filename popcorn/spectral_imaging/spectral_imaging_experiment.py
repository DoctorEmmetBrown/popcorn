import popcorn.input_output as in_out
import popcorn.image_processing.resampling as resampling
import material_decomposition
import popcorn.registration.registration as registration

import numpy as np

from skimage import filters

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
        self.resolution = resolution
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
        ref_img_filenames = in_out.create_list_of_files(self.path + self.materials[0] + "\\Below_Acquisition\\",
                                                        "tif")
        ref_img = in_out.open_sequence(ref_img_filenames[0:len(ref_img_filenames)//2])
        ref_thresh = filters.threshold_otsu(ref_img)
        ref_mask = np.copy(ref_img)
        ref_mask[ref_mask >= ref_thresh] = 1
        ref_mask[ref_mask < ref_thresh] = 0
        moving_img_filenames = in_out.create_list_of_files(self.path + self.materials[0] + "\\Above_Acquisition\\",
                                                           "tif")
        moving_img = in_out.open_sequence(moving_img_filenames[0:len(ref_img_filenames)//2])
        moving_thresh = filters.threshold_otsu(moving_img)
        moving_mask = np.copy(moving_img)
        moving_mask[moving_mask >= moving_thresh] = 1
        moving_mask[moving_mask < moving_thresh] = 0
        transformation = registration.registration_computation(moving_img, ref_img, reference_mask=ref_mask,
                                                               moving_mask=moving_mask, is_translation_needed=True,
                                                               is_rotation_needed=False, verbose=True)
        moving_img = registration.apply_itk_transformation(moving_img, transformation)
        in_out.save_tif_sequence(moving_img, self.path + self.materials[0] + "\\Above_Acquisition_Registered\\")

        # if len(self.materials) > 1:
        #     moving_img_filenames = in_out.create_list_of_files(self.path + self.materials[1] + "\\Above_Acquisition\\",
        #                                                        "tif")
        #     moving_img = in_out.open_sequence(moving_img_filenames)
        #     transformation = registration.registration_computation(moving_img, ref_img, is_translation_needed=False,
        #                                                            is_rotation_needed=True)
        #     moving_img = registration.apply_itk_transformation(moving_img, transformation)
        #     in_out.save_tif_sequence(moving_img, self.path + self.materials[1] + "\\Above_Acquisition_Registered\\")
        #
        #     moving_img_filenames = in_out.create_list_of_files(self.path + self.materials[1] + "\\Below_Acquisition\\",
        #                                                        "tif")
        #     moving_img = in_out.open_sequence(moving_img_filenames)
        #     transformation = registration.registration_computation(moving_img, ref_img, is_translation_needed=False,
        #                                                            is_rotation_needed=True)
        #     moving_img = registration.apply_itk_transformation(moving_img, transformation)
        #     in_out.save_tif_sequence(moving_img, self.path + self.materials[1] + "\\Below_Acquisition_Registered\\")

    def material_decomposition(self):
        if self.type == "phantom":
            if len(self.materials) == 1:
                material = self.materials[0]
                above_filenames = in_out.create_list_of_files(self.path + self.materials[0] + "\\Above_Acquisition\\",
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
                                                                                                  volume_fraction_hypothesis=True,
                                                                                                  verbose=True)
                    in_out.save_tif_image(concentration_maps[0], self.path + material + "\\"
                                          + material + "_decomposition\\" + '{:04d}'.format(filename_index))
                    in_out.save_tif_image(concentration_maps[1], self.path + material + "\\"
                                          + "Water_decomposition\\" + '{:04d}'.format(filename_index))
            elif len(self.materials) == 2:
                for material in self.materials:
                    above_filenames = in_out.create_list_of_files(
                        self.path + material + "\\Above_Acquisition\\",
                        "tif")
                    below_filenames = in_out.create_list_of_files(
                        self.path + material + "\\Below_Acquisition\\",
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
                                                                                                      verbose=True)
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

                # au_above_filenames = in_out.create_list_of_files(self.path + "Au" + "\\Above_Acquisition\\",
                #                                                  "tif")
                # au_below_filenames = in_out.create_list_of_files(self.path + "Au" + "\\Below_Acquisition\\",
                #                                                  "tif")
                # i_above_filenames = in_out.create_list_of_files(self.path + "I" + "\\Above_Acquisition\\",
                #                                                 "tif")
                # i_below_filenames = in_out.create_list_of_files(self.path + "I" + "\\Below_Acquisition\\",
                #                                                 "tif")
                # densities = np.array([19.3, 4.93, 1.0, 7.874])
                #
                # material_attenuations = np.array([[165.8642, 016.3330, 000.1827, 4.4735],
                #                                   [040.7423, 016.8852, 000.1835, 4.5973],
                #                                   [389.5878, 170.9231, 000.3188, 45.1628],
                #                                   [421.9694, 033.6370, 000.3307, 49.1556]])
                #
                # for filename_index in range(0, min(len(au_above_filenames), len(au_below_filenames), len(i_above_filenames), len(i_below_filenames))):
                #     au_above_image = in_out.open_image(au_above_filenames[filename_index])
                #     au_below_image = in_out.open_image(au_below_filenames[filename_index])
                #     i_above_image = in_out.open_image(i_above_filenames[filename_index])
                #     i_below_image = in_out.open_image(i_below_filenames[filename_index])
                #     images = np.stack((au_above_image, au_below_image, i_above_image, i_below_image), axis=0)
                #     concentration_maps = material_decomposition.decomposition_equation_resolution(images, densities,
                #                                                                                   material_attenuations,
                #                                                                                   volume_fraction_hypothesis=False,
                #                                                                                   verbose=True)
                #     in_out.save_tif_image(concentration_maps[0], self.path + "Au" + "_decomposition\\" + '{:04d}'.format(filename_index))
                #     in_out.save_tif_image(concentration_maps[1], self.path + "I" + "_decomposition\\" + '{:04d}'.format(filename_index))
                #     in_out.save_tif_image(concentration_maps[2], self.path + "Water" + "_decomposition\\" + '{:04d}'.format(filename_index))
                #     in_out.save_tif_image(concentration_maps[3], self.path + "Random" + "_decomposition\\" + '{:04d}'.format(filename_index))


if __name__ == '__main__':
    BiColor_B1toB9__ = SpectralImagingExperiment("BiColor_B1toB9__", "D:\\md1237\\BiColor_B1toB9__\\", "phantom",
                                                ["Au", "I"], 21.4, bin_factor=2)
    BiColor_B1toB9__.register_volumes()
    #
    # pellet_bicolore = SpectralImagingExperiment("BiColor_Cell_Pellet__", "D:\\md1237\\BiColor_Cell_Pellet__\\", "phantom",
    #                                            ["Au", "I"], 21.4, bin_factor=2)
    # pellet_bicolore.material_decomposition()
    #
    # Gamme_Cellules_13_14_16_11__ = SpectralImagingExperiment("Gamme_Cellules_13_14_16_11__", "D:\\md1237\\Gamme_Cellules_13_14_16_11__\\", "phantom",
    #                                            ["Au"], 21.4, bin_factor=2)
    # Gamme_Cellules_13_14_16_11__.material_decomposition()
    #
    # Gamme_Cellules_13_14_16_11_whitedot_ = SpectralImagingExperiment("Gamme_Cellules_13_14_16_11_whitedot_", "D:\\md1237\\Gamme_Cellules_13_14_16_11_whitedot_\\", "phantom",
    #                                            ["Au"], 21.4, bin_factor=2)
    # Gamme_Cellules_13_14_16_11_whitedot_.material_decomposition()
    #
    # GammeAu_0_to_14__ = SpectralImagingExperiment("GammeAu_0_to_14__", "D:\\md1237\\GammeAu_0_to_14__\\", "phantom",
    #                                            ["Au"], 21.4, bin_factor=2)
    # GammeAu_0_to_14__.material_decomposition()

    # GammeI_0_to_10__ = SpectralImagingExperiment("GammeI_0_to_10__", "D:\\md1237\\GammeI_0_to_10__\\", "phantom",
    #                                             ["I"], 21.4, bin_factor=2)
    # GammeI_0_to_10__.register_volumes()
    #
    # GammeI_HI__ = SpectralImagingExperiment("GammeI_HI__", "D:\\md1237\\GammeI_HI__\\", "phantom",
    #                                            ["I"], 21.4, bin_factor=2)
    # GammeI_HI__.material_decomposition()
    #
    # GammeI_MS_301020_ = SpectralImagingExperiment("GammeI_MS_301020_", "D:\\md1237\\GammeI_MS_301020_\\", "phantom",
    #                                            ["I"], 21.4, bin_factor=2)
    # GammeI_MS_301020_.material_decomposition()
    #
    # GammeI_PBS_301020_ = SpectralImagingExperiment("GammeI_PBS_301020_", "D:\\md1237\\GammeI_PBS_301020_\\", "phantom",
    #                                            ["I"], 21.4, bin_factor=2)
    # GammeI_PBS_301020_.material_decomposition()
