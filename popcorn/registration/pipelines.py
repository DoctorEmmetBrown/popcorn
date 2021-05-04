import PyIPSDK
import PyIPSDK.IPSDKIPLBinarization as Bin

import numpy as np

from popcorn.image_processing import resampling, segmentation, mathematical_morphology
from popcorn.spectral_imaging import registration

from popcorn import input_output
from popcorn.spectral_imaging.material_decomposition import three_materials_decomposition


def conversion_pipeline(above_folder, below_folder, bin_factor, above_min, above_max, below_min, below_max):
    """Opens 16 bit Above and Below images in main folder, converts them into float32 using min and max and saves them

    Args:
        above_folder (str): above folder path
        below_folder (str): below folder path
        bin_factor (int):   binning factor
        above_min (float):  min value of above image
        above_max (float):  max value of above image
        below_min (float):  min value of below image
        below_max (float):  max value of below image

    Returns:
        None
    """
    above_list_of_files = input_output.create_list_of_files(above_folder, extension="tif")
    above_image_int = input_output.open_sequence(above_list_of_files)
    above_image_float = resampling.conversion_uint16_to_float32(above_image_int, above_min, above_max)
    above_image_float = resampling.bin_resize(above_image_float, bin_factor)
    input_output.save_tif_sequence(above_image_float, input_output.remove_last_folder_in_path(above_folder)
                                   + "\\Above_Acquisition\\")

    below_list_of_files = input_output.create_list_of_files(below_folder, extension="tif")
    below_image_int = input_output.open_sequence(below_list_of_files)
    below_image_float = resampling.conversion_uint16_to_float32(below_image_int, below_min, below_max)
    below_image_float = resampling.bin_resize(below_image_float, bin_factor)
    input_output.save_tif_sequence(below_image_float, input_output.remove_last_folder_in_path(below_folder)
                                   + "\\Below_Acquisition\\")


def new_registration_pipeline(input_folder, main_element="Au", second_element="I", translation_bool = False, rotation_bool=True):
    """

    Args:
        input_folder ():
        main_element ():
        second_element ():
        translation_bool ():
        rotation_bool ():

    Returns:

    """

    above_list_of_files = input_output.create_list_of_files(input_folder + "Above_Acquisition", 'tif')
    above_image = input_output.open_seq(above_list_of_files)
    print("Above image opened")
    below_list_of_files = input_output.create_list_of_files(input_folder + "Below_Acquisition", 'tif')
    below_image = input_output.open_seq(below_list_of_files)
    print("Below image opened")

    above_img_ipsdk = PyIPSDK.fromArray(above_image)
    below_img_ipsdk = PyIPSDK.fromArray(below_image)

    threshold_value = segmentation.find_threshold_value(main_element)

    # -- Threshold computation
    above_thresholded_img_ipsdk = Bin.thresholdImg(above_img_ipsdk, threshold_value, 3)
    below_thresholded_img_ipsdk = Bin.thresholdImg(below_img_ipsdk, threshold_value, 3)

    # -- Extracting skulls
    above_skull, above_skull_bbox = segmentation.extract_skull(above_thresholded_img_ipsdk)
    below_skull, below_skull_bbox = segmentation.extract_skull(below_thresholded_img_ipsdk)

    translation_transform, rotation_transform = registration.registration_computation_with_mask(above_image,
                                                                                                below_image,
                                                                                                above_skull,
                                                                                                below_skull,
                                                                                                is_translation_needed=translation_bool,
                                                                                                is_rotation_needed=rotation_bool,
                                                                                                verbose=True)

    # 1) Registering the above image
    if translation_bool:
        above_image = registration.apply_itk_transformation_with_ref_img(above_image, below_image, translation_transform,
                                                                         "linear")
    if rotation_bool:
        above_image = registration.apply_itk_transformation(above_image, rotation_transform, "linear")

    main_concentration_map, second_concentration_map, water_concentration_map\
        = three_materials_decomposition(above_image, below_image, kedge_element=main_element, second_element=second_element)

    input_output.save_tif_sequence(above_image, input_folder + "registered_above_acquisition/")
    input_output.save_tif_sequence(main_concentration_map, input_folder + "main_concentration_map/")


def aligning_skull_pipeline(input_above_folder, input_below_folder, element="Au"):
    """computes all the skull segmentation and rat aligning with z axis calculations

    Args:
        input_above_folder (str): input above image path
        input_below_folder (str): input below image path
        element (str):            k-edge element (Au, I, Gd...)

    Returns:
        None
    """

    above_list_of_files = input_output.create_list_of_files(input_above_folder, 'tif')
    above_image = input_output.open_sequence(above_list_of_files)
    print("Above image opened")
    below_list_of_files = input_output.create_list_of_files(input_below_folder, 'tif')
    below_image = input_output.open_sequence(below_list_of_files)
    print("Below image opened")

    binning_factor = 2
    binned_image = resampling.bin_resize(above_image, binning_factor)

    threshold_value = segmentation.find_threshold_value(element)
    binned_mask = binned_image > threshold_value

    # -- Extracting skull
    above_skull, skull_bbox, \
        barycenter_jaw_one, barycenter_jaw_two, \
        y_max_jaw_one, y_max_jaw_two = segmentation.extract_skull_and_jaws(binned_mask)

    # 1) First rotation based on the position of skull/jaws
    print("... Beginning the first rotation ...")
    straightened_image, above_skull, triangle_angle = registration.straight_triangle_rotation(np.copy(binned_image),
                                                                                              above_skull,
                                                                                              skull_bbox,
                                                                                              barycenter_jaw_one,
                                                                                              barycenter_jaw_two)

    bbox = segmentation.skull_bounding_box_retriever(above_skull)

    # 2) Second rotation based on the position of the throat
    print("... Beginning the second rotation ...")
    post_mortem = True
    input_output.save_tif_sequence(straightened_image, input_output.remove_last_folder_in_path(input_above_folder) + "for_post_mortem\\")
    # segmentation of the throat
    if post_mortem:
        vector_director = np.array([-1., 4., 59.])
        throat_coordinates = np.array([313, 321, 0]).astype(np.uint32)
        # vector_director = np.array([2., 10., 59.]²0]).astype(np.uint32)
        straightened_image, rotation_matrix, throat_coordinates, offset = \
            registration.straight_throat_rotation(straightened_image, direction_vector=vector_director,
                                                  throat_coordinates=throat_coordinates, manual=True)
    else:
        throat_mask = segmentation.throat_segmentation(straightened_image, bbox, element)
        straightened_image, rotation_matrix, throat_coordinates, offset = \
            registration.straight_throat_rotation(straightened_image, throat_mask_img=throat_mask)

    # We re-segment the skull/jaws
    binned_mask = straightened_image > threshold_value

    above_skull, skull_bbox, \
        barycenter_jaw_one, barycenter_jaw_two, \
        y_max_jaw_one, y_max_jaw_two = segmentation.extract_skull_and_jaws(binned_mask)

    # 3) Third rotation based on the symmetry of the skull
    print("... Beginning the third rotation ...")
    final_image, final_skull, symmetry_angle = registration.symmetry_based_registration(straightened_image,
                                                                                        above_skull,
                                                                                        skull_bbox,
                                                                                        throat_coordinates,
                                                                                        20)

    # -- Apply the rotations to the original images
    throat_coordinates = np.array(throat_coordinates)
    throat_coordinates *= 2
    final_above_image = registration.apply_rotation_pipeline(above_image, triangle_angle, rotation_matrix,
                                                             throat_coordinates,
                                                             offset * binning_factor, symmetry_angle)

    final_above_image = final_above_image.astype(np.float32)
    thresholded_final_above_image = final_above_image > threshold_value
    final_above_skull, final_above_skull_bbox = segmentation.extract_skull(thresholded_final_above_image)
    print(" -> Applying transformations on original images")

    final_below_image = registration.apply_rotation_pipeline(below_image, triangle_angle, rotation_matrix,
                                                             throat_coordinates,
                                                             offset * binning_factor, symmetry_angle)

    final_below_image = final_below_image.astype(np.float32)
    thresholded_final_below_image = final_below_image > threshold_value
    final_below_skull, final_below_skull_bbox = segmentation.extract_skull(thresholded_final_below_image)

    # We take a bounding box containing both above and below bounding boxes
    final_bounding_box = np.array([min(final_above_skull_bbox[0], final_below_skull_bbox[0]),
                                   max(final_above_skull_bbox[1], final_below_skull_bbox[1]),
                                   min(final_above_skull_bbox[2], final_below_skull_bbox[2]),
                                   max(final_above_skull_bbox[3], final_below_skull_bbox[3]),
                                   min(final_above_skull_bbox[4], final_below_skull_bbox[4]),
                                   max(final_above_skull_bbox[5], final_below_skull_bbox[5])])

    # The bounding box needs to be centered on the throat coordinates
    if throat_coordinates[0] - final_bounding_box[0] > final_bounding_box[1] - throat_coordinates[0]:
        final_bounding_box[1] = int(final_bounding_box[0] + (throat_coordinates[0] - final_bounding_box[0]) * 2)
    else:
        final_bounding_box[0] = int(final_bounding_box[1] - (final_bounding_box[1] - throat_coordinates[0]) * 2)

    output_folder = input_output.remove_last_folder_in_path(input_above_folder)

    input_output.save_tif_sequence_and_crop(final_above_image, final_bounding_box,
                                            output_folder + "Aligned_Above_Acquisition\\")
    input_output.save_tif_sequence_and_crop(final_above_skull, final_bounding_box,
                                            output_folder + "Aligned_Above_Skull\\")

    input_output.save_tif_sequence_and_crop(final_below_image, final_bounding_box,
                                            output_folder + "Aligned_Below_Acquisition\\")
    input_output.save_tif_sequence_and_crop(final_below_skull, final_bounding_box,
                                            output_folder + "Aligned_Below_Skull\\")
    print("-------------------------------------")


def different_energies_registration_pipeline(input_folder, kedge_material="Au", secondary_material="I",
                                             translation_bool=False, rotation_bool=True):
    """registers above and below images and computes concentration maps

    Args:
        input_folder (str):       input folder path
        kedge_material (str):     element of interest (Au, I, Gd...)
        secondary_material (str): secondary material (Au, I, Gd...)
        translation_bool (bool):  True: computes 3D translation
        rotation_bool (bool):     True: computes 3D euler rotation

    Returns:
        None
    """
    print("Starting registration...")
    above_img_list_of_files = input_output.create_list_of_files(input_folder + "Above_img_for_registration\\", 'tif')
    above_image = input_output.open_sequence(above_img_list_of_files)
    above_skull_list_of_files = input_output.create_list_of_files(input_folder + "Above_skull_for_registration\\",
                                                                  'tif')
    above_skull = input_output.open_sequence(above_skull_list_of_files)

    below_img_list_of_files = input_output.create_list_of_files(input_folder + "Below_img_for_registration\\", 'tif')
    below_image = input_output.open_sequence(below_img_list_of_files)
    below_skull_list_of_files = input_output.create_list_of_files(input_folder + "Below_skull_for_registration\\",
                                                                  'tif')
    below_skull = input_output.open_sequence(below_skull_list_of_files)

    translation_transform, rotation_transform \
        = registration.registration_computation_with_mask(above_image,
                                                          below_image,
                                                          above_skull,
                                                          below_skull,
                                                          is_translation_needed=translation_bool,
                                                          is_rotation_needed=rotation_bool,
                                                          verbose=True)

    above_image_unnecessary_voxels = mathematical_morphology.dilate((above_image == 0).astype(np.int16), 3)
    below_image_unnecessary_voxels = mathematical_morphology.dilate((below_image == 0).astype(np.int16), 3)

    # 1) Registering the above image
    if translation_bool:
        above_image = registration.apply_itk_transformation(above_image, translation_transform, "linear",
                                                            ref_img=below_image)
    if rotation_bool:
        above_image = registration.apply_itk_transformation(above_image, rotation_transform, "linear")

    # 2) Registering the black voxels in the above image
    if translation_bool:
        above_image_unnecessary_voxels = registration.apply_itk_transformation(above_image_unnecessary_voxels,
                                                                               translation_transform, "linear",
                                                                               ref_img=below_image)
    if rotation_bool:
        above_image_unnecessary_voxels = registration.apply_itk_transformation(above_image_unnecessary_voxels,
                                                                               rotation_transform, "linear")

    # 3) Registering the skull mask of the above image
    if translation_bool:
        above_skull = registration.apply_itk_transformation(above_skull, translation_transform, "nearest",
                                                            ref_img=below_skull)
    if rotation_bool:
        above_skull = registration.apply_itk_transformation(above_skull, rotation_transform, "nearest")

    main_concentration_map, second_concentration_map, water_concentration_map\
        = three_materials_decomposition(above_image, below_image, kedge_material=kedge_material,
                                        secondary_material=secondary_material)

    above_image_unnecessary_voxels = (above_image_unnecessary_voxels > 0)
    both_images_unnecessary_voxels = above_image_unnecessary_voxels | below_image_unnecessary_voxels
    main_concentration_map = (both_images_unnecessary_voxels == 0) * main_concentration_map
    second_concentration_map = (both_images_unnecessary_voxels == 0) * second_concentration_map
    water_concentration_map = (both_images_unnecessary_voxels == 0) * water_concentration_map

    print("... Registration ended. Saving results in \"concentration map\" folder")
    input_output.save_tif_sequence(above_image, input_folder + "Above_registered_image\\")
    input_output.save_tif_sequence(above_skull, input_folder + "Above_registered_skull\\")
    input_output.save_tif_sequence(main_concentration_map, input_folder + "main_element_concentration_map\\")
    input_output.save_tif_sequence(second_concentration_map, input_folder + "second_element_concentration_map\\")
    input_output.save_tif_sequence(water_concentration_map, input_folder + "water_concentration_map\\")


def quantify_nanoparticles_in_brain_pipeline(input_folder, kedge_material):
    """segments and quantifies nanoparticles using input concentration map

    Args:
        input_folder (str):   input concentration map path
        kedge_material (str): material of interest (Au, I, Gd...)

    Returns:
        None
    """

    concentration_map_list_of_files = input_output.create_list_of_files(input_folder
                                                                        + "main_element_concentration_map\\", 'tif')
    concentration_map = input_output.open_sequence(concentration_map_list_of_files)

    skull_list_of_files = input_output.create_list_of_files(input_folder + "Above_registered_skull\\", 'tif')
    skull = input_output.open_sequence(skull_list_of_files)

    segmented_cells = segmentation.brain_nanoparticles_segmentation(concentration_map, skull, 0.0)
    # animationCreation.convert_image_to_mesh(segmented_cells) TODO
    # animationCreation.saveGold(input_folder, concentration_map, segmented_cells, 0, main_element) TODO
