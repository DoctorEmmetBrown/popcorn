# -- IPSDK Library --
import PyIPSDK
import PyIPSDK.IPSDKIPLMorphology as Morpho
import PyIPSDK.IPSDKIPLAdvancedMorphology as AdvMorpho
import PyIPSDK.IPSDKIPLShapeSegmentation as ShapeSegmentation
import PyIPSDK.IPSDKIPLShapeAnalysis as ShapeAnalysis
import PyIPSDK.IPSDKIPLArithmetic as Arithm
import PyIPSDK.IPSDKIPLBinarization as Bin
import PyIPSDK.IPSDKIPLLogical as Logic
import PyIPSDK.IPSDKIPLGlobalMeasure as GblMsr

from skimage import morphology

import numpy as np
import math


def find_threshold_value(energy_element):
    """We return the energy corresponding attenuation value used for bone segmentation

    Args:
        energy_element (str): what k-edge element are we trying to quantify (Au, I, Gd..)

    Returns:
        (float) threshold value
    """
    if energy_element == "Au":
        return 0.26
    elif energy_element == "I":
        return 0.69

    return 0


def extract_skull(thresholded_image):
    """extracts the skull from the volume

    Args:
        thresholded_image (np.ndarray): input thresholded image (binary)

    Returns:
        (numpy.ndarray, list[int]) skull mask and skull bounding box
    """
    # Conversion to binary IPSDK image
    thresholded_image_ipsdk = Bin.lightThresholdImg(PyIPSDK.fromArray(thresholded_image), 1)

    # We start with a 3d opening image computation
    morpho_mask = PyIPSDK.sphericalSEXYZInfo(0)  # 3D sphere (r=1) structuring element
    opened_image = Morpho.opening3dImg(thresholded_image_ipsdk, morpho_mask)

    # We extract the biggest shape from the volume (the skull)
    extracted_skull = AdvMorpho.keepBigShape3dImg(opened_image, 1)

    # We'll now analyze its shape by labeling it
    in_label_img_3d = AdvMorpho.connectedComponent3dImg(extracted_skull)

    # and then making it a shape object
    in_shape_3d_coll = ShapeSegmentation.labelShapeExtraction3d(in_label_img_3d)

    # these measures will help us cropping the full volume, getting rid of non-interesting parts of the imaged rat/mice
    # definition of proceeded measure
    in_measure_info_set_3d = PyIPSDK.createMeasureInfoSet3d()
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMinXMsrInfo", "BoundingBoxMinXMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMaxXMsrInfo", "BoundingBoxMaxXMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMinYMsrInfo", "BoundingBoxMinYMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMaxYMsrInfo", "BoundingBoxMaxYMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMinZMsrInfo", "BoundingBoxMinZMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMaxZMsrInfo", "BoundingBoxMaxZMsr")

    # shape analysis computation
    out_measure_set = ShapeAnalysis.shapeAnalysis3d(thresholded_image_ipsdk, in_shape_3d_coll, in_measure_info_set_3d)

    out_bounding_box_min_x_msr_info = out_measure_set.getMeasure("BoundingBoxMinXMsrInfo")
    out_bounding_box_max_x_msr_info = out_measure_set.getMeasure("BoundingBoxMaxXMsrInfo")
    out_bounding_box_min_y_msr_info = out_measure_set.getMeasure("BoundingBoxMinYMsrInfo")
    out_bounding_box_max_y_msr_info = out_measure_set.getMeasure("BoundingBoxMaxYMsrInfo")
    out_bounding_box_min_z_msr_info = out_measure_set.getMeasure("BoundingBoxMinZMsrInfo")
    out_bounding_box_max_z_msr_info = out_measure_set.getMeasure("BoundingBoxMaxZMsrInfo")

    # retrieving the extracted skull bounding box
    bounding_box = [int(out_bounding_box_min_x_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                    int(out_bounding_box_max_x_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                    int(out_bounding_box_min_y_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                    int(out_bounding_box_max_y_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                    int(out_bounding_box_min_z_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                    int(out_bounding_box_max_z_msr_info.getMeasureResult().getColl(0)[1] + 0.5)]

    # we return the mask of the skull and its 3D bounding box
    return np.copy(extracted_skull.array), bounding_box


def extract_skull_and_jaws(thresholded_image):
    """extracts skull and left/right parts of the jaw from a thresholded image

    Args:
        thresholded_image (np.ndarray): thresholded image (binary)

    Returns:
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float) skull mask, skull bounding box (X and Y), jaw one
         barycenter, jaw two barycenter, jaw one max Y value, jaw two max Y value
    """

    # Conversion to binary IPSDK image
    thresholded_image_ipsdk = Bin.lightThresholdImg(PyIPSDK.fromArray(thresholded_image), 1)

    # We extract the biggest shape from the volume (the skull)
    extracted_skull = AdvMorpho.keepBigShape3dImg(thresholded_image_ipsdk, 1)

    remaining_objects = Arithm.subtractImgImg(thresholded_image_ipsdk, extracted_skull)
    remaining_objects = Bin.lightThresholdImg(remaining_objects, 1)
    jaw_one = AdvMorpho.keepBigShape3dImg(remaining_objects, 1)

    remaining_objects = Arithm.subtractImgImg(remaining_objects, jaw_one)
    remaining_objects = Bin.lightThresholdImg(remaining_objects, 1)
    jaw_two = AdvMorpho.keepBigShape3dImg(remaining_objects, 1)

    in_measure_info_set_3d = PyIPSDK.createMeasureInfoSet3d()
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMinXMsrInfo", "BoundingBoxMinXMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMaxXMsrInfo", "BoundingBoxMaxXMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMinYMsrInfo", "BoundingBoxMinYMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMaxYMsrInfo", "BoundingBoxMaxYMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BarycenterXMsrInfo", "BarycenterXMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BarycenterYMsrInfo", "BarycenterYMsr")

    # Skull
    # We'll now analyze its shape by labeling it
    extracted_skull_in_label_img_3d = AdvMorpho.connectedComponent3dImg(extracted_skull)

    # and then making it a shape object
    extracted_skull_in_shape_3d_coll = ShapeSegmentation.labelShapeExtraction3d(extracted_skull_in_label_img_3d)

    # shape analysis computation
    out_measure_set = ShapeAnalysis.shapeAnalysis3d(thresholded_image_ipsdk, extracted_skull_in_shape_3d_coll,
                                                    in_measure_info_set_3d)

    out_bounding_box_min_x_msr_info = out_measure_set.getMeasure("BoundingBoxMinXMsrInfo")
    out_bounding_box_max_x_msr_info = out_measure_set.getMeasure("BoundingBoxMaxXMsrInfo")
    out_bounding_box_min_y_msr_info = out_measure_set.getMeasure("BoundingBoxMinYMsrInfo")
    out_bounding_box_max_y_msr_info = out_measure_set.getMeasure("BoundingBoxMaxYMsrInfo")

    # retrieving the extracted skull bounding box
    skull_bounding_box = [int(out_bounding_box_min_x_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                          int(out_bounding_box_max_x_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                          int(out_bounding_box_min_y_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                          int(out_bounding_box_max_y_msr_info.getMeasureResult().getColl(0)[1] + 0.5)]

    # Jaw one
    # We'll now analyze its shape by labeling it
    jaw_one_in_label_img_3d = AdvMorpho.connectedComponent3dImg(jaw_one)

    # and then making it a shape object
    jaw_one_in_shape_3d_coll = ShapeSegmentation.labelShapeExtraction3d(jaw_one_in_label_img_3d)

    # shape analysis computation
    out_measure_set = ShapeAnalysis.shapeAnalysis3d(thresholded_image_ipsdk, jaw_one_in_shape_3d_coll,
                                                    in_measure_info_set_3d)

    out_barycenter_x_msr_info = out_measure_set.getMeasure("BarycenterXMsrInfo")
    out_barycenter_y_msr_info = out_measure_set.getMeasure("BarycenterYMsrInfo")
    out_bounding_box_min_x_msr_info = out_measure_set.getMeasure("BoundingBoxMinXMsrInfo")
    out_bounding_box_max_x_msr_info = out_measure_set.getMeasure("BoundingBoxMaxXMsrInfo")
    out_bounding_box_min_y_msr_info = out_measure_set.getMeasure("BoundingBoxMinYMsrInfo")

    # retrieving the extracted skull bounding box
    jaw_one_barycenter = [int(out_barycenter_x_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                          int(out_barycenter_y_msr_info.getMeasureResult().getColl(0)[1] + 0.5)]

    jaw_one_max_x_y = [int(out_bounding_box_min_x_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                       int(out_bounding_box_max_x_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                       int(out_bounding_box_min_y_msr_info.getMeasureResult().getColl(0)[1] + 0.5)]

    # Jaw Two
    # We'll now analyze its shape by labeling it
    jaw_two_in_label_img_3d = AdvMorpho.connectedComponent3dImg(jaw_two)

    # and then making it a shape object
    jaw_two_in_shape_3d_coll = ShapeSegmentation.labelShapeExtraction3d(jaw_two_in_label_img_3d)

    # shape analysis computation
    out_measure_set = ShapeAnalysis.shapeAnalysis3d(thresholded_image_ipsdk, jaw_two_in_shape_3d_coll,
                                                    in_measure_info_set_3d)

    out_barycenter_x_msr_info = out_measure_set.getMeasure("BarycenterXMsrInfo")
    out_barycenter_y_msr_info = out_measure_set.getMeasure("BarycenterYMsrInfo")
    out_bounding_box_min_x_msr_info = out_measure_set.getMeasure("BoundingBoxMinXMsrInfo")
    out_bounding_box_max_x_msr_info = out_measure_set.getMeasure("BoundingBoxMaxXMsrInfo")
    out_bounding_box_min_y_msr_info = out_measure_set.getMeasure("BoundingBoxMinYMsrInfo")

    # retrieving the extracted skull bounding box
    jaw_two_barycenter = [int(out_barycenter_x_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                          int(out_barycenter_y_msr_info.getMeasureResult().getColl(0)[1] + 0.5)]

    jaw_two_max_x_y = [int(out_bounding_box_min_x_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                       int(out_bounding_box_max_x_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                       int(out_bounding_box_min_y_msr_info.getMeasureResult().getColl(0)[1] + 0.5)]

    return np.copy(extracted_skull.array), skull_bounding_box, jaw_one_barycenter, jaw_two_barycenter, \
        jaw_one_max_x_y, jaw_two_max_x_y


def skull_bounding_box_retriever(skull_image):
    """retrieves skull bounding box from input mask

    Args:
        skull_image (np.ndarray): input skull mask

    Returns:
        (np.ndarray) 3D skull bounding box
    """

    skull_image_ipsdk = Bin.lightThresholdImg(PyIPSDK.fromArray(skull_image), 1)

    # We'll now analyze its shape by labeling it
    in_label_img_3d = AdvMorpho.connectedComponent3dImg(skull_image_ipsdk)

    # and then making it a shape object
    in_shape_3d_coll = ShapeSegmentation.labelShapeExtraction3d(in_label_img_3d)

    # these measures will help us cropping the full volume, getting rid of non-interesting parts of the imaged rat/mice
    # definition of proceeded measure
    in_measure_info_set_3d = PyIPSDK.createMeasureInfoSet3d()
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMinXMsrInfo", "BoundingBoxMinXMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMaxXMsrInfo", "BoundingBoxMaxXMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMinYMsrInfo", "BoundingBoxMinYMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMaxYMsrInfo", "BoundingBoxMaxYMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMinZMsrInfo", "BoundingBoxMinZMsr")
    PyIPSDK.createMeasureInfo(in_measure_info_set_3d, "BoundingBoxMaxZMsrInfo", "BoundingBoxMaxZMsr")

    # shape analysis computation
    out_measure_set = ShapeAnalysis.shapeAnalysis3d(skull_image_ipsdk, in_shape_3d_coll, in_measure_info_set_3d)

    out_bounding_box_min_x_msr_info = out_measure_set.getMeasure("BoundingBoxMinXMsrInfo")
    out_bounding_box_max_x_msr_info = out_measure_set.getMeasure("BoundingBoxMaxXMsrInfo")
    out_bounding_box_min_y_msr_info = out_measure_set.getMeasure("BoundingBoxMinYMsrInfo")
    out_bounding_box_max_y_msr_info = out_measure_set.getMeasure("BoundingBoxMaxYMsrInfo")
    out_bounding_box_min_z_msr_info = out_measure_set.getMeasure("BoundingBoxMinZMsrInfo")
    out_bounding_box_max_z_msr_info = out_measure_set.getMeasure("BoundingBoxMaxZMsrInfo")

    # retrieving the extracted skull bounding box
    bounding_box = [int(out_bounding_box_min_x_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                    int(out_bounding_box_max_x_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                    int(out_bounding_box_min_y_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                    int(out_bounding_box_max_y_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                    int(out_bounding_box_min_z_msr_info.getMeasureResult().getColl(0)[1] + 0.5),
                    int(out_bounding_box_max_z_msr_info.getMeasureResult().getColl(0)[1] + 0.5)]

    return bounding_box


def throat_segmentation(image, bbox, element):
    """segmentation of the rat's throat

    Args:
        image (np.ndarray): input image
        bbox (np.ndarray):  skull bounding box
        element (str):      which element's k-edge the images were acquired around

    Returns:
        (np.ndarray) throat mask
    """

    for_throat_image = np.copy(image)
    for_throat_image[:, 0:bbox[2] + int((bbox[3] - bbox[2]) / 2), :] = 1
    for_throat_image[:, bbox[3]:for_throat_image.shape[1], :] = 1
    for_throat_image[:, :, 0:bbox[0]] = 1
    for_throat_image[:, :, bbox[1]:for_throat_image.shape[2]] = 1

    for_throat_image_ipsdk = PyIPSDK.fromArray(for_throat_image)

    if element == "Au":
        throat = Bin.darkThresholdImg(for_throat_image_ipsdk, 0.16)
        structure_element = PyIPSDK.sphericalSEXYZInfo(3)  # 3D sphere (r=1) structuring element
    else:
        throat = Bin.darkThresholdImg(for_throat_image_ipsdk, 0.25)
        structure_element = PyIPSDK.sphericalSEXYZInfo(2)  # 3D sphere (r=1) structuring element
    throat = Morpho.opening3dImg(throat, structure_element)
    structure_element = PyIPSDK.sphericalSEXYZInfo(6)  # 3D sphere (r=1) structuring element
    throat = Morpho.closing3dImg(throat, structure_element)

    return np.copy(throat.array)


def brain_nanoparticles_segmentation(concentration_map, skull, pixel_size, threshold=None):
    """segments brain injected nanoparticles from concentration map

    Args:
        concentration_map (np.ndarray): input concentration map
        skull (np.ndarray):             input skull mask
        pixel_size (float):             pixel size (in µm)
        threshold (float):              forced threshold value

    Returns:
        (np.ndarray) segmentation result
    """

    middle_slice_nb = int(skull.shape[0]/2)
    index = 0
    min_nb_of_zeros = skull.shape[1] * skull.shape[2]
    nb_of_zeros = skull.shape[1] * skull.shape[2]
    best_index = middle_slice_nb
    # Since we can't compute skull's 3D convex hull, we look for the best reference slice to compute 2D convex hull on
    while nb_of_zeros > 0 and index < int(skull.shape[0]/2):
        nb_of_zeros = (concentration_map[middle_slice_nb + index] == 0).sum()
        if nb_of_zeros < min_nb_of_zeros:
            best_index = middle_slice_nb + index
            min_nb_of_zeros = nb_of_zeros

        nb_of_zeros = (concentration_map[middle_slice_nb - index] == 0).sum()
        if nb_of_zeros < min_nb_of_zeros:
            best_index = middle_slice_nb - index
            min_nb_of_zeros = nb_of_zeros

        index += 1
    # We compute the 2D convex hull acting as a mask on the full 3D image
    skull_convex_hull = morphology.convex_hull_image(skull[best_index, :, :])
    skull_convex_hull_3d = np.repeat(skull_convex_hull[np.newaxis, :, :], skull.shape[0], axis=0)
    skull_convex_hull_3d_ipsdk = PyIPSDK.fromArray(skull_convex_hull_3d)
    skull_convex_hull_3d_ipsdk = Bin.lightThresholdImg(skull_convex_hull_3d_ipsdk, 0.5)

    concentration_map_ipsdk = PyIPSDK.fromArray(concentration_map)
    skull_ipsdk = PyIPSDK.fromArray(skull)

    # We erode/open the skull's inverted mask (the inside and outside of the skull will be True)
    inverted_skull_ipsdk = Bin.darkThresholdImg(skull_ipsdk, 0.5)

    eroding_sphere = PyIPSDK.sphericalSEXYZInfo(7)
    inverted_skull_ipsdk = Morpho.erode3dImg(inverted_skull_ipsdk, eroding_sphere)

    opening_sphere = PyIPSDK.sphericalSEXYZInfo(80)
    inverted_skull_ipsdk = Morpho.opening3dImg(inverted_skull_ipsdk, opening_sphere)

    # AND operation on convex hull and previously computed inverted masks gives us: the brain
    brain_mask_ipsdk = AdvMorpho.keepBigShape3dImg(Logic.bitwiseAndImgImg(skull_convex_hull_3d_ipsdk,
                                                                          inverted_skull_ipsdk), 1)
    # Standard deviation computation
    brain_measure_results = GblMsr.statsMaskMsr3d(concentration_map_ipsdk, brain_mask_ipsdk)
    brain_concentration_mask_ipsdk = Logic.maskImg(concentration_map_ipsdk, brain_mask_ipsdk)

    # if threshold is not specified, we use 3*standard_deviation as a threshold
    if threshold is None:
        print("Threshold :", 3 * brain_measure_results.stdDev, "mg/mL")
        segmented_cells_ipsdk = Bin.lightThresholdImg(brain_concentration_mask_ipsdk, 3 * brain_measure_results.stdDev)
    else:
        print("Threshold :", threshold, "mg/mL")
        segmented_cells_ipsdk = Bin.lightThresholdImg(brain_concentration_mask_ipsdk, threshold)

    segmented_cells_analysis(concentration_map, segmented_cells_ipsdk.array, pixel_size)

    return np.copy(segmented_cells_ipsdk.array)


def segmented_cells_analysis(material_concentration_map, cells_mask, pixel_size=21.4):
    """Computes mean concentration and other values on segmented nanoparticles

    Args:
        material_concentration_map (np.ndarray): concentration map
        cells_mask (np.ndarray):                 segmented cells mask
        pixel_size (float):                      pixel size (in µm)

    Returns:
        None
    """
    segmented_cells = material_concentration_map[cells_mask > 0]

    mean_concentration = np.mean(segmented_cells)
    nb_of_pixels = segmented_cells.size
    print("***************** Results of segmentation *****************")
    print("Segmented volume   :", nb_of_pixels * (pixel_size * pixel_size * pixel_size * 0.000000000001), "mL")
    print("                  ->", nb_of_pixels * (pixel_size * pixel_size * pixel_size * 0.000000001), "µL")
    print("Mean concentration :", np.mean(segmented_cells), "mg/mL")
    total_sum = 0
    for pixel in segmented_cells:
        total_sum += (pixel - mean_concentration) * (pixel - mean_concentration)
    print("Standard deviation :", math.sqrt(total_sum / nb_of_pixels), "mg/mL")
    print("Mass of gold : ", 1000 * sum(segmented_cells) * (pixel_size * pixel_size * pixel_size * 0.000000000001))
