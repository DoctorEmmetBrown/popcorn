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

# -- numpy --
import numpy as np
import math

from popcorn import input_output
from popcorn import animation_creation


def find_threshold_value(energy_element):
    """We return the energy corresponding attenuation value used for bone segmentation

    Args:
        energy_element (str): what k-edge element are we trying to quantify (Au, I, Gd..)

    Returns:
        (float) threshold value
    """
    if energy_element == "Au":
        return 0.26
    else:
        return 0.69

    return 0


def extract_skull(thresholded_ipsdk_image):
    """extracts the skull from the volume

    Args:
        thresholded_ipsdk_image (): input image

    Returns:
        (numpy.ndarray, list[int]) skull mask and skull bounding box
    """
    """
    
    :param thresholded_ipsdk_image: 
    :return: 
    """
    # We start with a 3d opening image computation
    morpho_mask = PyIPSDK.sphericalSEXYZInfo(0)  # 3D sphere (r=1) structuring element
    opened_image = Morpho.opening3dImg(thresholded_ipsdk_image, morpho_mask)

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
    out_measure_set = ShapeAnalysis.shapeAnalysis3d(thresholded_ipsdk_image, in_shape_3d_coll, in_measure_info_set_3d)

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


def extract_skull_and_jaws(thresholded_ipsdk_image):
    """
    extracting the skull from the volume
    :param thresholded_ipsdk_image: input volume
    :return: skull mask, skull bounding box (X and Y), jaw one and jaw two barycenter, jaw one and jaw two max Y value
    """
    # We start with a 3d opening image computation

    # We extract the biggest shape from the volume (the skull)
    extracted_skull = AdvMorpho.keepBigShape3dImg(thresholded_ipsdk_image, 1)

    remaining_objects = Arithm.subtractImgImg(thresholded_ipsdk_image, extracted_skull)
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
    out_measure_set = ShapeAnalysis.shapeAnalysis3d(thresholded_ipsdk_image, extracted_skull_in_shape_3d_coll,
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
    out_measure_set = ShapeAnalysis.shapeAnalysis3d(thresholded_ipsdk_image, jaw_one_in_shape_3d_coll, in_measure_info_set_3d)

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
    out_measure_set = ShapeAnalysis.shapeAnalysis3d(thresholded_ipsdk_image, jaw_two_in_shape_3d_coll, in_measure_info_set_3d)

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

    return np.copy(extracted_skull.array), skull_bounding_box, jaw_one_barycenter, jaw_two_barycenter, jaw_one_max_x_y, jaw_two_max_x_y


def skull_bounding_box_retriever(skull_image):
    """
    Calculating the skull's bounding box (3D)
    :param skull_image: input skull image
    :return: skull's bounding box ([minX, maxX, minY, maxY, minZ, maxZ])
    """
    skull_image = Bin.lightThresholdImg(skull_image, 1)

    # We'll now analyze its shape by labeling it
    in_label_img_3d = AdvMorpho.connectedComponent3dImg(skull_image)

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
    out_measure_set = ShapeAnalysis.shapeAnalysis3d(skull_image, in_shape_3d_coll, in_measure_info_set_3d)

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
    """
    segmentation of the rat's throat
    :param image: input image
    :param bbox: bounding box of the skull
    :param element: what energy element the image were acquired around
    :return: throat mask
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


def brain_nanoparticles_segmentation(concentration_map, skull, threshold=0.0):

    middle_slice_nb = int(skull.shape[0]/2)
    index = 0
    min_nb_of_zeros = skull.shape[1] * skull.shape[2]
    nb_of_zeros = skull.shape[1] * skull.shape[2]

    while nb_of_zeros > 0 and index < int(skull.shape[0]/2):
        nb_of_zeros = (concentration_map[middle_slice_nb + index] == 0).sum()
        if nb_of_zeros < min_nb_of_zeros:
            best_index = middle_slice_nb + index
            min_nb_of_zeros = nb_of_zeros

        nb_of_zeros = (concentration_map[middle_slice_nb - index] == 0).sum()
        if nb_of_zeros < min_nb_of_zeros:
            best_index = middle_slice_nb - index
            min_nb_of_zeros = nb_of_zeros

        index +=1
    skull_convex_hull = morphology.convex_hull_image(skull[best_index, :, :])
    skull_convex_hull_3d = np.repeat(skull_convex_hull[np.newaxis, :, :], skull.shape[0], axis=0)
    skull_convex_hull_3d_ipsdk = PyIPSDK.fromArray(skull_convex_hull_3d)
    skull_convex_hull_3d_ipsdk = Bin.lightThresholdImg(skull_convex_hull_3d_ipsdk, 0.5)
    concentration_map_ipsdk = PyIPSDK.fromArray(concentration_map)

    skull_ipsdk = PyIPSDK.fromArray(skull)
    inverted_skull_ipsdk = Bin.darkThresholdImg(skull_ipsdk, 0.5)

    eroding_sphere = PyIPSDK.sphericalSEXYZInfo(7)
    inverted_skull_ipsdk = Morpho.erode3dImg(inverted_skull_ipsdk, eroding_sphere)

    opening_sphere = PyIPSDK.sphericalSEXYZInfo(80)
    inverted_skull_ipsdk = Morpho.opening3dImg(inverted_skull_ipsdk, opening_sphere)


    brain_mask_ipsdk = AdvMorpho.keepBigShape3dImg(Logic.bitwiseAndImgImg(skull_convex_hull_3d_ipsdk, inverted_skull_ipsdk), 1)

    brain_measure_results = GblMsr.statsMaskMsr3d(concentration_map_ipsdk, brain_mask_ipsdk)
    brain_concentration_mask_ipsdk = Logic.maskImg(concentration_map_ipsdk, brain_mask_ipsdk)


    if threshold == 0.0:
        print("Threshold :", 3 * brain_measure_results.stdDev, "mg/mL")
        segmented_cells_ipsdk = Bin.lightThresholdImg(brain_concentration_mask_ipsdk, 3 * brain_measure_results.stdDev)
    else:
        print("Threshold :", threshold, "mg/mL")
        segmented_cells_ipsdk = Bin.lightThresholdImg(brain_concentration_mask_ipsdk, threshold)

    segmented_cells_analysis(concentration_map, segmented_cells_ipsdk.array, 21.4 * 2)

    return np.copy(segmented_cells_ipsdk.array)

def segmented_cells_analysis(material_concentration_map, cells_mask, resolution = 21.4):
    segmented_cells = material_concentration_map[cells_mask > 0]

    mean_concentration = np.mean(segmented_cells)
    nb_of_pixels = segmented_cells.size
    print("***************** Results of segmentation *****************")
    print("Segmented volume   :", nb_of_pixels * (resolution * resolution * resolution * 0.000000000001), "mL")
    print("                  ->", nb_of_pixels * (resolution * resolution * resolution * 0.000000001), "µL")
    print("Mean concentration :", np.mean(segmented_cells), "mg/mL")
    totalSum = 0
    for pixel in segmented_cells:
        totalSum += (pixel - mean_concentration) * (pixel - mean_concentration)
    print("Standard deviation :", math.sqrt(totalSum / nb_of_pixels), "mg/mL")
    print("Mass of gold : ", 1000 * sum(segmented_cells) * (resolution * resolution * resolution * 0.000000000001))