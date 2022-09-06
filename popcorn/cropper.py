import math

from popcorn.input_output import open_sequence, open_image, create_list_of_files, save_tif_image, save_tif_sequence
from popcorn.spectral_imaging.registration import compute_2d_rotation

import numpy as np

from scipy import ndimage as ndi

from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu


def retrieve_info_from_sample_name(sample_name):
    """Looks for useful informations in sample name

    Args:
        sample_name (str): name of the sample

    Returns:
        (dict): useful informations
    """
    if "above" in sample_name.lower():
        sample_info = sample_name.lower().split("above")[0]
        acquisition_info = sample_name.lower().split("above")[1]
        above_or_below = "above"
    elif "below" in sample_name.lower():
        sample_info = sample_name.lower().split("below")[0]
        acquisition_info = sample_name.lower().split("below")[1]
        above_or_below = "below"

    list_of_materials = []
    if "au" in sample_info:
        list_of_materials.append("Au")
    if "i" in sample_info:
        list_of_materials.append("I")

    if sample_info[0:2] == "ha":
        half_acquisition = True
    else:
        half_acquisition = False

    sample_number = int(sample_info.split("_")[1])

    material_kedge = ""
    if "au" in acquisition_info:
        material_kedge = "au"
    elif "i" in acquisition_info:
        material_kedge = "i"
    elif "gd" in acquisition_info:
        material_kedge = "gd"

    floor_nb = int(acquisition_info.split("00")[-1].split("_")[0])

    if "pag" in acquisition_info:
        paganin = True
    else:
        paganin = False

    infos = {"above_or_below": above_or_below,
             "half_acquisition":half_acquisition,
             "paganin": paganin,
             "material_kedge": material_kedge,
             "floor_nb": floor_nb,
             "sample_number": sample_number,
             "list_of_materials": list_of_materials}

    return infos


def affine_linear_regression(x_values, y_values):
    """computes affine function linear regression (ax + b = y) with a given list of x,y points

    Args:
        x_values (list[float]): list of x values
        y_values (list[float]): list of y values

    Returns:
        (float, float): return a,b in solution to y = ax + b such that root mean square distance between trend line and
        original points is minimized
    """
    n = len(x_values)
    sx = sy = sxx = syy = sxy = 0.0
    for x, y in zip(x_values, y_values):
        sx = sx + x
        sy = sy + y
        sxx = sxx + x * x
        syy = syy + y * y
        sxy = sxy + x * y
    det = sxx * n - sx * sx
    return (sxy * n - sy * sx) / det, (sxx * sy - sx * sxy) / det


def compute_distances_between_2d_points(list_of_points):
    """computes distances between all the 2D points of a list

    Args:
        list_of_points (list[list[float]]): complete path

    Returns:
        2 entries matrix of distances between points
    """
    nb_points = len(list_of_points)
    output = np.zeros((nb_points, nb_points))

    for i in range(0, nb_points):
        ref_point = list_of_points[i]
        for j in range(i + 1, nb_points):
            point = list_of_points[j]
            distance = ((ref_point[0] - point[0])**2 + (ref_point[1] - point[1])**2)**0.5
            output[i, j] = distance
            output[j, i] = distance
    return output


def look_for_scattered_images(input_image):
    """parses all slices of a 3D image to find images of interest

    Args:
        input_image (numpy.ndarray): input 3D image

    Returns:
        None
    """
    for slice_nb in range(input_image.shape[0]):
        print(slice_nb, np.median(input_image[slice_nb, :, :]))


def retrieve_bbox_from_multiple_samples(input_image):
    """retrieves bounding box of elements of interest in an image

    Args:
        input_image (numpy.ndarray): input 2D image of interest

    Returns:
        (list[list[int]]): list of all element of interest bounding boxes [[x_min, x_max, y_min, y_max], ...]
    """
    # Using Otsu to look for elements of interest
    thresh = threshold_otsu(input_image)
    binary_image = input_image > thresh
    distance = ndi.distance_transform_edt(binary_image)
    binary_image = distance > 30  # We compute a distance map and look for elements that are big enough (>30px radius)

    label_img = label(binary_image, connectivity=binary_image.ndim)
    props = regionprops(label_img)  # We label them using connected components

    list_of_centroids = []
    for prop in props:
        list_of_centroids.append(prop.centroid)  # We retrieve their mass center

    elt_radius = int(np.max(distance) + 5)  # Approximate estimation of elements of interest's size (radius in px)

    # Using linear trend regression, we have an approximate orientation of the sample
    x_list = [x[0] for x in list_of_centroids]
    y_list = [x[1] for x in list_of_centroids]
    a, b = affine_linear_regression(x_list, y_list)
    angle_offset = 0.15
    angle = math.acos(-a) + angle_offset

    # We rotate the image based on the orientation of the samples and how we're supposed to count them,
    # here : horizontal + 0.15 rad
    rotated_image = compute_2d_rotation(input_image, -angle)

    # We again segment the element of interest (We could have rotated the binary image)
    binary_image = rotated_image > thresh
    distance = ndi.distance_transform_edt(binary_image)
    binary_image = distance > 30

    # We retrieve the new centroids, but now in correct order (helps identifying which element is which)
    label_img = label(binary_image, connectivity=binary_image.ndim)
    props = regionprops(label_img)

    list_of_centroids = []
    for prop in props:
        list_of_centroids.append(prop.centroid)

    # Now that we have centroids in the correct order, we reverse-rotate their (x,y) position
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    rotation_matrix = np.array([[c, s], [-s, c]])
    points_matrix = np.array(list_of_centroids) - input_image.shape[0]//2

    rotated_points = (np.matmul(rotation_matrix, np.transpose(points_matrix)) + input_image.shape[0]//2).tolist()
    list_of_centroids = []
    for i in range(len(rotated_points[0])):
        list_of_centroids.append([rotated_points[0][i], rotated_points[1][i]])

    # We now have the position of all the elements in correct order, we return their bbox based on guessed radius
    list_of_bbox = [[int(point[0]) - elt_radius,
                     int(point[0]) + elt_radius,
                     int(point[1]) - elt_radius,
                     int(point[1]) + elt_radius] for point in list_of_centroids]
    return list_of_bbox


def crop_image_based_on_bbox(input_image, list_of_bbox, output_folder, sample_infos = None):
    """Crops multiple subparts of an image based on the bounding boxes given as input and saves the results

    Args:
        input_image (numpy.ndarray):    input 3D image
        list_of_bbox (list[list[int]]): list of bounding boxes of elements of interest
        output_folder (str):            path to output folder
        sample_infos (dict):            sample infos (optional)

    Returns:
        (list[list[int]]): list of all element of interest bounding boxes [[x_min, x_max, y_min, y_max], ...]
    """
    if sample_infos is not None and sample_infos["sample_number"] is not None:
        sample_nb = sample_infos["sample_number"]
        above_or_below = sample_infos["above_or_below"]
        material_kedge = sample_infos["material_kedge"]
        floor_nb = sample_infos["floor_nb"]
        if sample_infos["paganin"]:
            paganin = "pag"
        else:
            paganin = ""
    else:
        sample_nb = 1
        above_or_below = ""
        material_kedge = ""
        floor_nb = "001"
        paganin = ""

    for bbox in list_of_bbox:
        cropped_image = input_image[:,bbox[0]:bbox[1],bbox[2]:bbox[3]]
        save_tif_sequence(cropped_image, output_folder + "cropped_sample_" + str(sample_nb) + "_" + above_or_below +
                          material_kedge + "_00" + str(floor_nb) + "_" + paganin + "\\")
        sample_nb += 1


if __name__ == '__main__':
    input_folder = "C:\\Users\\UA7\\Desktop\\HA200_31_38_AuIgel_D1_D8_AboveAu__002_pag\\"
    sample_infos = retrieve_info_from_sample_name("HA200_31_38_AuIgel_D1_D8_AboveAu__002_pag")

    list_of_files = create_list_of_files(input_folder, extension="edf")
    middle_slice = open_image(list_of_files[len(list_of_files) // 2])

    bbox_list = retrieve_bbox_from_multiple_samples(middle_slice)

    image = open_sequence(list_of_files)
    crop_image_based_on_bbox(image, bbox_list, "C:\\Users\\UA7\\Desktop\\result_popcorn\\", sample_infos=sample_infos)
