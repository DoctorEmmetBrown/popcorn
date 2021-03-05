import os
import glob

import numpy as np

import popcornIO
import Resampling


def three_materials_decomposition(above_kedge_image, below_kedge_image, kedge_material="Au", secondary_material="I"):
    """Temporary 3 material decomposition for (I, Ba, Gd, Au) + Water

    Args:
        above_kedge_image (numpy.ndarray): above K-edge acquisition image
        below_kedge_image (numpy.ndarray): below K-edge acquisition image
        kedge_material (str):              K-edge element (among I, Ba, Gd and Au)
        secondary_material (str):          secondary element (among I, Ba, Gd and Au)

    Returns:
        (numpy.ndarray, numpy.ndarray, numpy.ndarray): concentration maps (in g/L) of both input materials and water
    """
    indices_dict = {
        "i": 1,
        "iodine": 1,
        "ba": 2,
        "barium": 2,
        "gd": 3,
        "gadolinium": 3,
        "au": 4,
        "gold": 4
    }

    #          I-        I+       Ba-       Ba+       Gd-       Gd+       Au-       Au+
    muz = [[000.3307, 000.3188, 000.2940, 000.2790, 000.2290, 000.2240, 000.1835, 000.1827],  # Water
           [033.6370, 170.9231, 028.2000, 024.5600, 012.8300, 011.5500, 016.8852, 016.3330],  # Iodine
           [008.2170, 007.0000, 005.9230, 027.3200, 014.3500, 012.9300, 004.0000, 003.7450],  # Barium
           [012.3200, 010.4900, 008.8730, 007.7050, 004.0150, 017.7000, 005.6240, 005.2720],  # Gadolinium
           [421.9694, 389.5878, 016.5900, 014.4300, 007.5510, 006.8090, 040.7423, 165.8642]]  # Gold

    densities = [4.93, 3.5, 7.9, 19.3]

    kedge_material_index = indices_dict.get(kedge_material.lower())
    secondary_material_index = indices_dict.get(secondary_material.lower())

    # Water attenuation
    mu_water_below = muz[0][(kedge_material_index - 1) * 2]
    mu_water_above = muz[0][(kedge_material_index - 1) * 2 + 1]

    # Secondary material attenuation
    mu_secondary_material_below = muz[secondary_material_index][(kedge_material_index - 1) * 2]
    mu_secondary_material_above = muz[secondary_material_index][(kedge_material_index - 1) * 2 + 1]

    # Kedge material attenuation
    mu_kedge_material_below = muz[kedge_material_index][(kedge_material_index - 1) * 2]
    mu_kedge_material_above = muz[kedge_material_index][(kedge_material_index - 1) * 2 + 1]

    images = np.stack((above_kedge_image, below_kedge_image), axis=0)
    material_densities = np.array([densities[kedge_material_index - 1], densities[secondary_material_index - 1], 1.0])
    mus = np.array([[mu_kedge_material_above, mu_secondary_material_above, mu_water_above],
                    [mu_kedge_material_below, mu_secondary_material_below, mu_water_below]])

    main_material_concentration_map, second_material_concentration_map, water_concentration_map = \
        decomposition_equation_resolution(images, material_densities, mus)

    return main_material_concentration_map.copy(), \
        second_material_concentration_map.copy(), \
        water_concentration_map.copy()


def decomposition_equation_resolution(images, densities, material_attenuations, volume_fraction_hypothesis=True,
                                      verbose=False):
    """solves the element decomposition system

    Args:
        images (numpy.ndarray): N dim array, each N-1 dim array is an image acquired at 1 given energy (can be 2D or 3D,
        K energies in total)
        densities (numpy.ndarray): 1D array, one density per elements inside a voxel (P elements in total)
        material_attenuations (numpy.ndarray): 2D array, linear attenuation of each element at each energy (K * P array)
        volume_fraction_hypothesis (bool):
        verbose (bool):

    Returns:
        (numpy.ndarray): material decomposition maps, N-dim array composed of P * N-1-dim arrays
    """
    number_of_energies = images.shape[0]
    number_of_materials = densities.size

    if verbose:
        print("-- Material decomposition --")
        print(">Number of energies: ", number_of_energies)
        print(">Number of materials: ", number_of_materials)
        print(">Sum of materials volume fraction equal to 1 hypothesis :", volume_fraction_hypothesis)

    system_2d_matrix = np.ones((number_of_energies + volume_fraction_hypothesis * 1, number_of_materials))

    system_2d_matrix[0:number_of_energies, :] = material_attenuations
    vector_2d_matrix = np.ones((number_of_energies + volume_fraction_hypothesis * 1, images[0, :].size))
    energy_index = 0
    for image in images:
        vector_2d_matrix[energy_index] = image.flatten()
        energy_index += 1

    vector_2d_matrix = np.transpose(vector_2d_matrix)

    solution_matrix = None
    if number_of_energies + volume_fraction_hypothesis * 1 == number_of_materials:
        system_3d_matrix = np.repeat(system_2d_matrix[np.newaxis, :], images[0, :].size, axis=0)
        solution_matrix = np.linalg.solve(system_3d_matrix, vector_2d_matrix)
    else:
        for vector in vector_2d_matrix:
            solution_vector = np.linalg.lstsq(system_2d_matrix, vector, rcond=None)
            if solution_matrix:
                if solution_matrix.ndim == 2:
                    solution_matrix = np.vstack([solution_matrix, solution_vector[0]])
                else:
                    solution_matrix = np.stack((solution_matrix, solution_vector[0]), axis=0)
            else:
                solution_matrix = solution_vector[0]

    m1_image = np.reshape(solution_matrix[:, 0] * densities[0] * 1000.0, images[0, :].shape).astype(np.float32)
    m2_image = np.reshape(solution_matrix[:, 1] * densities[1] * 1000.0, images[0, :].shape).astype(np.float32)
    m3_image = np.reshape(solution_matrix[:, 2] * densities[2] * 1000.0, images[0, :].shape).astype(np.float32)

    return m1_image, m2_image, m3_image


if __name__ == '__main__':
    radix = "BiColor_B1toB9__"
    mainFolder = "/data/visitor/md1237/id17/voltif/"
    mainMaterial = "Au"

    aboveFileNames = glob.glob(mainFolder + radix + "*Above*" + mainMaterial + "*.*" + '/*.tif') + glob.glob(
        mainFolder + radix + "*Above*" + mainMaterial + "*.*" + '/*.edf')
    aboveFileNames.sort()

    belowFileNames = glob.glob(mainFolder + radix + "*Below*" + mainMaterial + "*.*" + '/*.tif') + glob.glob(
        mainFolder + radix + "*Below*" + mainMaterial + "*.*" + '/*.edf')
    belowFileNames.sort()

    aboveFolder = aboveFileNames[0].split("/")[-2]
    aboveMinFloat = float(aboveFolder.split("_")[-2])
    aboveMaxFloat = float(aboveFolder.split("_")[-1])
    print("Found above folder :", aboveFolder)
    print("min Float value :", aboveMinFloat)
    print("max Float value :", aboveMaxFloat)
    belowFolder = belowFileNames[0].split("/")[-2]
    belowMinFloat = float(belowFolder.split("_")[-2])
    belowMaxFloat = float(belowFolder.split("_")[-1])
    print("Found below folder :", belowFolder)
    print("min Float value :", belowMinFloat)
    print("max Float value :", belowMaxFloat)

    if not os.path.exists(mainFolder + "material_decomposition/"):
        os.makedirs(mainFolder + "material_decomposition/")

    if not os.path.exists(mainFolder + radix + "/"):
        os.makedirs(mainFolder + radix + "material_decomposition/")

    if not os.path.exists(mainFolder + radix + "material_decomposition/" + "/Au_decomposition"):
        os.makedirs(mainFolder + radix + "material_decomposition/" + "/Au_decomposition")
    if not os.path.exists(mainFolder + radix + "material_decomposition/" + "/I_decomposition"):
        os.makedirs(mainFolder + radix + "material_decomposition/" + "/I_decomposition")
    if not os.path.exists(mainFolder + radix + "material_decomposition/" + "/Water_decomposition"):
        os.makedirs(mainFolder + radix + "material_decomposition/" + "/Water_decomposition")

    for fileNameIndex in range(0, min(len(aboveFileNames), len(belowFileNames))):
        belowImage = popcornIO.open_image(belowFileNames[fileNameIndex])
        aboveImage = popcornIO.open_image(aboveFileNames[fileNameIndex])

        aboveImage = Resampling.conversion_from_uint16_to_float32(aboveImage, aboveMinFloat, aboveMaxFloat)

        belowImage = Resampling.conversion_from_uint16_to_float32(belowImage, belowMinFloat, belowMaxFloat)

        if mainMaterial == "Au":
            AuImage, IImage, WaterImage = three_materials_decomposition(aboveImage, belowImage, "Au", "I")
        else:
            IImage, AuImage, WaterImage = three_materials_decomposition(aboveImage, belowImage, "I", "Au")

        percent = "{0:.1f}".format(100 * (fileNameIndex / float(min(len(aboveFileNames), len(belowFileNames)) - 1)))
        filled_length = int(100 * fileNameIndex // min(len(aboveFileNames), len(belowFileNames)) - 1)
        bar = '#' * filled_length + '-' * (100 - filled_length)
        print("\r |" + bar + "| " + percent + "% ", end="\r")
        # Print New Line on Complete
        if fileNameIndex == min(len(aboveFileNames), len(belowFileNames)) - 1:
            print()

        textSlice = '%4.4d' % fileNameIndex

        popcornIO.save_edf_image(AuImage,
                                 mainFolder + radix + "material_decomposition/" + "/Au_decomposition/"
                                 + radix + "Au_decomposition_" + textSlice + '.edf')
        popcornIO.save_edf_image(IImage,
                                 mainFolder + radix + "material_decomposition/" + "/I_decomposition/"
                                 + radix + "I_decomposition_" + textSlice + '.edf')
        popcornIO.save_edf_image(WaterImage,
                                 mainFolder + radix + "material_decomposition/" + "/Water_decomposition/"
                                 + radix + "Water_decomposition_" + textSlice + '.edf')
