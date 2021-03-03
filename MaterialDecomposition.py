import numpy as np
import glob, os
import popcornIO


def convert_int_to_float(image, minFloat, maxFloat):
    """
    Conversion from [0:65535] uint16 to [minFloat:maxFloat] float32
    :param image: input uint image
    :param minFloat: min val after conversion
    :param maxFloat: max val after conversion
    :return: converted float32 image
    """
    image = image.astype(np.float32)
    return image / 65535 * (maxFloat - minFloat) + minFloat


def three_materials_decomposition(above_kedge_image, below_kedge_image, kedge_element="Au", second_element="I"):
    """
    Calculating the concentration (in mg/mL) of the kedge_element based on the assumption that each voxel is only
    composed of water, the kedge element and the second element.
    :param above_kedge_image: image acquired with an energy right above the element kedge
    :param below_kedge_image: image acquired with an energy right below the element kedge
    :param kedge_element: the kedge element (I, Ba, Gd, Au)
    :param second_element: the second element
    :return: concentration map (in mg/mL) of the kedge element
    """
    indexesDict = {
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

    main_element_index = indexesDict.get(kedge_element.lower())
    second_element_index = indexesDict.get(second_element.lower())
    # E1 = Below, E2 = Above
    # Water
    mu_water_below = muz[0][(main_element_index - 1) * 2]
    mu_water_above = muz[0][(main_element_index - 1) * 2 + 1]

    # Second element
    mu_second_element_below = muz[second_element_index][(main_element_index - 1) * 2]
    mu_second_element_above = muz[second_element_index][(main_element_index - 1) * 2 + 1]

    # Main element
    mu_main_element_below = muz[main_element_index][(main_element_index - 1) * 2]
    mu_main_element_above = muz[main_element_index][(main_element_index - 1) * 2 + 1]

    images = np.stack((above_kedge_image, below_kedge_image), axis=0)
    material_densities = np.array([densities[main_element_index - 1], densities[second_element_index - 1], 1.0])
    mus = np.array([[mu_main_element_above, mu_second_element_above, mu_water_above],
                       [mu_main_element_below, mu_second_element_below, mu_water_below]])

    main_element_concentration_map, second_element_concentration_map, water_concentration_map = \
        decomposition_equation_resolution(images, material_densities, mus)

    return main_element_concentration_map.copy(), \
           second_element_concentration_map.copy(), \
           water_concentration_map.copy()


def decomposition_equation_resolution(images, densities, materialAttenuations, volumeFractionHypothesis=True):
    """
    solving the element decomposition system : images.ndim energies
    :param images: N dim array, each N-1 dim array is an image acquired at 1 given energy (can be 2D or 3D, K energies in total)
    :param densities: 1D array, one density per elements inside a voxel (P elements in total)
    :param materialAttenuations: 2D array, linear attenuation of each element at each energy (K * P array)
    :return: material decomposition maps, N-dim array composed of P * N-1-dim arrays
    """
    print("-- Material decomposition --")
    numberOfEnergies = images.shape[0]
    print(">Number of energies: ", numberOfEnergies)
    numberOfMaterials = densities.size
    print(">Number of materials: ", numberOfMaterials)
    print(">Sum of materials volume fraction equal to 1 hypothesis :", volumeFractionHypothesis)
    system_2d_matrix = np.ones((numberOfEnergies + volumeFractionHypothesis * 1, numberOfMaterials))

    system_2d_matrix[0:numberOfEnergies, :] = materialAttenuations
    vector_2d_matrix = np.ones((numberOfEnergies + volumeFractionHypothesis * 1, images[0, :].size))
    energyIndex = 0
    for image in images:
        vector_2d_matrix[energyIndex] = image.flatten()
        energyIndex += 1

    vector_2d_matrix = np.transpose(vector_2d_matrix)

    if numberOfEnergies + volumeFractionHypothesis * 1 == numberOfMaterials:
        system_3d_matrix = np.repeat(system_2d_matrix[np.newaxis, :], images[0, :].size, axis=0)
        solution_matrix = np.linalg.solve(system_3d_matrix, vector_2d_matrix)
    else:
        for vector in vector_2d_matrix:
            solution_vector = np.linalg.lstsq(system_2d_matrix, vector, rcond=None)
            if 'solution_matrix' in locals():
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
    for fileNameIndex in range(0, min(len(aboveFileNames), len(belowFileNames))):
        belowImage = popcornIO.openImage(belowFileNames[fileNameIndex])
        aboveImage = popcornIO.openImage(aboveFileNames[fileNameIndex])

        aboveImage = convert_int_to_float(aboveImage, aboveMinFloat, aboveMaxFloat)

        belowImage = convert_int_to_float(belowImage, belowMinFloat, belowMaxFloat)

        if mainMaterial == "Au":
            AuImage, IImage, WaterImage = three_materials_decomposition(aboveImage, belowImage, "Au", "I")
        else:
            IImage, AuImage, WaterImage = three_materials_decomposition(aboveImage, belowImage, "I", "Au")

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

        textSlice = '%4.4d' % fileNameIndex

        popcornIO.saveEdf(AuImage, mainFolder + radix + "material_decomposition/" + "/Au_decomposition/" + radix + "Au_decomposition_" + textSlice + '.edf')
        popcornIO.saveEdf(IImage, mainFolder + radix + "material_decomposition/" + "/I_decomposition/" + radix + "I_decomposition_" + textSlice + '.edf')
        popcornIO.saveEdf(WaterImage, mainFolder + radix + "material_decomposition/" + "/Water_decomposition/" + radix + "Water_decomposition_" + textSlice + '.edf')


