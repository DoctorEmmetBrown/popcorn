import numpy as np
import sys
from pathlib import Path
path_root = str(Path(__file__).parents[1])
if path_root not in sys.path:
    sys.path.append(path_root)
from popcorn import input_output
from scipy.ndimage import gaussian_filter

def create_phantom_line(nb_slices, height, width, first_point, last_point, type_of_structure="square", size=5, gray_value=255, lowpass_filter=False, sigma=0):
    """function creating squares/circles from point A to point B

    Args:
        nb_slices (int):         z dimension
        height (int):            y dimension
        width (int):             x dimension
        first_point (numpy.ndarray):  position of the starting point
        last_point (numpy.ndarray):   position of the ending point
        type_of_structure (str): circle ? square ?
        size (int):              size of the structure (half of side for square, radius for circle)
        gray_value (float):      gray value of the pixels
        lowpass_filter (bool):   do we compute lowpass filter?

    Returns:
        (numpy.ndarray) The created phantom
    """
    image = np.zeros((nb_slices, height, width)).astype(np.float32)
    positions = np.linspace(first_point, last_point, nb_slices)
    for slice_nb in range(0, nb_slices):
        point2d = positions[slice_nb]
        if type_of_structure == "square":
            for x in range(int(point2d[0] - size), int(point2d[0] + size)):
                for y in range(int(point2d[1] - size), int(point2d[1] + size)):

                    image[slice_nb, y, x] = gray_value
        elif type_of_structure == "circle":
            for x in range(int(point2d[0] - size), int(point2d[0] + size)):
                for y in range(int(point2d[1] - size), int(point2d[1] + size)):
                    if ((point2d[0] - x) ** 2 + (point2d[1] - y) ** 2) ** 0.5 < size:
                        image[slice_nb, y, x] = gray_value

    if lowpass_filter:
        if sigma == 0:
            sigma = size/5
        image = gaussian_filter(image, sigma=sigma)
    return np.copy(image)


if __name__ == "__main__" :

    phantom_circles_straight = create_phantom_line(100,
                                                   200,
                                                   200,
                                                   np.array([100, 100]),
                                                   np.array([100, 100]),
                                                   type_of_structure="circle",
                                                   size=50, gray_value=0.123,
                                                   lowpass_filter=True,
                                                   sigma=2)
    input_output.save_tif_sequence(phantom_circles_straight, "C:\\Users\\ctavakol\\Desktop\\test_for_popcorn\\circles_straight\\")

    phantom_circles_bias = create_phantom_line(100,
                                               200,
                                               200,
                                               np.array([99, 100]),
                                               np.array([100, 100]),
                                               type_of_structure="circle",
                                               size=50,
                                               gray_value=0.123,
                                               lowpass_filter=True,
                                               sigma=2)

    input_output.save_tif_sequence(phantom_circles_bias, "C:\\Users\\ctavakol\\Desktop\\test_for_popcorn\\circles_bias\\")