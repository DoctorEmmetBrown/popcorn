import numpy as np
import ratGistrationIO
import registration

def create_phantom_line(nb_slices, height, width, first_point, last_point, type_of_structure = "square", radius = 5):
    """
    function creating squares from point A to point B
    :param nb_slices: z dim
    :param height: y dim
    :param width: x dim
    :param first_point: position of the first structure element
    :param last_point: position of the first structure element
    :param type_of_structure: circle ? square ?
    :param radius: size of the structure
    :return:
    """
    image = np.zeros((nb_slices, height, width)).astype(np.uint16)
    positions = np.linspace(first_point, last_point, nb_slices)
    for slice_nb in range(0, nb_slices):
        point2d = positions[slice_nb]
        for x in range(int(point2d[0] - radius), int(point2d[0] + radius)):
            for y in range(int(point2d[1] - radius), int(point2d[1] + radius)):
                image[slice_nb, y, x] = 255
    return np.copy(image)


if __name__ == "__main__" :

    phantom_x = create_phantom_line(50, 200, 200, np.array([50, 100]), np.array([100, 100]))
    #ratGistrationIO.save_tif_sequence(phantom_x, "C:\\Users\\ctavakol\\Desktop\\test_for_ratGistration\\test_phantoms\\phantom_x\\")

    phantom_y = create_phantom_line(50, 200, 200, np.array([100, 50]), np.array([100, 100]))
    #ratGistrationIO.save_tif_sequence(phantom_y, "C:\\Users\\ctavakol\\Desktop\\test_for_ratGistration\\test_phantoms\\phantom_y\\")

    phantom_xy = create_phantom_line(50, 200, 200, np.array([50, 50]), np.array([100, 100]))
    #ratGistrationIO.save_tif_sequence(phantom_xy, "C:\\Users\\ctavakol\\Desktop\\test_for_ratGistration\\test_phantoms\\phantom_xy\\")

    reoriented_phantom_x = registration.two_vectors_3d_rotation(phantom_x, np.array([-0.714, 0, -0.699]), np.array([0, 0, -1]), [100, 100, 50])
    #ratGistrationIO.save_tif_sequence(reoriented_phantom_x, "C:\\Users\\ctavakol\\Desktop\\test_for_ratGistration\\test_phantoms\\result_phantom_x\\")

    reoriented_phantom_y = registration.two_vectors_3d_rotation(phantom_y, np.array([0, -0.714, -0.699]), np.array([0, 0, -1]), [100, 100, 50])
    #ratGistrationIO.save_tif_sequence(reoriented_phantom_y, "C:\\Users\\ctavakol\\Desktop\\test_for_ratGistration\\test_phantoms\\result_phantom_y\\")

    reoriented_phantom_xy = registration.two_vectors_3d_rotation(phantom_xy, np.array([-0.581, -0.581, -0.569]), np.array([0, 0, -1]), [100, 100, 50])
    #ratGistrationIO.save_tif_sequence(reoriented_phantom_xy, "C:\\Users\\ctavakol\\Desktop\\test_for_ratGistration\\test_phantoms\\result_phantom_xy\\")