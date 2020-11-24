import meshio
import numpy as np


def read_off(filepath):
    mesh = meshio.read(filepath)
    v = np.array(mesh.points)
    f = np.array(mesh.cells, dtype='object')[0, 1]

    return v, f


def write_off(data, save_path):
    points = data[0]
    cells = [("triangle", data[1])]
    meshio.write_points_cells(save_path, points, cells)
