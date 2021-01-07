import meshio
import numpy as np
import pyvista as pv


def read_mesh(filepath):
    mesh = meshio.read(filepath)
    v = np.array(mesh.points)
    f = np.array(mesh.cells, dtype='object')[0, 1]

    return v, f


def write_off(data, save_path):
    points = data[0]
    cells = [("triangle", data[1])]
    meshio.write_points_cells(save_path, points, cells)


def numpy_to_pyvista(v, f=None):
    if f is None:
        return pv.PolyData(v)
    else:
        return pv.PolyData(v, np.concatenate((np.full(
            (f.shape[0], 1), 3), f), 1))
