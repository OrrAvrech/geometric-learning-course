import os
from mesh import Mesh
import numpy as np
import pyvista as pv


def numpy_to_pyvista(v, f=None):
    if f is None:
        return pv.PolyData(v)
    else:
        return pv.PolyData(v, np.concatenate((np.full(
            (f.shape[0], 1), 3), f), 1))


def render_wireframe(mesh):
    obj = numpy_to_pyvista(mesh.v, mesh.f)
    wireframe_plotter = pv.Plotter()
    wireframe_plotter.add_mesh(obj, style='wireframe')
    return wireframe_plotter


def render_pointcloud(scalar_func, mesh):
    obj = numpy_to_pyvista(mesh.v, mesh.f)
    point_cloud_plotter = pv.Plotter()
    point_cloud_plotter.add_mesh(obj, style='points', scalars=scalar_func,
                                 render_points_as_spheres=True, stitle='')
    return point_cloud_plotter


def render_surface(scalar_func, mesh):
    obj = numpy_to_pyvista(mesh.v, mesh.f)
    surf_plotter = pv.Plotter()
    surf_plotter.add_mesh(obj, scalars=scalar_func, stitle='')
    return surf_plotter


def viz_face_normals(mesh, normalized, plotter=None):
    fn = mesh.get_face_normals(normalized=normalized)
    if plotter is None:
        plotter = pv.Plotter()
    centers = mesh.get_face_barycenters()
    plotter.add_arrows(centers, fn)
    return plotter


def viz_vertex_normals(mesh, normalized, plotter=None):
    vn = mesh.get_vertex_normals(normalized=normalized)
    if plotter is None:
        plotter = pv.Plotter()
    centers = mesh.v
    plotter.add_arrows(centers, vn)
    return plotter


def main():

    src_dir = os.path.join('..', '..', 'example_off_files')
    files_list = [os.path.join(src_dir, f) for f in os.listdir(src_dir)]
    mesh = Mesh(files_list[2])
    vertex_centroid = np.mean(mesh.v, axis=0)
    scalar_func = np.linalg.norm(mesh.v - vertex_centroid, axis=1)
    plotter = render_pointcloud(scalar_func, mesh)
    plotter.show()


if __name__ == "__main__":
    main()
