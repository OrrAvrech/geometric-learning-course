import os
import gdist
from scipy.sparse.linalg import inv
from q2.mesh import Mesh
from q2.utils import numpy_to_pyvista, save_fig
import pyvista as pv
import numpy as np


def compute_dirac_scalars(src_idx, mesh):
    scalar_dirac_func = np.zeros(len(mesh.v))
    scalar_dirac_func[src_idx] = 1
    return scalar_dirac_func


def compute_geodesic_scalars(src_idx, mesh):
    source_indices = np.array([src_idx], dtype=np.int32)
    target_indices = np.array(range(len(mesh.v)), dtype=np.int32)
    scalar_geodesic_func = gdist.compute_gdist(mesh.v.astype(np.float64), mesh.f, source_indices, target_indices)
    return scalar_geodesic_func


def main():

    # load FAUST dataset
    src_dir = os.path.join('..', 'FAUST')
    files_list = [os.path.join(src_dir, f) for f in os.listdir(src_dir)]
    num_meshes = 3
    meshes = []
    for i in range(num_meshes):
        meshes.append(Mesh(files_list[i]))
    k = 5
    # Section 3: isometry invariance
    # for cls in ['half_cotangent', 'uniform']:
    #     pv.set_plot_theme('document')
    #     p = pv.Plotter(shape=(num_meshes, k))
    #     p.link_views()
    #     for i, mesh in enumerate(meshes):
    #         eig_val, eig_vec = mesh.laplacian_spectrum(k=k, cls=cls)
    #         for j in range(k):
    #             p.subplot(i, j)
    #             p.add_mesh(numpy_to_pyvista(mesh.v, mesh.f), scalars=eig_vec[:, j], colormap='jet')
    # A single mesh for sections 4-5
    mesh = meshes[0]
    # Section 4: signed mean curvature
    # for cls in ['half_cotangent', 'uniform']:
    #     laplacian = mesh.laplacian(cls=cls)
    #     mean_curvature_normal = 0.5 * laplacian * mesh.v / mesh.get_barycentric_vertex_areas()[:, None]
    #     unsigned_mean_curvature = np.linalg.norm(mean_curvature_normal, axis=1)
    #     # clipping mean curvature unsigned values
    #     unsigned_mean_curvature = np.clip(unsigned_mean_curvature, a_min=np.percentile(unsigned_mean_curvature, 10),
    #                                       a_max=np.percentile(unsigned_mean_curvature, 90))
    #     vertex_normals = mesh.get_vertex_normals()
    #     signed_mean_curvature = unsigned_mean_curvature * np.sign(np.sum(vertex_normals*mean_curvature_normal, axis=1))
    #     p = mesh.render_surface(scalar_func=signed_mean_curvature)

    # Section 5: Laplacian applications on scalar functions
    src_idx = 4811
    scalar_dirac_func = compute_dirac_scalars(src_idx, mesh)
    scalar_geodesic_func = compute_geodesic_scalars(src_idx, mesh)
    scalar_funcs = {'dirac': scalar_dirac_func, 'geodesic': scalar_geodesic_func}
    vertex_mass_mat = mesh.barycenter_vertex_mass_matrix()
    k_values = [2, 4, 8, 32, 64]
    for cls in ['half_cotangent', 'uniform']:
        for func_name, scalar_func in scalar_funcs.items():
            pv.set_plot_theme('document')
            p = pv.Plotter(shape=(1, len(k_values)))
            p.link_views()
            for i, k in enumerate(k_values):
                eig_val, eig_vec = mesh.laplacian_spectrum(k=k, cls=cls)
                approx_func = eig_vec.dot(eig_vec.T.dot(vertex_mass_mat.dot(scalar_func)))
                p.subplot(0, i)
                p.add_mesh(numpy_to_pyvista(mesh.v, mesh.f), scalars=approx_func, colormap='jet')


if __name__ == "__main__":
    main()
