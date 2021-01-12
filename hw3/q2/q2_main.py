import os
import gdist
from scipy.sparse.linalg import inv
from sklearn.manifold import MDS
from q2.mesh import Mesh
from q2.utils import numpy_to_pyvista, save_fig
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt


def compute_dirac_scalars(src_idx, mesh):
    scalar_dirac_func = np.zeros(len(mesh.v))
    scalar_dirac_func[src_idx] = 1
    return scalar_dirac_func


def compute_geodesic_scalars(src_idx, mesh):
    source_indices = np.array([src_idx], dtype=np.int32)
    target_indices = np.array(range(len(mesh.v)), dtype=np.int32)
    scalar_geodesic_func = gdist.compute_gdist(mesh.v.astype(np.float64), mesh.f, source_indices, target_indices)
    return scalar_geodesic_func


def compute_pairwise_dist(meshes, sig_name, k, **hks_kwargs):
    signatures = []
    for mesh in meshes:
        if sig_name == 'shape_dna':
            sig = mesh.shape_dna_signature(k=k)
        elif sig_name == 'gps':
            sig = mesh.global_point_signature(k=k)
        elif sig_name == 'hks':
            sig = mesh.heat_kernel_signature(k=k, **hks_kwargs)
        else:
            sig = mesh.get_signed_mean_curvature()
        signatures.append(sig)

    x = np.array(signatures)
    pairwise_dist = np.linalg.norm(x - x[:, None], axis=-1)
    return pairwise_dist, signatures


def load_dataset(src_dir):
    file_paths, subject_labels, pose_labels = [], [], []
    for file_name in os.listdir(src_dir):
        file_path = os.path.join(src_dir, file_name)
        file_paths.append(file_path)
        idx = int(os.path.splitext(file_name)[0][-3:])
        subject_id = int(idx / 10)
        pose_id = int(idx % 10)
        subject_labels.append(subject_id)
        pose_labels.append(pose_id)

    return file_paths, subject_labels, pose_labels


def apply_mds(signatures, labels, show=False):
    embedding = MDS(n_components=2, dissimilarity='precomputed')
    transformed_sig = embedding.fit_transform(signatures)
    if show:
        plt.scatter(transformed_sig[:, 0], transformed_sig[:, 1], c=labels)
        plt.show()


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
    # mesh = meshes[0]
    # Section 4: signed mean curvature
    # for cls in ['half_cotangent', 'uniform']:
    #     signed_mean_curvature = mesh.get_signed_mean_curvature(cls=cls)
    #     p = mesh.render_surface(scalar_func=signed_mean_curvature)

    # Section 5: Laplacian applications on scalar functions
    # src_idx = 4811
    # scalar_dirac_func = compute_dirac_scalars(src_idx, mesh)
    # scalar_geodesic_func = compute_geodesic_scalars(src_idx, mesh)
    # scalar_funcs = {'dirac': scalar_dirac_func, 'geodesic': scalar_geodesic_func}
    # for func_name, scalar_func in scalar_funcs.items():
    #     p_func = mesh.render_surface(scalar_func)
    # vertex_mass_mat = mesh.barycenter_vertex_mass_matrix()
    # k_values = [2, 4, 8, 32, 64]
    # for cls in ['half_cotangent', 'uniform']:
    #     for func_name, scalar_func in scalar_funcs.items():
    #         pv.set_plot_theme('document')
    #         multi_p = pv.Plotter(shape=(1, len(k_values)))
    #         multi_p.link_views()
    #         for i, k in enumerate(k_values):
    #             eig_val, eig_vec = mesh.laplacian_spectrum(k=k, cls=cls)
    #             approx_func = eig_vec.dot(eig_vec.T.dot(vertex_mass_mat.dot(scalar_func)))
    #             multi_p.subplot(0, i)
    #             multi_p.add_mesh(numpy_to_pyvista(mesh.v, mesh.f), scalars=approx_func, colormap='jet')
    #         normalized_laplacian_func = mesh.laplacian(cls=cls) * scalar_func
    #         normalized_laplacian_func /= mesh.get_barycentric_vertex_areas()
    #         p = mesh.render_surface(scalar_func=normalized_laplacian_func)

    # Section 6: non-rigid shape classification
    file_paths, subject_labels, pose_labels = load_dataset(src_dir)
    ds_meshes = [Mesh(f) for f in file_paths]
    pairwise_dist, signatures = compute_pairwise_dist(ds_meshes, sig_name='shape_dna', k=1000)
    apply_mds(pairwise_dist, labels=subject_labels, show=True)


if __name__ == "__main__":
    main()
