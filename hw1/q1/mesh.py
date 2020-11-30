from scipy.sparse import csr_matrix, identity
from q1.utils import read_off, numpy_to_pyvista
import numpy as np
import pyvista as pv
from numpy.linalg import norm


class Mesh:

    def __init__(self, filepath):

        self.v, self.f = read_off(filepath)

    def vertex_face_adjacency(self):
        num_vertices = self.v.shape[0]
        num_faces = self.f.shape[0]
        face_len = self.f.shape[1]
        rows = np.concatenate(list(self.f))
        cols = np.arange(num_faces * face_len) // face_len
        assert np.shape(rows) == np.shape(cols)
        data = np.ones_like(rows)
        vf_adj = csr_matrix((data, (rows, cols)), shape=(num_vertices,
                                                         num_faces), dtype=bool)
        return vf_adj

    def vertex_vertex_adjacency(self):
        vf_adj = self.vertex_face_adjacency()
        vv_mat = vf_adj.dot(vf_adj.transpose())
        vv_adj = (vv_mat - identity(len(self.v))).astype(bool)
        return vv_adj

    def vertex_degree(self):
        vv_adj = self.vertex_vertex_adjacency()
        v_degree = np.sum(vv_adj.toarray(), axis=0)
        return v_degree

    @staticmethod
    def get_rows_norm(mat):
        l2_norm = np.linalg.norm(mat, axis=1)
        return l2_norm

    def get_face_normals(self, normalized=True):
        v, f = self.v, self.f
        a = v[f[:, 0], :]
        b = v[f[:, 1], :]
        c = v[f[:, 2], :]
        fn = np.cross(b - a, c - a)
        if normalized:
            norms = self.get_rows_norm(fn)
            fn = fn / norms[:, None]
        return fn

    def get_face_barycenters(self):
        v, f = self.v, self.f
        face_coordinates = v[f[:], :]
        face_centers = np.mean(face_coordinates, axis=1)
        return face_centers

    def get_face_areas(self):
        fn = self.get_face_normals(normalized=False)
        fn_magnitude = np.linalg.norm(fn, axis=1)
        face_areas = fn_magnitude / 2
        return face_areas

    def get_barycentric_vertex_areas(self):
        vf_adj = self.vertex_face_adjacency()
        face_areas = self.get_face_areas()
        vertex_areas = vf_adj.dot(face_areas) / 3
        return vertex_areas

    def get_vertex_normals(self, normalized=True):
        vf_adj = self.vertex_face_adjacency()
        face_normals = self.get_face_normals(normalized=False)
        face_areas = self.get_face_areas()
        face_normals_areas = face_normals * face_areas[:, None]
        vertex_normals = vf_adj.dot(face_normals_areas)
        if normalized:
            norms = self.get_rows_norm(vertex_normals)
            vertex_normals = vertex_normals / norms[:, None]
        return vertex_normals

    def get_centroid(self):
        centroid = np.mean(self.v, axis=0)
        euclidean_dist = np.linalg.norm(self.v - centroid, axis=1)
        return centroid, euclidean_dist

    def get_gaussian_curvature(self):
        a = self.v[self.f[:, 0], :]
        b = self.v[self.f[:, 1], :]
        c = self.v[self.f[:, 2], :]
        dot_prod = [np.sum((b - a) * (c - a), axis=1),
                    np.sum((a - b) * (c - b), axis=1),
                    np.sum((a - c) * (b - c), axis=1)]
        ab_norm = np.linalg.norm(a - b, axis=1)
        bc_norm = np.linalg.norm(b - c, axis=1)
        ac_norm = np.linalg.norm(a - c, axis=1)
        theta = [np.arccos(dot_prod[0] / (ab_norm * ac_norm)),
                 np.arccos(dot_prod[1] / (ab_norm * bc_norm)),
                 np.arccos(dot_prod[2] / (ac_norm * bc_norm))]
        num_vertices = self.v.shape[0]
        num_faces = self.f.shape[0]
        face_len = self.f.shape[1]
        rows = np.concatenate(list(self.f))
        cols = np.arange(num_faces * face_len) // face_len
        data = np.transpose(np.hstack([theta])).flatten()
        theta_mat = csr_matrix((data, (rows, cols)), shape=(num_vertices, num_faces))
        sum_theta = np.array(theta_mat.sum(axis=1)).squeeze()
        gaussian_curvature = (2 * np.pi - sum_theta) / self.get_barycentric_vertex_areas()
        return gaussian_curvature

    def render_wireframe(self):
        obj = numpy_to_pyvista(self.v, self.f)
        wireframe_plotter = pv.Plotter()
        wireframe_plotter.add_mesh(obj, style='wireframe')
        return wireframe_plotter

    def render_pointcloud(self, scalar_func, centroid):
        obj = numpy_to_pyvista(self.v, self.f)
        point_cloud_plotter = pv.Plotter()
        point_cloud_plotter.add_mesh(obj, style='points', scalars=scalar_func,
                                     render_points_as_spheres=True)
        if centroid:
            point_obj = numpy_to_pyvista(self.get_centroid()[0])
            point_cloud_plotter.add_mesh(point_obj, style='points', point_size=50,
                                         color='black', render_points_as_spheres=True)
        return point_cloud_plotter

    def render_surface(self, scalar_func):
        obj = numpy_to_pyvista(self.v, self.f)
        surf_plotter = pv.Plotter()
        surf_plotter.add_mesh(obj, scalars=scalar_func)
        return surf_plotter

    def viz_face_normals(self, normalized, plotter=None):
        fn = self.get_face_normals(normalized=normalized)
        if plotter is None:
            plotter = pv.Plotter()
        centers = self.get_face_barycenters()
        plotter.add_arrows(centers, fn)
        return plotter

    def viz_vertex_normals(self, normalized, plotter=None):
        vn = self.get_vertex_normals(normalized=normalized)
        if plotter is None:
            plotter = pv.Plotter()
        centers = self.v
        plotter.add_arrows(centers, vn)
        return plotter
