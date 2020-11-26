from scipy.sparse import coo_matrix
from utils import read_off
import numpy as np


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
        vf_adj = coo_matrix((data, (rows, cols)), shape=(num_vertices,
                                                         num_faces)).toarray()

        return vf_adj

    def vertex_vertex_adjacency(self):
        vf_adj = self.vertex_face_adjacency()
        vv_mat = np.matmul(vf_adj, np.transpose(vf_adj))
        diag = np.matmul(vv_mat, np.identity(len(vv_mat)))
        vv_adj = np.where(vv_mat - diag > 0, 1, 0)

        return vv_adj

    def vertex_degree(self):
        vv_adj = self.vertex_vertex_adjacency()
        v_degree = np.sum(vv_adj, axis=0)

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
        vertex_areas = np.dot(vf_adj, face_areas) / 3
        return vertex_areas

    def get_vertex_normals(self, normalized=True):
        vf_adj = self.vertex_face_adjacency()
        face_normals = self.get_face_normals(normalized=False)
        face_areas = self.get_face_areas()
        vf_adj_areas = vf_adj * face_areas
        vertex_normals = np.matmul(vf_adj_areas, face_normals)
        if normalized:
            norms = self.get_rows_norm(vertex_normals)
            vertex_normals = vertex_normals / norms[:, None]
        return vertex_normals
