import os
from q1.mesh import Mesh
import numpy as np


def main():

    src_dir = os.path.join('off_files')
    files_list = [os.path.join(src_dir, f) for f in os.listdir(src_dir)]
    meshes = {'moomoo': Mesh(files_list[0]),
              'vase': Mesh(files_list[1]),
              'teddy': Mesh(files_list[2])}
    for name, mesh in meshes.items():
        mesh.render_wireframe()
        scalar_random = np.random.random(len(mesh.v))
        mesh.render_pointcloud(scalar_random)
        scalar_vertex_degree = mesh.vertex_degree()
        mesh.render_surface(scalar_vertex_degree)
        scalar_face_areas = mesh.get_face_areas()
        mesh.render_surface(scalar_face_areas)
        scalar_vertex_areas = mesh.get_barycentric_vertex_areas()
        mesh.render_surface(scalar_vertex_areas)
        face_normals = mesh.get_face_normals(normalized=False)
        mesh.viz_face_normals(normalized=True, plotter=mesh.render_surface(face_normals))
        vertex_normals = mesh.get_vertex_normals(normalized=False)
        mesh.viz_vertex_normals(normalized=True, plotter=mesh.render_surface(vertex_normals))
        _, euclidean_scalar_func = mesh.get_centroid()
        mesh.render_pointcloud(euclidean_scalar_func, centroid=True)
        gc = mesh.get_gaussian_curvature()
        scalar_gc = np.clip(gc, a_min=np.percentile(gc, 5), a_max=np.percentile(gc, 95))
        mesh.render_surface(scalar_func=scalar_gc)


if __name__ == "__main__":
    main()
