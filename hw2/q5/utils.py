from scipy import sparse
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils.graph_shortest_path import graph_shortest_path
import numpy as np


def affinity_mat(z, kernel_method='linear', epsilon=None, n_neighbor=None):

    num_samples, num_features = z.shape
    if epsilon is None:
        epsilon = num_features

    if kernel_method == 'gaussian':
        kernel = np.exp(-(np.linalg.norm(z - z[:, None], axis=-1)**2) / epsilon)
    else:
        kernel = np.linalg.norm(z - z[:, None], axis=-1)**2

    if n_neighbor:
        # apply kNN
        knn_graph = kneighbors_graph(z, n_neighbors=n_neighbor)
        # turn knn-graph into a symmetric matrix
        knn_symmetric = (knn_graph + knn_graph.T).astype(bool)
        pairwise_dist = knn_symmetric.multiply(kernel)
    else:
        pairwise_dist = sparse.csr_matrix(kernel)

    return pairwise_dist


def diffusion_maps(z, n_dim, epsilon=None, n_neighbor=None):
    adj_mat = affinity_mat(z, kernel_method='gaussian', epsilon=epsilon, n_neighbor=n_neighbor)
    degree_mat = sparse.diags(np.array(adj_mat.sum(axis=0)), [0]).tocsc()
    walk_mat = sparse.linalg.inv(degree_mat).dot(adj_mat)
    eigen_values, eigen_vectors = np.linalg.eigh(walk_mat.toarray())
    idx = eigen_values.argsort()[::-1]
    eigen_values, eigen_vectors = eigen_values[idx], eigen_vectors[:, idx]
    eigen_values, eigen_vectors = eigen_values[1:n_dim+1], eigen_vectors[:, 1:n_dim+1]
    embedding = np.dot(eigen_vectors, np.diag(eigen_values))
    return embedding


def isomap(z, n_dim, n_neighbor=None):
    num_samples, num_features = z.shape
    adj_mat = affinity_mat(z, n_neighbor=n_neighbor)
    shortest_paths = graph_shortest_path(adj_mat)
    h = np.eye(num_samples) - (1/num_samples)*np.ones((num_samples, num_samples))
    k = -0.5 * h.dot(shortest_paths**2).dot(h)
    eigen_values, eigen_vectors = np.linalg.eigh(k)
    idx = eigen_values.argsort()[::-1]
    eigen_values, eigen_vectors = eigen_values[idx], eigen_vectors[:, idx]
    eigen_values, eigen_vectors = eigen_values[:n_dim], eigen_vectors[:, :n_dim]
    embedding = np.dot(eigen_vectors, np.diag(eigen_values**(1/2)))
    return embedding


def locally_linear_embedding(z, n_dim, n_neighbor=None):
    num_samples, num_features = z.shape
    knn = NearestNeighbors(n_neighbors=n_neighbor).fit(z)
    neighbors_indices = knn.kneighbors(z, return_distance=False)
    iw = (sparse.eye(num_samples) - adj_mat)
    b = iw.T.dot(iw)
    eigen_values, eigen_vectors = np.linalg.eigh(b.toarray())
    idx = eigen_values.argsort()
    eigen_values, eigen_vectors = eigen_values[idx], eigen_vectors[:, idx]
    embedding = eigen_vectors[:, :n_dim]
    return embedding
