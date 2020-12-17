from scipy import sparse
from sklearn.neighbors import kneighbors_graph
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import eigh

# class Dataset
# add normalization by degree matrix according to some parameter alpha


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
        knn_symmetric = (knn_graph + knn_graph.transpose()).astype(bool)
        pairwise_dist = knn_symmetric.multiply(kernel)
    else:
        pairwise_dist = sparse.csr_matrix(kernel)

    return pairwise_dist


def diffusion_maps(z, n_dim, epsilon=None, n_neighbor=None):
    adj_mat = affinity_mat(z, kernel_method='gaussian', epsilon=epsilon, n_neighbor=n_neighbor)
    degree_mat = sparse.diags(np.array((1/adj_mat.sum(axis=0))), [0])
    walk_mat = adj_mat.dot(degree_mat)
    eigen_values, eigen_vectors = np.linalg.eigh(walk_mat.toarray())
    idx = eigen_values.argsort()[::-1]
    eigen_values, eigen_vectors = eigen_values[idx], eigen_vectors[idx]
    embedding = eigen_values[:n_dim] * eigen_vectors[:, :n_dim]
    return embedding


# def isomap(n_neighbor=None):
