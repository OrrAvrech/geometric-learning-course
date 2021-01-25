import os
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

SEED = 0
np.random.seed(SEED)


def plot_3d(data, clusters=None, show=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=clusters)
    if show:
        plt.show()


def spectral_clustering(x, k, sigma=0.2, normalized=False):
    kernel = np.exp(-(np.linalg.norm(x - x[:, None], axis=-1) ** 2) / (2*sigma**2))
    affinity_mat = kernel - np.eye(len(x))
    degree_vec = affinity_mat.sum(axis=0)
    degree_mat = np.diag(degree_vec)
    laplacian = degree_mat - affinity_mat
    if normalized:
        norm_degree = np.diag(1/np.sqrt(degree_vec))
        laplacian = norm_degree.dot(laplacian).dot(norm_degree)
        eig_values, eig_vectors = np.linalg.eigh(laplacian)
        embedding = (eig_vectors[:, :k] / eig_values[None, :k])
    else:
        eig_values, eig_vectors = np.linalg.eigh(laplacian)
        embedding = (eig_vectors[:, 1:k+1] / eig_values[None, 1:k+1])
    return embedding


def main():
    rings_dir = os.path.join('Rings')
    ring_5_path = os.path.join(rings_dir, 'ring5.npy')
    ring_2_path = os.path.join(rings_dir, 'ring2.npy')
    # Section 4
    k = 5
    ring_5_np = np.load(ring_5_path).T
    plot_3d(ring_5_np)
    k_means = KMeans(n_clusters=k)
    euc_cluster_indices = k_means.fit_predict(ring_5_np)
    plot_3d(ring_5_np, euc_cluster_indices)
    spectral_embedding = spectral_clustering(ring_5_np, k=k)
    spectral_cluster_indices = k_means.fit_predict(spectral_embedding)
    plot_3d(ring_5_np, spectral_cluster_indices)
    plot_3d(spectral_embedding[:, :3])

    # Section 5
    k = 2
    ring_2_np = np.load(ring_2_path).T
    plot_3d(ring_2_np)
    k_means = KMeans(n_clusters=k)
    euc_cluster_indices = k_means.fit_predict(ring_2_np)
    plot_3d(ring_2_np, euc_cluster_indices)
    spectral_embedding = spectral_clustering(ring_2_np, k=k, sigma=0.05)
    spectral_cluster_indices = k_means.fit_predict(spectral_embedding)
    plot_3d(ring_2_np, spectral_cluster_indices)
    norm_spectral_embedding = spectral_clustering(ring_2_np, k=k, sigma=0.05, normalized=True)
    norm_spectral_cluster_indices = k_means.fit_predict(norm_spectral_embedding)
    plot_3d(ring_2_np, norm_spectral_cluster_indices)


if __name__ == "__main__":
    main()
