from sklearn.datasets import load_digits
from q5.utils import affinity_mat, diffusion_maps
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from pydiffmap import diffusion_map as dm


def main():

    # 3.a
    n_samples = 2000
    r = 4
    R = 10
    x, y = np.random.random((2, n_samples))
    s = np.array([(R + r * np.cos(2*np.pi*y)) * np.cos(2*np.pi*x),
                  (R + r * np.cos(2*np.pi*y)) * np.sin(2*np.pi*x),
                  r * np.sin(2*np.pi*y)]).transpose()
    b = diffusion_maps(s, n_dim=2, epsilon=3)
    neighbor_params = {'n_jobs': -1, 'algorithm': 'ball_tree'}

    # mydmap = dm.DiffusionMap.from_sklearn(n_evecs=2, k=200, epsilon='bgh', alpha=1.0, neighbor_params=neighbor_params)
    # fit to data and return the diffusion map.
    # a = mydmap.fit_transform(s)


    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(s[:, 0], s[:, 1], s[:, 2])
    plt.scatter(b[:, 0], b[:, 1])

    # 3.b
    # digits_datasets = [load_digits(n_class=i) for i in (3, 5, 7)]
    # a = digits_datasets[2]
    # b = mydmap.fit_transform(a.data)
    # b = LocallyLinearEmbedding(n_neighbors=30, n_components=2).fit_transform(a.data)
    # b = diffusion_maps(a.data, n_dim=2, epsilon=64)
    # plt.scatter(b[:, 0], b[:, 1], c=a.target)
    plt.show()


if __name__ == "__main__":
    main()
