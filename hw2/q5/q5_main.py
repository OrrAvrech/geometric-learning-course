from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from q5.utils import diffusion_maps, isomap, locally_linear_embedding
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


def generate_torus(n_samples, r, R):
    x, y = np.random.random((2, n_samples))
    s = np.array([(R + r * np.cos(2 * np.pi * y)) * np.cos(2 * np.pi * x),
                  (R + r * np.cos(2 * np.pi * y)) * np.sin(2 * np.pi * x),
                  r * np.sin(2 * np.pi * y)]).transpose()
    return s, x, y


def torus_color_subplots(res, s, x, y, title=''):
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.scatter(s[:, 0], s[:, 1], s[:, 2], c=res[:, 0])
    ax.set_title('$\psi_1$')
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.scatter(s[:, 0], s[:, 1], s[:, 2], c=res[:, 1])
    ax.set_title('$\psi_2$')
    ax = fig.add_subplot(2, 2, 3)
    ax.scatter(res[:, 0], res[:, 1], c=x)
    ax.set_title('x')
    ax = fig.add_subplot(2, 2, 4)
    ax.scatter(res[:, 0], res[:, 1], c=y)
    ax.set_title('y')
    fig.suptitle(title)
    plt.show()


def digits_ds_subplots(ds, n_neighbors_list, title=''):
    data, classes = ds.data, ds.target
    fig_diff = plt.figure()
    fig_isomap = plt.figure()
    for i, n_neighbor in enumerate(n_neighbors_list):
        diffusion_res = diffusion_maps(data, n_dim=2, epsilon=64, n_neighbor=n_neighbor)
        ax = fig_diff.add_subplot(1, len(n_neighbors_list), i + 1)
        ax.scatter(diffusion_res[:, 0], diffusion_res[:, 1], c=classes)
        ax.set_title(f"{n_neighbor}NN")
        fig_diff.suptitle(f"diffusion-map, {title}")

        isomap_res = isomap(data, n_dim=2, n_neighbor=n_neighbor)
        ax = fig_isomap.add_subplot(1, len(n_neighbors_list), i + 1)
        ax.scatter(isomap_res[:, 0], isomap_res[:, 1], c=classes)
        ax.set_title(f"{n_neighbor}NN")
        fig_isomap.suptitle(f"isomap, {title}")
    plt.show()


def main():

    # Section 3 #
    # 3.a
    n_samples, r, R = (2000, 4, 10)
    torus = generate_torus(n_samples, r, R)
    s, x, y = torus
    n_neighbors = [10, 50, 100]
    for n_neighbor in n_neighbors:
        diffusion_res = diffusion_maps(s, n_dim=2, epsilon=3, n_neighbor=n_neighbor)
        torus_color_subplots(diffusion_res, *torus, title=f"diffusion-map, {n_neighbor}-nearest-neighbors")
        isomap_res = isomap(s, n_dim=2, n_neighbor=n_neighbor)
        torus_color_subplots(isomap_res, *torus, title=f"isomap, {n_neighbor}-nearest-neighbors")
    #
    # # 3.b.i
    # n_classes = [3, 5, 7]
    # digits_datasets = [load_digits(n_class=i) for i in n_classes]
    # n_neighbors = [10, 50, 100]
    # for i, ds in enumerate(digits_datasets):
    #     digits_ds_subplots(ds, n_neighbors, title=f"n_class={n_classes[i]}")

    # 3.b.ii
    full_digits_ds = load_digits(n_class=10)
    diffusion_embedding = diffusion_maps(full_digits_ds.data, n_dim=20, epsilon=64)
    # isomap_embedding = isomap(full_digits_ds.data, n_dim=2, n_neighbor=10)
    clf = SVC()
    cv_scores = cross_val_score(clf, diffusion_embedding, full_digits_ds.target, cv=5)
    print(cv_scores)


if __name__ == "__main__":
    main()
