import numpy as np


def power_method(B, num_iter):
    u_0 = np.random.rand(B.shape[1])
    u_t = u_0
    for _ in range(num_iter):
        u_t = np.dot(B, u_0)
        u_t = u_t / np.linalg.norm(u_t)
    vec = u_t
    value = vec.T.dot(B).dot(vec) / vec.T.dot(vec)
    return np.abs(vec), value


def np_eigen(B):
    eigenvalues, eigenvectors = np.linalg.eig(B)
    idx = eigenvalues.argsort()[::-1]
    eigenvalue_max = eigenvalues[idx][0]
    eigenvector_max = np.abs(eigenvectors[idx][0])
    return eigenvector_max, eigenvalue_max


def main():
    # b = np.random.randint(-2, 2, size=(3, 3))
    # B = (b + b.T)
    B = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    pm_vec, pm_val = power_method(B, num_iter=500)
    np_vec, np_val = np_eigen(B)
    print(pm_val)
    print(np_val)


if __name__ == "__main__":
    main()
