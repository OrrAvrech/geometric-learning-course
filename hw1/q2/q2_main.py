import os
import logging
import numpy as np
import q2.utils as utils
import matplotlib.pyplot as plt


def plot_curve(c, name='', save=False, img_dir='images', iteration=None):
    x, y = c
    plt.plot(x, y)
    plt.grid()
    title = name
    if iteration:
        title = f"{title}_{iteration}"
    plt.title(title)
    if save:
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(os.path.join(img_dir, name), exist_ok=True)
        file_path = os.path.join(img_dir, name, f"{title}.png")
        plt.savefig(file_path)
        logging.info(f"saved image to {file_path}")
        plt.close()
    else:
        plt.show()


def plot_kappa_values(c, k, num_max_indices):
    x, y = c
    max_indices = np.argsort(k)[-num_max_indices:]

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(x, y)
    ax[0].plot(x[max_indices], y[max_indices], 'ro')
    ax[0].title.set_text('C(t)')
    ax[1].plot(k)
    ax[1].plot(max_indices, k[max_indices], 'ro')
    ax[1].title.set_text('$\kappa(s)$')
    plt.show()


def plot_curvature_signatures(k_orig, k_transformed, name='', save=False, img_dir='images'):
    plt.plot(k_orig, label='original')
    plt.plot(k_transformed, label='transformed')
    plt.legend()
    plt.grid()
    if save:
        os.makedirs(img_dir, exist_ok=True)
        file_path = os.path.join(img_dir, f"{name}.png")
        plt.savefig(file_path)
        logging.info(f"saved image to {file_path}")
        plt.close()
    else:
        plt.show()


def plot_cartan_signatures(c_orig, c_transformed, name='', save=False, img_dir='images'):
    k_orig, ks_orig = c_orig
    k_trans, ks_trans = c_transformed
    plt.plot(k_orig, ks_orig, label='original')
    plt.plot(k_trans, ks_trans, label='transformed')
    plt.legend()
    plt.grid()
    if save:
        os.makedirs(img_dir, exist_ok=True)
        file_path = os.path.join(img_dir, f"{name}.png")
        plt.savefig(file_path)
        logging.info(f"saved image to {file_path}")
        plt.close()
    else:
        plt.show()


def compute_curvature_flow(c):
    tangent = np.gradient(c, axis=1)
    tangent_norm = np.linalg.norm(tangent, axis=0)
    unit_tangent = tangent / tangent_norm
    # quarter-rotation counter-clockwise
    unit_normal = np.matmul(np.array([[0, -1], [1, 0]]), unit_tangent)
    x_t, y_t = tangent
    x_tt, y_tt = np.gradient(x_t), np.gradient(y_t)
    k = (y_tt * x_t - x_tt * y_t) / np.power(x_t**2 + y_t**2, 3 / 2)
    return k, unit_normal


def filter_func(f, window_size=5):
    window = np.ones(window_size)/window_size
    f_smooth = np.convolve(f, window, mode='valid')
    f[window_size//2:-window_size//2+1] = f_smooth
    return f


def compute_curvature_flow_bonus(c):
    c = np.stack([filter_func(f) for f in c], axis=0)
    tangent = np.gradient(c, axis=1)
    tangent_norm = np.linalg.norm(tangent, axis=0)
    unit_tangent = tangent / tangent_norm
    # quarter-rotation counter-clockwise
    unit_normal = np.matmul(np.array([[0, -1], [1, 0]]), unit_tangent)
    x_t, y_t = tangent
    x_tt, y_tt = np.gradient(tangent, axis=1)
    k = (y_tt * x_t - x_tt * y_t) / np.power(x_t**2 + y_t**2, 3 / 2)
    k = np.clip(k, a_min=np.percentile(k, 10), a_max=np.percentile(k, 90))
    return k, unit_normal


def curvature_flow(c, smooth=False):
    if smooth:
        k, unit_normal = compute_curvature_flow_bonus(c)
    else:
        k, unit_normal = compute_curvature_flow(c)
    c_ss = k * unit_normal
    return c_ss


def rotate(c, theta):
    rot = np.array([[np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)]])
    rot_c = np.matmul(rot, c)
    return rot_c


def get_arc_length(c):
    x_diff, y_diff = np.diff(c, axis=1)
    arc_length = np.sum(np.hypot(x_diff, y_diff))
    return arc_length


def main():

    logging.basicConfig(level=logging.INFO)
    # 1.(a)
    astroid = utils.get_astroid()
    folium_of_descrates = utils.get_folium_of_descrates()
    talbots = utils.get_talbots()
    witch_of_agensi = utils.get_witch_of_agensi()

    curves = [astroid, folium_of_descrates, talbots, witch_of_agensi]
    names = ['astroid', 'folium_of_descrates', 'talbots', 'witch_of_agensi']
    dt = 0.1
    iter_plot = 4
    # 1.(b)
    for i, curve in enumerate(curves):
        curr_curve = curve.copy()
        for j in range(20):
            if j % iter_plot == 0:
                plot_curve(curr_curve, names[i], save=True, iteration=j)
            next_curve = curr_curve + dt * curvature_flow(curr_curve)
            curr_curve = next_curve

    # 1.(c)
    dt = 1e-3
    iter_plot = 1000
    for i, curve in enumerate(curves):
        curr_curve = curve.copy()
        curve_arc_lengths = []
        for j in range(5000):
            if j % iter_plot == 0:
                plot_curve(curr_curve, f"{names[i]}_smooth", save=True, iteration=j)
            arc_len = get_arc_length(curr_curve)
            curve_arc_lengths.append(arc_len)
            next_curve = curr_curve + dt * curvature_flow(curr_curve, smooth=True)
            curr_curve = next_curve
        plt.plot(curve_arc_lengths)
        plt.title('Arc-length vs. Iterations')
        plt.show()

    # 2.
    cardioid = utils.get_cardioid()
    curvature, _ = compute_curvature_flow_bonus(cardioid)
    # sample from a new starting point
    cardioid_new = utils.get_cardioid(start=-2*np.pi)
    cardioid_trans = rotate(cardioid_new, theta=np.pi/2)
    curvature_trans, _ = compute_curvature_flow_bonus(cardioid_trans)
    plot_curvature_signatures(curvature, curvature_trans, save=False, name='curvature_signatures')

    # 3.
    curvature_grad = np.gradient(curvature)
    curvature_trans_grad = np.gradient(curvature_trans)
    sig = np.stack([curvature, curvature_grad], axis=0)
    sig_trans = np.stack([curvature_trans, curvature_trans_grad], axis=0)
    plot_cartan_signatures(sig, sig_trans, save=False, name='cartan_signatures')


if __name__ == "__main__":
    main()
