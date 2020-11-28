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
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    if save:
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(os.path.join(img_dir, name), exist_ok=True)
        file_path = os.path.join(img_dir, name, f"{title}.png")
        plt.savefig(file_path)
        logging.info(f"saved image to {file_path}")
        plt.close()
    else:
        plt.show()


def plot_curvature_signatures(k_orig, k_transformed):
    plt.plot(k_orig, label='original')
    plt.plot(k_transformed, label='transformed')
    plt.legend()
    plt.grid()
    plt.show()


def plot_cartan_signatures(c_orig, c_transformed):
    x_orig, y_orig = c_orig
    x_trans, y_trans = c_transformed
    plt.plot(x_orig, y_orig, label='original')
    plt.plot(x_trans, y_trans, label='transformed')
    plt.legend()
    plt.grid()
    plt.show()


def compute_curvature_flow(c):
    tangent = np.gradient(c, axis=1)
    tangent_norm = np.linalg.norm(tangent, axis=0)
    unit_tangent = tangent / tangent_norm
    # quarter-rotation counter-clockwise
    unit_normal = np.matmul(np.array([[0, -1], [1, 0]]), unit_tangent)
    x_t, y_t = tangent
    x_tt, y_tt = np.gradient(x_t), np.gradient(y_t)
    k = (y_tt * x_t - x_tt * y_t) / np.power(x_t ** 2 + y_t ** 2, 3 / 2)
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
    x_tt, y_tt = np.gradient(x_t), np.gradient(y_t)
    k = (y_tt * x_t - x_tt * y_t) / np.power(x_t ** 2 + y_t ** 2, 3 / 2)
    k = np.clip(k, a_min=np.percentile(k, 1), a_max=np.percentile(k, 99))
    k = filter_func(k, window_size=10)
    return k, unit_normal


def curvature_flow(c, smooth=False):
    if smooth:
        k, unit_normal = compute_curvature_flow_bonus(c)
        c_ss = k * unit_normal
    else:
        k, unit_normal = compute_curvature_flow(c)
        c_ss = k * unit_normal
    return c_ss


def rotate(c, theta):
    rot = np.array([[np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)]])
    rot_c = np.matmul(rot, c)
    return rot_c


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
    # for i, curve in enumerate(curves):
    #     curr_curve = curve.copy()
    #     for j in range(20):
    #         if j % iter_plot == 0:
    #             plot_curve(curr_curve, names[i], save=True, iteration=j)
    #         next_curve = curr_curve + dt * curvature_flow(curr_curve)
    #         curr_curve = next_curve

    # 1.(c)
    # for i, curve in enumerate(curves):
    #     curr_curve = curve.copy()
    #     for j in range(20):
    #         if j % iter_plot == 0:
    #             plot_curve(curr_curve, f"{names[i]}_smooth", save=True, iteration=j)
    #         next_curve = curr_curve + dt * curvature_flow(curr_curve, smooth=True)
    #         curr_curve = next_curve

    # 2.
    cardioid = utils.get_cardioid()
    curvature, _ = compute_curvature_flow_bonus(cardioid)
    # sample from a new starting point
    cardioid_new = utils.get_cardioid(start=-4)
    cardioid_trans = rotate(cardioid_new, theta=np.pi/2)
    curvature_trans, _ = compute_curvature_flow_bonus(cardioid_trans)
    plot_curvature_signatures(curvature, curvature_trans)

    # 3.
    curvature_grad = np.gradient(curvature)
    curvature_trans_grad = np.gradient(curvature_trans)
    sig = np.stack([curvature, curvature_grad], axis=0)
    sig_trans = np.stack([curvature_trans, curvature_trans_grad])
    plot_cartan_signatures(sig, sig_trans)


if __name__ == "__main__":
    main()
