import imageio
import numpy as np
import pyvista as pv


def update_step(img, i, j, dt):
    laplacian = img[i + 1, j] + img[i - 1, j] + img[i, j + 1] + img[i, j - 1] - 4 * img[i, j]
    pix_update = img[i, j] + dt * laplacian
    return pix_update


def update_image(img, dt):
    img_update = np.zeros_like(img)
    height, width = np.shape(img)[:2]
    for i in range(1, height-1):
        for j in range(1, width-1):
            img_update[i, j] = update_step(img, i, j, dt)
    return img_update


def create_heat_kernel(xx, yy, t):
    kernel = np.exp(-np.expand_dims(t, axis=(1, 2))*(np.sqrt(xx**2 + yy**2) ** 2))
    return kernel


def get_fourier_transform(u):
    x = np.fft.fft2(u)
    f_u = np.fft.fftshift(x)
    return f_u


def get_inverse_fourier_transform(f_u):
    x = np.fft.ifftshift(f_u)
    u = np.fft.ifft2(x)
    u_mag = np.abs(u)**2
    return u_mag


def main():
    # load and pad cameraman image
    img = imageio.imread('cameraman.tif') / 255.
    img_pad = np.pad(img, 1)
    # define grid for surface plot and heat kernel
    height, width = np.shape(img_pad)[:2]
    x, y = range(height), range(width)
    xx, yy = np.meshgrid(x, y)
    img_to_show = (255. * img_pad).astype('uint8')
    num_iter = 30

    # Section (b): explicit form
    img_grid = pv.StructuredGrid(xx, yy, img_to_show)
    plotter = pv.Plotter()
    plotter.add_mesh(img_grid, scalars=np.transpose(img_to_show).ravel())
    print('Orient the view, then press "q" to close window and produce GIF')
    plotter.show(auto_close=False)
    plotter.open_gif('surface_explicit.gif')
    pts = img_grid.points.copy()
    curr_img = img_pad.copy()
    images_in_time = []
    for _ in range(num_iter):
        next_img = update_image(curr_img, dt=0.1)
        next_img_to_show = (255. * next_img).astype('uint8')
        pts[:, -1] = np.transpose(next_img_to_show).ravel()
        plotter.update_coordinates(pts, render=False)
        plotter.update_scalars(np.transpose(next_img_to_show).ravel(), render=False)
        plotter.render()
        plotter.write_frame()
        images_in_time.append(next_img_to_show)
        curr_img = next_img
    imageio.mimsave('image_explicit.gif', images_in_time, duration=0.5)
    plotter.close()

    # Section (b): heat kernel
    f_img = get_fourier_transform(img_pad)
    t = np.linspace(0, 0.01, num_iter)
    center_x, center_y = np.shape(img_pad) * np.array([0.5, 0.5])
    a = np.array(range(height)) - int(center_x)
    b = np.array(range(width)) - int(center_y)
    xx, yy = np.meshgrid(a, b)
    f_gaussian = create_heat_kernel(xx, yy, t)
    f_sol = np.multiply(f_img, f_gaussian)
    sol = get_inverse_fourier_transform(f_sol)[num_iter//2:]
    imageio.mimsave('image_heat_kernel.gif', sol, duration=0.5)


if __name__ == "__main__":
    main()
