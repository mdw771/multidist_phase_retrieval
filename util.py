import matplotlib.pyplot as plt
import numpy as np
from pyfftw.interfaces.numpy_fft import fft2, ifft2
from pyfftw.interfaces.numpy_fft import fftshift as np_fftshift
from pyfftw.interfaces.numpy_fft import ifftshift as np_ifftshift
import tensorflow as tf

PI = 3.1415927


def plot(arr):
    if arr.dtype in [np.complex64, np.complex128]:
        arr = np.abs(arr)
    plt.imshow(arr)
    plt.show()
    return


def get_kernel(dist_nm, lmbda_nm, voxel_nm, grid_shape):
    """Get Fresnel propagation kernel for TF algorithm.

    Parameters:
    -----------
    simulator : :class:`acquisition.Simulator`
        The Simulator object.
    dist : float
        Propagation distance in cm.
    """
    # k = 2 * PI / lmbda_nm
    u_max = 1. / (2. * voxel_nm[0])
    v_max = 1. / (2. * voxel_nm[1])
    u, v = gen_mesh([v_max, u_max], grid_shape[0:2])
    # H = np.exp(1j * k * dist_nm * np.sqrt(1 - lmbda_nm**2 * (u**2 + v**2)))
    H = np.exp(-1j * PI * lmbda_nm * dist_nm * (u ** 2 + v ** 2))

    return H


def get_kernel_ir(dist_nm, lmbda_nm, voxel_nm, grid_shape):
    """
    Get Fresnel propagation kernel for IR algorithm.

    Parameters:
    -----------
    simulator : :class:`acquisition.Simulator`
        The Simulator object.
    dist : float
        Propagation distance in cm.
    """
    size_nm = np.array(voxel_nm) * np.array(grid_shape)
    k = 2 * PI / lmbda_nm
    ymin, xmin = np.array(size_nm)[:2] / -2.
    x = np.linspace(xmin, -xmin, grid_shape[1])
    y = np.linspace(ymin, -ymin, grid_shape[0])
    x, y = np.meshgrid(x, y)
    h = np.exp(1j * k * dist_nm) / (1j * lmbda_nm * dist_nm) * np.exp(1j * k / (2. * dist_nm) * (x ** 2 + y ** 2))
    H = np_fftshift(fft2(h)) * voxel_nm[0] * voxel_nm[1]

    return H


def gen_mesh(max, shape):
    """Generate mesh grid.
    """
    yy = np.linspace(-max[0], max[0], shape[0])
    xx = np.linspace(-max[1], max[1], shape[1])
    res = np.meshgrid(xx, yy)
    return res


def fftshift(tensor):
    ndim = len(tensor.shape)
    dim_ls = range(ndim - 2, ndim)
    for i in dim_ls:
        n = tensor.shape[i].value
        p2 = (n + 1) // 2
        begin1 = [0] * ndim
        begin1[i] = p2
        size1 = tensor.shape.as_list()
        size1[i] = size1[i] - p2
        begin2 = [0] * ndim
        size2 = tensor.shape.as_list()
        size2[i] = p2
        t1 = tf.slice(tensor, begin1, size1)
        t2 = tf.slice(tensor, begin2, size2)
        tensor = tf.concat([t1, t2], axis=i)
    return tensor


def ifftshift(tensor):
    ndim = len(tensor.shape)
    dim_ls = range(ndim - 2, ndim)
    for i in dim_ls:
        n = tensor.shape[i].value
        p2 = n - (n + 1) // 2
        begin1 = [0] * ndim
        begin1[i] = p2
        size1 = tensor.shape.as_list()
        size1[i] = size1[i] - p2
        begin2 = [0] * ndim
        size2 = tensor.shape.as_list()
        size2[i] = p2
        t1 = tf.slice(tensor, begin1, size1)
        t2 = tf.slice(tensor, begin2, size2)
        tensor = tf.concat([t1, t2], axis=i)
    return tensor


def fresnel_propagate_numpy(wavefront, energy_ev, psize_cm, dist_cm):
    lmbda_nm = 1240. / energy_ev
    lmbda_cm = 0.000124 / energy_ev
    psize_nm = psize_cm * 1e7
    dist_nm = dist_cm * 1e7

    if dist_cm == 'inf':
        wavefront = np_fftshift(fft2(wavefront))
    else:
        n = np.mean(wavefront.shape)
        z_crit_cm = (psize_cm * n) ** 2 / (lmbda_cm * n)
        algorithm = 'TF' if dist_cm < z_crit_cm else 'IR'
        if algorithm == 'TF':
            h = get_kernel(dist_nm, lmbda_nm, [psize_nm, psize_nm], wavefront.shape)
            wavefront = ifft2(np_ifftshift(np_fftshift(fft2(wavefront)) * h))
        else:
            h = get_kernel_ir(dist_nm, lmbda_nm, [psize_nm, psize_nm], wavefront.shape)
            wavefront = np_ifftshift(ifft2(np_fftshift(fft2(wavefront)) * h))

    return wavefront


def fresnel_propagate(wavefront_real, wavefront_imag, energy_ev, psize_cm, dist_cm):
    lmbda_nm = 1240. / energy_ev
    lmbda_cm = 0.000124 / energy_ev
    psize_nm = psize_cm * 1e7
    dist_nm = dist_cm * 1e7

    wavefront = wavefront_real + 1j * wavefront_imag
    wave_shape = wavefront.get_shape().as_list()

    if dist_cm == 'inf':
        wavefront = fftshift(tf.fft2d(wavefront))
    else:
        n = np.mean(wave_shape)
        z_crit_cm = (psize_cm * n) ** 2 / (lmbda_cm * n)
        algorithm = 'TF' if dist_cm < z_crit_cm else 'IR'
        if algorithm == 'TF':
            h = get_kernel(dist_nm, lmbda_nm, [psize_nm, psize_nm], wave_shape)
            h = tf.convert_to_tensor(h, dtype=tf.complex64)
            wavefront = tf.ifft2d(ifftshift(fftshift(tf.fft2d(wavefront)) * h))
        else:
            h = get_kernel_ir(dist_nm, lmbda_nm, [psize_nm, psize_nm], wave_shape)
            h = tf.convert_to_tensor(h, dtype=tf.complex64)
            wavefront = ifftshift(tf.ifft2d(fftshift(tf.fft2d(wavefront)) * h))

    return wavefront


def image_entropy(arr, multiplier=1, vmin=None, vmax=None):
    if vmin is not None and vmax is not None:
        arr = tf.clip_by_value(arr, vmin, vmax)
    arr = arr * multiplier
    arr = tf.cast(arr, tf.int32)
    hist = tf.bincount(arr)
    hist /= tf.reduce_sum(hist)
    loghist = tf.log(hist)
    histnotnan = tf.is_finite(loghist)
    s = -tf.reduce_sum(tf.boolean_mask(hist, histnotnan) * tf.boolean_mask(loghist, histnotnan))
    return tf.cast(s, tf.float32)


def gaussian_blur(arr, size, sigma):
    kernel = get_gaussian_kernel(size, sigma)
    arr = tf.expand_dims(tf.expand_dims(arr, 0), -1)
    kernel = tf.expand_dims(tf.expand_dims(kernel, -1), -1)
    arr = tf.nn.conv2d(arr, kernel, [1, 1, 1, 1], 'SAME')
    arr = tf.squeeze(arr)
    return arr


def get_gaussian_kernel(size, sigma):
    xmin = (size - 1) / 2.
    x = np.linspace(-xmin, xmin, size)
    xx, yy = np.meshgrid(x, x)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel.astype('float32')

if __name__ == '__main__':

    import dxchange
    a = dxchange.read_tiff('data/cameraman_512_dp.tiff')
    a = tf.constant(a)
    sess = tf.Session()
    a = sess.run(gaussian_blur(a, 5, 2) - a)
    plt.imshow(a)
    plt.show()
    # print(sess.run(image_entropy(a)))
    # print(get_gaussian_kernel(5, sigma=2))
