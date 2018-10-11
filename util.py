import matplotlib.pyplot as plt
import numpy as np
from pyfftw.interfaces.numpy_fft import fft2, ifft2
from pyfftw.interfaces.numpy_fft import fftshift as np_fftshift
from pyfftw.interfaces.numpy_fft import ifftshift as np_ifftshift
import tensorflow as tf
from scipy.misc import imresize
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift

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


def convert_cone_to_parallel(data, source_to_det_dist_cm, z_d_cm, psize=None, crop=False):
    z_d_cm = np.array(z_d_cm)
    z_s_cm = source_to_det_dist_cm - z_d_cm
    d_para_cm = z_s_cm * z_d_cm / source_to_det_dist_cm
    mag = source_to_det_dist_cm / z_s_cm
    print(mag)
    new_data = []

    if psize is not None:
        psize_norm = np.array(psize) / np.min(psize)
        ind_ref = np.argmin(psize)
        shape_ref = data[ind_ref].shape
        shape_ref_half = (np.array(shape_ref) / 2).astype('int')
        for i, img in enumerate(data):
            if i != ind_ref:
                zoom = psize_norm[i]
                img = imresize(img, zoom, interp='bilinear', mode='F')
                if crop:
                    center = (np.array(img.shape) / 2).astype('int')
                    img = img[center[0] - shape_ref_half[0]:center[0] - shape_ref_half[0] + shape_ref[0],
                          center[1] - shape_ref_half[1]:center[1] - shape_ref_half[1] + shape_ref[1]]
            new_data.append(img)
    else:
        # unify zooming of all images to the one with largest magnification
        mag_norm = mag / mag.max()
        print(mag_norm)
        ind_ref = np.argmax(mag_norm)
        shape_ref = data[ind_ref].shape
        shape_ref_half = (np.array(shape_ref) / 2).astype('int')
        for i, img in enumerate(data):
            if i != ind_ref:
                zoom = 1. / mag_norm[i]
                img = imresize(img, zoom, interp='bilinear', mode='F')
                if crop:
                    center = (np.array(img.shape) / 2).astype('int')
                    img = img[center[0] - shape_ref_half[0]:center[0] - shape_ref_half[0] + shape_ref[0],
                              center[1] - shape_ref_half[1]:center[1] - shape_ref_half[1] + shape_ref[1]]
            new_data.append(img)
    return new_data, d_para_cm


def realign_image(arr, shift, angle=0):
    """
    Translate and rotate image via Fourier

    Parameters
    ----------
    arr : ndarray
        Image array.

    shift: float
        Mininum and maximum values to rescale data.

    angle: float, optional
        Mininum and maximum values to rescale data.

    Returns
    -------
    ndarray
        Output array.
    """
    # if both shifts are integers, do circular shift; otherwise perform Fourier shift.
    if np.count_nonzero(np.abs(np.array(shift) - np.round(shift)) < 0.01) == 2:
        temp = np.roll(arr, int(shift[0]), axis=0)
        temp = np.roll(temp, int(shift[1]), axis=1)
        temp = temp.astype('float32')
    else:
        temp = fourier_shift(np.fft.fftn(arr), shift)
        temp = np.fft.ifftn(temp)
        temp = np.abs(temp).astype('float32')
    return temp


def register_data(data, ref_ind=0):
    ref = data[ref_ind]
    for i, img in enumerate(data):
        if i != ref_ind:
            shift, _, _ = register_translation(img, ref, upsample_factor=100)
            print(shift)
            img = realign_image(img, shift)
            data[i] = img
    return data


def shift_data(data, shifts, ref_ind=0):
    for i, img in enumerate(data):
        if i != ref_ind:
            shift = shifts[i]
            img = realign_image(img, shift)
            data[i] = img
    return data


def fourier_shift_tf(arr, shift, image_shape):

    wy = np.fft.fftfreq(image_shape[0])
    wx = np.fft.fftfreq(image_shape[1])
    wxx, wyy = np.meshgrid(wx, wy)
    w = np.zeros([image_shape[0], image_shape[1], 2], dtype='float32')
    w[:, :, 0] = wyy
    w[:, :, 1] = wxx
    k = tf.reduce_sum(tf.constant(w) * shift, axis=2)
    k = tf.exp(-1j * 2 * PI * tf.cast(k, tf.complex64))
    res = tf.ifft2d(tf.fft2d(tf.cast(arr, tf.complex64)) * k)
    return tf.abs(res)


def rescale_image(arr, m, original_shape):

    arr_shape = tf.cast(arr.shape, tf.float32)
    y_newlen = arr_shape[0] / m
    x_newlen = arr_shape[1] / m
    # tf.linspace shouldn't be used since it does not support gradient
    y = tf.range(0, arr_shape[0], 1, dtype=tf.float32)
    y = y / m + (original_shape[0] - y_newlen) / 2.
    x = tf.range(0, arr_shape[1], 1, dtype=tf.float32)
    x = x / m + (original_shape[1] - x_newlen) / 2.
    # y = tf.linspace((original_shape[0] - y_newlen) / 2., (original_shape[0] + y_newlen) / 2. - 1, arr.shape[0])
    # x = tf.linspace((original_shape[1] - x_newlen) / 2., (original_shape[1] + x_newlen) / 2. - 1, arr.shape[1])
    y = tf.clip_by_value(y, 0, arr_shape[0])
    x = tf.clip_by_value(x, 0, arr_shape[1])
    x_resample, y_resample = tf.meshgrid(x, y, indexing='ij')
    warp = tf.transpose(tf.stack([x_resample, y_resample]))
    # warp = tf.transpose(tf.stack([tf.reshape(y_resample, (np.prod(original_shape), )), tf.reshape(x_resample, (np.prod(original_shape), ))]))
    # warp = tf.cast(warp, tf.int32)
    # arr = arr * tf.reshape(warp[:, 0], original_shape)
    # arr = tf.gather_nd(arr, warp)
    warp = tf.expand_dims(warp, 0)
    arr = tf.contrib.resampler.resampler(tf.expand_dims(tf.expand_dims(arr, 0), -1), warp)
    arr = tf.reshape(arr, original_shape)

    return arr


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
