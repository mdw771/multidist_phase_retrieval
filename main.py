import tensorflow as tf
from pyfftw.interfaces.numpy_fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
import dxchange
import numpy as np
import os
import time

from util import *


def retrieve_phase_near_field(data, save_path, energy_ev, dist_cm_ls, psize_cm,
                              output_fname=None, pad_length=18, n_epoch=100, learning_rate=0.001,
                              gamma=1.):

    prj_np_ls = data

    if output_fname is None:
        output_fname = 'recon'

    # take modulus and inverse shift
    prj_np_ls = np.sqrt(prj_np_ls)
    prj_shape = list(prj_np_ls.shape[1:])
    half_prj_shape = [int(x / 2) for x in prj_shape]

    # get first estimation from direct backpropagation
    prj_back = np.zeros(prj_shape, dtype='complex64')
    for i, dist_cm in enumerate(dist_cm_ls):
        prj_back += fresnel_propagate_numpy(prj_np_ls[i], energy_ev, psize_cm, -dist_cm)
    prj_back /= len(dist_cm_ls)
    obj_init = np.zeros(prj_shape + [2])
    obj_init[:, :, 0] = prj_back.real
    obj_init[:, :, 1] = prj_back.imag


    obj_init[:, :, 0] = np.random.normal(0.7, 0.1, prj_shape)
    obj_init[:, :, 1] = np.random.normal(0.8, 0.1, prj_shape)



    obj = tf.Variable(obj_init, dtype=tf.float32, name='obj')
    obj_real = tf.cast(obj[:, :, 0], dtype=tf.complex64)
    obj_imag = tf.cast(obj[:, :, 1], dtype=tf.complex64)
    prj_ls = tf.constant(prj_np_ls, name='prj')

    obj_pad = tf.pad(obj, [[pad_length, pad_length], [pad_length, pad_length], [0, 0]], mode='SYMMETRIC')
    obj_real_pad = tf.cast(obj_pad[:, :, 0], dtype=tf.complex64)
    obj_imag_pad = tf.cast(obj_pad[:, :, 1], dtype=tf.complex64)

    center = [int((x + 2 * pad_length) / 2) for x in prj_shape]
    half_probe_size = [int(x / 2) for x in prj_shape]
    quat_probe_size = [int(x / 2) for x in half_probe_size]
    obj_norm = tf.norm(obj, axis=2)

    loss = tf.constant(0, dtype=tf.float32)
    reg_term = tf.constant(0, dtype=tf.float32)
    for i, dist_cm in enumerate(dist_cm_ls):
        det = fresnel_propagate(obj_real_pad, obj_imag_pad, energy_ev, psize_cm, dist_cm)
        # remove padded margins
        det = det[center[0] - half_probe_size[0]:center[0] - half_probe_size[0] + prj_np_ls[0].shape[0],
                  center[1] - half_probe_size[1]:center[1] - half_probe_size[1] + prj_np_ls[0].shape[1]]
        loss += tf.reduce_mean(tf.squared_difference(tf.abs(det), prj_ls[i], name='loss'))
    loss /= len(dist_cm_ls)

    sess = tf.Session()

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = optimizer.minimize(loss)

    sess.run(tf.global_variables_initializer())

    for i_epoch in range(n_epoch):
        t0 = time.time()
        _, current_loss, current_reg = sess.run([optimizer, loss, reg_term])
        print('Iteration {}: loss = {}, reg = {}, Δt = {} s.'.format(i_epoch, current_loss, current_reg, time.time() - t0))

    # det_final = sess.run(det)
    obj_final = sess.run(obj)
    obj_final = obj_final[:, :, 0] + 1j * obj_final[:, :, 1]
    dxchange.write_tiff(np.abs(obj_final), os.path.join(save_path, output_fname + '_mag'), dtype='float32', overwrite=True)
    dxchange.write_tiff(np.angle(obj_final), os.path.join(save_path, output_fname + '_phase'), dtype='float32', overwrite=True)
    # dxchange.write_tiff(fftshift(np.angle(det_final)), os.path.join(save_path, 'detector_phase'), dtype='float32', overwrite=True)
    # dxchange.write_tiff(fftshift(np.abs(det_final) ** 2), os.path.join(save_path, 'detector_mag'), dtype='float32', overwrite=True)

    return


def retrieve_phase_far_field(src_fname, save_path, output_fname=None, pad_length=256, n_epoch=100, learning_rate=0.001):

    # raw data is assumed to be centered at zero frequency
    prj_np = dxchange.read_tiff(src_fname)
    if output_fname is None:
        output_fname = os.path.basename(os.path.splitext(src_fname)[0]) + '_recon'

    # take modulus and inverse shift
    prj_np = ifftshift(np.sqrt(prj_np))

    obj_init = np.random.normal(50, 10, list(prj_np.shape) + [2])

    obj = tf.Variable(obj_init, dtype=tf.float32, name='obj')
    prj = tf.constant(prj_np, name='prj')

    obj_real = tf.cast(obj[:, :, 0], dtype=tf.complex64)
    obj_imag = tf.cast(obj[:, :, 1], dtype=tf.complex64)

    # obj_pad = tf.pad(obj, [[pad_length, pad_length], [pad_length, pad_length], [0, 0]], mode='SYMMETRIC')
    det = tf.fft2d(obj_real + 1j * obj_imag, name='detector_plane')

    loss = tf.reduce_mean(tf.squared_difference(tf.abs(det), prj, name='loss'))

    sess = tf.Session()

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = optimizer.minimize(loss)

    sess.run(tf.global_variables_initializer())

    for i_epoch in range(n_epoch):
        t0 = time.time()
        _, current_loss = sess.run([optimizer, loss])
        print('Iteration {}: loss = {}, Δt = {} s.'.format(i_epoch, current_loss, time.time() - t0))

    det_final = sess.run(det)
    obj_final = sess.run(obj)
    res = np.linalg.norm(obj_final, 2, axis=2)
    dxchange.write_tiff(res, os.path.join(save_path, output_fname), dtype='float32', overwrite=True)
    dxchange.write_tiff(fftshift(np.angle(det_final)), os.path.join(save_path, 'detector_phase'), dtype='float32', overwrite=True)
    dxchange.write_tiff(fftshift(np.abs(det_final) ** 2), os.path.join(save_path, 'detector_mag'), dtype='float32', overwrite=True)

    return


if __name__ == '__main__':

    params_cameraman = {'src_fname_root': 'data/cameraman_512_dp',
                        'save_path': 'data',
                        'output_fname': None,
                        'pad_length': 18,
                        'n_epoch': 100,
                        'learning_rate': 1,
                        'energy_ev': 25000,
                        'psize_cm': 1e-4,
                        'dist_cm_ls': [40, 60, 80, 100],
                        'gamma': 1}

    params_brain = {'save_path': 'data/vincent/test',
                    'output_fname': None,
                    'pad_length': 18,
                    'n_epoch': 50,
                    'learning_rate': 0.01,
                    'energy_ev': 17500,
                    'psize_cm': 0.1e-4,
                    'dist_cm_ls': [47.4848, 47.3848, 46.9848, 45.9847],
                    'gamma': 1}

    params = params_brain

    prj0 = np.squeeze(dxchange.read_tiff('data/vincent/test/proj_0.tiff'))
    prj1 = np.squeeze(dxchange.read_tiff('data/vincent/test/proj_1.tiff'))
    prj2 = np.squeeze(dxchange.read_tiff('data/vincent/test/proj_2.tiff'))
    prj3 = np.squeeze(dxchange.read_tiff('data/vincent/test/proj_3.tiff'))
    data = [prj0, prj1, prj2, prj3]

    retrieve_phase_near_field(data=data,
                              save_path=params['save_path'],
                              energy_ev=params['energy_ev'],
                              dist_cm_ls=params['dist_cm_ls'],
                              psize_cm=params['psize_cm'],
                              output_fname=params['output_fname'],
                              pad_length=params['pad_length'],
                              n_epoch=params['n_epoch'],
                              learning_rate=params['learning_rate'],
                              gamma=params['gamma'])
