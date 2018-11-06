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
                              gamma=1., phase_limit=None, cpu_only=False, allow_shift=False,
                              allow_scaling=False, allow_intensity_scaling=False, allow_refine_distances=True,
                              registration_iter_limit=200, fin_sup_mask=None):

    prj_np_ls = data

    if output_fname is None:
        output_fname = 'recon'

    # take modulus and inverse shift
    prj_np_ls = np.sqrt(prj_np_ls)
    prj_shape = list(prj_np_ls.shape[1:])

    # get first estimation from direct backpropagation
    n_dists = len(dist_cm_ls)
    prj_back = np.zeros(prj_shape, dtype='complex64')
    for i, dist_cm in enumerate(dist_cm_ls):
        prj_back += fresnel_propagate_numpy(prj_np_ls[i], energy_ev, psize_cm, -dist_cm)
    prj_back /= n_dists
    obj_init = np.zeros(prj_shape + [2])
    obj_init[:, :, 0] = prj_back.real
    obj_init[:, :, 1] = prj_back.imag


    obj_init[:, :, 0] = np.random.normal(0.7, 0.1, prj_shape)
    obj_init[:, :, 1] = np.random.normal(0.8, 0.1, prj_shape)

    if allow_shift:
        prj_shift = tf.Variable(np.zeros([n_dists, 2]), dtype=tf.float32)
        # prj_shift = tf.constant([[4.828, 0.466], [-0.560, -1.880], [-1.040, -3.100], [-6.229, 8.523]])
    else:
        prj_shift = tf.zeros([n_dists, 2], dtype=tf.int32)
    if allow_scaling:
        prj_scaling = tf.Variable(np.ones(len(dist_cm_ls)), dtype=tf.float32)
        # prj_scaling = tf.constant([ 1.00439858, 1.0057919, 1.00345397, 1.01061535])
    else:
        prj_scaling = tf.ones(n_dists, tf.float32)
    if allow_intensity_scaling:
        prj_inten = tf.Variable(np.ones(n_dists), dtype=tf.float32)
    else:
        prj_inten = tf.ones(n_dists, tf.float32)
    if allow_refine_distances:
        dist_cm_ls = tf.Variable(dist_cm_ls)

    if fin_sup_mask is not None:
        # fin_sup_mask = fin_sup_mask[:, :, np.newaxis]
        # fin_sup_mask = np.tile(fin_sup_mask, [1, 1, 2])
        fin_sup_mask = tf.constant(fin_sup_mask, dtype=tf.float32)

    obj_real = tf.Variable(obj_init[:, :, 0], dtype=tf.float32, name='obj')
    obj_imag = tf.Variable(obj_init[:, :, 1], dtype=tf.float32, name='obj')
    if fin_sup_mask is not None:
        obj_real_masked = (obj_real - 1.) * fin_sup_mask + 1.
        obj_imag_masked = obj_imag * fin_sup_mask
    else:
        obj_real_masked = obj_real
        obj_imag_masked = obj_imag
    obj_real_complex = tf.cast(obj_real_masked, dtype=tf.complex64)
    obj_imag_complex = tf.cast(obj_imag_masked, dtype=tf.complex64)
    prj_ls = tf.constant(prj_np_ls, name='prj')

    obj_real_pad = tf.pad(obj_real_complex, [[pad_length, pad_length], [pad_length, pad_length]], mode='SYMMETRIC')
    obj_imag_pad = tf.pad(obj_imag_complex, [[pad_length, pad_length], [pad_length, pad_length]], mode='SYMMETRIC')

    center = [int((x + 2 * pad_length) / 2) for x in prj_shape]
    half_probe_size = [int(x / 2) for x in prj_shape]

    loss = tf.constant(0, dtype=tf.float32)
    reg_term = tf.constant(0, dtype=tf.float32)
    modified_prj_ls = [None] * n_dists
    for i in range(n_dists):
        dist_cm = dist_cm_ls[i]
        det = fresnel_propagate(obj_real_pad, obj_imag_pad, energy_ev, psize_cm, dist_cm)
        # remove padded margins
        det = det[center[0] - half_probe_size[0]:center[0] - half_probe_size[0] + prj_np_ls[0].shape[0],
                  center[1] - half_probe_size[1]:center[1] - half_probe_size[1] + prj_np_ls[0].shape[1]]
        this_prj = prj_ls[i]
        if allow_scaling:
            m = prj_scaling[i]
            this_prj = rescale_image(this_prj, m, prj_shape)
        if allow_shift:
            this_prj = fourier_shift_tf(this_prj, prj_shift[i], image_shape=prj_shape)
        if allow_intensity_scaling:
            this_prj = this_prj * prj_inten[i]
        modified_prj_ls[i] = this_prj
        loss += tf.reduce_mean(tf.squared_difference(tf.abs(det), this_prj, name='loss'))
    loss /= n_dists

    sess = tf.Session()

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = optimizer.minimize(loss, var_list=[obj_real, obj_imag])
    optimizer_ls = [optimizer]
    if allow_shift:
        optimizer_shift = tf.train.AdamOptimizer(learning_rate=0.5)
        optimizer_shift = optimizer_shift.minimize(loss, var_list=[prj_shift])
        optimizer_ls.append(optimizer_shift)
    if allow_scaling:
        optimizer_scaling = tf.train.AdamOptimizer(learning_rate=0.005)
        optimizer_scaling = optimizer_scaling.minimize(loss, var_list=[prj_scaling])
        optimizer_ls.append(optimizer_scaling)
    if allow_intensity_scaling:
        optimizer_inten = tf.train.AdamOptimizer(learning_rate=0.01)
        optimizer_inten = optimizer_inten.minimize(loss, var_list=[prj_inten])
        optimizer_ls.append(optimizer_inten)
    if allow_refine_distances:
        optimizer_dist = tf.train.AdamOptimizer(learning_rate=0.005)
        optimizer_dist = optimizer_dist.minimize(loss, var_list=[dist_cm_ls])
        optimizer_ls.append(optimizer_dist)

    sess.run(tf.global_variables_initializer())
    loss_ls = []

    for i_epoch in range(n_epoch):
        current_obj_real, current_obj_imag = sess.run([obj_real_complex, obj_imag_complex])
        dxchange.write_tiff(np.abs(current_obj_real + 1j * current_obj_imag), 'data/vincent/test/scan2/test/mag/mag_before', dtype='float32')
        dxchange.write_tiff(np.angle(current_obj_real + 1j * current_obj_imag), 'data/vincent/test/scan2/test/phase/phase_before', dtype='float32')
        t0 = time.time()
        if i_epoch < registration_iter_limit:
            runres = sess.run([*optimizer_ls, loss, reg_term, prj_shift, prj_scaling, prj_inten, dist_cm_ls, det])
            current_loss =runres[len(optimizer_ls)]
            current_reg =runres[len(optimizer_ls) + 1]
            current_shift =runres[len(optimizer_ls) + 2]
            current_scaling =runres[len(optimizer_ls) + 3]
            current_inten = runres[len(optimizer_ls) + 4]
            current_dist = runres[len(optimizer_ls) + 5]
            current_det = runres[len(optimizer_ls) + 6]
        else:
            _, current_loss, current_reg, current_shift, current_scaling, current_inten, transformed_prj = sess.run([optimizer, loss, reg_term, prj_shift, prj_scaling, prj_inten, dist_cm_ls])
        loss_ls.append(current_loss)
        # dxchange.write_tiff(transformed_prj, os.path.join(save_path, 'temp', 'trans_prj'), dtype='float32')
        print('Iteration {}: loss = {}, reg = {}, Δt = {} s.'.format(i_epoch, current_loss, current_reg, time.time() - t0))

        current_obj_real, current_obj_imag = sess.run([obj_real, obj_imag])
        dxchange.write_tiff(np.sqrt(current_obj_real ** 2 + current_obj_imag ** 2), 'data/vincent/test/scan2/test/mag/mag', dtype='float32')
        dxchange.write_tiff(np.angle(current_obj_real + 1j * current_obj_imag), 'data/vincent/test/scan2/test/phase/phase', dtype='float32')
        dxchange.write_tiff(current_det, 'data/vincent/test/scan2/test/det/det', dtype='float32')
        if allow_shift:
            print('Current shift: \n{}'.format(current_shift))
        if allow_scaling:
            print('Current scaling: \n{}'.format(current_scaling))
        if allow_intensity_scaling:
            print('Current intensity scaling: \n{}'.format(current_inten))
        if allow_refine_distances:
            print('Current refined distances: \n{}'.format(current_dist))

        if phase_limit:
            mag = tf.sqrt(obj_real ** 2 + obj_imag ** 2)
            obj_real = tf.clip_by_value(obj_real, mag * tf.cos(float(phase_limit)), mag)
            obj_imag = tf.clip_by_value(obj_imag, -mag * tf.sin(float(phase_limit)), mag * tf.sin(float(phase_limit)))

    # det_final = sess.run(det)
    obj_final_real, obj_final_imag = sess.run([obj_real, obj_imag])
    obj_final = obj_final_real + 1j * obj_final_imag
    dxchange.write_tiff(np.abs(obj_final), os.path.join(save_path, output_fname + '_mag'), dtype='float32', overwrite=True)
    dxchange.write_tiff(np.angle(obj_final), os.path.join(save_path, output_fname + '_phase'), dtype='float32', overwrite=True)

    np.save(os.path.join(save_path, 'loss'), loss_ls)

    det_final = sess.run(det)
    dxchange.write_tiff(np.angle(det_final), os.path.join(save_path, 'detector_phase'), dtype='float32', overwrite=True)
    dxchange.write_tiff(np.abs(det_final) ** 2, os.path.join(save_path, 'detector_mag'), dtype='float32', overwrite=True)

    # prj_final_ls = sess.run(modified_prj_ls)
    # dxchange.write_tiff(np.array(prj_final_ls), os.path.join(save_path, 'final_prj', 'prj'), dtype='float32')

    return


def retrieve_phase_near_field_er(data, save_path, energy_ev, dist_cm_ls, psize_cm,
                                  output_fname=None, pad_length=18, n_epoch=100, learning_rate=0.001,
                                  gamma=1., phase_limit=None, cpu_only=False, fin_sup_mask=None):

    prj_np_ls = data

    if output_fname is None:
        output_fname = 'recon'

    # take modulus and inverse shift
    prj_np_ls = np.sqrt(prj_np_ls)
    prj_shape = list(prj_np_ls.shape[1:])

    # get first estimation from direct backpropagation
    prj_back = np.zeros(prj_shape, dtype='complex64')
    for i, dist_cm in enumerate(dist_cm_ls):
        prj_back += fresnel_propagate_numpy(prj_np_ls[i], energy_ev, psize_cm, -dist_cm)
    prj_back /= len(dist_cm_ls)
    obj = np.zeros(prj_shape + [2])
    obj[:, :, 0] = prj_back.real
    obj[:, :, 1] = prj_back.imag

    # obj[:, :, 0] = np.random.normal(0.7, 0.1, prj_shape)
    # obj[:, :, 1] = np.random.normal(0.8, 0.1, prj_shape)

    wavefront = obj[:, :, 0] + 1j * obj[:, :, 1]
    wavefront = np.pad(wavefront, [[pad_length, pad_length], [pad_length, pad_length]], mode='symmetric')

    loss_ls = []

    for i_epoch in range(n_epoch):
        t0 = time.time()
        loss = []
        for i_dist, dist_cm in enumerate(dist_cm_ls):
            wavefront = fresnel_propagate_numpy(wavefront, energy_ev, psize_cm, dist_cm)
            wavefront = wavefront[pad_length:pad_length + prj_shape[0], pad_length:pad_length + prj_shape[1]]
            loss.append(np.mean((abs(wavefront) - prj_np_ls[i_dist]) ** 2))
            wavefront = wavefront / np.abs(wavefront) * prj_np_ls[i_dist]
            wavefront = np.pad(wavefront, [[pad_length, pad_length], [pad_length, pad_length]], mode='symmetric')
            wavefront = fresnel_propagate_numpy(wavefront, energy_ev, psize_cm, -dist_cm)
        if fin_sup_mask is not None:
            valid_region = wavefront[pad_length:pad_length + prj_shape[0], pad_length:pad_length + prj_shape[1]]
            valid_region = (np.real(valid_region) - 1) * fin_sup_mask + 1 + 1j * np.imag(valid_region) * fin_sup_mask
            wavefront[pad_length:pad_length + prj_shape[0], pad_length:pad_length + prj_shape[1]] = valid_region
        loss = np.mean(loss)
        loss_ls.append(loss)
        print('Iteration {}: loss = {}, Δt = {} s.'.format(i_epoch, loss, time.time() - t0))
    wavefront = wavefront[pad_length:pad_length + prj_shape[0], pad_length:pad_length + prj_shape[1]]

    dxchange.write_tiff(np.abs(wavefront), os.path.join(save_path, output_fname + '_mag'), dtype='float32', overwrite=True)
    dxchange.write_tiff(np.angle(wavefront), os.path.join(save_path, output_fname + '_phase'), dtype='float32', overwrite=True)

    np.save(os.path.join(save_path, 'loss'), loss_ls)

    # det_final = sess.run(det)
    # dxchange.write_tiff(np.angle(det_final), os.path.join(save_path, 'detector_phase'), dtype='float32', overwrite=True)
    # dxchange.write_tiff(np.abs(det_final) ** 2, os.path.join(save_path, 'detector_mag'), dtype='float32', overwrite=True)

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

    fin_sup_mask_scan2 = dxchange.read_tiff('data/vincent/test/scan2/mask.tiff')

    params_cameraman = {'save_path': 'data/cameraman',
                        'output_fname': None,
                        'pad_length': 18,
                        'n_epoch': 200,
                        'learning_rate': 0.01,
                        'energy_ev': 17500.,
                        'psize_cm': 1e-4,
                        'dist_cm_ls': [40, 60, 80, 100],
                        'fin_sup_mask': None,
                        'gamma': 1,
                        'allow_shift': True,
                        'allow_scaling': True,
                        'allow_intensity_scaling': True,
                        'allow_refine_distances': True}

    params_brain_scan2 = {'save_path': 'data/vincent/test/scan2',
                          'output_fname': None,
                          'pad_length': 18,
                          'n_epoch': 200,
                          'learning_rate': 0.01,
                          'energy_ev': 17500,
                          'psize_cm': 99.8e-7,
                          'dist_cm_ls': [7.39, 7.46, 7.74, 8.39],
                          'fin_sup_mask': fin_sup_mask_scan2,
                          # 'dist_cm_ls': [7.291, 3.976],
                          # 'dist_cm_ls': [11.04, 7.29, 3.97],
                          'gamma': 1,
                          'allow_shift': True,
                          'allow_scaling': True,
                          'allow_intensity_scaling': True,
                          'allow_refine_distances': True}

    params_brain_comb = {'save_path': 'data/vincent/test/bin',
                         'output_fname': None,
                         'pad_length': 18,
                         'n_epoch': 400,
                         'learning_rate': 0.01,
                         'energy_ev': 17500,
                         'psize_cm': 49.9e-7,
                         # 'psize_cm': 99.8e-7,
                         # 'dist_cm_ls': [7.29, 7.36, 7.66, 8.28],
                         # 'dist_cm_ls': [7.291, 3.976],
                         'dist_cm_ls': [11.04, 7.29, 3.97],
                         'fin_sup_mask': fin_sup_mask_scan2,
                         'gamma': 1,
                         'allow_shift': True,
                         'allow_scaling': True,
                         'allow_intensity_scaling': True,
                         'allow_refine_distances': True}

    params_32id = {'save_path': 'data/32id',
                   'output_fname': None,
                   'pad_length': 18,
                   'n_epoch': 400,
                   'learning_rate': 0.01,
                   'energy_ev': 25000,
                   'psize_cm': 800e-7,
                   'dist_cm_ls': [30.],
                   'fin_sup_mask': None,
                   'gamma': 1,
                   'allow_shift': False,
                   'allow_scaling': False,
                   'allow_intensity_scaling': False,
                   'allow_refine_distances': True}

    params = params_32id

    # prj0 = np.squeeze(dxchange.read_tiff('data/vincent/test/scan2/final_prj/prj_00000.tiff'))
    # prj1 = np.squeeze(dxchange.read_tiff('data/vincent/test/scan2/final_prj/prj_00001.tiff'))
    # prj2 = np.squeeze(dxchange.read_tiff('data/vincent/test/scan2/final_prj/prj_00002.tiff'))
    # prj3 = np.squeeze(dxchange.read_tiff('data/vincent/test/scan2/final_prj/prj_00003.tiff'))
    # prj3 = np.squeeze(dxchange.read_tiff('data/vincent/test/converted/converted_00003.tiff'))
    # prj0 = np.squeeze(dxchange.read_tiff('data/cameraman/cameraman_512_dp_40.tiff'))
    # prj1 = np.squeeze(dxchange.read_tiff('data/cameraman/cameraman_512_dp_60.tiff'))
    # prj2 = np.squeeze(dxchange.read_tiff('data/cameraman/cameraman_512_dp_80.tiff'))
    # prj3 = np.squeeze(dxchange.read_tiff('data/cameraman/cameraman_512_dp_100.tiff'))
    prj0 = np.squeeze(dxchange.read_tiff('data/32id/octopus.tiff'))

    # data = [prj0, prj1, prj2, prj3]
    data = [prj0]

    retrieve_phase_near_field(data=data,
                              save_path=params['save_path'],
                              energy_ev=params['energy_ev'],
                              dist_cm_ls=params['dist_cm_ls'],
                              psize_cm=params['psize_cm'],
                              output_fname=params['output_fname'],
                              pad_length=params['pad_length'],
                              n_epoch=params['n_epoch'],
                              learning_rate=params['learning_rate'],
                              gamma=params['gamma'],
                              fin_sup_mask=params['fin_sup_mask'],
                              allow_shift=params['allow_shift'],
                              allow_scaling=params['allow_scaling'],
                              allow_intensity_scaling=params['allow_intensity_scaling'],
                              allow_refine_distances=params['allow_refine_distances'])
