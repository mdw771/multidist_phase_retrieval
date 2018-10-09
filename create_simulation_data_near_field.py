import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import dxchange
import matplotlib.pyplot as plt

from util import *


src_mag_fname = 'data/cameraman/cameraman_530.tiff'
src_phase_fname = 'data/cameraman/baboon_530.tiff'
actual_size = [512, 512]
energy_ev = 17500
psize_cm = 1e-4
dist_cm_ls = [40, 60, 80, 100]
src_mag = np.squeeze(dxchange.read_tiff(src_mag_fname))
src_mag = src_mag / src_mag.max() * 0.1 + 0.9
src_mag = src_mag.astype(np.complex64)

src_phase = np.squeeze(dxchange.read_tiff(src_phase_fname))
src_phase = src_phase / 255.
src_phase = src_phase.astype(np.complex64)

# pure phase
# src_mag = np.ones_like(src_mag)

# pure abs
# src_phase = np.zeros_like(src_phase)

src = src_mag * np.exp(1j * src_phase)
# print(src)

# wavefront = np.zeros([512, 512])
# wavefront[128:384, 128:384] = 1

for dist_cm in dist_cm_ls:
    wavefront = fresnel_propagate_numpy(src, energy_ev, psize_cm, dist_cm)
    center = [int(x / 2) for x in wavefront.shape]
    half_probe_size = [int(x / 2) for x in actual_size]
    wavefront = wavefront[center[0] - half_probe_size[0]:center[0] - half_probe_size[0] + actual_size[0],
                          center[1] - half_probe_size[1]:center[1] - half_probe_size[1] + actual_size[1]]
    rand_y, rand_x = np.random.randint(0, 5, [2, ])
    wavefront = np.roll(wavefront, rand_y, axis=0)
    wavefront = np.roll(wavefront, rand_x, axis=1)
    print(rand_y, rand_x)

    dxchange.write_tiff(np.abs(wavefront) ** 2, 'data/cameraman/cameraman_512_dp_{}'.format(dist_cm), dtype='float32', overwrite=True)
    dxchange.write_tiff(np.angle(wavefront), 'data/cameraman/cameraman_512_phase_{}'.format(dist_cm), dtype='float32', overwrite=True)
