import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import dxchange
import matplotlib.pyplot as plt

from util import *


src_fname = 'data/cameraman_530.tiff'
actual_size = [512, 512]
energy_ev = 25000
psize_cm = 1e-4
dist_cm_ls = [40, 60, 80, 100]
src = dxchange.read_tiff(src_fname)
src = src.astype(np.complex64)

# wavefront = np.zeros([512, 512])
# wavefront[128:384, 128:384] = 1

for dist_cm in dist_cm_ls:
    wavefront = fresnel_propagate_numpy(src, energy_ev, psize_cm, dist_cm)
    center = [int(x / 2) for x in wavefront.shape]
    half_probe_size = [int(x / 2) for x in actual_size]
    wavefront = wavefront[center[0] - half_probe_size[0]:center[0] - half_probe_size[0] + actual_size[0],
                          center[1] - half_probe_size[1]:center[1] - half_probe_size[1] + actual_size[1]]

    dxchange.write_tiff(np.abs(wavefront) ** 2, 'data/cameraman_512_dp_{}'.format(dist_cm), dtype='float32', overwrite=True)
    dxchange.write_tiff(np.angle(wavefront), 'data/cameraman_512_phase_{}'.format(dist_cm), dtype='float32', overwrite=True)
