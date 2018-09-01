import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import dxchange
import matplotlib.pyplot as plt


src_fname = 'data/cameraman_512.tiff'
probe_type = 'square'
probe_size = [128, 128]

obj = np.squeeze(dxchange.read_tiff(src_fname))

if probe_type == 'square':
    half_probe_size = [int(x / 2) for x in probe_size]
    center = [int(x / 2) for x in obj.shape]
    probe = np.zeros_like(obj, dtype='float32')
    probe[center[0] - half_probe_size[0]:center[0] - half_probe_size[0] + probe_size[0],
          center[1] - half_probe_size[1]:center[1] - half_probe_size[1] + probe_size[1]] = 1
else:
    raise Exception('You screwed it up.')

probe *= obj
probe = fftshift(fft2(probe))
probe = probe[::4, ::4]
dxchange.write_tiff(np.abs(probe) ** 2, 'data/cameraman_128_dp', dtype='float32', overwrite=True)
dxchange.write_tiff(np.angle(probe), 'data/cameraman_128_phase', dtype='float32', overwrite=True)
dxchange.write_tiff(np.abs(ifftshift(ifft2(probe))), 'data/cameraman_128', dtype='float32', overwrite=True)
