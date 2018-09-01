import dxchange
import numpy as np
import tomopy

src_fname = 'data/cameraman_512_dp.tiff'
actual_size = [512, 512]
energy_ev = 25000.
psize_cm = 1e-4
dist_cm = 50

img = dxchange.read_tiff(src_fname)
img = np.sqrt(img)
img = img[np.newaxis, :, :]
res = np.squeeze(tomopy.retrieve_phase(img, psize_cm, dist_cm, energy_ev / 1000, alpha=5e-2))
dxchange.write_tiff(res, 'data/cameraman_512_paganin', dtype='float32', overwrite=True)