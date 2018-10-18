import numpy as np
from util import *
import dxchange
import os
import tomopy


energy_ev = 17500
psize_cm_ls = [99.8e-7, 101e-7, 106e-7, 117e-7]
psize_cm = 100.e-7
dist_cm_ls = [7.29074313846154, 7.36286892307692, 7.66213661538462, 8.277246]
prj0 = np.squeeze(dxchange.read_tiff('data/vincent/test/scan2/final_prj/prj_00000.tiff'))
prj1 = np.squeeze(dxchange.read_tiff('data/vincent/test/scan2/final_prj/prj_00001.tiff'))
prj2 = np.squeeze(dxchange.read_tiff('data/vincent/test/scan2/final_prj/prj_00002.tiff'))
prj3 = np.squeeze(dxchange.read_tiff('data/vincent/test/scan2/final_prj/prj_00003.tiff'))
data_ls = [prj0, prj1, prj2, prj3]

# for i, prj in enumerate(data_ls):
#     prj_back = tomopy.retrieve_phase(prj[np.newaxis, :, :], pixel_size=psize_cm_ls[i], dist=dist_cm_ls[i], energy=energy_ev * 1e-3, alpha=5e-3)
#     # prj_back = fresnel_propagate_numpy(prj, energy_ev, psize_cm_ls[i], -dist_cm_ls[i])
#     # prj_back = np.squeeze(prj_back)
#     dxchange.write_tiff(np.abs(prj_back), os.path.join('data/vincent/test/scan2/paganin_5e-3', 'back_{}'.format(i)), dtype='float32', overwrite=True)

phase = multidistance_ctf(data_ls, dist_cm_ls, psize_cm, energy_ev * 1e-3, kappa=200, alpha_1=10.)
dxchange.write_tiff(phase, os.path.join('data/vincent/test/scan2/ctf', 'ctf'), dtype='float32', overwrite=True)