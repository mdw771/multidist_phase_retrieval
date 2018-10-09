import numpy as np
import dxchange
from util import *


prj0 = np.squeeze(dxchange.read_tiff('data/vincent/test/converted/converted_00000.tiff'))
prj1 = np.squeeze(dxchange.read_tiff('data/vincent/test/converted/converted_00001.tiff'))
prj2 = np.squeeze(dxchange.read_tiff('data/vincent/test/converted/converted_00002.tiff'))
# prj3 = np.squeeze(dxchange.read_tiff('data/vincent/test/proj_3.tiff'))
data = [prj0, prj1, prj2]

# data, d_para_cm = convert_cone_to_parallel(data, 390, np.array([41.0234, 47.4848, 51.7924]), psize=[175, 98.8, 49.9], crop=False)
# data = register_data(data)
data = shift_data(data, [[-22, -14], [0, 18], [22, 22]], ref_ind=None)
# dxchange.write_tiff_stack(data, 'data/vincent/test/converted/converted', dtype='float32', overwrite=True)

for i, img in enumerate(data):
    dxchange.write_tiff(img, 'data/vincent/test/registered/converted_{:05}'.format(i), dtype='float32', overwrite=True)
# print(d_para_cm)
