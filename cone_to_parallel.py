import numpy as np
import dxchange
from util import *


prj0 = np.squeeze(dxchange.read_tiff('data/vincent/test/proj_0.tiff'))
prj1 = np.squeeze(dxchange.read_tiff('data/vincent/test/proj_1.tiff'))
prj2 = np.squeeze(dxchange.read_tiff('data/vincent/test/proj_2.tiff'))
prj3 = np.squeeze(dxchange.read_tiff('data/vincent/test/proj_3.tiff'))
data = [prj0, prj1, prj2]

data, d_para_cm = convert_cone_to_parallel(data, 56.1, np.array([47.4848, 47.3848, 46.9848]), psize=[99.8, 101, 106], crop=True)
# data = register_data(data)
data = shift_data(data, [[5, 3], [2.78, 1.31], [2.85, 1.77], [0, 5]], ref_ind=-1)
# dxchange.write_tiff_stack(data, 'data/vincent/test/converted/converted', dtype='float32', overwrite=True)

for i, img in enumerate(data):
    dxchange.write_tiff(img, 'data/vincent/test/converted/converted_{:05}'.format(i), dtype='float32', overwrite=True)
print(d_para_cm)
