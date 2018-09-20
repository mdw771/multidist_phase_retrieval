import numpy as np
import dxchange
from util import *


prj0 = np.squeeze(dxchange.read_tiff('data/vincent/test/proj_0.tiff'))
prj1 = np.squeeze(dxchange.read_tiff('data/vincent/test/proj_1.tiff'))
prj2 = np.squeeze(dxchange.read_tiff('data/vincent/test/proj_2.tiff'))
prj3 = np.squeeze(dxchange.read_tiff('data/vincent/test/proj_3.tiff'))
data = [prj0, prj1, prj2, prj3]

data, d_para_cm = convert_cone_to_parallel(data, 390, [47.4848, 47.3848, 46.9848, 45.9847])
dxchange.write_tiff(data, 'data/vincent/test/converted', dtype='float32', overwrite=True)
print(d_para_cm)