import os
import numpy as np
from hparam import HParam

hps = HParam()

# set_path = os.path.join(hps.dataset_target_path, '')
# for root, dirs, files in os.walk(set_path):
#     print(root, len(files))

data_path = os.path.join(hps.feature_path, 'test', 'Foxtrot.npy')
data = np.load(data_path)
print(data.shape)

