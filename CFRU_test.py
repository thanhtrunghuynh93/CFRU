import ujson
import torch
import pickle
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = "1,0"
import utils.fmodule
from utils import fmodule
from utils.fmodule import _modeldict_cp
import importlib
import utils.fflow as flw
import numpy as np
import io
import json
import ujson
import torch.nn as nn

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)
    
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == '__main__':
    task_name = './fedtasksave/movielens1m_cnum100_dist0_skew0_seed0'
    exp_name = 'CFRU_NCF_Mu8_R40_P0.30_alpha0.5_seed0_fedAttack'
    with open(os.path.join(task_name, exp_name, 'record', 'history40.pkl'), 'rb') as test_f:
        hist_unlearn = CPU_Unpickler(test_f).load()
    print("Hit Ratio and NDCG for top 10 recommendation!")
    print("HR@10: {}    || NDCG@10: {}".format(hist_unlearn['accuracy'][0], hist_unlearn['accuracy'][0]))
    # pass

