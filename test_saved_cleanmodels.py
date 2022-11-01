import ujson
import torch
import pickle
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = "1,0"
from utils import fmodule
from utils.fmodule import _modeldict_cp
import importlib
import utils.fflow as flw
# from eval.evaluate_utils import subtract_weight
import numpy as np
# from eval.evaluate_utils import *
import io
from test_saved_models import NumpyEncoder, CPU_Unpickler
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

if __name__ == '__main__':
    accuracy = []
    round = 100
    with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R200_P0.30_AP0.20_AlphaGen_6atk_minus1.0/record/history{}.pkl'.format(round), 'rb') as test_f:
        hist_clean = CPU_Unpickler(test_f).load()
    # print("Done " + str(round))
    import pdb; pdb.set_trace()
    # fedtask_name = 'R10_P0.30_AP0.20_clean1'
    # task_name = 'mnist_cnum100_dist2_skew0.5_seed0'
    # root_path = './fedtasksave/' + task_name + '/' + fedtask_name + '/record/'
    # device = torch.device('cuda:0')

    # with open(root_path + 'history0.pkl', 'rb') as test_f:
    #     history_clean = CPU_Unpickler(test_f).load()
    
    # m_dirty = history_clean['server_model'].to(device)
    # print(m_dirty.state_dict())
    # print(history_clean['selected_clients'])
    
    # num_round = 0

    # def test():
    #     global num_round
    #     if num_round == 2:
    #         import pdb
    #         pdb.set_trace()
    #         print("test")

    # def test2():
    #     for i in range(10):
    #         global num_round
    #         num_round += 1
    #         print(num_round)
    #         test()

    # # test2()
    # for i in range(10):
    #     print(i)
    #     import pdb
    #     pdb.set_trace()
    #     print(i**2)

# (Pdb) pdb.set_trace = lambda: None  # This replaces the set_trace() function!
# (Pdb) continue 

