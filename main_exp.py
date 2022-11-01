import utils.fflow as flw
import numpy as np
import torch
import os
import multiprocessing
import json
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
# from eval.evaluate_utils import subtract_weight
import numpy as np
import random
# from eval.evaluate_utils import *
from torch.utils.data import Dataset, DataLoader
from test_saved_models import NumpyEncoder, CPU_Unpickler
from offline_unlearn import compute_alpha_cutoff
class MyLogger(flw.Logger):
    def log(self, server=None):
        if server==None: return
        if self.output == {}:
            self.output = {
                "meta":server.option,
                "mean_curve":[],
                "var_curve":[],
                "train_losses":[],
                "test_accs":[],
                "backdoor_accs":[],
                "test_losses":[],
                "valid_accs":[],
                "client_accs":{},
                "mean_valid_accs":[],
                "train_uncertainty_on_clients":[],
                "all_selected_clients":[]
            }
        if "mp_" in server.name:
            test_metric, test_loss, test_backdoor = server.test(device=torch.device('cuda:0'))
        else:
            test_metric, test_loss, test_backdoor = server.test()
        
        valid_metrics, valid_losses = server.test_on_clients(self.current_round, 'valid')
        train_metrics, train_losses = server.test_on_clients(self.current_round, 'train')
        self.output['train_losses'].append(1.0*sum([ck * closs for ck, closs in zip(server.client_vols, train_losses)])/server.data_vol)
        self.output['valid_accs'].append(valid_metrics)
        self.output['test_accs'].append(test_metric)
        self.output['backdoor_accs'].append(test_backdoor)
        self.output['test_losses'].append(test_loss)
        self.output['mean_valid_accs'].append(1.0*sum([ck * acc for ck, acc in zip(server.client_vols, valid_metrics)])/server.data_vol)
        self.output['mean_curve'].append(np.mean(valid_metrics))
        self.output['var_curve'].append(np.std(valid_metrics))
        self.output['train_uncertainty_on_clients'].append(server.uncertainty_round)
        self.output['all_selected_clients'].append([int(id) for id in server.selected_clients])
        
        for cid in range(server.num_clients):
            self.output['client_accs'][server.clients[cid].name]=[self.output['valid_accs'][i][cid] for i in range(len(self.output['valid_accs']))]
        print(self.temp.format("Training Loss:", self.output['train_losses'][-1]))
        print(self.temp.format("Testing Loss:", self.output['test_losses'][-1]))
        print(self.temp.format("Testing Accuracy:", self.output['test_accs'][-1]))
        print(self.temp.format("Backdoor Accuracy:", self.output['backdoor_accs'][-1]))
        print(self.temp.format("Validating Accuracy:", self.output['mean_valid_accs'][-1]))
        print(self.temp.format("Mean of Client Accuracy:", self.output['mean_curve'][-1]))
        print(self.temp.format("Std of Client Accuracy:", self.output['var_curve'][-1]))
        print("Uncertainty of all clients in this round:")
        print(self.output['train_uncertainty_on_clients'][-1])
        print("Selected clients in this round:")
        print(self.output['all_selected_clients'][-1])

logger = MyLogger()

def main():
    multiprocessing.set_start_method('spawn')
    # read options
    option = flw.read_option()
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = '8888'
    os.environ['WORLD_SIZE'] = str(3)
    # set random seed
    flw.setup_seed(option['seed'])
    print(option['num_rounds'])
    atk_pct = [0.05]#[0.05, 0.1] #, 0.2]
    theta = [1]
    gamma = [1]
    algo = [3, 1, 0, 2] #[0, 1, 2, 3]
    for pct in atk_pct:
        for the_ta in theta:
            for gm in gamma:
                for ag in algo:
                    option['attacker_pct'] = pct
                    option['theta_delta'] = the_ta
                    option['gamma_epsilon'] = gm
                    option['unlearn_algorithm'] = ag
                    server = flw.initialize(option)
                                # start federated optimization
                    server.run()
                    print(server.round_selected)
                    del server

def plot_alpha_cutoff(cutoff, round = 200, layer = 'fc2.weight'):
    input_path = os.path.join('./result','alpha_cutoff', 'round{}'.format(round), layer, 'cutoff{}.json'.format(cutoff))
    output_path = os.path.join('./result','alpha_cutoff', 'round{}'.format(round), layer, 'plot')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    with open(input_path, 'r') as f:
        data = json.load(f)
    x0 = data['alpha_cutoff']
    x = []
    for i in range(round):
        sum = 0
        for client in range(6, 30):
            sum += x0[client][i]
        x.append(sum/24)
    # import pdb; pdb.set_trace()
    # x = data['alpha_cutoff'][9][5:]
    plt.plot(x, label="alpha over rounds - cut 5%")
    plt.xlabel('round')
    plt.ylabel('alpha')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'avg_alpha_p.png'))
    # print(x[100])
    # # print(x[200])
    plt.close()
    import pdb; pdb.set_trace()
    
def plot_track():
    output_path = os.path.join('./result','unlearn_track', 'round100', 'plot')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for alp in [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2]:
        input_path = os.path.join('./result','unlearn_track', 'round100', 'alpha{}.json'.format(alp))
        with open(input_path, 'r') as f:
            data = json.load(f)
        x = data['backdoor']

        # import pdb; pdb.set_trace()
        # x = data['alpha_cutoff'][9][5:]
        plt.plot(x, label="backdoor accuracy-alpha{}".format(alp))
        plt.legend()
    plt.xlabel('round')
    plt.ylabel('accuracy')
    plt.savefig(os.path.join(output_path, 'backdoor_acc.png'))
    # print(x[100])
    # # print(x[200])
    plt.close()
    # import pdb; pdb.set_trace()
    

if __name__ == '__main__':
    task_name = 'mnist_cnum100_dist2_skew0.5_seed0'
    task_model = 'cnn'
    bmk_name = task_name[:task_name.find('cnum')-1].lower()
    bmk_model_path = '.'.join(['benchmark', bmk_name, 'model', task_model])
    bmk_core_path = '.'.join(['benchmark', bmk_name, 'core'])
    # utils.fmodule.device = torch.device('cuda:{}'.format(option['server_gpu_id']) if torch.cuda.is_available() and option['server_gpu_id'] != -1 else 'cpu')
    utils.fmodule.Model = getattr(importlib.import_module(bmk_model_path), 'Model')
    device = 'cpu'
    # plot_params_alpha('./result', device, 99, 'fc2.weight')
    # print(compute_alpha_over_rounds('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0', max_iters=10, list_layers= ['fc2.weight']))
    # offline_track('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0', alp = 0.01)
    # compute_alpha_cutoff('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0', max_iters=10)
    ##########
    # my_mod = utils.fmodule.Model()
    # # print(len(np.array(torch.flatten(my_mod.state_dict()['fc2.weight']))))
    # for layer in ['conv1.weight', 'conv2.weight', 'fc1.weight', 'fc2.weight']:
    #     my_len = len(np.array(torch.flatten(my_mod.state_dict()[layer])))
    #     for cutoff in [int(my_len*0.025), int(my_len*0.05), 0, 1, 2, 3, 4, 5, 10]:
    #         compute_alpha_cutoff('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0', max_iters=200, cutoff=cutoff, layer=layer)
    ###
    plot_track()



