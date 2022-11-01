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
# from eval.evaluate_utils import subtract_weight
import numpy as np
import random
# from eval.evaluate_utils import *
import io
import statistics
import json
from torch.utils.data import Dataset, DataLoader
from test_saved_models import NumpyEncoder, CPU_Unpickler

lossfunc = torch.nn.CrossEntropyLoss()

@torch.no_grad()
def data_to_device(data, device=None):
    if device is None:
        return data[0].to('cpu'), data[1].to('cpu')
    else:
        return data[0].to(device), data[1].to(device)

def test(model, data, device=None):
    """Metric = Accuracy"""
    tdata = data_to_device(data, device)
    model = model.to(device)
    outputs = model(tdata[0])
    loss = lossfunc(outputs, tdata[-1])
    y_pred = outputs.data.max(1, keepdim=True)[1]
    correct = y_pred.eq(tdata[1].data.view_as(y_pred)).long().cpu().sum()
    return (1.0 * correct / len(tdata[1])).item(), loss.item()

def accuracy_track(num_round, log_folder= './fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R200_P0.30_AP0.20_Alpha1.0_6atk_minus0.05/record/'):
    # init fedtask
    task_name = 'mnist_cnum100_dist2_skew0.5_seed0'
    task_model = 'cnn'
    bmk_name = task_name[:task_name.find('cnum')-1].lower()
    bmk_model_path = '.'.join(['benchmark', bmk_name, 'model', task_model])
    bmk_core_path = '.'.join(['benchmark', bmk_name, 'core'])
    utils.fmodule.device = torch.device('cpu')
    utils.fmodule.TaskCalculator = getattr(importlib.import_module(bmk_core_path), 'TaskCalculator')
    utils.fmodule.TaskCalculator.setOP(getattr(importlib.import_module('torch.optim'), 'SGD'))
    utils.fmodule.Model = getattr(importlib.import_module(bmk_model_path), 'Model')
    task_reader = getattr(importlib.import_module(bmk_core_path), 'TaskReader')(taskpath=os.path.join('fedtask', 'mnist_cnum100_dist2_skew0.5_seed0'))
    train_datas, train_datas_attack, valid_datas, test_data, backtask_data, client_names = task_reader.read_data()
    num_clients = len(client_names)
    
    # get dataloader
    def seed_worker():
        worker_seed = 0
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(0) # 0
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, drop_last=False, worker_init_fn=seed_worker, generator=g)
    backdoor_loader = DataLoader(backtask_data, batch_size=64, shuffle=True, drop_last=False, worker_init_fn=seed_worker, generator=g)
    
    main_acc = []
    backdoor_acc = []
    ## get model from pkl file
    for round_idx in range(num_round):
        with open(log_folder + 'history{}.pkl'.format(round_idx), 'rb') as test_f:
            history_dirty = CPU_Unpickler(test_f).load()
        
        model_test = history_dirty['server_model']
        model_test.eval()
        loss = 0
        eval_metric = 0
        for batch_id, batch_data in enumerate(test_dataloader):
            bmean_eval_metric, bmean_loss = test(model_test, batch_data)
            loss += bmean_loss * len(batch_data[1])
            eval_metric += bmean_eval_metric * len(batch_data[1])
        eval_metric /= len(test_data)
        main_acc.append(eval_metric)
        
        eval_backdoor = -1
        if backtask_data:
            eval_backdoor = 0
            model_test.eval()
            for batch_id, batch_data in enumerate(backdoor_loader):
                backdoor_eval_metric, backdoor_loss = test(model_test, batch_data)
                eval_backdoor += backdoor_eval_metric * len(batch_data[1])
            eval_backdoor /= len(backtask_data)
            backdoor_acc.append(eval_backdoor)
        print("Done round {}".format(round_idx))   
        
    return main_acc, backdoor_acc

def offline_test(model_test, test_loader, backdoor_loader, len_test, len_backdoor):
    model_test.eval()
    loss = 0
    eval_metric = 0
    for batch_id, batch_data in enumerate(test_loader):
        bmean_eval_metric, bmean_loss = test(model_test, batch_data)
        loss += bmean_loss * len(batch_data[1])
        eval_metric += bmean_eval_metric * len(batch_data[1])
    eval_metric /= len_test
        
    eval_backdoor = 0
    model_test.eval()
    for batch_id, batch_data in enumerate(backdoor_loader):
        backdoor_eval_metric, backdoor_loss = test(model_test, batch_data)
        eval_backdoor += backdoor_eval_metric * len(batch_data[1])
    eval_backdoor /= len_backdoor
    return eval_metric, eval_backdoor

def offline_track(log_folder, alp, attackers = [0, 1, 2, 3, 4, 5]):
    end_round_unlearn = 100 # attacked_round = start_round -> end_round - 1
    start_round_unlearn = 0
    main_test_all_round = []
    backdoor_test_all_round = []
    hist_dirty_name = 'R200_P0.30_AP0.20_Alpha1.0_6atk_minus0.01'
    with open(os.path.join(log_folder, hist_dirty_name, 'record', 'history{}.pkl'.format(end_round_unlearn)), 'rb') as test_f:
        prototype_dirty = CPU_Unpickler(test_f).load()
    prototype_model = prototype_dirty['server_model']
    
    # init fedtask
    task_name = 'mnist_cnum100_dist2_skew0.5_seed0'
    task_model = 'cnn'
    bmk_name = task_name[:task_name.find('cnum')-1].lower()
    bmk_model_path = '.'.join(['benchmark', bmk_name, 'model', task_model])
    bmk_core_path = '.'.join(['benchmark', bmk_name, 'core'])
    utils.fmodule.device = torch.device('cpu')
    utils.fmodule.TaskCalculator = getattr(importlib.import_module(bmk_core_path), 'TaskCalculator')
    utils.fmodule.TaskCalculator.setOP(getattr(importlib.import_module('torch.optim'), 'SGD'))
    utils.fmodule.Model = getattr(importlib.import_module(bmk_model_path), 'Model')
    task_reader = getattr(importlib.import_module(bmk_core_path), 'TaskReader')(taskpath=os.path.join('fedtask', 'mnist_cnum100_dist2_skew0.5_seed0'))
    train_datas, train_datas_attack, valid_datas, test_data, backtask_data, client_names = task_reader.read_data()
    num_clients = len(client_names)
    
    # get dataloader
    def seed_worker():
        worker_seed = 0
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(0) # 0
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, drop_last=False, worker_init_fn=seed_worker, generator=g)
    backdoor_loader = DataLoader(backtask_data, batch_size=64, shuffle=True, drop_last=False, worker_init_fn=seed_worker, generator=g)
    len_test = len(test_data)
    len_backdoor = len(backtask_data)
    ## offline track
    for round_track in range(start_round_unlearn, end_round_unlearn):
        # track from round_track to end_round - 1
        round_attack = [rid for rid in range(round_track, end_round_unlearn)]
        attackers_round = [attackers for rid in range(round_track, end_round_unlearn)]
        # unlearn 
        unlearning_term3 = fmodule._create_new_model(prototype_model) * 0.0
        # unlearning_term2 = fmodule._create_new_model(prototype_model) * 0.0
        alpha = - alp
        for idx in range(len(round_attack)):
            round_id = round_attack[idx]
            with open(os.path.join(log_folder, hist_dirty_name, 'record', 'history{}.pkl'.format(round_id)), 'rb') as test_f:
                hist_dirty = CPU_Unpickler(test_f).load()
            beta = 0.0
            for pid in range(len(hist_dirty['p'])):
                if hist_dirty['selected_clients'][pid] not in attackers_round[idx]:
                    beta += hist_dirty['p'][pid]
            beta = beta * alpha + 1
            # 
            unlearning_term3 = unlearning_term3 * beta
            for pid in range(len(hist_dirty['p'])):
                if hist_dirty['selected_clients'][pid] in attackers_round[idx]:
                    unlearning_term3 += 1.0 * hist_dirty['p'][pid] * (hist_dirty['server_model'] - hist_dirty['models'][pid])
                    # unlearning_term2 += 1.0 * hist_dirty['p'][pid] * (hist_dirty['server_model'] - hist_dirty['models'][pid])
        # compute unlearn_model
        unlearn_model3 = prototype_model + unlearning_term3
        # unlearn_model2 = prototype_model + unlearning_term2
        # test unlearn_model
        main_acc, backdoor_acc = offline_test(unlearn_model3, test_dataloader, backdoor_loader, len_test, len_backdoor)
        main_test_all_round.append(main_acc)
        backdoor_test_all_round.append(backdoor_acc)                
        print("Done track {}".format(round_track))
    out_path = os.path.join('./result', 'unlearn_track', 'round{}'.format(end_round_unlearn))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    outp = {}
    outp['main'] = main_test_all_round
    outp['backdoor'] = backdoor_test_all_round
    dumped = json.dumps(outp, cls=NumpyEncoder)
    with open(os.path.join(out_path, 'alpha{}.json'.format(alp)), 'w') as outf:
        outf.write(dumped + '\n')
        
def compute_alpha_over_rounds(log_folder, max_iters, list_layers= []):
    dict_alpha = {'conv1.weight':[], 'conv1.bias':[], 'conv2.weight':[], 'conv2.bias':[], 
                  'fc1.weight':[], 'fc1.bias':[], 'fc2.weight':[], 'fc2.bias':[]}
    list_keys = list(dict_alpha.keys())
    for round_idx in range(max_iters + 1):
        with open(os.path.join(log_folder, 'R5_P0.30_AP0.20_Alpha1.0_6atk_minus1.0', 'record', 'history{}.pkl'.format(round_idx)), 'rb') as test_f:
            hist_dirty = CPU_Unpickler(test_f).load()
        with open(os.path.join(log_folder, 'R5_P0.30_AP0.20_Alpha1.0_6atk_minus0.0', 'record', 'history{}.pkl'.format(round_idx)), 'rb') as test_f:
            hist_clean = CPU_Unpickler(test_f).load()
        # compute grad 
        grads_dirty = [(hist_dirty['server_model'] - hist_dirty['models'][idx]) for idx in range(len(hist_dirty['selected_clients']))]
        grads_clean = [(hist_clean['server_model'] - hist_clean['models'][idx]) for idx in range(len(hist_clean['selected_clients']))]
        delta = hist_clean['server_model'] - hist_dirty['server_model']
        
        list_avg_alpha = [[] for i in range(8)]
        for clean_idx in range(len(hist_clean['selected_clients'])):
            idx = hist_dirty['selected_clients'].index(hist_clean['selected_clients'][clean_idx])
            res = fmodule.abs(fmodule._model_elementwise_divide(grads_clean[clean_idx] - grads_dirty[idx], delta))
            for idx in range(len(list_keys)):
                list_avg_alpha[idx].append(fmodule._model_avg_param(res, [list_keys[idx]]))
        for idx in range(len(list_keys)):
            dict_alpha[list_keys[idx]].append(statistics.fmean(list_avg_alpha[idx]))
        print('Done round {}'.format(round_idx))
    return dict_alpha

def compute_alpha_cutoff(log_folder, max_iters, cutoff = 256, layer = 'fc2.weight'):
    list_alpha = [[] for i in range(30)]
    for round_idx in range(max_iters + 1):
        with open(os.path.join(log_folder, 'R200_P0.30_AP0.20_Alpha1.0_6atk_minus0.01', 'record', 'history{}.pkl'.format(round_idx)), 'rb') as test_f:
            hist_dirty = CPU_Unpickler(test_f).load()
        with open(os.path.join(log_folder, 'R200_P0.30_AP0.20_Alpha0.0_6atk_minus0.0', 'record', 'history{}.pkl'.format(round_idx)), 'rb') as test_f:
            hist_clean = CPU_Unpickler(test_f).load()
        # compute grad 
        grads_dirty = [(hist_dirty['server_model'] - hist_dirty['models'][idx]) for idx in range(len(hist_dirty['selected_clients']))]
        grads_clean = [(hist_clean['server_model'] - hist_clean['models'][idx]) for idx in range(len(hist_clean['selected_clients']))]
        delta = hist_clean['server_model'] - hist_dirty['server_model']
        list_avg_alpha = []
        for clean_idx in range(len(hist_clean['selected_clients'])):
            idx = hist_dirty['selected_clients'].index(hist_clean['selected_clients'][clean_idx])
            res = fmodule.abs(fmodule._model_elementwise_divide(grads_clean[clean_idx] - grads_dirty[idx], delta))
            x = np.array(torch.flatten(res.state_dict()[layer]).cpu())
            x = np.mean(np.sort(x)[cutoff : len(x) - cutoff])
            list_alpha[idx].append(x)
        print('Done round {}'.format(round_idx))
    out_path = os.path.join('./result', 'alpha_cutoff', 'round{}'.format(max_iters), layer)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    outp = {}
    outp['alpha_cutoff'] = list_alpha
    dumped = json.dumps(outp, cls=NumpyEncoder)
    with open(os.path.join(out_path, 'cutoff{}.json'.format(cutoff)), 'w') as outf:
        outf.write(dumped + '\n')
    return list_alpha

def plot_params_alpha(log_folder, device, round, layer):
    with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R100_P0.30_AP0.20_Alpha1.0_6atk_minus0.09/record/history{}.pkl'.format(round), 'rb') as test_f:
        hist_dirty = CPU_Unpickler(test_f).load()
    
    with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R200_P0.30_AP0.20_Alpha0.0_6atk_minus0.0/record/history{}.pkl'.format(round), 'rb') as test_f:
        hist_clean = CPU_Unpickler(test_f).load()
    
    output_path = os.path.join(log_folder, "params_alpha", layer, "cross-entropy", "round{}".format(round))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    hist_dirty['server_model'].to(device)
    hist_clean['server_model'].to(device)
    grads_dirty = [(hist_dirty['server_model'] - hist_dirty['models'][idx].to(device)) for idx in range(len(hist_dirty['selected_clients']))]
    grads_clean = [(hist_clean['server_model'] - hist_clean['models'][idx].to(device)) for idx in range(len(hist_clean['selected_clients']))]
    delta = hist_clean['server_model'] - hist_dirty['server_model']
    clean_params_alpha = []
    for clean_idx in range(len(hist_clean['selected_clients'])):
        idx = hist_dirty['selected_clients'].index(hist_clean['selected_clients'][clean_idx])
        res = fmodule.abs(fmodule._model_elementwise_divide(grads_clean[clean_idx] - grads_dirty[idx], delta))
        # for avg alpha 
        clean_params_alpha.append(res)
        #
        x = np.array(torch.flatten(res.state_dict()[layer]).cpu())
        fig,ax=plt.subplots(figsize = (65, 30))
        plt.plot(x)
        ax.set_xticks([i for i in range(0, len(x)+1, 512)])
        plt.grid(axis = 'x', ls = '--')
        plt.title('Parameter of alpha-vector for {}'.format(layer), fontsize=28)
        plt.xlabel('Parameter index in tensor of alpha', fontsize = 22)
        plt.ylabel('Value of parameter', fontsize = 22)
        plt.savefig(os.path.join(output_path, 'client{}.png'.format(hist_clean['selected_clients'][clean_idx])))
        plt.close()
    ## plot for avg alpha over clean clients
    res = fmodule._model_average(ms = clean_params_alpha, p = hist_clean['p'])
    x = np.array(torch.flatten(res.state_dict()[layer]).cpu())
    import pdb; pdb.set_trace()
    print("Median of alpha: {}".format(np.median(x)))
    fig,ax=plt.subplots(figsize = (65, 30))
    plt.plot(x)
    ax.set_xticks([i for i in range(0, len(x)+1, 512)])
    plt.grid(axis = 'x', ls = '--')
    plt.title('Parameter of average alpha-vector'.format(layer), fontsize=28)
    plt.xlabel('Parameter index in tensor of alpha', fontsize = 22)
    plt.ylabel('Value of parameter', fontsize = 22)
    plt.savefig(os.path.join(output_path, 'avg_alpha_p.png'))
    plt.close()

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
    # for alp in [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.15, 0.2]:
    #     offline_track('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0', alp = alp)
    # dict_alpha = compute_alpha_over_rounds('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0', 5)
    main_acc, back_acc = accuracy_track(num_round= 201)
    import pdb; pdb.set_trace()
    # compute_alpha_cutoff('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0', max_iters=10)
    # my_mod = utils.fmodule.Model()
    # print(len(np.array(torch.flatten(my_mod.state_dict()['fc2.weight']))))
    