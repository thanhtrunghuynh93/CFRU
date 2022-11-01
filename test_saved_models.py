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
# from eval.evaluate_utils import *
import io
import statistics
import json

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def inverse_func(x):
    if x == 0: return 0
    else: return 1.0/x

def compute_alpha(hist_dirty, hist_clean, device, list_layers = []):
    hist_dirty['server_model'].to(device)
    hist_clean['server_model'].to(device)
    grads_dirty = [(hist_dirty['server_model'] - hist_dirty['models'][idx].to(device)) for idx in range(len(hist_dirty['selected_clients']))]
    grads_clean = [(hist_clean['server_model'] - hist_clean['models'][idx].to(device)) for idx in range(len(hist_clean['selected_clients']))]
    delta = hist_clean['server_model'] - hist_dirty['server_model']
    list_avg_alpha = []
    list_max_alpha = []
    for clean_idx in range(len(hist_clean['selected_clients'])):
        idx = hist_dirty['selected_clients'].index(hist_clean['selected_clients'][clean_idx])
        res = fmodule.abs(fmodule._model_elementwise_divide(grads_clean[clean_idx] - grads_dirty[idx], delta))
        list_avg_alpha.append(fmodule._model_avg_param(res, list_layers))
        list_max_alpha.append(fmodule._model_max_element(res, list_layers))
    return statistics.fmean(list_avg_alpha), statistics.fmean(list_max_alpha), torch.Tensor.numpy(max(list_avg_alpha).cpu()), torch.Tensor.numpy(max(list_max_alpha).cpu())

def save_alpha_result(log_folder, diff_list_alpha, list_layers = []):
    if list_layers == []:
        out_task = 'all_params'
    else:
        out_task = ''
        for l_name in list_layers:
            out_task += l_name
    output_path = os.path.join(log_folder, "log", "cross-entropy", out_task)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # 'avg_params-avg_clients', 'max_params-avg_clients', 'avg_params-max_clients', 'max_params-max_clients'
    output = {}
    output['avg_params-avg_clients'] = diff_list_alpha[0]
    output['max_params-avg_clients'] = diff_list_alpha[1]
    output['avg_params-max_clients'] = diff_list_alpha[2]
    output['max_params-max_clients'] = diff_list_alpha[3]
    dumped = json.dumps(output, cls=NumpyEncoder)
    with open(os.path.join(output_path, 'alpha.json'), 'w') as outf:
        outf.write(dumped + '\n')
        # ujson.dump(output, outf)

def plot_alpha(log_folder, diff_list_alpha, list_layers = []):
    if list_layers == []:
        out_task = 'all_params'
    else:
        out_task = ''
        for l_name in list_layers:
            out_task += l_name 
        out_task += "_10atk"
    output_path = os.path.join(log_folder, "alpha", "cross-entropy", out_task) ### remove seed43
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    pic_label = ['avg_params-avg_clients', 'max_params-avg_clients', 'avg_params-max_clients', 'max_params-max_clients']
    for idx in range(4):
        plt.plot(diff_list_alpha[idx], label='Compute alpha constance by ' + pic_label[idx])
        plt.legend()
        plt.xlabel('round')
        plt.ylabel('alpha')
        plt.savefig(os.path.join(output_path, "{}.png".format(pic_label[idx])))
        plt.close()

def plot_params_alpha(log_folder, device, round, layer):
    with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R250_P0.30_AP0.20_6atk_clean0/record/history{}.pkl'.format(round), 'rb') as test_f:
        hist_dirty = CPU_Unpickler(test_f).load()
    
    with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R250_P0.30_AP0.20_6atk_clean1/record/history{}.pkl'.format(round), 'rb') as test_f:
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

def plot_median_alpha(log_folder, device, layer, max_iters = 250):
    output_path = os.path.join(log_folder, "median_alpha", layer, "max_iter{}".format(max_iters))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    med_alpha = []
    for iter in range(max_iters+1):
        with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R250_seed43_clean0/record/history{}.pkl'.format(iter), 'rb') as test_f:
            hist_dirty = CPU_Unpickler(test_f).load()
        
        with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R250_seed43_clean1/record/history{}.pkl'.format(iter), 'rb') as test_f:
            hist_clean = CPU_Unpickler(test_f).load()
    
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
        ## plot for avg alpha over clean clients
        res = fmodule._model_average(ms = clean_params_alpha, p = hist_clean['p'])
        x = np.array(torch.flatten(res.state_dict()[layer]).cpu())
        med_alpha.append(np.median(x))
        print("Round {} done".format(iter))
        # print("Median of alpha: {}".format(np.median(x)))
    print("Median of alpha: {}".format(med_alpha))
    plt.plot(med_alpha)
    plt.title('Median of alpha over rounds ({})'.format(layer), fontsize=28)
    plt.xlabel('Round index', fontsize = 22)
    plt.ylabel('Median of alpha', fontsize = 22)
    plt.savefig(os.path.join(output_path, 'med_alpha.png'))
    plt.close()

def plot_params_grads(log_folder, device, round, layer):
    with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R250_P0.30_AP0.20_6atk_clean0/record/history{}.pkl'.format(round), 'rb') as test_f:
        hist_dirty = CPU_Unpickler(test_f).load()
    
    with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R250_P0.30_AP0.20_6atk_clean1/record/history{}.pkl'.format(round), 'rb') as test_f:
        hist_clean = CPU_Unpickler(test_f).load()
    
    output_path = os.path.join(log_folder, "params_grads", layer, "cross-entropy", "round{}".format(round))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    hist_dirty['server_model'].to(device)
    hist_clean['server_model'].to(device)
    grads_dirty = [(hist_dirty['server_model'] - hist_dirty['models'][idx].to(device)) for idx in range(len(hist_dirty['selected_clients']))]
    grads_clean = [(hist_clean['server_model'] - hist_clean['models'][idx].to(device)) for idx in range(len(hist_clean['selected_clients']))]
    delta = hist_clean['server_model'] - hist_dirty['server_model']
    for clean_idx in range(len(hist_clean['selected_clients'])):
        idx = hist_dirty['selected_clients'].index(hist_clean['selected_clients'][clean_idx])
        res = fmodule.abs(grads_clean[clean_idx] - grads_dirty[idx])
        x = np.array(torch.flatten(res.state_dict()[layer]).cpu())
        fig,ax=plt.subplots(figsize = (65, 30))
        plt.plot(x)
        ax.set_xticks([i for i in range(0, len(x)+1, 512)])
        plt.grid(axis = 'x', ls = '--')
        plt.title('Parameter of diff between gradients for {}'.format(layer), fontsize=28)
        plt.xlabel('Parameter index in diff between gradients', fontsize = 22)
        plt.ylabel('Value of parameter', fontsize = 22)
        plt.savefig(os.path.join(output_path, 'client{}.png'.format(hist_clean['selected_clients'][clean_idx])))
        plt.close()
    # plot for delta
    res = fmodule.abs(delta)
    x = np.array(torch.flatten(res.state_dict()[layer]).cpu())
    fig,ax=plt.subplots(figsize = (65, 30))
    plt.plot(x)
    ax.set_xticks([i for i in range(0, len(x)+1, 512)])
    plt.grid(axis = 'x', ls = '--')
    plt.title('Parameter of delta for {}'.format(layer), fontsize=28)
    plt.xlabel('Parameter index in diff between gradients', fontsize = 22)
    plt.ylabel('Value of parameter', fontsize = 22)
    plt.savefig(os.path.join(output_path, 'delta.png'))
    plt.close()

def plot_clean(log_folder, device, round, layer):  
    with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R250_P0.30_AP0.20_6atk_clean1/record/history{}.pkl'.format(round), 'rb') as test_f:
        hist_clean = CPU_Unpickler(test_f).load()
    
    output_path = os.path.join(log_folder, "clean_models", layer, "cross-entropy", "round{}".format(round))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    hist_clean['server_model'].to(device)
    # print(hist_clean['server_model'].state_dict().keys())
    grads_clean = [(hist_clean['server_model'] - hist_clean['models'][idx].to(device)) for idx in range(len(hist_clean['selected_clients']))]
    for clean_idx in range(len(hist_clean['selected_clients'])):
        res = fmodule.abs(grads_clean[clean_idx])
        x = np.array(torch.flatten(res.state_dict()[layer]).cpu())
        fig,ax=plt.subplots(figsize = (65, 30))
        plt.plot(x)
        ax.set_xticks([i for i in range(0, len(x)+1, 512)])
        plt.grid(axis = 'x', ls = '--')
        plt.title('Parameter of clean grads for {}'.format(layer), fontsize=28)
        plt.xlabel('Parameter index in clean grads', fontsize = 22)
        plt.ylabel('Value of parameter', fontsize = 22)
        plt.savefig(os.path.join(output_path, 'client{}.png'.format(hist_clean['selected_clients'][clean_idx])))
        plt.close()
    # plot for clean_model
    res = fmodule.abs(hist_clean['server_model'])
    x = np.array(torch.flatten(res.state_dict()[layer]).cpu())
    fig,ax=plt.subplots(figsize = (65, 30))
    plt.plot(x)
    ax.set_xticks([i for i in range(0, len(x)+1, 512)])
    plt.grid(axis = 'x', ls = '--')
    plt.title('Parameter of clean model for {}'.format(layer), fontsize=28)
    plt.xlabel('Parameter index in clean model', fontsize = 22)
    plt.ylabel('Value of parameter', fontsize = 22)
    plt.savefig(os.path.join(output_path, 'clean_model.png'))
    plt.close()

def plot_dirty(log_folder, device, round, layer):
    with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R250_P0.30_AP0.20_6atk_clean0/record/history{}.pkl'.format(round), 'rb') as test_f:
        hist_dirty = CPU_Unpickler(test_f).load()
    
    # with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R250_P0.30_AP0.20_6atk_clean1/record/history{}.pkl'.format(round), 'rb') as test_f:
    #     hist_clean = CPU_Unpickler(test_f).load()
    
    output_path = os.path.join(log_folder, "dirty_models", layer, "cross-entropy", "round{}".format(round))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    hist_dirty['server_model'].to(device)
    grads_dirty = [(hist_dirty['server_model'] - hist_dirty['models'][idx].to(device)) for idx in range(len(hist_dirty['selected_clients']))]
    for clean_idx in range(len(hist_dirty['selected_clients'])):
        # idx = hist_dirty['selected_clients'].index(hist_clean['selected_clients'][clean_idx])
        res = fmodule.abs(grads_dirty[clean_idx])
        x = np.array(torch.flatten(res.state_dict()[layer]).cpu())
        fig,ax=plt.subplots(figsize = (65, 30))
        plt.plot(x)
        ax.set_xticks([i for i in range(0, len(x)+1, 512)])
        plt.grid(axis = 'x', ls = '--')
        plt.title('Parameter of dirty grads for {}'.format(layer), fontsize=28)
        plt.xlabel('Parameter index in dirty grads', fontsize = 22)
        plt.ylabel('Value of parameter', fontsize = 22)
        plt.savefig(os.path.join(output_path, 'client{}.png'.format(hist_dirty['selected_clients'][clean_idx])))
        plt.close()
    # plot for delta
    res = fmodule.abs(hist_dirty['server_model'])
    x = np.array(torch.flatten(res.state_dict()[layer]).cpu())
    fig,ax=plt.subplots(figsize = (65, 30))
    plt.plot(x)
    ax.set_xticks([i for i in range(0, len(x)+1, 512)])
    plt.grid(axis = 'x', ls = '--')
    plt.title('Parameter of dirty model for {}'.format(layer), fontsize=28)
    plt.xlabel('Parameter index in dirty model', fontsize = 22)
    plt.ylabel('Value of parameter', fontsize = 22)
    plt.savefig(os.path.join(output_path, 'dirty_model.png'))
    plt.close()

# def plot_unlearn_term(log_folder, max_iters, layer="fc2.weight"):
#     output_path = os.path.join(log_folder, 'unlearn_term', layer)
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#     avg_unlearn2 = [0]
#     avg_unlearn3 = [0]
#     avg_delta = []
#     for rid in range(max_iters):
#         with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R5_P0.30_AP0.20_ALGO0_clean0/record/history{}.pkl'.format(rid), 'rb') as test_f:
#             hist_dirty = CPU_Unpickler(test_f).load()
#         with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R5_P0.30_AP0.20_ALGO0_clean1/record/history{}.pkl'.format(rid), 'rb') as test_f:
#             hist_clean = CPU_Unpickler(test_f).load()
#         delta = hist_clean['server_model'] - hist_dirty['server_model']
#         avg_delta.append(fmodule._model_avg_param(delta, list_layers = [layer]))
#         ## unlearn term
#         avg_unlearn2.append(fmodule._model_avg_param(hist_dirty['unlearn_term_algo2'], list_layers = [layer]))
#         avg_unlearn3.append(fmodule._model_avg_param(hist_dirty['unlearn_term_algo3'], list_layers = [layer]))
#     # last round
#     with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R5_P0.30_AP0.20_ALGO0_clean0/record/history{}.pkl'.format(max_iters), 'rb') as test_f:
#         hist_dirty = CPU_Unpickler(test_f).load()
#     with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R5_P0.30_AP0.20_ALGO0_clean1/record/history{}.pkl'.format(max_iters), 'rb') as test_f:
#         hist_clean = CPU_Unpickler(test_f).load()
#     delta = hist_clean['server_model'] - hist_dirty['server_model']
#     avg_delta.append(fmodule._model_avg_param(delta, list_layers = [layer]))
#     plt.plot(avg_delta, label= "{}-diff_clean_dirty".format(layer))
#     plt.plot(avg_unlearn2, label= "{}-upperbound-unlearn2".format(layer))
#     plt.plot(avg_unlearn3, label= "{}-upperbound-unlearn3".format(layer))
#     plt.legend()
#     plt.xlabel('round')
#     plt.ylabel('weight')
#     plt.savefig(os.path.join(output_path, "{}.png".format(layer)))
#     plt.close()

#### ### abs(mean) ### ##### 
# def plot_unlearn_term(log_folder, dirty_name, clean_name, max_iters, output_file, layer="fc2.weight"):
#     output_path = os.path.join(log_folder, 'fig', 'unlearn_term', 'alpha-func')
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#     avg_unlearn2 = [0]
#     avg_unlearn3 = [0]
#     avg_delta = []
#     for rid in range(max_iters):
#         with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/'+ clean_name +'/record/history{}.pkl'.format(rid), 'rb') as test_f:
#             hist_clean = CPU_Unpickler(test_f).load()
#         with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/'+ dirty_name +'/record/history{}.pkl'.format(rid), 'rb') as test_f:
#             hist_dirty = CPU_Unpickler(test_f).load()
#         print(rid)
#         delta = hist_clean['server_model'] - hist_dirty['server_model']
#         delta_weight = delta.fc2.weight.cpu().detach().numpy()
#         delta_weight = delta_weight.reshape(-1,)
#         # avg_delta.append(abs(fmodule._model_avg_param(delta, list_layers = [layer])))
#         avg_delta.append(abs(np.mean(delta_weight)))
#         ## unlearn term
#         dirty_weight_2 = hist_dirty['unlearn_term_algo2'].fc2.weight.cpu().detach().numpy()
#         dirty_weight_2 = dirty_weight_2.reshape(-1,)
#         dirty_weight_3 = hist_dirty['unlearn_term_algo3'].fc2.weight.cpu().detach().numpy()
#         dirty_weight_3 = dirty_weight_3.reshape(-1,)
#         # avg_unlearn2.append(abs(fmodule._model_avg_param(hist_dirty['unlearn_term_algo2'], list_layers = [layer])))
#         # avg_unlearn3.append(abs(fmodule._model_avg_param(hist_dirty['unlearn_term_algo3'], list_layers = [layer])))
#         avg_unlearn2.append(abs(np.mean(dirty_weight_2)))
#         avg_unlearn3.append(abs(np.mean(dirty_weight_3)))
#         # if avg_unlearn3[-1] < avg_unlearn2[-1]:
#         #     import pdb; pdb.set_trace();
    
#     delta = hist_clean['server_model'] - hist_dirty['server_model']
#     delta_weight = delta.fc2.weight.cpu().detach().numpy()
#     delta_weight = delta_weight.reshape(-1,)
#     # avg_delta.append(abs(fmodule._model_avg_param(delta, list_layers = [layer])))
#     avg_delta.append(abs(np.mean(delta_weight)))
#     # avg_delta.append(abs(fmodule._model_avg_param(delta, list_layers = [layer])))
#     plt.plot(avg_delta, label= "{}-diff_clean_dirty".format(layer))
#     plt.plot(avg_unlearn2, label= "{}-upperbound-unlearn2".format(layer))
#     plt.plot(avg_unlearn3, label= "{}-upperbound-unlearn3".format(layer))
#     plt.legend()
#     plt.xlabel('round')
#     plt.ylabel('weight')
#     plt.savefig(os.path.join(output_path, "{}.png".format(output_file)))
#     plt.close()

#### ### mean(abs) ### ##### 
def plot_unlearn_term(log_folder, dirty_name, clean_name, max_iters, output_file, layer="fc2.weight"):
    output_path = os.path.join(log_folder, 'fig', 'unlearn_term', 'alpha-gen-law')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    avg_unlearn2 = [0]
    avg_unlearn3 = [0]
    avg_delta = []
    for rid in range(max_iters):
        with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/'+ clean_name +'/record/history{}.pkl'.format(rid), 'rb') as test_f:
            hist_clean = CPU_Unpickler(test_f).load()
        with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/'+ dirty_name +'/record/history{}.pkl'.format(rid), 'rb') as test_f:
            hist_dirty = CPU_Unpickler(test_f).load()
        print(rid)
        delta = hist_clean['server_model'] - hist_dirty['server_model']
        delta_weight = delta.fc2.bias.cpu().detach().numpy()
        delta_weight = delta_weight.reshape(-1,)
        # avg_delta.append(abs(fmodule._model_avg_param(delta, list_layers = [layer])))
        avg_delta.append(np.mean(np.absolute(delta_weight)))
        ## unlearn term
        dirty_weight_2 = hist_dirty['unlearn_term_algo2'].fc2.bias.cpu().detach().numpy()
        dirty_weight_2 = dirty_weight_2.reshape(-1,)
        dirty_weight_3 = hist_dirty['unlearn_term_algo3'].fc2.bias.cpu().detach().numpy()
        dirty_weight_3 = dirty_weight_3.reshape(-1,)
        # avg_unlearn2.append(abs(fmodule._model_avg_param(hist_dirty['unlearn_term_algo2'], list_layers = [layer])))
        # avg_unlearn3.append(abs(fmodule._model_avg_param(hist_dirty['unlearn_term_algo3'], list_layers = [layer])))
        avg_unlearn2.append(np.mean(np.absolute(dirty_weight_2)))
        avg_unlearn3.append(np.mean(np.absolute(dirty_weight_3)))
        # if avg_unlearn3[-1] < avg_unlearn2[-1]:
        #     import pdb; pdb.set_trace();
    
    delta = hist_clean['server_model'] - hist_dirty['server_model']
    delta_weight = delta.fc2.bias.cpu().detach().numpy()
    delta_weight = delta_weight.reshape(-1,)
    # avg_delta.append(abs(fmodule._model_avg_param(delta, list_layers = [layer])))
    avg_delta.append(np.mean(np.absolute(delta_weight)))
    # avg_delta.append(abs(fmodule._model_avg_param(delta, list_layers = [layer])))
    plt.plot(avg_delta, label= "{}-diff_clean_dirty".format(layer))
    plt.plot(avg_unlearn2, label= "{}-upperbound-unlearn2".format(layer))
    plt.plot(avg_unlearn3, label= "{}-upperbound-unlearn3".format(layer))
    plt.legend()
    plt.xlabel('round')
    plt.ylabel('weight')
    plt.savefig(os.path.join(output_path, "{}.png".format(output_file)))
    plt.close()

def plot_accuracy(log_folder, dirty_name, max_iters, main_acc, output_file):
    output_path = os.path.join(log_folder, 'fig', 'accuracy', 'acc-alpha_gen_law')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    unlearn2_acc = []
    unlearn3_acc = []
    for rid in range(max_iters):
        with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/'+ dirty_name +'/record/history{}.pkl'.format(rid), 'rb') as test_f:
            hist_dirty = CPU_Unpickler(test_f).load()
        print(rid)
        unlearn2_acc.append(hist_dirty["accuracy2"][0])
        unlearn3_acc.append(hist_dirty["accuracy3"][0])
    
    plt.plot(unlearn2_acc, label= "{}-acc-unlearn2".format(output_file))
    plt.plot(unlearn3_acc, label= "{}-acc-unlearn3".format(output_file))
    plt.plot(main_acc, label= "{}-acc-w".format(output_file))
    plt.legend()
    plt.xlabel('round')
    plt.ylabel('accuracy')
    plt.savefig(os.path.join(output_path, "{}.png".format(output_file)))
    plt.close()
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    task_name = 'mnist_cnum100_dist2_skew0.5_seed0'
    task_model = 'cnn'
    bmk_name = task_name[:task_name.find('cnum')-1].lower()
    bmk_model_path = '.'.join(['benchmark', bmk_name, 'model', task_model])
    bmk_core_path = '.'.join(['benchmark', bmk_name, 'core'])
    # utils.fmodule.device = torch.device('cuda:{}'.format(option['server_gpu_id']) if torch.cuda.is_available() and option['server_gpu_id'] != -1 else 'cpu')
    utils.fmodule.Model = getattr(importlib.import_module(bmk_model_path), 'Model')
    device = torch.device('cuda:0')
    max_iters = 250
    list_layers = ['fc2.weight']
    
    # list_alpha_avg_avg = []
    # list_alpha_avg_max = []
    # list_alpha_max_avg = []
    # list_alpha_max_max = []
    # for round_idx in range(max_iters + 1):
    #     with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R250_P0.30_AP0.20_6atk_clean0/record/history{}.pkl'.format(round_idx), 'rb') as test_f:
    #         history_dirty = CPU_Unpickler(test_f).load()
    
    #     with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R250_P0.30_AP0.20_6atk_clean1/record/history{}.pkl'.format(round_idx), 'rb') as test_f:
    #         history_clean = CPU_Unpickler(test_f).load()
        
    #     avg_avg, max_avg, avg_max, max_max = compute_alpha(history_dirty, history_clean, device, list_layers)
    #     list_alpha_avg_avg.append(avg_avg)
    #     list_alpha_avg_max.append(avg_max)
    #     list_alpha_max_avg.append(max_avg)
    #     list_alpha_max_max.append(max_max)
    #     if avg_avg > 10: print(round_idx)
    #     print("Done round {}".format(round_idx))
    
    # plot_alpha('./result', [list_alpha_avg_avg, list_alpha_max_avg, list_alpha_avg_max, list_alpha_max_max], list_layers)
    # save_alpha_result('./result', [list_alpha_avg_avg, list_alpha_max_avg, list_alpha_avg_max, list_alpha_max_max], list_layers)
    # plot_params_alpha('./result', device, 250, 'fc2.weight')
    # r_count = [10, 25, 50]
    # alp = []
    # for rid in r_count:
    #     if rid == 50:
    #         alp = [0.01, 0.02, 0.05, 0.07, 0.1, 0.2]
    #     elif rid == 25:
    #         alp = [0.5, 0.7, 1.0, 1.5, 2.0, 5.0, 7.0]
    #     else: 
    #         alp = [10.0, 32.0]
    #     for alpha in alp:
    # alpha = 0.04
    # plot_unlearn_term('./result', 'R50_P0.30_AP0.20_Alpha{}_10atk'.format(alpha), 50, "alpha{}-10atker".format(alpha), layer="fc2.weight")
            # with open("run_experiments.sh", "w") as file:
            #     file.write(f"CUDA_VISIBLE_DEVICES=0 python main.py --task mnist_cnum100_dist2_skew0.5_seed0 --unlearn_algorithm 0 --proportion 0.3 --attacker_pct 0.2 --theta_delta 1 --gamma_epsilon {alpha} --model cnn --algorithm fedavg --aggregate weighted_com --num_rounds {rid} --num_epochs 5 --learning_rate 0.001 --batch_size 10 --clean_model 0 --eval_interval 1\n")
            # # bash run_experiments.sh
            # os.system('bash run_experiments.sh')
            # save figure
    alpha = 1.0
    rid = 100
    atk = 6
    alp = 0.05
    lay = 'fc2.bias'
    # with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/check_nan_model/record/model2.pkl'.format(rid), 'rb') as test_f:
    #     hist = CPU_Unpickler(test_f).load()
    # xzero = torch.zeros(10, 1, 28, 28)
    # model_nan = hist['model']
    # import pdb; pdb.set_trace()
    # print(model_nan(xzero))
    # for alp in [0.05, 0.09]:
    #     with open("run_experiments.sh", "w") as file:
    #         file.write(f"CUDA_VISIBLE_DEVICES=0 python main.py --task mnist_cnum100_dist2_skew0.5_seed0 --unlearn_algorithm 0 --proportion 0.3 --attacker_pct 0.2 --theta_delta 1 --gamma_epsilon {alp} --model cnn --algorithm fedavg --aggregate weighted_com --num_rounds 100 --num_epochs 5 --learning_rate 0.001 --batch_size 10 --clean_model 0 --eval_interval 1\n")
    #     # bash run_experiments.sh
    #     os.system('bash run_experiments.sh')
    # plot_unlearn_term('./result', 'R2_P0.30_AP0.20_Alpha1.0_{}atk_minus{}'.format(atk, alp), 'R100_P0.30_AP0.20_Alpha0.0_{}atk'.format(atk), rid, lay + "-{}atker_minus{}".format(atk, alp), layer=lay)
    # main_acc = [0.0972, 0.1145, 0.1225, 0.1135, 0.1135, 0.1135, 0.1733, 0.1135, 0.1783, 0.192, 0.1419, 0.2551, 0.281, 0.4246, 0.5188, 0.598, 0.3902, 0.5164, 0.5993, 0.5083, 0.607, 0.5133, 0.617, 0.5276, 0.5926, 0.6318, 0.5707, 0.6573, 0.7187, 0.6084, 0.6768, 0.7117, 0.7278, 0.7439, 0.7615, 0.777, 0.792, 0.8029, 0.8109, 0.8175, 0.8248, 0.7815, 0.811, 0.8284, 0.8382, 0.8428, 0.8495, 0.8539, 0.7034, 0.7655, 0.8077, 0.8325, 0.8496, 0.8605, 0.8669, 0.8713, 0.8734, 0.8758, 0.8759, 0.8761, 0.8781, 0.8787, 0.8804, 0.8823, 0.8846, 0.8884, 0.8901, 0.8932, 0.8949, 0.8968, 0.8987, 0.8992, 0.9003, 0.9008, 0.9013, 0.9021, 0.9039, 0.9044, 0.9065, 0.9075, 0.908, 0.8227, 0.8506, 0.8713, 0.8858, 0.8963, 0.9032, 0.9077, 0.9115, 0.9134, 0.9152, 0.917, 0.9183, 0.9193, 0.9194, 0.9201, 0.9209, 0.9209, 0.9213, 0.9216, 0.9222, 0.9229, 0.9234, 0.9236, 0.9239, 0.9247, 0.9252, 0.9254, 0.926, 0.926, 0.926, 0.9263, 0.9267, 0.9271, 0.9274, 0.9275, 0.9279,0.9284, 0.9287, 0.9289, 0.9293, 0.9295, 0.93, 0.9304, 0.9309, 0.9314, 0.9318, 0.9323, 0.9326, 0.9329, 0.9331, 0.9335, 0.9339, 0.934, 0.9341, 0.9343, 0.935, 0.9352, 0.9357, 0.9358, 0.9364, 0.9365, 0.9365, 0.9368, 0.9369, 0.9369, 0.9372, 0.9373, 0.9373, 0.9374, 0.9376, 0.9383, 0.9384, 0.9387, 0.9391, 0.9392,0.9394, 0.9394, 0.9395, 0.9396, 0.9398, 0.9401, 0.9402, 0.9402, 0.9405, 0.9408, 0.9408, 0.9411, 0.9414, 0.9417, 0.9417, 0.9425, 0.9433, 0.8842, 0.9102, 0.8226, 0.861, 0.8863, 0.904, 0.9173, 0.9266, 0.9326, 0.9367, 0.9398, 0.9422, 0.9433, 0.9445, 0.9445, 0.9453, 0.946, 0.9468, 0.9474, 0.9479, 0.9484, 0.9487, 0.9493, 0.9495, 0.9503, 0.9506, 0.9509, 0.9513]
    # main_acc = [0.1145,0.1245,0.1372,0.1137,0.1449,0.1137,0.1722,0.1161,0.1669,0.1927,0.3435,0.3327,0.2506,0.3975,0.5234,0.6069,0.6615,0.4618,0.5706,0.4747,0.5896,0.6561,0.6921,0.5951,0.6305,0.6665,0.5757,0.6436,0.69,0.7255,0.623,0.6854,0.7162,0.7343,0.7503,0.7682,0.784,0.797,0.6311,0.7125,0.7649,0.7993,0.8204,0.8323,0.8408,0.8453,0.8493,0.8541,0.8592,0.8617,0.8648,0.8672,0.87,0.8724,0.7995,0.8388,0.8583,0.871,0.8785,0.8821,0.8848,0.8861,0.888,0.8903,0.8931,0.7727,0.7974,0.8162,0.8342,0.852,0.8681,0.8782,0.8864,0.8933,0.8975,0.9004,0.903,0.9052,0.9059,0.9084,0.9096,0.911,0.9124,0.9136,0.9143,0.9161,0.9166,0.9177,0.9183,0.9192,0.9195,0.92,0.9213,0.9217,0.9221,0.9225,0.9228,0.923,0.9233,0.924,0.9248]
    # plot_accuracy('./result', 'R100_P0.30_AP0.20_Alpha1.0_{}atk_minus{}'.format(atk, alp), rid, main_acc, "{}atker_test_minus{}".format(atk, alp))
    # acc = []
    # for id in range(35):
    #     with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R100_P0.30_AP0.20_Alpha1.0_6atk_minus0.09/record/history{}.pkl'.format(id), 'rb') as test_f:
    #         hist_dirty = CPU_Unpickler(test_f).load()
    #         acc.append(hist_dirty["accuracy3"][0])
    # print(acc)
    # ### Bang Nguyen Trong - test
    # # fedtask_name = 'R300_P0.30_AP0.20_clean0'
    # # # task_name = 'mnist_cnum100_dist2_skew0.5_seed0'
    # # root_path = './fedtasksave/' + task_name + '/' + fedtask_name + '/record/'

    # # with open(root_path + 'history0.pkl', 'rb') as test_f:
    # #     history_dirty = CPU_Unpickler(test_f).load()
    
    # # with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R300_P0.30_AP0.20_clean1/record/history0.pkl', 'rb') as test_f:
    # #     history_clean = CPU_Unpickler(test_f).load()
    
    