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

def plot_accuracy(log_folder, dirty_name, max_iters, output_file):
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
    rid = 2
    atk = 3
    alp = 1.0
    lay = 'fc2.bias'
    # with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/check_nan_model/record/model2.pkl'.format(rid), 'rb') as test_f:
    #     hist = CPU_Unpickler(test_f).load()
    # xzero = torch.zeros(10, 1, 28, 28)
    # model_nan = hist['model']
    # import pdb; pdb.set_trace()
    # print(model_nan(xzero))
    for alp in [0.1, 0.2]:
        with open("run_experiments.sh", "w") as file:
            file.write(f"CUDA_VISIBLE_DEVICES=1 python main.py --task mnist_cnum100_dist2_skew0.5_seed0 --unlearn_algorithm 0 --proportion 0.3 --attacker_pct 0.2 --theta_delta 1 --gamma_epsilon {alp} --model cnn --algorithm fedavg --aggregate weighted_com --num_rounds 200 --num_epochs 5 --learning_rate 0.001 --batch_size 10 --clean_model 0 --eval_interval 1\n")
        # bash run_experiments.sh
        os.system('bash run_experiments.sh')
    # plot_unlearn_term('./result', 'R2_P0.30_AP0.20_Alpha1.0_{}atk_minus{}'.format(atk, alp), 'R100_P0.30_AP0.20_Alpha0.0_{}atk'.format(atk), rid, lay + "-{}atker_minus{}".format(atk, alp), layer=lay)
    # plot_accuracy('./result', 'R100_P0.30_AP0.20_AlphaGen_{}atk_minus{}'.format(atk, alp), rid, "{}atker_test_minus{}".format(atk, alp))
    # ### Bang Nguyen Trong - test
    # # fedtask_name = 'R300_P0.30_AP0.20_clean0'
    # # # task_name = 'mnist_cnum100_dist2_skew0.5_seed0'
    # # root_path = './fedtasksave/' + task_name + '/' + fedtask_name + '/record/'

    # # with open(root_path + 'history0.pkl', 'rb') as test_f:
    # #     history_dirty = CPU_Unpickler(test_f).load()
    
    # # with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R300_P0.30_AP0.20_clean1/record/history0.pkl', 'rb') as test_f:
    # #     history_clean = CPU_Unpickler(test_f).load()
    
    # # a, b, c, d = compute_alpha(history_dirty, history_clean, device)
    # # print(a)
    # # print(b)
    # # print(c)
    # # print(d)
    # # plot_alpha('./result', [a, b, c, d], list_layers)
    # for i in [223]:
    # plot_clean('./result', device, round=25, layer='fc2.weight')
    # plot_unlearn_term('./result', max_iters= 5, layer='fc2.weight')
    # with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R5_P0.30_AP0.20_ALGO0_clean0/record/history{}.pkl'.format(0), 'rb') as test_f:
    #     hist_dirty = CPU_Unpickler(test_f).load()
    # with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R5_P0.30_AP0.20_ALGO0_clean1/record/history{}.pkl'.format(0), 'rb') as test_f:
    #     hist_clean = CPU_Unpickler(test_f).load()

    # with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R5_P0.30_AP0.20_ALGO0_clean0/record/history{}.pkl'.format(1), 'rb') as test_f:
    #     hist_dirty1 = CPU_Unpickler(test_f).load()
    # with open('./fedtasksave/mnist_cnum100_dist2_skew0.5_seed0/R5_P0.30_AP0.20_ALGO0_clean1/record/history{}.pkl'.format(1), 'rb') as test_f:
    #     hist_clean1 = CPU_Unpickler(test_f).load()
    # print((hist_dirty['p'][0]*(hist_dirty['server_model'] - hist_dirty['models'][0]) - (hist_clean1['server_model']- hist_dirty1['server_model'])).state_dict())
    # w = fmodule._model_sum([model_k * pk for model_k, pk in zip(hist_clean['models'], hist_clean['p'])])
    # w1 = (hist_clean['p'][0] * hist_clean['models'][0]) + (hist_clean['p'][1] * hist_clean['models'][1]) + (hist_clean['p'][2] * hist_clean['models'][2])
    # w1 = p0 + p1 + p2
    # w2 = (hist_dirty['p'][0] * hist_dirty['models'][0]) + q0 + q1 + q2
    # print(((1.0-sum(hist_clean['p']))*hist_clean['server_model'] + w1 - hist_clean1['server_model']).state_dict())
    # print(hist_dirty['p'][3] - hist_clean['p'][2])
    # w2 = (hist_dirty['p'][0] * hist_dirty['models'][0]) + (hist_dirty['p'][1] * hist_dirty['models'][1]) + (hist_dirty['p'][2] * hist_dirty['models'][2]) + (hist_dirty['p'][3] * hist_dirty['models'][3])
    # print(((1.0-sum(hist_dirty['p']))*hist_dirty['server_model'] + w2 - hist_dirty1['server_model']).state_dict())
    # print((hist_clean['server_model']- hist_dirty['server_model']).state_dict())

    # print((w2 - w1 - (hist_dirty['p'][0] * hist_dirty['models'][0])).state_dict())
    # idx = 2
    # p0 = (hist_dirty['p'][1] * hist_dirty['models'][1])
    # q0 = (hist_clean['p'][0] * hist_clean['models'][0])

    # p1 = (hist_dirty['p'][2] * hist_dirty['models'][2]) 
    # q1 = (hist_clean['p'][1] * hist_clean['models'][1])

    # p2 = (hist_dirty['p'][3] * hist_dirty['models'][3]) 
    # q2 = (hist_clean['p'][2] * hist_clean['models'][2])

    # x = (hist_dirty['p'][0] * hist_dirty['models'][0])
    # w1 = p0 + p1 + p2
    # w2 = x + q0 + q1 + q2
    # print((p0 + p1 - q0 - q1).state_dict())

    # print((p0 - q0).state_dict())
    # print((p1 - q1).state_dict())
    # print((p2 - q2).state_dict())

    # print(((hist_dirty['p'][idx + 1] * hist_dirty['models'][idx + 1]) - (hist_clean['p'][idx] * hist_clean['models'][idx])).state_dict())
