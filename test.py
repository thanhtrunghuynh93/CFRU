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
import copy
from torch.utils.data import Dataset, DataLoader
from test_saved_models import NumpyEncoder, CPU_Unpickler
import torch.nn as nn

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def offline_unlearn(log_folder, task_name, hist_dirty_name, alp, num_round, num_neg, model = 'NCF', atk_method = 'fedAttack'):
    # init 
    task_model = model
    bmk_name = task_name[:task_name.find('cnum')-1].lower()
    bmk_model_path = '.'.join(['benchmark', bmk_name, 'model', task_model])
    bmk_core_path = '.'.join(['benchmark', bmk_name, 'core'])
    utils.fmodule.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.fmodule.TaskCalculator = getattr(importlib.import_module(bmk_core_path), 'TaskCalculator')
    utils.fmodule.TaskCalculator.setOP(getattr(importlib.import_module('torch.optim'), 'SGD'))
    utils.fmodule.Model = getattr(importlib.import_module(bmk_model_path), 'Model')
    task_reader = getattr(importlib.import_module(bmk_core_path), 'TaskReader')(taskpath=os.path.join('fedtask', task_name))
    client_train_datas, test_data, backdoor_data, users_per_client, data_conf, clients_config, client_names= task_reader.read_data(num_neg, task_model)
    # get test_dataloader
    def seed_worker():
        worker_seed = 0
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(0) # 0
    test_dataloader = DataLoader(test_data, batch_size=100, shuffle=False, drop_last=False, worker_init_fn=seed_worker, generator=g)
    # unlearn process
    # track from 0 to end_round - 1
    attackers =  list(range(10)) # fix attacker from 0 to 9
    round_attack = [rid for rid in range(0, num_round+1)]
    attackers_round = [attackers for rid in range(0, num_round+1)]
    # unlearn 
    with open(os.path.join(log_folder, hist_dirty_name, 'record', 'history0.pkl'), 'rb') as test_f:
        hist_dirty = CPU_Unpickler(test_f).load()
    unlearning_term3 = copy.deepcopy(hist_dirty['server_model']).to(fmodule.device) * 0.0
    alpha = - alp
    test_all_round = {}
    for idx in range(len(round_attack)):
        round_id = round_attack[idx]
        with open(os.path.join(log_folder, hist_dirty_name, 'record', 'history{}.pkl'.format(round_id)), 'rb') as test_f:
            hist_dirty = CPU_Unpickler(test_f).load()
        beta = 0.0
        for pid in range(len(hist_dirty['p'])):
            if hist_dirty['selected_clients'][pid] not in attackers_round[idx]:
                beta += hist_dirty['p'][pid]
        beta = beta * alpha + 1
        # compute unlearning term with inductive method
        unlearning_term3 = unlearning_term3 * beta
        for pid in range(len(hist_dirty['p'])):
            if hist_dirty['selected_clients'][pid] in attackers_round[idx]:
                unlearning_term3 += 1.0 * hist_dirty['p'][pid] * hist_dirty['updates'][str(pid)].to(fmodule.device) ### ????
        # get unlearned-model and test
        unlearn_model3 = hist_dirty['server_model'] + unlearning_term3
        test_results = offline_test_on_clients(test_dataloader, hist_dirty['client_model'], users_per_client, unlearn_model3)
        test_all_round['round{}'.format(round_id)] = test_results
    # compute unlearn_model
    task_save_path = os.path.join('./result', task_name)
    if not os.path.exists(task_save_path): os.makedirs(task_save_path, exist_ok=True)
    with open(os.path.join(task_save_path, 'R{}_M{}_AM{}.json'.format(num_round, model, atk_method))) as json_file:
        json.dump(test_all_round, json_file)
    

def offline_test_on_clients(test_dataloader, client_models, users_sets, server_model=None):
    # init 
    list_test_metrics = {
        'top1': [],
        'top5': [],
        'top10': []
    }
    num_clients = len(client_models)
    attackers = list(range(10))
    # test on each client[idx]
    for idx in range(num_clients):
        if idx in attackers: continue
        if server_model == None: model=client_models[idx]
        else:
            model = copy.deepcopy(client_models[idx])
            fmodule._model_merge_(model, server_model)

        if test_dataloader:
            model.to(fmodule.device)
            #
            model.eval()
            for top_k in [1, 5, 10]:
                test_metric = test(model, test_dataloader, top_k, users_sets[idx])
                # add test metric of client idx to metric list
                list_test_metrics['top{}'.format(top_k)].append(test_metric)
            #
            model.to('cpu')
    # compute average test value for topK
    test_results = {}
    for topK in ['top1', 'top5', 'top10']:
        HR = 0.0
        NDCG = 0.0
        for metric in list_test_metrics[topK]:
            HR = HR + metric[0]
            NDCG = NDCG + metric[1]
        mean_hr = float(HR)/len(list_test_metrics[topK])
        mean_ndcg = float(NDCG)/len(list_test_metrics[topK])
        test_results[topK] = [mean_hr, mean_ndcg]
    return test_results

def test(model, test_loader, top_k, users_test=None):
    def hit(gt_item, pred_items):
        if gt_item in pred_items:
            return 1
        return 0


    def ndcg(gt_item, pred_items):
        if gt_item in pred_items:
            index = pred_items.index(gt_item)
            return np.reciprocal(np.log2(index+2))
        return 0
    
    HR, NDCG = [], []
    for user, item_i, item_j in test_loader:
        if user[0].item() not in users_test:
            continue
        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j.cuda() # not useful when testing
        
        # prediction_i, prediction_j = model((user, item_i, item_j))
        # _, indices = torch.topk(prediction_i, top_k)
        output = model((user, item_i, item_j))
        indices = model.handle_test(output, top_k)
        recommends = torch.take(
                item_i, indices).cpu().numpy().tolist()

        gt_item = item_i[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))
    return np.mean(HR), np.mean(NDCG)

if __name__ == '__main__':
    # for atk in ['fedAttack', 'fedFlipGrads']:
    #     with open('./fedtasksave/movielens1m_cnum100_dist11_skew0.0_seed0/ULKD_R30_P0.30_alpha0.03_seed0_{}/record/history30.pkl'.format(atk), 'rb') as test_f:
    #         hist = CPU_Unpickler(test_f).load()
    #     print(hist)
    #     # print('HR: {} | NDCG: {}'.format(hist['HR_on_clients'], hist['NDCG_on_clients']))

    for alp in [8, 9, 10, 11, 12]:
        with open('./fedtasksave/movielens1m_cnum10_dist11_skew0.0_seed0/FedEraser_BPR_R12_P0.30_alpha0.09_clean2_seed0/record/history{}.pkl'.format(alp), 'rb') as test_f:
            hist = CPU_Unpickler(test_f).load()
        print('HR: {} | NDCG: {}'.format(hist['HR_on_clients'], hist['NDCG_on_clients']))