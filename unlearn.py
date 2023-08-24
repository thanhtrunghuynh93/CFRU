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
import ujson
import copy
from torch.utils.data import Dataset, DataLoader
# from test_saved_models import NumpyEncoder, CPU_Unpickler
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
    utils.fmodule.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    with open(os.path.join(log_folder, task_name, hist_dirty_name, 'record', 'history0.pkl'), 'rb') as test_f:
        hist_dirty = CPU_Unpickler(test_f).load()
    unlearning_term3 = copy.deepcopy(hist_dirty['server_model']).to(fmodule.device) * 0.0
    alpha = - alp
    test_all_round = {}
    for idx in range(len(round_attack)):
        round_id = round_attack[idx]
        with open(os.path.join(log_folder, task_name, hist_dirty_name, 'record', 'history{}.pkl'.format(round_id)), 'rb') as test_f:
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
    task_save_path = os.path.join('./result', hist_dirty_name)
    if not os.path.exists(task_save_path): os.makedirs(task_save_path, exist_ok=True)
    with open(os.path.join(task_save_path, 'R{}_M{}_AM{}.json'.format(num_round, model, atk_method))) as json_file:
        json.dump(test_all_round, json_file)
    
"""
"selected_clients": self.selected_clients,
"updates": important_weights,
"p": [1.0 * self.client_vols[cid] / self.data_vol for cid in self.selected_clients],
"server_model": self.model,
"client_model": models,
"not_top": not_tops
"""
import time
def offline_unlearn_important(log_folder, task_name, hist_dirty_name, alp, num_round, num_neg, model = 'NCF', atk_method = 'fedAttack'):
    # init 
    task_model = model
    bmk_name = task_name[:task_name.find('cnum')-1].lower()
    bmk_model_path = '.'.join(['benchmark', bmk_name, 'model', task_model])
    bmk_core_path = '.'.join(['benchmark', bmk_name, 'core'])
    utils.fmodule.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    utils.fmodule.TaskCalculator = getattr(importlib.import_module(bmk_core_path), 'TaskCalculator')
    utils.fmodule.TaskCalculator.setOP(getattr(importlib.import_module('torch.optim'), 'SGD'))
    utils.fmodule.Model = getattr(importlib.import_module(bmk_model_path), 'Model')
    task_reader = getattr(importlib.import_module(bmk_core_path), 'TaskReader')(taskpath=os.path.join('fedtask', task_name))
    client_train_datas, test_data, backdoor_data, users_per_client, data_conf, clients_config, client_names= task_reader.read_data(num_neg, task_model)
    # set config in fmodule
    option = {
        'dropout': 0.2,
        'embedding.size': 64,
        'n_layer': 2
    }
    utils.fmodule.data_conf = data_conf
    utils.fmodule.option = option
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
    with open(os.path.join(log_folder, task_name, hist_dirty_name, 'record', 'history0.pkl'), 'rb') as test_f:
        hist_dirty = CPU_Unpickler(test_f).load()
    unlearning_term3 = copy.deepcopy(hist_dirty['server_model']).to(fmodule.device) * 0.0
    unlearning_term_top1 = copy.deepcopy(unlearning_term3)
    unlearning_term_top5 = copy.deepcopy(unlearning_term3)
    unlearning_term_top10 = copy.deepcopy(unlearning_term3)
    # import pdb; pdb.set_trace()
    alpha = - alp
    test_all_round = {}
    for idx in range(len(round_attack)):
        round_id = round_attack[idx]
        with open(os.path.join(log_folder, task_name, hist_dirty_name, 'record', 'history{}.pkl'.format(round_id)), 'rb') as test_f:
            hist_dirty = CPU_Unpickler(test_f).load()
        ##
        w = fmodule._model_sum([model_k * pk for model_k, pk in zip(hist_dirty['client_model'], hist_dirty['p'])])
        server_model = (1.0-sum(hist_dirty['p']))*hist_dirty['server_model'] + w
        ##
        # start_time = time.time()
        beta = 0.0
        for pid in range(len(hist_dirty['p'])):
            if hist_dirty['selected_clients'][pid] not in attackers_round[idx]:
                beta += hist_dirty['p'][pid]
        beta = beta * alpha + 1
        # compute unlearning term with inductive method
        # unlearning_term3 = unlearning_term3 * beta
        unlearning_term_top1 = unlearning_term_top1 * beta
        unlearning_term_top5 = unlearning_term_top5 * beta
        unlearning_term_top10 = unlearning_term_top10 * beta

        for pid in range(len(hist_dirty['p'])):
            if hist_dirty['selected_clients'][pid] in attackers_round[idx]:
                update = hist_dirty['updates'][pid].to(fmodule.device)
                # M_v.embed_item.weight[not_topk_items, :] = 0
                update_top1 = - copy.deepcopy(update)
                for param in update_top1.parameters(): param.requires_grad = False
                update_top1.embed_item.weight[torch.tensor(hist_dirty['not_top'][pid]['p0.2']).cuda(), :] = 0
                for param in update_top1.parameters(): param.requires_grad = True
                # # import pdb; pdb.set_trace()
                update_top5 = - copy.deepcopy(update)
                for param in update_top5.parameters(): param.requires_grad = False
                update_top5.embed_item.weight[torch.tensor(hist_dirty['not_top'][pid]['p0.5']).cuda(), :] = 0
                for param in update_top5.parameters(): param.requires_grad = True
                # #
                update_top10 = - copy.deepcopy(update)
                for param in update_top10.parameters(): param.requires_grad = False
                update_top10.embed_item.weight[torch.tensor(hist_dirty['not_top'][pid]['p1.0']).cuda(), :] = 0
                for param in update_top10.parameters(): param.requires_grad = True
                # unlearning_term3 += 1.0 * hist_dirty['p'][pid] * hist_dirty['updates'][str(pid)].to(fmodule.device) ### ????
                unlearning_term_top1 += 1.0 * hist_dirty['p'][pid] * update_top1
                unlearning_term_top5 += 1.0 * hist_dirty['p'][pid] * update_top5
                unlearning_term_top10 += 1.0 * hist_dirty['p'][pid] * update_top10
        # get unlearned-model and test
        # unlearn_model3 = server_model + unlearning_term3
        unlearn_model_top1 = server_model + unlearning_term_top1
        unlearn_model_top5 = server_model + unlearning_term_top5
        unlearn_model_top10 = server_model + unlearning_term_top10
        # end_time = time.time()
        print('Done round{}'.format(round_id))
        ## test phase
        if round_id == 40:
            test_results_top1 = offline_test_on_clients(test_dataloader, hist_dirty['client_model'], users_per_client, unlearn_model_top1)
            test_results_top5 = offline_test_on_clients(test_dataloader, hist_dirty['client_model'], users_per_client, unlearn_model_top5)
            test_results_top10 = offline_test_on_clients(test_dataloader, hist_dirty['client_model'], users_per_client, unlearn_model_top10)
            # save all results
            test_all_round['round{}'.format(round_id)] = {
                '20p': test_results_top1,
                '50p': test_results_top5,
                '100p': test_results_top10,
            }
            print('Done round{}'.format(round_id))
            print(test_all_round['round{}'.format(round_id)])
            task_save_path = os.path.join('./result', hist_dirty_name)
            if not os.path.exists(task_save_path): os.makedirs(task_save_path, exist_ok=True)
            with open(os.path.join(task_save_path, 'R{}_M{}_AM{}_alp{}.json'.format(num_round, model, atk_method, alp)), 'w') as json_file:
                ujson.dump(test_all_round, json_file)

# hyperparam tuning
def offline_unlearn_track(log_folder, task_name, hist_dirty_name, alp, num_round, num_neg, model = 'NCF', atk_method = 'fedAttack'):
    # init 
    main_test = []
    task_model = model
    bmk_name = task_name[:task_name.find('cnum')-1].lower()
    bmk_model_path = '.'.join(['benchmark', bmk_name, 'model', task_model])
    bmk_core_path = '.'.join(['benchmark', bmk_name, 'core'])
    utils.fmodule.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    utils.fmodule.TaskCalculator = getattr(importlib.import_module(bmk_core_path), 'TaskCalculator')
    utils.fmodule.TaskCalculator.setOP(getattr(importlib.import_module('torch.optim'), 'SGD'))
    utils.fmodule.Model = getattr(importlib.import_module(bmk_model_path), 'Model')
    task_reader = getattr(importlib.import_module(bmk_core_path), 'TaskReader')(taskpath=os.path.join('fedtask', task_name))
    client_train_datas, test_data, backdoor_data, users_per_client, data_conf, clients_config, client_names= task_reader.read_data(num_neg, task_model)
    # set config in fmodule
    option = {
        'dropout': 0.2,
        'embedding.size': 64,
        'n_layer': 2
    }
    utils.fmodule.data_conf = data_conf
    utils.fmodule.option = option
    # get test_dataloader
    def seed_worker():
        worker_seed = 0
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(0) # 0
    test_dataloader = DataLoader(test_data, batch_size=100, shuffle=False, drop_last=False, worker_init_fn=seed_worker, generator=g)
    # unlearn process
    with open(os.path.join(log_folder, task_name, hist_dirty_name, 'record', 'history{}.pkl'.format(num_round)), 'rb') as test_f:
        hist_dirty = CPU_Unpickler(test_f).load()
    w = fmodule._model_sum([model_k * pk for model_k, pk in zip(hist_dirty['client_model'], hist_dirty['p'])])
    server_model = (1.0-sum(hist_dirty['p']))*hist_dirty['server_model'] + w
    ul_term = copy.deepcopy(hist_dirty['server_model']).to(fmodule.device) * 0.0
    # tracking
    for round_track in range(0, num_round+1):
        # track from 0 to end_round - 1
        attackers =  list(range(10)) # fix attacker from 0 to 9
        round_attack = [rid for rid in range(round_track, num_round+1)]
        attackers_round = [attackers for rid in range(round_track, num_round+1)]
        # unlearn 
        unlearning_term = copy.deepcopy(ul_term)
        # import pdb; pdb.set_trace()
        alpha = - alp
        test_all_round = {}
        for idx in range(len(round_attack)):
            round_id = round_attack[idx]
            with open(os.path.join(log_folder, task_name, hist_dirty_name, 'record', 'history{}.pkl'.format(round_id)), 'rb') as test_f:
                hist_dirty = CPU_Unpickler(test_f).load()
            ##
            # start_time = time.time()
            beta = 0.0
            for pid in range(len(hist_dirty['p'])):
                if hist_dirty['selected_clients'][pid] not in attackers_round[idx]:
                    beta += hist_dirty['p'][pid]
            beta = beta * alpha + 1
            # compute unlearning term with inductive method
            unlearning_term = unlearning_term * beta

            for pid in range(len(hist_dirty['p'])):
                if hist_dirty['selected_clients'][pid] in attackers_round[idx]:
                    update = hist_dirty['updates'][pid].to(fmodule.device)
                    # M_v.embed_item.weight[not_topk_items, :] = 0
                    update_top10 = - copy.deepcopy(update)
                    for param in update_top10.parameters(): param.requires_grad = False
                    update_top10.embed_item.weight[torch.tensor(hist_dirty['not_top'][pid]['p1.0']).cuda(), :] = 0
                    for param in update_top10.parameters(): param.requires_grad = True
                    # unlearning_term3 += 1.0 * hist_dirty['p'][pid] * hist_dirty['updates'][str(pid)].to(fmodule.device) ### ????
                    unlearning_term += 1.0 * hist_dirty['p'][pid] * update_top10
            ## test phase
            if round_id == num_round:
                unlearn_model = server_model + unlearning_term
                # end_time = time.time()
                print('Done unlearn')
                test_results_top10 = offline_test_on_clients(test_dataloader, hist_dirty['client_model'], users_per_client, unlearn_model)
                # save all results
                main_test.append(test_results_top10)
                print('Done round{}'.format(round_track))

        task_save_path = os.path.join('./result', hist_dirty_name, 'save_track')
        save_logs = {
            'accuracy': main_test
        }
        if not os.path.exists(task_save_path): os.makedirs(task_save_path, exist_ok=True)
        with open(os.path.join(task_save_path, 'R{}_M{}_AM{}_alp{}.json'.format(num_round, model, atk_method, alp)), 'w') as json_file:
            ujson.dump(save_logs, json_file)

def offline_test(log_folder, task_name, hist_dirty_name, alp, num_round, num_neg, model = 'NCF', atk_method = 'fedAttack'):
    # init 
    task_model = model
    bmk_name = task_name[:task_name.find('cnum')-1].lower()
    bmk_model_path = '.'.join(['benchmark', bmk_name, 'model', task_model])
    bmk_core_path = '.'.join(['benchmark', bmk_name, 'core'])
    utils.fmodule.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    utils.fmodule.TaskCalculator = getattr(importlib.import_module(bmk_core_path), 'TaskCalculator')
    utils.fmodule.TaskCalculator.setOP(getattr(importlib.import_module('torch.optim'), 'SGD'))
    utils.fmodule.Model = getattr(importlib.import_module(bmk_model_path), 'Model')
    task_reader = getattr(importlib.import_module(bmk_core_path), 'TaskReader')(taskpath=os.path.join('fedtask', task_name))
    client_train_datas, test_data, backdoor_data, users_per_client, data_conf, clients_config, client_names= task_reader.read_data(num_neg, task_model)
    # set config in fmodule
    option = {
        'dropout': 0.2,
        'embedding.size': 64,
        'n_layer': 2
    }
    utils.fmodule.data_conf = data_conf
    utils.fmodule.option = option
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

    alpha = - alp
    test_all_round = {}
    for idx in range(len(round_attack)):
        round_id = round_attack[idx]
        with open(os.path.join(log_folder, task_name, hist_dirty_name, 'record', 'history{}.pkl'.format(round_id)), 'rb') as test_f:
            hist_dirty = CPU_Unpickler(test_f).load()
        ##
        w = fmodule._model_sum([model_k * pk for model_k, pk in zip(hist_dirty['client_model'], hist_dirty['p'])])
        server_model = (1.0-sum(hist_dirty['p']))*hist_dirty['server_model'] + w
        
        ## test phase
        test_results_top1 = offline_test_on_clients(test_dataloader, hist_dirty['client_model'], users_per_client, server_model)
        # save all results
        test_all_round['round{}'.format(round_id)] = {
            '20p': test_results_top1
        }
        print('Done round{}'.format(round_id))
        print(test_all_round['round{}'.format(round_id)])
        # compute unlearn_model
        # if round_id % 5 == 0:
        #     task_save_path = os.path.join('./result', hist_dirty_name)
        #     if not os.path.exists(task_save_path): os.makedirs(task_save_path, exist_ok=True)
        #     with open(os.path.join(task_save_path, 'R{}_M{}_AM{}.json'.format(num_round, model, atk_method)), 'w') as json_file:
        #         ujson.dump(test_all_round, json_file)

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

    # for alp in [8, 9, 10, 11, 12]:
        # with open('./fedtasksave/movielens1m_cnum10_dist11_skew0.0_seed0/FedEraser_BPR_R12_P0.30_alpha0.09_clean2_seed0/record/history{}.pkl'.format(alp), 'rb') as test_f:
        #     hist = CPU_Unpickler(test_f).load()
        # print('HR: {} | NDCG: {}'.format(hist['HR_on_clients'], hist['NDCG_on_clients']))
    for mt in ['fedAttack']:#['fedFlipGrads', 'fedAttack']:
        offline_unlearn_track(
            log_folder='./fedtasksave', 
            task_name='pinterest_cnum100_dist11_skew0.0_seed0', 
            hist_dirty_name='ULOpt_NCF_R40_P0.30_alpha0.1_seed0_' + mt, 
            alp=0.8,
            num_round=40, 
            num_neg=1, 
            model = 'NCF', 
            atk_method = mt
            )

