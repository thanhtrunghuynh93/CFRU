from .fedbase import BasicServer, BasicClient
import torch
import pickle
import os
import copy
from utils import fmodule
from tqdm import tqdm
from multiprocessing import Pool as ThreadPool
from main import logger
import numpy as np
import torch.nn.functional as F 

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None, backtask_data = None):
        super(Server, self).__init__(option, model, clients, test_data, backtask_data)
        self.path_save = os.path.join('fedtasksave', self.option['task'],
                                    "FedPruning_R{}_P{:.2f}_AP{:.2f}_1atk_seed{}_eta{}".format(
                                        option['num_rounds'],
                                        option['proportion'],
                                        option['attacker_pct'],
                                        option['seed'],
                                        option['theta_delta']
                                        # option['clean_model']
                                    ),
                                    'record')
        self.unlearn_term_algo2 = None
        self.unlearn_term_algo3 = None
        # tuning param
        # eta = 0.05
        self.eta = option['theta_delta']
        # self.seed = self.option['seed']
        # FLerase
        self.local_epoch = self.option['num_epochs']
        self.global_epoch = self.option['num_rounds']
        self.target_class = 0
        if self.option['task'].startswith("mnist"):
            self.target_class = 7
        elif self.option['task'].startswith("cifar10"):
            self.target_class = 0
        elif self.option['task'].startswith("medmnist") or self.option['task'].startswith("tissue"):
            self.target_class = 0
        else:
            raise Exception("Invalid value for task name")
        # self.N_client = self.option['N_client']
        self.N_client = 10
        self.unlearn_interval = 1
        self.forget_local_epoch_ratio = 0.5
        
        # create folder for saving model
        print(self.path_save)
        if not os.path.exists(self.path_save):
            os.makedirs(self.path_save, exist_ok=True)
    
    def save_models(self, main_accuracy, backdoor_accuracy, unlearn_time, unlearn_model):
        # log
        save_logs = {
            "main": main_accuracy,
            "backdoor": backdoor_accuracy,
            "time": unlearn_time,
            "unlearn_model": unlearn_model
        }
        pickle.dump(save_logs,
                    open(os.path.join(self.path_save, "history" + str(self.global_epoch) + ".pkl"), 'wb'),
                    pickle.HIGHEST_PROTOCOL)
        print("Save  ", self.global_epoch)
    
    def run(self):
        # all_global_models = list()
        # all_client_models = list()
        # global_model = self.model
        # all_global_models.append(copy.deepcopy(self.model))
        for round in tqdm(range(self.num_rounds)):
            print("--------------Round {}--------------".format(round))
            self.round = round
            _global_model, _client_models = self.iterate(round)
            # all_client_models += client_models
            # all_global_models.append(copy.deepcopy(global_model))
        global_model = copy.deepcopy(_global_model)
        client_models = copy.deepcopy(_client_models)
        logger.time_start('unlearning time')
        unlearn_GM = self.unlearning(global_model, client_models)
        unlearn_time = logger.time_end('unlearning time')
        eval_metric, loss, eval_backdoor = self.test(unlearn_GM)
        print(eval_metric)
        print(eval_backdoor)
        self.save_models(eval_metric, eval_backdoor, unlearn_time, unlearn_GM)
        print("------------End---------------")
        return unlearn_GM
    
    def unlearning(self, old_GMs, old_CMs):
        global_model = copy.deepcopy(old_GMs)
        client_models = copy.deepcopy(old_CMs)
        # global_model = old_global_models[-1]
        # client_models = old_client_models[-len(self.selected_clients):]
        A_locals = self._local_proc(client_models)
        tf_idf = self._global_proc(A_locals)
        unlearn_global_model = self._pruning(global_model, tf_idf)
        return unlearn_global_model
    
    def _pruning(self, global_model, tf_idf):
        def index_to_prune(tf_idf):
            return (tf_idf >= self.eta).nonzero()
        unlearn_global_model = copy.deepcopy(global_model)
        return_state_dict = unlearn_global_model.state_dict()
        for layer in unlearn_global_model.state_dict().keys():
            if "conv1" in layer:
                return_state_dict[layer][index_to_prune(tf_idf[0])] = 0
            if "conv2" in layer:
                return_state_dict[layer][index_to_prune(tf_idf[1])] = 0
        unlearn_global_model.load_state_dict(return_state_dict)
        return unlearn_global_model
    
    def _global_proc(self, A_locals):
        # 2 is cnn layers
        A_l_1 = []
        A_l_2 = []
        for client in A_locals:
            A_l_1.append(client[0])
            A_l_2.append(client[1])
        A_l_1 = torch.mean(torch.stack(A_l_1), dim=0)
        A_l_2 = torch.mean(torch.stack(A_l_2), dim=0)
        A_star = [A_l_1, A_l_2]
        TF_u = []
        IDF_u = []
        TF_IDF_u = []
        target_class = self.target_class
        for layer in range(len(A_star)):
            A_u_quotaion = A_star[layer][target_class]
            TF_u.append(torch.div(A_u_quotaion, torch.sum(A_u_quotaion)))
            IDF_u_c = []
            for j in range(A_u_quotaion.shape[0]):
                count_u = 0
                for u in range(10):
                    _temp_A = A_star[layer][u]
                    if _temp_A[j] >= torch.mean(_temp_A):
                        count_u += 1 
                IDF_u_c.append(torch.log(torch.div(11, 1+count_u)))
            IDF_u.append(IDF_u_c)
        for layer in range(len(A_star)):
            TF_IDF_layer = []
            for c in range(len(IDF_u[layer])): 
                TF_IDF_layer.append(torch.mul(TF_u[layer][c], IDF_u[layer][c]))
            TF_IDF_u.append(torch.stack(TF_IDF_layer))
        return TF_IDF_u

    def _local_proc(self, client_models):
        A_locals = list()
        for client_id in range(len(client_models)):
            model = client_models[client_id]
            client = self.clients[client_id]
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output.detach()
                return hook
            for name, layer in model.named_modules():
                layer.register_forward_hook(get_activation(name))
            data_loader = client.calculator.get_data_loader(client.train_data, batch_size=client.batch_size)
            gts = []
            A_l_1 = []
            A_l_2 = []
            for batch_id, batch_data in enumerate(data_loader):
                _ = model(batch_data[0].cuda()) # batchsize * num_class
                for layer in activation:
                    if layer == "conv1":
                        A_l_1.append(F.avg_pool2d(F.relu(activation[layer]), (activation[layer].shape[2], activation[layer].shape[3])).squeeze())
                    if layer == "conv2":
                        A_l_2.append(F.avg_pool2d(F.relu(activation[layer]), (activation[layer].shape[2], activation[layer].shape[3])).squeeze())
                gts.append(batch_data[1])
            A_l_1 = torch.cat(A_l_1)
            A_l_2 = torch.cat(A_l_2)
            gts = torch.cat(gts)
            A_l_1_u = list()
            A_l_2_u = list()
            for i in range(10):
                indexes = (batch_data[1] == i).nonzero(as_tuple=True)[0]
                if len(indexes) > 0:
                    a_i_1 = torch.mean(A_l_1[indexes], dim=0)
                    a_i_2 = torch.mean(A_l_2[indexes], dim=0)
                else:
                    a_i_1 = torch.zeros(A_l_1.shape[1]).to(A_l_1.device)
                    a_i_2 = torch.zeros(A_l_2.shape[1]).to(A_l_2.device)
                A_l_1_u.append(a_i_1)
                A_l_2_u.append(a_i_2)
            A_l_1_u = torch.stack(A_l_1_u)
            A_l_2_u = torch.stack(A_l_2_u)
            A_locals.append([A_l_1_u, A_l_2_u])
        return A_locals   

    def communicate(self, selected_clients):
        """
        The whole simulating communication procedure with the selected clients.
        This part supports for simulating the client dropping out.
        :param
            selected_clients: the clients to communicate with
        :return
            :the unpacked response from clients that is created ny self.unpack()
        """
        packages_received_from_clients = []
        if self.num_threads <= 1:
            # computing iteratively
            for client_id in selected_clients:
                response_from_client_id = self.communicate_with(client_id)
                packages_received_from_clients.append(response_from_client_id)
        else:
            # computing in parallel
            pool = ThreadPool(min(self.num_threads, len(selected_clients)))
            packages_received_from_clients = pool.map(self.communicate_with, selected_clients)
            pool.close()
            pool.join()
        # count the clients not dropping
        self.selected_clients = [selected_clients[i] for i in range(len(selected_clients)) if packages_received_from_clients[i]]
        packages_received_from_clients = [pi for pi in packages_received_from_clients if pi]
        models, x, y = self.unpack(packages_received_from_clients)
        return models, x, y
 
    def iterate(self, t):
        # sample clients: MD sampling as default but with replacement=False
        self.selected_clients = self.sample(t)
        attack_clients = []

        for cid in self.selected_clients: 
            if cid in self.option['attacker']:
                attack_clients.append(cid) 
        
        for idx in self.selected_clients:
            self.round_selected[idx].append(t)

        models, _, _ = self.communicate(self.selected_clients)

        self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        return self.model, models
    
# original sample clients
    def sample(self, t):
        ##
        if self.option['clean_model'] == 0: 
            if self.option['attacker_pct'] == 2:
                selected_clients = [0, 3, 5, 6, 8, 16, 19, 22, 24, 25]
            elif self.option['attacker_pct'] == 3:
                selected_clients = [0, 1, 2, 4, 5, 8, 14, 18, 21, 22]
            elif self.option['attacker_pct'] == 4:
                selected_clients = [0, 1, 2, 3, 5, 6, 14, 16, 18, 25]
            else:
                selected_clients = [i for i in range(10)]
        else:
            raise Exception("Invalid value for attribute clean_model")
        # selected_clients = [10]
        return selected_clients


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)


