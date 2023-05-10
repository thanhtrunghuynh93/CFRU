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
class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None, backtask_data = None):
        super(Server, self).__init__(option, model, clients, test_data, backtask_data)
        self.path_save = os.path.join('fedtasksave', self.option['task'],
                                    "FedKD_R{}_P{:.2f}_AP{:.2f}_1atk".format(
                                        option['num_rounds'],
                                        option['proportion'],
                                        option['attacker_pct']
                                        # option['clean_model']
                                    ),
                                    'record')
        self.unlearn_term_algo2 = None
        self.unlearn_term_algo3 = None
        
        # FLerase
        self.local_epoch = self.option['num_epochs']
        self.global_epoch = self.option['num_rounds']
        self.if_unlearning = False
        self.forget_client_idx = self.option['attacker'][0]
        # self.N_client = self.option['N_client']
        self.N_client = 10
        self.unlearn_interval = 1
        self.forget_local_epoch_ratio = 0.5
        self.batch_size = option['batch_size']
        # create folder for saving model
        print(self.path_save)
        if not os.path.exists(self.path_save):
            os.makedirs(self.path_save, exist_ok=True)
    
    def save_models(self, main_accuracy, backdoor_accuracy, unlearn_time, unlearn_model, global_model, grads_target):
        # log
        save_logs = {
            "main": main_accuracy,
            "backdoor": backdoor_accuracy,
            "time": unlearn_time,
            "unlearn_model": unlearn_model,
            "global_model": global_model,
            "grads": grads_target
        }
        pickle.dump(save_logs,
                    open(os.path.join(self.path_save, "history" + str(self.global_epoch) + ".pkl"), 'wb'),
                    pickle.HIGHEST_PROTOCOL)
        print("Save  ", self.global_epoch)
    
    def run(self):
        grads_target_client = []
        for round in tqdm(range(self.num_rounds)):
            print("--------------Round {}--------------".format(round))
            self.round = round
            _global_model, grad_target_client = self.iterate(round)
            grads_target_client.append(grad_target_client)
        global_model = copy.deepcopy(_global_model)
        logger.time_start('unlearning time')
        unlearn_GM = self.unlearning(global_model, grads_target_client)
        unlearn_time = logger.time_end('unlearning time')
        # self.if_unlearning = True
        # unlearn_GM, _ = self.iterate(self.num_rounds+1, w_u_i)
        eval_metric, loss, eval_backdoor = self.test(unlearn_GM)
        print(eval_metric)
        print(eval_backdoor)
        self.save_models(eval_metric, eval_backdoor, unlearn_time, unlearn_GM, global_model, grads_target_client)
        print("------------End---------------")
        return unlearn_GM
        
    def unlearning(self, global_model, grads_target_client):
        # Input   
        MF = copy.deepcopy(global_model)
        MF_apos = copy.deepcopy(global_model)
        grads_target_client = copy.deepcopy(grads_target_client)
        temp_state_dict = MF.state_dict()
        grads_layer = {layer: [] for layer in MF.state_dict().keys()}
        for i in range(len(grads_target_client)):
            for layer, value in temp_state_dict.items():
                if "norm" in layer:
                    continue
                grads_layer[layer].append(grads_target_client[i].state_dict()[layer])
                
        for layer in grads_layer:
            if "norm" in layer:
                continue
            grads_layer[layer] = sum(grads_layer[layer])/len(grads_layer[layer])
            
        mf_apos_state_dict = {}
        for layer, value in temp_state_dict.items():
            if "norm" in layer:
                mf_apos_state_dict[layer] = value
                continue
            mf_apos_state_dict[layer] = MF.state_dict()[layer].cpu() + grads_layer[layer]
        MF_apos.load_state_dict(mf_apos_state_dict)
        
        # Data X
        # rawdata_path='./benchmark/mnist/data'
        from torchvision import datasets, transforms
        from torch.autograd import Variable
        import torch.utils.data as data
        import medmnist
        from medmnist import INFO, Evaluator    
        from utils import fmodule
        # from benchmark.uncertainty_loss import one_hot_embedding
        import torch.nn.functional as F
        if self.option['task'].startswith("mnist"):
            rawdata_path='./benchmark/mnist/data'
            self.kd_data = data.random_split(datasets.MNIST(rawdata_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                                            [10000, 50000])
        elif self.option['task'].startswith("cifar10"):
            rawdata_path='./benchmark/cifar10/data'
            transform_train = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: F.pad(
                                                    Variable(x.unsqueeze(0), requires_grad=False),
                                                    (4,4,4,4),mode='reflect').data.squeeze()),
                                transforms.ToPILImage(),
                                transforms.RandomCrop(32),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                                    std=[x/255.0 for x in [63.0, 62.1, 66.7]])
                                ])
            self.kd_data = data.random_split(datasets.CIFAR10(rawdata_path, train=True, download=True, transform=transform_train),
                                                [10000, 40000])
        elif self.option['task'].startswith("medmnist"):
            rawdata_path='./benchmark/medmnist/data'
            data_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[.5], std=[.5])
                ])
            # 0, 12, 13
            data_flag = 'octmnist'
            info = INFO[data_flag]
            DataClass = getattr(medmnist, info['python_class'])
            self.kd_data = data.random_split(DataClass(split='train', transform=data_transform, download=True, root=rawdata_path),
                                                                        [10000, 87477])
        elif self.option['task'].startswith("tissue"):
            rawdata_path='./benchmark/tissue/data'
            data_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[.5], std=[.5])
                ])
            # 0, 12, 13
            data_flag = 'tissuemnist'
            info = INFO[data_flag]
            DataClass = getattr(medmnist, info['python_class'])
            self.kd_data = data.random_split(DataClass(split='train', transform=data_transform, download=True, root=rawdata_path),
                                                                        [10000, 155466])
        else:
            raise Exception("Invalid value for task name")
        # self.kd_data = data.random_split(datasets.MNIST(rawdata_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),[10000, 50000])
        calculator = fmodule.TaskCalculator(device=fmodule.device)
        optimizer = calculator.get_optimizer(self.option['optimizer'], MF_apos, lr = self.option['learning_rate'], weight_decay=self.option['weight_decay'], momentum=self.option['momentum'])
        data_loader = calculator.get_data_loader(self.kd_data[0], batch_size = self.batch_size)
        self.lossfunc = torch.nn.CrossEntropyLoss()
        kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
        
        def data_to_device(data):
            return data[0].to(calculator.device), data[1].to(calculator.device)
        def dist_loss(teacher, student, T=1):
            prob_t = F.softmax(teacher/T, dim=1)
            log_prob_s = F.log_softmax(student/T, dim=1)
            # dist_loss = -(prob_t*log_prob_s).sum(dim=1).mean()
            dist_loss = kl_loss(log_prob_s, prob_t)
            return dist_loss
        MF.eval()
        for iter in range(5):
            for batch_id, batch_data in enumerate(data_loader):
                tdata = data_to_device(batch_data)
                teacher_output = MF(tdata[0])
                
                MF_apos.zero_grad()
                tdata = data_to_device(batch_data)
                student_output = MF_apos(tdata[0])
                loss_student = self.lossfunc(student_output, tdata[1].squeeze())
                loss = loss_student + dist_loss(teacher_output, student_output)
                loss.backward()
                optimizer.step()
        return MF_apos

    def communicate(self, selected_clients, global_model):
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
                response_from_client_id = self.communicate_with(client_id, global_model)
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
        if((self.if_unlearning) and (self.forget_client_idx in range(self.N_client))):
            models.pop(self.forget_client_idx)
            x.pop(self.forget_client_idx)
            x.pop(self.forget_client_idx)
        return models, x, y
    
    def communicate_with(self, client_id, global_model):
        """
        Pack the information that is needed for client_id to improve the global model
        :param
            client_id: the id of the client to communicate with
        :return
            client_package: the reply from the client and will be 'None' if losing connection
        """
        # package the necessary information
        svr_pkg = self.pack(client_id, global_model)
        # listen for the client's response and return None if the client drops out
        if self.clients[client_id].is_drop(): return None
        return self.clients[client_id].reply(svr_pkg)
    
    def pack(self, client_id, global_model):
        """
        Pack the necessary information for the client's local training.
        Any operations of compression or encryption should be done here.
        :param
            client_id: the id of the client to communicate with
        :return
            a dict that only contains the global model as default.
        """
        if global_model:
            model = global_model
        else:
            model = self.model
        return {
            "model" : copy.deepcopy(model),
            "round" : self.round,
        }
    
    def iterate(self, t, global_model=None):
        # sample clients: MD sampling as default but with replacement=False
        self.selected_clients = self.sample(t)
        attack_clients = []

        for cid in self.selected_clients: 
            if cid in self.option['attacker']:
                attack_clients.append(cid) 
        
        for idx in self.selected_clients:
            self.round_selected[idx].append(t)

        models, _, _ = self.communicate(self.selected_clients, global_model)
        grads_target_client = (self.model - models[0]).cpu()
        self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        return self.model, grads_target_client
    
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


