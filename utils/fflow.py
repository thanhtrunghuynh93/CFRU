import numpy as np
import argparse
import random
import torch
import os.path
import importlib
import os
import utils.fmodule
import ujson
import time

sample_list=['uniform', 'md', 'active']
agg_list=['uniform', 'weighted_scale', 'weighted_com', 'none']
optimizer_list=['SGD', 'Adam']
attack_methods = ['fedAttack', 'fedFlipGrads', 'none']

def read_option():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument('--task', help='name of fedtask;', type=str, default='mnist_cnum100_dist0_skew0_seed0')
    parser.add_argument('--algorithm', help='name of algorithm;', type=str, default='fedavg')
    parser.add_argument('--model', help='name of model;', type=str, default='cnn')
    parser.add_argument('--atk_method', help='method for poisoning global model', type=str, choices=attack_methods, default='none')

    # methods of server side for sampling and aggregating
    parser.add_argument('--sample', help='methods for sampling clients', type=str, choices=sample_list, default='uniform')
    parser.add_argument('--aggregate', help='methods for aggregating models', type=str, choices=agg_list, default='weighted_com')
    parser.add_argument('--learning_rate_decay', help='learning rate decay for the training process;', type=float, default=0.998)
    parser.add_argument('--weight_decay', help='weight decay for the training process', type=float, default=0)
    parser.add_argument('--lr_scheduler', help='type of the global learning rate scheduler', type=int, default=-1)
    # hyper-parameters of training in server side
    parser.add_argument('--num_rounds', help='number of communication rounds', type=int, default=20)
    
    ## hyper-parameters of training
    parser.add_argument('--proportion', help='proportion of clients sampled per round', type=float, default=0.3) # 0.3 / 1
    parser.add_argument('--theta_delta', help='coefficient multiply with delta each round', type=float, default=1) # 0.5 / 0.8 / 1
    parser.add_argument('--alpha', help='coefficient multiply with epsilon (difference between grad of U and grad of W)', type=float, default=1) # 0.5 / 0.8 / 1
    ## end
    
    
    # hyper-parameters of local training
    parser.add_argument('--num_epochs', help='number of epochs when clients trainset on data;', type=int, default=5)
    parser.add_argument('--learning_rate', help='learning rate for inner solver;', type=float, default=0.1)
    parser.add_argument('--batch_size', help='batch size when clients trainset on data;', type=int, default=64)
    parser.add_argument('--optimizer', help='select the optimizer for gd', type=str, choices=optimizer_list, default='Adam')
    parser.add_argument('--momentum', help='momentum of local update', type=float, default=0)

    # machine environment settings
    parser.add_argument('--seed', help='seed for random initialization;', type=int, default=0)
    parser.add_argument('--eval_interval', help='evaluate every __ rounds;', type=int, default=1)
    parser.add_argument('--num_threads', help='the number of threads;', type=int, default=1)
    parser.add_argument('--num_threads_per_gpu', help="the number of threads per gpu in the clients computing session;", type=int, default=1)
    parser.add_argument('--num_gpus', default=3, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    # the simulating system settings of clients
    parser.add_argument('--S1', type=int, default=8)
    # hyperparams for GNN
    parser.add_argument('--embedding.size', type=int, default=64)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--reg.lambda', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--topN', type=int, default=10)
    parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
    parser.add_argument("--test_num_ng", type=int, default=99, help="sample part of negative items for testing")

    
    # constructing the heterogeity of the network
    parser.add_argument('--net_drop', help="controlling the dropout of clients after being selected in each communication round according to distribution Beta(drop,1)", type=float, default=0)
    parser.add_argument('--net_active', help="controlling the probability of clients being active and obey distribution Beta(active,1)", type=float, default=99999)
    # constructing the heterogeity of computing capability
    parser.add_argument('--capability', help="controlling the difference of local computing capability of each client", type=float, default=0)
    
    # clean or not
    parser.add_argument('--clean_model', help='clean_model equals 1 in order to run clean model and 0 otherwise', type=int, default=2)
    
    # server gpu
    parser.add_argument('--server_gpu_id', help='server process on this gpu', type=int, default=0)
    
    try: option = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    return option

def setup_seed(seed):
    random.seed(1+seed)
    np.random.seed(21+seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(12+seed) 
    torch.cuda.manual_seed_all(123+seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize(option):
    # init fedtask
    print("init fedtask...", end='')
    # dynamical initializing the configuration with the benchmark
    bmk_name = option['task'][:option['task'].find('cnum')-1].lower()
    bmk_model_path = '.'.join(['benchmark', bmk_name, 'model', option['model']])
    bmk_core_path = '.'.join(['benchmark', bmk_name, 'core'])
    utils.fmodule.device = torch.device('cuda:{}'.format(option['server_gpu_id']) if torch.cuda.is_available() and option['server_gpu_id'] != -1 else 'cpu')
    utils.fmodule.TaskCalculator = getattr(importlib.import_module(bmk_core_path), 'TaskCalculator')
    utils.fmodule.TaskCalculator.setOP(getattr(importlib.import_module('torch.optim'), option['optimizer']))
    utils.fmodule.Model = getattr(importlib.import_module(bmk_model_path), 'Model')
    task_reader = getattr(importlib.import_module(bmk_core_path), 'TaskReader')(taskpath=os.path.join('fedtask', option['task']))
    client_train_datas, test_data, backdoor_data, users_per_client, data_conf, clients_config, client_names= task_reader.read_data(option['num_ng'], option['model'])
    num_clients = len(client_names)
    # import pdb; pdb.set_trace()
    print("done")
    if users_per_client == None:
        users_per_client = [None] * num_clients
    # init data of attackers
    if num_clients == 100:
        s_atk = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    elif num_clients == 10:
        s_atk = [0]
    else:
        raise ValueError('Not enough clients')
    malicious_users = []
    for cid in s_atk:
        malicious_users = malicious_users + users_per_client[cid]
    option['attacker'] = s_atk
    option['malicious_users'] = malicious_users
    
    # set config in fmodule
    utils.fmodule.data_conf = data_conf
    utils.fmodule.option = option

    # init client
    # import pdb; pdb.set_trace()
    print('init clients...', end='')
    client_path = '%s.%s' % ('algorithm', option['algorithm'])
    Client=getattr(importlib.import_module(client_path), 'Client')
    clients = [Client(option, name = client_names[cid], model = utils.fmodule.Model(clients_config[cid], option).to(utils.fmodule.device), 
                    train_data = client_train_datas[cid], users_set = users_per_client[cid]) for cid in range(num_clients)]

    # init server
    print("init server...", end='')
    #
    server_path = '%s.%s' % ('algorithm', option['algorithm'])
    server = getattr(importlib.import_module(server_path), 'Server')(option, utils.fmodule.Model(data_conf, option).to(utils.fmodule.device), clients, test_data = test_data, backtask_data = backdoor_data)
    print('done')
    return server

def output_filename(option, server):
    header = "{}_".format(option["algorithm"])
    for para in server.paras_name: header = header + para + "{}_".format(option[para])
    output_name = header + "M{}_ES{}_RL{}_R{}_E{}_LR{}_B{}_top{}_seed{}_alp{}_clean{}_cap{}.json".format(
        option['model'],
        option['embedding.size'],
        option['reg.lambda'],
        option['num_rounds'],
        option['num_epochs'],
        option['learning_rate'],
        option['batch_size'],
        option['topN'],
        option['seed'],
        option['alpha'],
        option['clean_model'],
        option['capability']
        )
    return output_name

class Logger:
    def __init__(self):
        self.output = {}
        self.current_round = -1
        self.temp = "{:<30s}{:.4f}"
        self.time_costs = []
        self.time_buf={}

    def check_if_log(self, round, eval_interval=-1):
        """For evaluating every 'eval_interval' rounds, check whether to log at 'round'."""
        self.current_round = round
        return eval_interval > 0 and (round == 0 or round % eval_interval == 0)

    def time_start(self, key = ''):
        """Create a timestamp of the event 'key' starting"""
        if key not in [k for k in self.time_buf.keys()]:
            self.time_buf[key] = []
        self.time_buf[key].append(time.time())

    def time_end(self, key = ''):
        """Create a timestamp that ends the event 'key' and print the time interval of the event."""
        if key not in [k for k in self.time_buf.keys()]:
            raise RuntimeError("Timer end before start.")
        else:
            self.time_buf[key][-1] =  time.time() - self.time_buf[key][-1]
            print("{:<30s}{:.4f}".format(key+":", self.time_buf[key][-1]) + 's')

    def save(self, filepath):
        ## print accuracy
        print(self.output["test_accs"])
        
        """Save the self.output as .json file"""
        if self.output=={}: return
        with open(filepath, 'w') as outf:
            ujson.dump(self.output, outf)
            
    def write(self, var_name=None, var_value=None):
        """Add variable 'var_name' and its value var_value to logger"""
        if var_name==None: raise RuntimeError("Missing the name of the variable to be logged.")
        if var_name in [key for key in self.output.keys()]:
            self.output[var_name] = []
        self.output[var_name].append(var_value)
        return

    def log(self, server=None):
        pass
    
def convert(o):
    if isinstance(o, np.generic): return o.item()  
    raise TypeError
