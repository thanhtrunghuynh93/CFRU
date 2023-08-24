import utils.fflow as flw
import numpy as np
import torch
import os
import multiprocessing

class MyLogger(flw.Logger):
    def log(self, server=None):
        if server==None: return
        if self.output == {}:
            self.output = {
                "meta":server.option,
                "test_accs":[],
                "backdoor_accs":[],
                "client_test_accs":[],
                "client_backdoor_accs":[],
                "all_selected_clients":[]
            }
        # if "mp_" in server.name:
        #     test_metric, backdoor_metric = server.test(device=torch.device('cuda:0'))
        # else:
        #     test_metric, backdoor_metric = server.test()
        
        client_test_metrics, client_backdoor_metrics = server.test_on_clients(self.current_round)
        # compute HR and NDCG for test
        HR = 0.0
        NDCG = 0.0
        for metric in client_test_metrics:
            HR = HR + metric[0]
            NDCG = NDCG + metric[1]
        self.output['client_test_accs'].append([float(HR)/len(client_test_metrics), float(NDCG)/len(client_test_metrics)])
        # compute HR and NDCG for backdoor
        # HR = 0.0
        # NDCG = 0.0
        # for metric in client_backdoor_metrics:
        #     HR = HR + metric[0]
        #     NDCG = NDCG + metric[1]
        # self.output['client_backdoor_accs'].append([float(HR)/len(client_backdoor_metrics), float(NDCG)/len(client_backdoor_metrics)])
        # 
        # self.output['test_accs'].append(test_metric)
        # self.output['backdoor_accs'].append(backdoor_metric)
        self.output['all_selected_clients'].append([int(id) for id in server.selected_clients])

        print("Hit Ratio and NDCG for top {} recommendation!".format(server.topN))
        # print("Testing Metric:")
        # print(self.output['test_accs'][-1])
        # print("Backdoor Metric:")
        # print(self.output['backdoor_accs'][-1])
        print("Client test: main test {}".format(self.output['client_test_accs'][-1]))
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
    server = flw.initialize(option)
                   # start federated optimization
    server.run()
    # print(server.round_selected)

if __name__ == '__main__':
    main()




