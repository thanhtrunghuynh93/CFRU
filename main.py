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
                "client_test_accs":[],
                "all_selected_clients":[]
            }
        
        test_result = server.test_on_clients()

        self.output['client_test_accs'].append(test_result)
        
        self.output['all_selected_clients'].append([int(id) for id in server.selected_clients])

        print("Hit Ratio and NDCG for top {} recommendation!".format(server.topN))
        print("HR@{}:{} and  NDCG@{}:{}".format(server.topN, test_result[0], server.topN, test_result[1]))
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




