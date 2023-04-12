from torchvision import datasets, transforms
from benchmark.toolkits import ClassifyCalculator, DefaultTaskGen, XYTaskReader
from yaml import load
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os.path as osp
import os
import ujson

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5):
        super(TaskGen, self).__init__(benchmark='pinterest',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/pinterest/data',
                                      )
        self.num_classes = 10
        self.save_data = self.XYData_to_json

    def load_data(self):
        self.train_data, self.test_data, self.user_num, self.item_num, self.train_mat = self.load_all(os.path.join(self.rawdata_path, 'pinterest-20.train.rating'),
                                                                                                      os.path.join(self.rawdata_path, 'pinterest-20.test.negative'))
        print('----Start loading removal data----')
        self.removal_data = self.load_removal(user_removal=0)
        print('----Finish loading removal data----')

    def load_all(self, train_path, test_path, test_num=100):
        """ We load all the three file here to save time in each epoch. """
        train_data = pd.read_csv(
            train_path, 
            sep='\t', header=None, names=['user', 'item'], 
            usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
        
        user_num = train_data['user'].max() + 1
        item_num = train_data['item'].max() + 1

        train_data = train_data.values.tolist()

        # load ratings as a dok matrix
        train_mat = {} # sp.dok_matrix((user_num, item_num), dtype=np.float32)
        for x in train_data:
            # train_mat[x[0], x[1]] = 1.0
            if train_mat.get(x[0]) is None:
                train_mat[x[0]] = [x[1]]
            else:
                train_mat[x[0]].append(x[1])

        test_data = []
        with open(test_path, 'r') as fd:
            line = fd.readline()
            while line != None and line != '':
                arr = line.split('\t')
                u = eval(arr[0])[0]
                test_data.append([u, eval(arr[0])[1]])
                for i in arr[1:]:
                    test_data.append([u, int(i)])
                line = fd.readline()
        return train_data, test_data, user_num, item_num, train_mat
    
    def load_removal(self, user_removal):
        removal_data = []
        for data in self.test_data:
            if data[0] == user_removal:
                removal_data.append(data)
        # randomly sample 100 
        list_interacted = []
        list_non_interacted = []
        # latest item interaction
        latest_item = removal_data[0][1]
        list_interacted.append(latest_item)
        for data in self.train_data:
            if data[0] == user_removal:
                list_interacted.append(data[1])
        
        for item in range(self.item_num):
            if item not in list_interacted:
                list_non_interacted.append(item)
        # randomly generate test set
        import random
        random.seed(42)
        for user in range(199):
            removal_data.append([user_removal, latest_item])
            random_selection = random.sample(list_non_interacted, 99)
            for neg_item in random_selection:
                removal_data.append([user_removal, neg_item])
        # import pdb; pdb.set_trace()
        return removal_data

class TaskReader(XYTaskReader):
    def __init__(self, taskpath=''):
        super(TaskReader, self).__init__(taskpath)

class TaskCalculator(ClassifyCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)

