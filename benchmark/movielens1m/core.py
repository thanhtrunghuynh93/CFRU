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
        super(TaskGen, self).__init__(benchmark='movielens1m',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/movielens1m/data',
                                      )
        self.num_classes = 10
        self.save_data = self.XYData_to_json

    def load_data(self):
        self.train_data, self.test_data, self.user_num, self.item_num, self.train_mat = self.load_all(os.path.join(self.rawdata_path, 'ml-1m.train.rating'),
                                                                                                      os.path.join(self.rawdata_path, 'ml-1m.test.negative'))

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

class TaskReader(XYTaskReader):
    def __init__(self, taskpath=''):
        super(TaskReader, self).__init__(taskpath)

class TaskCalculator(ClassifyCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)

