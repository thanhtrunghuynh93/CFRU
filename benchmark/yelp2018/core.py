from torchvision import datasets, transforms
from benchmark.toolkits import ClassifyCalculator, DefaultTaskGen, XYTaskReader
from graph.interaction import Interaction
from benchmark.loader import FileIO
from yaml import load
import numpy as np
import os.path as osp
import os
import ujson

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5):
        super(TaskGen, self).__init__(benchmark='yelp2018',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/yelp2018/data',
                                      )
        self.num_classes = 10
        self.save_data = self.XYData_to_json

    def load_data(self):
        self.train_data = FileIO.load_data_set(os.path.join(self.rawdata_path, 'train.txt'), 'graph')
        self.test_data = FileIO.load_data_set(os.path.join(self.rawdata_path, 'test.txt'), 'graph')

    def convert_data_for_saving(self):
        self.data = Interaction(self.train_data, self.test_data)
        return

class TaskReader(XYTaskReader):
    def __init__(self, taskpath=''):
        super(TaskReader, self).__init__(taskpath)

class TaskCalculator(ClassifyCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)

