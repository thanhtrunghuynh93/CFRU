"""
DISTRIBUTION OF DATASET
-----------------------------------------------------------------------------------
balance:
	iid:            0 : identical and independent distributions of the dataset among clients
	label skew:     1 Quantity:  each party owns data samples of a fixed number of labels.
					2 Dirichlet: each party is allocated a proportion of the samples of each label according to Dirichlet distribution.
					3 Shard: each party is allocated the same numbers of shards that is sorted by the labels of the data
-----------------------------------------------------------------------------------
depends on partitions:
	feature skew:   4 Noise: each party owns data samples of a fixed number of labels.
					5 ID: For Shakespeare\FEMNIST, we divide and assign the writers (and their characters) into each party randomly and equally.
-----------------------------------------------------------------------------------
imbalance:
	iid:            6 Vol: only the vol of local dataset varies.
	niid:           7 Vol: for generating synthetic data
"""
import torch
import ujson
import numpy as np
import os.path
import random
import urllib
import zipfile
import os
import ssl
import math
import heapq
import copy
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import torch
from torch import nn
import torch.nn.functional as F
ssl._create_default_https_context = ssl._create_unverified_context
import importlib
import torchvision.transforms as transforms
from random import shuffle,randint,choice,sample
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import concurrent
# import warnings
# warnings.filterwarnings('error')


def set_random_seed(seed=0):
	"""Set random seed"""
	random.seed(3 + seed)
	np.random.seed(97 + seed)
	os.environ['PYTHONHASHSEED'] = str(seed)

def download_from_url(url= None, filepath = '.'):
	"""Download dataset from url to filepath."""
	if url: urllib.request.urlretrieve(url, filepath)
	return filepath

def extract_from_zip(src_path, target_path):
	"""Unzip the .zip file (src_path) to target_path"""
	f = zipfile.ZipFile(src_path)
	f.extractall(target_path)
	targets = f.namelist()
	f.close()
	return [os.path.join(target_path, tar) for tar in targets]

def find_k_largest(K, candidates):
	n_candidates = []
	for iid, score in enumerate(candidates[:K]):
		n_candidates.append((score, iid))

	heapq.heapify(n_candidates)

	for iid, score in enumerate(candidates[K:]):
		if score > n_candidates[0][0]:
			# ...
			heapq.heapreplace(n_candidates, (score, iid + K))
	n_candidates.sort(key=lambda d: d[0], reverse=True)
	ids = [item[1] for item in n_candidates]
	k_largest_scores = [item[0] for item in n_candidates]
	return ids, k_largest_scores

class Metric(object):
	def __init__(self):
		pass

	@staticmethod
	def hits(origin, res):
		hit_count = {}
		for user in origin:
			items = list(origin[user].keys())
			predicted = [item[0] for item in res[user]]
			hit_count[user] = len(set(items).intersection(set(predicted)))
		return hit_count

	@staticmethod
	def hit_ratio(origin, hits):
		"""
		Note: This type of hit ratio calculates the fraction:
		 (# retrieved interactions in the test set / #all the interactions in the test set)
		"""
		total_num = 0
		for user in origin:
			items = list(origin[user].keys())
			total_num += len(items)
		hit_num = 0
		for user in hits:
			hit_num += hits[user]
		return round(hit_num/total_num,5)

	@staticmethod
	def precision(hits, N):
		prec = sum([hits[user] for user in hits])
		return round(prec / (len(hits) * N),5)

	@staticmethod
	def recall(hits, origin):
		recall_list = [hits[user]/len(origin[user]) for user in hits]
		recall = round(sum(recall_list) / len(recall_list),5)
		return recall

	@staticmethod
	def F1(prec, recall):
		if (prec + recall) != 0:
			return round(2 * prec * recall / (prec + recall),5)
		else:
			return 0

	@staticmethod
	def MAE(res):
		error = 0
		count = 0
		for entry in res:
			error+=abs(entry[2]-entry[3])
			count+=1
		if count==0:
			return error
		return round(error/count,5)

	@staticmethod
	def RMSE(res):
		error = 0
		count = 0
		for entry in res:
			error += (entry[2] - entry[3])**2
			count += 1
		if count==0:
			return error
		return round(math.sqrt(error/count),5)

	@staticmethod
	def NDCG(origin,res,N):
		sum_NDCG = 0
		for user in res:
			DCG = 0
			IDCG = 0
			#1 = related, 0 = unrelated
			for n, item in enumerate(res[user]):
				if item[0] in origin[user]:
					DCG+= 1.0/math.log(n+2,2)
			for n, item in enumerate(list(origin[user].keys())[:N]):
				IDCG+=1.0/math.log(n+2,2)
			sum_NDCG += DCG / IDCG
		return round(sum_NDCG / len(res),5)

class BasicTaskGen:
	_TYPE_DIST = {
		0: 'iid',
		1: 'label_skew_quantity',
		2: 'label_skew_dirichlet',
		3: 'label_skew_shard',
		4: 'feature_skew_noise',
		5: 'feature_skew_id',
		6: 'iid_volumn_skew',
		7: 'niid_volumn_skew',
		8: 'concept skew',
		9: 'concept and feature skew and balance',
		10: 'concept and feature skew and imbalance',
		11: 'iid division by users (fedRec)',
		12: 'randomly 1 user per client',
		13: 'top 100 correlation user selected to 100 clients',
		14: 'top 100 by correlation of pairs'
	}
	_TYPE_DATASET = ['2DImage', '3DImage', 'Text', 'Sequential', 'Graph', 'Tabular']

	def __init__(self, benchmark, dist_id, skewness, rawdata_path, seed=0):
		self.benchmark = benchmark
		self.rootpath = './fedtask'
		self.rawdata_path = rawdata_path
		self.dist_id = dist_id
		self.dist_name = self._TYPE_DIST[dist_id]
		self.skewness = 0 if dist_id==0 else skewness
		self.num_clients = -1
		self.seed = seed
		set_random_seed(self.seed)
		if not os.path.exists(self.rootpath):
			os.makedirs(self.rootpath, exist_ok=True)

	def run(self):
		"""The whole process to generate federated task. """
		pass

	def load_data(self):
		"""Download and load dataset into memory."""
		pass

	def partition(self):
		"""Partition the data according to 'dist' and 'skewness'"""
		pass

	def save_data(self):
		"""Save the federated dataset to the task_path/data.
		This algorithm should be implemented as the way to read
		data from disk that is defined by DataReader.read_data()
		"""
		pass

	def save_info(self):
		"""Save the task infomation to the .json file stored in taskpath"""
		pass

	def get_taskname(self):
		"""Create task name and return it."""
		taskname = '_'.join([self.benchmark, 'cnum' +  str(self.num_clients), 'dist' + str(self.dist_id), 'skew' + str(self.skewness).replace(" ", ""), 'seed'+str(self.seed)])
		return taskname

	def get_client_names(self):
		k = str(len(str(self.num_clients)))
		return [('Client{:0>' + k + 'd}').format(i) for i in range(self.num_clients)]

	def create_task_directories(self):
		"""Create the directories of the task."""
		taskname = self.get_taskname()
		taskpath = os.path.join(self.rootpath, taskname)
		os.mkdir(taskpath)
		os.mkdir(os.path.join(taskpath, 'record'))

	def _check_task_exist(self):
		"""Check whether the task already exists."""
		taskname = self.get_taskname()
		return os.path.exists(os.path.join(self.rootpath, taskname))

class DefaultTaskGen(BasicTaskGen):
	def __init__(self, benchmark, dist_id, skewness, rawdata_path, num_clients=1, minvol=10, seed=0):
		super(DefaultTaskGen, self).__init__(benchmark, dist_id, skewness, rawdata_path, seed)
		self.minvol=minvol
		self.num_classes = -1
		# load from data
		self.train_data = None
		self.test_data = None
		self.user_num = None
		self.item_num = None
		self.local_user_idxs = None
		self.train_mat = None

		self.num_clients = num_clients
		self.cnames = self.get_client_names()
		self.taskname = self.get_taskname()
		self.taskpath = os.path.join(self.rootpath, self.taskname)
		self.save_data = self.XYData_to_json
		self.datasrc = {
			'lib': None,
			'class_name': None,
			'args':[]
		}

	def run(self):
		""" Generate federated task"""
		# check if the task exists
		if not self._check_task_exist():
			self.create_task_directories()
		else:
			print("Task Already Exists.")
			return
		# read raw_data into self.train_data and self.test_data
		print('-----------------------------------------------------')
		print('Loading...')
		self.load_data()
		print('Done.')
		# partition data and hold-out for each local dataset
		print('-----------------------------------------------------')
		print('Partitioning data...')
		local_datas = self.partition()
		train_cidxs, valid_cidxs = self.local_holdout(local_datas, rate=1.0, shuffle=True)
		#
		self.train_cidxs = train_cidxs
		self.valid_cidxs = valid_cidxs
		print('Length of valid dataset: {}'.format(len(self.valid_cidxs[-1])))
		print('Done.')
		# save task infomation as .json file and the federated dataset
		print('-----------------------------------------------------')
		print('Saving data...')
		self.save_info()
		self.save_data(train_cidxs, valid_cidxs) # XYData_to_json
		print('Done.')
		return

	def load_data(self):
		""" load and pre-process the raw data"""
		return

	def partition(self):
		# Partition self.train_data according to the delimiter and return indexes of data owned by each client as [c1data_idxs, ...] where the type of each element is list(int)
		if self.dist_id == 0:
			"""IID division by users in fedRec"""
			local_datas = [[] for _ in range(self.num_clients)]
			user_idxs = np.random.permutation(self.user_num)
			user_idxs_split = np.array_split(user_idxs, self.num_clients)
			self.local_user_idxs = [arr.tolist() for arr in user_idxs_split]
			user_client_dict = {}
			for client_id in range(len(self.local_user_idxs)):
				for user in self.local_user_idxs[client_id]:
					user_client_dict[str(user)] = client_id
			# randomly disjoin train data
			d_idxs = np.random.permutation(len(self.train_data))
			for idx in d_idxs:
				user = self.train_data[idx][0]
				local_datas[user_client_dict[str(user)]].append(idx)
		else:
			raise Exception("Distribution identification not found!")
		return local_datas

	def local_holdout(self, local_datas, rate=0.8, shuffle=False):
		"""split each local dataset into train data and valid data according the rate."""
		train_cidxs = []
		valid_cidxs = []
		for local_data in local_datas:
			if shuffle:
				np.random.shuffle(local_data)
			k = int(len(local_data) * rate)
			train_cidxs.append(local_data[:k])
			valid_cidxs.append(local_data[k:])
		return train_cidxs, valid_cidxs

	def save_info(self):
		info = {
			'benchmark': self.benchmark,  # name of the dataset
			'dist': self.dist_id,  # type of the partition way
			'skewness': self.skewness,  # hyper-parameter for controlling the degree of niid
			'num-clients': self.num_clients,  # numbers of all the clients
		}
		# save info.json
		with open(os.path.join(self.taskpath, 'info.json'), 'w') as outf:
			ujson.dump(info, outf)

	def convert_data_for_saving(self):
		"""Convert self.train_data and self.test_data to list that can be stored as .json file and the converted dataset={'x':[], 'y':[]}"""
		pass

	def XYData_to_json(self, train_cidxs, valid_cidxs):
		# self.convert_data_for_saving()
		# save federated dataset
		feddata = {
			'store': 'XY',
			'client_names': self.cnames,
			'dtrain': self.train_data,
			'dtest': self.test_data,
			'users_per_client': self.local_user_idxs,
			'user_num': int(self.user_num),
			'item_num': int(self.item_num),
			'train_mat': self.train_mat
		}
		# import pdb; pdb.set_trace()
		for cid in range(self.num_clients):
			feddata[self.cnames[cid]] = {
				'client_train': [self.train_data[did] for did in train_cidxs[cid]]
			}
		with open(os.path.join(self.taskpath, 'data.json'), 'w') as outf:
			ujson.dump(feddata, outf)
		# import pdb; pdb.set_trace()
		return

	def IDXData_to_json(self, train_cidxs, valid_cidxs):
		if self.datasrc ==None:
			raise RuntimeError("Attr datasrc not Found. Please define it in __init__() before calling IndexData_to_json")
		feddata = {
			'store': 'IDX',
			'client_names': self.cnames,
			'dtest': [i for i in range(len(self.test_data))],
			'datasrc': self.datasrc
		}
		for cid in range(self.num_clients):
			feddata[self.cnames[cid]] = {
				'dtrain': train_cidxs[cid],
				'dvalid': valid_cidxs[cid]
			}
		with open(os.path.join(self.taskpath, 'data.json'), 'w') as outf:
			ujson.dump(feddata, outf)
		return

class BasicTaskCalculator:

	_OPTIM = None

	def __init__(self, device):
		self.device = device
		self.lossfunc = None
		self.DataLoader = None

	def data_to_device(self, data):
		raise NotImplementedError

	def get_loss(self):
		raise NotImplementedError

	def get_evaluation(self):
		raise NotImplementedError

	def get_data_loader(self, data, batch_size = 64):
		return NotImplementedError
	# def generate_set(self):
	#     raise NotImplementedError

	def test(self):
		raise NotImplementedError

	def get_optimizer(self, name="sgd", model=None, lr=0.1, weight_decay=0, momentum=0):
		# if self._OPTIM == None:
		#     raise RuntimeError("TaskCalculator._OPTIM Not Initialized.")
		if name.lower() == 'sgd':
			return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
		elif name.lower() == 'adam':
			return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay, amsgrad=True)
		else:
			raise RuntimeError("Invalid Optimizer.")

	@classmethod
	def setOP(cls, OP):
		cls._OPTIM = OP

class ClassifyCalculator(BasicTaskCalculator):
	def __init__(self, device):
		super(ClassifyCalculator, self).__init__(device)
		self.num_classes = 10
		# self.lossfunc = torch.nn.CrossEntropyLoss()
		# self.lossfunc_Dir = DirichletLoss(num_classes= self.num_classes, annealing_step= 10, device= device)
		# self.lossMSE = torch.nn.MSELoss()
		self.DataLoader = DataLoader

	# def generate_set(self, data, test_data):
	#     test_set = defaultdict(dict)
	#     for entry in test_data:
	#         user, item, rating = entry
	#         if user not in data.user or item not in data.item:
	#             continue
	#         test_set[user][item] = rating
	#     return test_set

	@torch.no_grad()
	def get_evaluation(self, model, data):
		tdata = self.data_to_device(data)
		outputs = model(tdata)
		y_pred = outputs.data.max(1, keepdim=True)[1]
		correct = y_pred.eq(tdata[1].data.view_as(y_pred)).long().cpu().sum()
		return (1.0 * correct / len(tdata[1])).item()

	@torch.no_grad()
	def test(self, model, test_loader, top_k, users_test=None):
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
		if users_test == None:
			for user, item_i, item_j in test_loader:
				user = user.cuda()
				item_i = item_i.cuda()
				item_j = item_j.cuda() # not useful when testing
				
				output = model((user, item_i, item_j))
				indices = model.handle_test(output, top_k)
				# _, indices = torch.topk(prediction_i, top_k)
				recommends = torch.take(
						item_i, indices).cpu().numpy().tolist()

				gt_item = item_i[0].item()
				HR.append(hit(gt_item, recommends))
				NDCG.append(ndcg(gt_item, recommends))
		else:
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

	def data_to_device(self, data, device=None):
		if device is None:
			return data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
		else:
			return data[0].to(device), data[1].to(device), data[2].to(device)

	def get_loss(self, model, data, option, device=None):
		tdata = self.data_to_device(data, device)
		output = model(tdata)
		loss = model.handle_loss(output, option)
		return loss

	def get_loss_variance(self, model, data, users_set, var_config, epoch_cur, score_cand_all, score_pos_all, Mu_idx, candidate_cur, train_iddict, option=None):
		# map current batch to index in list users
		def find_indices(list_user_batch, list_user_client):
			indices = [list_user_client.index(element) for element in list_user_batch]
			return indices

		model.eval()
		device = next(model.parameters()).device
		tdata = self.data_to_device(data, device)
		user_batch = data[0].tolist()
		user_batch_indices = find_indices(user_batch, users_set)
		item_batch = data[1].tolist()

		negitems_candidates_all = np.array([Mu_idx[u] for u in user_batch_indices])
		ratings_positems = model.get_score([tdata[0], tdata[1]]).squeeze(-1).detach().cpu().numpy()
		Mu_items_all = torch.tensor(np.array([candidate_cur[user_batch_indices[i], negitems_candidates_all[i]] for i in range(len(user_batch))])).view(-1, 1).long().to(device)
		users = torch.tensor(user_batch).view(-1, 1).repeat(1, var_config['S1']).view(-1, 1).long().to(device)
		ratings_candidates_all = model.get_score([users, Mu_items_all]).view(-1, var_config['S1']).detach().cpu().numpy()

		hisscore_candidates_all = np.concatenate([score_cand_all[:, user_batch_indices[i]:user_batch_indices[i]+1, np.reshape(negitems_candidates_all[i], [-1])] for i in range(len(user_batch))], axis=1)
		hisscore_pos_all = np.expand_dims(np.concatenate([score_pos_all[:, user_batch_indices[i]:user_batch_indices[i]+1, train_iddict[user_batch_indices[i]][item_batch[i]]] for i in range(len(user_batch))], axis=1), -1)
		hislikelihood_candidates_all = 1 / (1 + np.exp(hisscore_pos_all - hisscore_candidates_all))
		mean_candidates_all = np.mean(hislikelihood_candidates_all, axis=0)

		variance_candidates_all = np.sqrt(np.mean((hislikelihood_candidates_all - mean_candidates_all) ** 2, axis=0))
		likelihood_candidates_all = 1 / (1 + np.exp(np.expand_dims(ratings_positems, -1) - ratings_candidates_all))
		epoch_scale = min(1, epoch_cur/var_config['warmup'])
		alpha = var_config['alpha']

		item_arg_all = np.argmax(likelihood_candidates_all + (alpha if alpha >= 0 else 0) * epoch_scale * variance_candidates_all, axis=1)
		# example_weight = np.ones((len(user_batch),1), dtype=np.float)
		negitems = [candidate_cur[user_batch_indices[i], negitems_candidates_all[i, item_arg_all[i]]] for i in range(len(user_batch))]

		for i in range(len(user_batch)):
			Mu_set = set(Mu_idx[user_batch_indices[i]])
			while len(Mu_idx[user_batch_indices[i]]) < var_config['S1'] * (1 + var_config['S2_div_S1']):
				random_item = random.randint(0, candidate_cur.shape[1] - 1)
				while random_item in Mu_set:
					random_item = random.randint(0, candidate_cur.shape[1] - 1)
				Mu_idx[user_batch_indices[i]].append(random_item)

		negitems_mu_candidates = np.array([Mu_idx[u] for u in user_batch_indices])
		negitems_mu = torch.tensor(np.array([candidate_cur[user_batch_indices[i], negitems_mu_candidates[i]] for i in range(len(user_batch))])).view(-1, 1).long().to(device)
		users = torch.tensor(user_batch).view(-1, 1).repeat(1, var_config['S1'] * (1 + var_config['S2_div_S1'])).view(-1, 1).long().to(device)
		ratings_mu_candidates = model.get_score([users, negitems_mu]).view(-1, var_config['S1'] * (1 + var_config['S2_div_S1'])).detach().cpu().numpy()

		# pre-process
		min_vals = np.min(ratings_mu_candidates, axis=1, keepdims=True)
		max_vals = np.max(ratings_mu_candidates, axis=1, keepdims=True)

		ratings_mu_candidates = (ratings_mu_candidates - min_vals) / (max_vals - min_vals)
		# ratings_mu_candidates = ratings_mu_candidates / var_config['temperature']
		ratings_mu_candidates = np.exp(ratings_mu_candidates) / np.reshape(np.sum(np.exp(ratings_mu_candidates), axis=1), [-1, 1])
		user_set = set()
		for i in range(len(user_batch)):
			if user_batch_indices[i] not in user_set:
				user_set.add(user_batch_indices[i])
				cache_arg = np.random.choice(var_config['S1'] * (1 + var_config['S2_div_S1']), var_config['S1'], p=ratings_mu_candidates[i], replace=False)
				Mu_idx[user_batch_indices[i]] = np.array(Mu_idx[user_batch_indices[i]])[cache_arg].tolist()

		model.train()
		negitems = torch.tensor(negitems).view(-1, 1).squeeze(-1)
		input_model = model.process_input([tdata[0], tdata[1], negitems.long().to(device)], device=device)
		# output = model([tdata[0], tdata[1], negitems.long().to(device)])
		output = model(input_model)
		loss = model.handle_loss(output, option)

		return loss, Mu_idx, negitems.tolist()

	def get_data_loader(self, dataset, batch_size=64, shuffle=True, droplast=False): # shuffle = True
		def seed_worker():
			worker_seed = 0
			numpy.random.seed(worker_seed)
			random.seed(worker_seed)
		g = torch.Generator()
		g.manual_seed(0) # 0
		if self.DataLoader == None:
			raise NotImplementedError("DataLoader Not Found.")
		return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=droplast, worker_init_fn=seed_worker, generator=g)
		
	
## Bang Nguyen Trong

class BasicTaskReader:
	def __init__(self, taskpath=''):
		self.taskpath = taskpath

	def read_data(self):
		"""
			Reading the spilted dataset from disk files and loading data into the class 'LocalDataset'.
			This algorithm should read three types of data from the processed task:
				train_sets = [client1_train_data, ...] where each item is an instance of 'LocalDataset'
				valid_sets = [client1_valid_data, ...] where each item is an instance of 'LocalDataset'
				test_set = test_dataset
			Return train_sets, valid_sets, test_set, client_names
		"""
		pass

class XYTaskReader(BasicTaskReader):
	def __init__(self, taskpath=''):
		super(XYTaskReader, self).__init__(taskpath)

	def read_data(self, num_ng, model_type):
		with open(os.path.join(self.taskpath, 'data.json'), 'r') as inf:
			feddata = ujson.load(inf)
		# test_data = feddata['dtest']
		clients_config = None
		if model_type == 'BPR':
			train_info = {
				'user_num': feddata['user_num'],
				'item_num': feddata['item_num']
				# 'adj_dict': feddata['train_mat']
			}
			
			clients_config = []
			for name in feddata['client_names']:
				c_conf = {
					'user_num': feddata['user_num'],
					'item_num': feddata['item_num']
				}
				clients_config.append(c_conf)
				
			train_datas = [BPRData(feddata[name]['client_train'], feddata['item_num'], feddata['train_mat'], num_ng, True) for name in feddata['client_names']]
			test_data = BPRData(feddata['dtest'], feddata['item_num'], feddata['train_mat'], 0, False)
		elif model_type == 'NCF':
			train_info = {
				'user_num': feddata['user_num'],
				'item_num': feddata['item_num']
				# 'adj_dict': feddata['train_mat']
			}
			
			clients_config = []
			for name in feddata['client_names']:
				c_conf = {
					'user_num': feddata['user_num'],
					'item_num': feddata['item_num']
				}
				clients_config.append(c_conf)
				
			train_datas = [NCFData(feddata[name]['client_train'], feddata['item_num'], feddata['train_mat'], num_ng, True) for name in feddata['client_names']]
			test_data = NCFData(feddata['dtest'], feddata['item_num'], feddata['train_mat'], 0, False)
		elif model_type == 'LightGCN':
			train_info = {
				'user_num': feddata['user_num'],
				'item_num': feddata['item_num'],
				'training_data': feddata['dtrain'],
				# 'adj_dict': feddata['train_mat']
			}
			clients_config = []
			for name in feddata['client_names']:
				c_conf = {
					'user_num': feddata['user_num'],
					'item_num': feddata['item_num'],
					'training_data': feddata[name]['client_train'],
				}
				clients_config.append(c_conf)
				
			train_datas = [GCNData(feddata[name]['client_train'], feddata['item_num'], feddata['train_mat'], num_ng, True) for name in feddata['client_names']]
			test_data = GCNData(feddata['dtest'], feddata['item_num'], feddata['train_mat'], 0, False)
		else:
			raise TypeError('Not exist model with name {}'.format(model_type))
		return train_datas, test_data, feddata['users_per_client'], train_info, clients_config, feddata['client_names']

class IDXTaskReader(BasicTaskReader):
	def __init__(self, taskpath=''):
		super(IDXTaskReader, self).__init__(taskpath)

	def read_data(self):
		with open(os.path.join(self.taskpath, 'data.json'), 'r') as inf:
			feddata = ujson.load(inf)
		DS = getattr(importlib.import_module(feddata['datasrc']['lib']), feddata['datasrc']['class_name'])
		arg_strings = '(' + ','.join(feddata['datasrc']['args'])
		train_args = arg_strings + ', train=True)'
		test_args = arg_strings + ', train=False)'
		DS.SET_DATA(eval(feddata['datasrc']['class_name'] + train_args))
		DS.SET_DATA(eval(feddata['datasrc']['class_name'] + test_args), key='TEST')
		test_data = IDXDataset(feddata['dtest'], key='TEST')
		train_datas = [IDXDataset(feddata[name]['dtrain']) for name in feddata['client_names']]
		valid_datas = [IDXDataset(feddata[name]['dvalid']) for name in feddata['client_names']]
		return train_datas, valid_datas, test_data, feddata['client_names']

class BPRData(Dataset):
	def __init__(self, features, 
				num_item, train_mat=None, num_ng=0, is_training=None):
		super(BPRData, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.features = features
		self.num_item = num_item
		self.train_mat = train_mat
		self.num_ng = num_ng
		self.is_training = is_training
		self.len_data = self.num_ng * len(self.features) if self.is_training else len(self.features)
		self.user_items = self._get_user_items()

	def _get_user_items(self):
		user_items = {}
		for user, item in self.features:
			if user not in user_items:
				user_items[user] = []
			user_items[user].append(item)
		return user_items

	def get_user_items(self):
		return self.user_items

	def get_pos_items(self, user):
		return self.train_mat[str(user)]

	def pos_sampling(self):
		assert self.is_training, 'no need to sampling when testing'
		self.features_fill = []
		for x in self.features:
			u, i = x[0], x[1]
			self.features_fill.append([u, i, i])
		self.len_data = len(self.features_fill)
		return 

	def ng_sample_original(self):
		assert self.is_training, 'no need to sampling when testing'
		self.features_fill = []
		all_neg_item = []
		for x in self.features:
			u, i = x[0], x[1]
			for t in range(self.num_ng):
				j = np.random.randint(self.num_item)
				while j in self.train_mat[str(u)]: #(u, j) in self.train_mat:
					j = np.random.randint(self.num_item)
				self.features_fill.append([u, i, j])
				all_neg_item.append(j)
		self.len_data = len(self.features_fill)
		return list(set(all_neg_item))
    
	def ng_sample_by_user(self, user):
		assert self.is_training, 'no need to sampling when testing'
		self.features_fill = []
		pos_items = self.train_mat[str(user)]
		for item in pos_items:
			for t in range(self.num_ng):
				j = np.random.randint(self.num_item)
				while j in pos_items:
					j = np.random.randint(self.num_item)
				self.features_fill.append([user, item, j])
		self.len_data = len(self.features_fill)
 
	def ng_sample_old(self, negative_samples):
		assert self.is_training, 'no need to sampling when testing'
		self.features_fill = []
		all_neg_item = []
		for x in self.features:
			u, i = x[0], x[1]
			# original semi-hard sampling
			# list_neg = negative_samples[str(u)]
			# randomly select num_ng negative samples for each positive sample
			list_neg = np.random.choice(negative_samples[str(u)], size=self.num_ng, replace=False)
			all_neg_item = all_neg_item + list(list_neg)
			for j in list_neg:
				self.features_fill.append([u, i, j])
		return list(set(all_neg_item))

	def ng_sample_chunk(self, negative_samples, chunk_start, chunk_end):
		assert self.is_training, 'no need to sampling when testing'
		features_fill_chunk = []
		all_neg_item = []
		for x in self.features[chunk_start:chunk_end]:
			u, i = x[0], x[1]
			list_neg = np.random.choice(negative_samples[str(u)], size=self.num_ng, replace=False)
			all_neg_item = all_neg_item + list(list_neg)
			for j in list_neg:
				features_fill_chunk.append([u, i, j])
		return features_fill_chunk, list(set(all_neg_item))

	def ng_sample(self, negative_samples, num_workers=16):
		assert self.is_training, 'no need to sampling when testing'
		
		# Calculate the chunk size based on the number of workers
		chunk_size = len(self.features) // num_workers

		# Create a list of chunk ranges
		chunk_ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_workers)]
		chunk_ranges[-1] = (chunk_ranges[-1][0], len(self.features))  # Make sure to include the last element
		
		# Use ProcessPoolExecutor to parallelize the processing
		with ProcessPoolExecutor(max_workers=num_workers) as executor:
			futures = [executor.submit(self.ng_sample_chunk, negative_samples, start, end) for start, end in chunk_ranges]
			
			results = [future.result() for future in futures]
		
		# Combine the results
		self.features_fill = [item for sublist, _ in results for item in sublist]
		all_neg_item = list(set(item for _, items in results for item in items))
		
		return all_neg_item

	def ng_sample_fedatk(self, model, topK, malicious_users):
		assert self.is_training, 'no need to sampling when testing'
		self.features_fill = []
		all_neg_item = []
		malicious_users_in_client = []
		# self.malicious_features = []
		for x in self.features:
			u, i = x[0], x[1]
			if u in malicious_users:
				malicious_users_in_client.append(u)
				# self.malicious_features.append(x)
				continue
			# negative sampling for normal users
			for t in range(self.num_ng):
				j = np.random.randint(self.num_item)
				while j in self.train_mat[str(u)]:
					j = np.random.randint(self.num_item)
				self.features_fill.append([u, i, j])
				all_neg_item.append(j)
		## 
		malicious_users_in_client = set(malicious_users_in_client)
		for user in malicious_users_in_client:
			top_items, bottom_items = model.predict_user(user, topK)
			# torch.manual_seed(42)
			perm1 = torch.randperm(len(top_items))
			perm2 = torch.randperm(len(bottom_items))
			for x, y in zip(top_items[perm1], bottom_items[perm2]):
				self.features_fill.append([user, y.item(), x.item()])
				#
				all_neg_item.append(x.item())
				all_neg_item.append(y.item())
		random.shuffle(self.features_fill)
		self.len_data = len(self.features_fill)
		return list(set(all_neg_item))
    
	def semi_hard_ng_sample_old(self, model, list_user):
		import time
		assert self.is_training, 'no need to sampling when testing'
		self.features_fill = []
		neg_sampling_user = {}
		# get candidate pool
		for u in list_user:
			start_time = time.time()
			V_neg_k = model.get_semi_hard_negative_items(u, pos_items=self.train_mat[str(u)], R=0.5)
			N = int(self.num_item * 0.5)
			# print("Hi{}".format(N))
			B = 0.05
			index_neg_sampling = np.random.choice(V_neg_k, size=int(N*B), replace=False)
			neg_sampling_user[str(u)] = index_neg_sampling
		return neg_sampling_user
		# for x in self.features:
		# 	u, i = x[0], x[1]
		# 	for j in neg_sampling_user[str(u)]:
		# 		self.features_fill.append([u, i, j])
	def get_neg_sampling(self, u, model):
		V_neg_k = model.get_semi_hard_negative_items(u, pos_items=self.train_mat[str(u)], R=0.5)
		N = int(self.num_item * 0.5)
		B = 0.05
		index_neg_sampling = np.random.choice(V_neg_k, size=int(N * B), replace=False)
		return str(u), index_neg_sampling

	def semi_hard_ng_sample(self, model, list_user, num_workers=16):
		assert self.is_training, 'no need to sampling when testing'
		self.features_fill = []
		neg_sampling_user = {}

		# Process nhanh hon Thread.

		with ProcessPoolExecutor(max_workers=num_workers) as executor:
			future_results = {executor.submit(self.get_neg_sampling, u, model): u for u in list_user}
			for future in concurrent.futures.as_completed(future_results):
				u = future_results[future]
				try:
					user, index_neg_sampling = future.result()
					neg_sampling_user[user] = index_neg_sampling
				except Exception as e:
					print(f"User {u} generated an exception: {e}")
		
		return neg_sampling_user

	def __len__(self):
		return self.len_data
		# return self.num_ng * len(self.features) if \
		# 		self.is_training else len(self.features)

	def __getitem__(self, idx):
		features = self.features_fill if \
				self.is_training else self.features
		# if len(features[idx]) < 3: import pdb; pdb.set_trace()
		user = features[idx][0]
		item_i = features[idx][1]
		item_j = features[idx][2] if \
				self.is_training else features[idx][1]
		return user, item_i, item_j 

class NCFData(Dataset):
	def __init__(self, features, 
				num_item, train_mat=None, num_ng=0, is_training=None):
		super(NCFData, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.features = features
		self.num_item = num_item
		self.train_mat = train_mat
		self.num_ng = num_ng
		self.is_training = is_training
		self.labels = [0 for _ in range(len(features))]
		self.len_data = (self.num_ng + 1) * len(self.labels) 
		
	def get_pos_items(self, user):
		return self.train_mat[str(user)]
	
	def pos_sampling(self):
		assert self.is_training, 'no need to sampling when testing'
		self.features_fill = self.features
		self.labels_fill = [1 for _ in range(len(self.features))]
		# for x in self.features:
		# 	u, i = x[0], x[1]
		# 	self.features_fill.append([u, i, i])
		self.len_data = len(self.features_fill)
		return 
	
	def ng_sample_original(self):
		assert self.is_training, 'no need to sampling when testing'
		all_neg_item = []
		self.features_ng = []
		for x in self.features:
			u = x[0]
			for t in range(self.num_ng):
				j = np.random.randint(self.num_item)
				while j in self.train_mat[str(u)]: #(u, j) in self.train_mat:
					j = np.random.randint(self.num_item)
				self.features_ng.append([u, j])
				all_neg_item.append(j)

		labels_ps = [1 for _ in range(len(self.features))]
		labels_ng = [0 for _ in range(len(self.features_ng))]

		self.features_fill = self.features + self.features_ng
		self.labels_fill = labels_ps + labels_ng
		self.len_data = len(self.labels_fill)
		return list(set(all_neg_item))
	
	def ng_sample_fedatk(self, model, topK, malicious_users):
		assert self.is_training, 'no need to sampling when testing'
		import copy
		ft_pos = copy.copy(self.features)
		all_neg_item = []
		self.features_ng = []
		malicious_users_in_client = []
		# self.malicious_features = []
		for x in self.features:
			u = x[0]
			if u in malicious_users:
				malicious_users_in_client.append(u)
				continue
			# negative sampling for normal users
			for t in range(self.num_ng):
				j = np.random.randint(self.num_item)
				while j in self.train_mat[str(u)]:
					j = np.random.randint(self.num_item)
				self.features_ng.append([u, j])
				all_neg_item.append(j)
		## 
		malicious_users_in_client = set(malicious_users_in_client)
		for user in malicious_users_in_client:
			top_items, bottom_items = model.predict_user(user, topK)
			# torch.manual_seed(42)
			perm1 = torch.randperm(len(top_items))
			perm2 = torch.randperm(len(bottom_items))
			for x, y in zip(top_items[perm1], bottom_items[perm2]):
				ft_pos.append([user, y.item()])
				self.features_ng.append([user, x.item()])
				all_neg_item.append(x.item())
				all_neg_item.append(y.item())
		
		labels_ps = [1 for _ in range(len(ft_pos))]
		labels_ng = [0 for _ in range(len(self.features_ng))]

		self.features_fill = ft_pos + self.features_ng
		self.labels_fill = labels_ps + labels_ng
		self.len_data = len(self.labels_fill)
		return list(set(all_neg_item))

	def __len__(self):
		return self.len_data
		# return (self.num_ng + 1) * len(self.labels)

	def __getitem__(self, idx):
		features = self.features_fill if self.is_training \
					else self.features
		labels = self.labels_fill if self.is_training \
					else self.labels

		user = features[idx][0]
		item = features[idx][1]
		label = labels[idx]
		return user, item ,label

class GCNData(Dataset):
	def __init__(self, features, 
				num_item, train_mat=None, num_ng=0, is_training=None):
		super(GCNData, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.features = features
		self.num_item = num_item
		self.train_mat = train_mat
		self.num_ng = num_ng
		self.is_training = is_training
		self.len_data = self.num_ng * len(self.features) if self.is_training else len(self.features)

	def get_pos_items(self, user):
		return self.train_mat[str(user)]
	
	def ng_sample(self):
		assert self.is_training, 'no need to sampling when testing'
		self.features_fill = []
		for x in self.features:
			u, i = x[0], x[1]
			for t in range(self.num_ng):
				j = np.random.randint(self.num_item)
				while j in self.train_mat[str(u)]: #(u, j) in self.train_mat:
					j = np.random.randint(self.num_item)
				self.features_fill.append([u, i, j])

	def pos_sampling(self):
		assert self.is_training, 'no need to sampling when testing'
		self.features_fill = []
		for x in self.features:
			u, i = x[0], x[1]
			self.features_fill.append([u, i, i])
		self.len_data = len(self.features_fill)
		return 
	
	def ng_sample_original(self):
		assert self.is_training, 'no need to sampling when testing'
		self.features_fill = []
		all_neg_item = []
		for x in self.features:
			u, i = x[0], x[1]
			for t in range(self.num_ng):
				j = np.random.randint(self.num_item)
				while j in self.train_mat[str(u)]: #(u, j) in self.train_mat:
					j = np.random.randint(self.num_item)
				self.features_fill.append([u, i, j])
				all_neg_item.append(j)
		self.len_data = len(self.features_fill)
		return list(set(all_neg_item))
	
	def ng_sample_fedatk(self, model, topK, malicious_users):
		assert self.is_training, 'no need to sampling when testing'
		self.features_fill = []
		all_neg_item = []
		malicious_users_in_client = []
		# self.malicious_features = []
		for x in self.features:
			u, i = x[0], x[1]
			if u in malicious_users:
				malicious_users_in_client.append(u)
				# self.malicious_features.append(x)
				continue
			# negative sampling for normal users
			for t in range(self.num_ng):
				j = np.random.randint(self.num_item)
				while j in self.train_mat[str(u)]:
					j = np.random.randint(self.num_item)
				self.features_fill.append([u, i, j])
				all_neg_item.append(j)
		## 
		malicious_users_in_client = set(malicious_users_in_client)
		for user in malicious_users_in_client:
			top_items, bottom_items = model.predict_user(user, topK)
			# torch.manual_seed(42)
			perm1 = torch.randperm(len(top_items))
			perm2 = torch.randperm(len(bottom_items))
			for x, y in zip(top_items[perm1], bottom_items[perm2]):
				self.features_fill.append([user, y.item(), x.item()])
				#
				all_neg_item.append(x.item())
				all_neg_item.append(y.item())
		random.shuffle(self.features_fill)
		self.len_data = len(self.features_fill)
		return list(set(all_neg_item))
	
	def __len__(self):
		return self.len_data

	def __getitem__(self, idx):
		features = self.features_fill if \
				self.is_training else self.features

		user = features[idx][0]
		item_i = features[idx][1]
		item_j = features[idx][2] if \
				self.is_training else features[idx][1]
		return user, item_i, item_j 

class IDXDataset(Dataset):
	# The source dataset that can be indexed by IDXDataset
	_DATA = {'TRAIN': None,'TEST': None}

	def __init__(self, idxs, key='TRAIN'):
		"""Init dataset with 'src_data' and a list of indexes that are used to position data in 'src_data'"""
		if not isinstance(idxs, list):
			raise RuntimeError("Invalid Indexes")
		self.idxs = idxs
		self.key = key

	@classmethod
	def SET_DATA(cls, dataset, key = 'TRAIN'):
		cls._DATA[key] = dataset

	@classmethod
	def ADD_KEY_TO_DATA(cls, key, value = None):
		if key==None:
			raise RuntimeError("Empty key when calling class algorithm IDXData.ADD_KEY_TO_DATA")
		cls._DATA[key]=value

	def __getitem__(self, item):
		idx = self.idxs[item]
		return self._DATA[self.key][idx]

class TupleDataset(Dataset):
	def __init__(self, X1=[], X2=[], Y=[], totensor=True):
		if totensor:
			try:
				self.X1 = torch.tensor(X1)
				self.X2 = torch.tensor(X2)
				self.Y = torch.tensor(Y)
			except:
				raise RuntimeError("Failed to convert input into torch.Tensor.")
		else:
			self.X1 = X1
			self.X2 = X2
			self.Y = Y

	def __getitem__(self, item):
		return self.X1[item], self.X2[item], self.Y[item]

	def __len__(self):
		return len(self.Y)

	def tolist(self):
		if not isinstance(self.X1, torch.Tensor):
			return self.X1, self.X2, self.Y
		return self.X1.tolist(), self.X2.tolist(), self.Y.tolist()
