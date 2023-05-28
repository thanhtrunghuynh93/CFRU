import numpy as np
from utils import fmodule
import copy
from multiprocessing import Pool as ThreadPool
from main import logger
from tqdm import tqdm
import pickle
import os
import utils.fflow as flw
import torch
import json
class Server():
	def __init__(self, option, model, clients, test_data = None, backtask_data = None):
		# basic setting
		self.task = option['task']
		self.name = option['algorithm']
		self.model = model
		# self.data = data
		# server calculator
		self.calculator = fmodule.TaskCalculator(fmodule.device)
		self.round = 0
		self.test_backdoor = backtask_data
		self.test_data = test_data
		# self.test_backdoor = self.calculator.generate_set(self.data, backtask_data)
		#
		self.eval_interval = option['eval_interval']
		self.num_threads = option['num_threads']
		# clients settings
		self.clients = clients
		self.num_clients = len(self.clients)
		self.client_vols = [c.datavol for c in self.clients]
		self.data_vol = sum(self.client_vols)
		self.clients_buffer = [{} for _ in range(self.num_clients)]
		self.selected_clients = []
		# hyper-parameters during training process
		self.num_rounds = option['num_rounds']
		self.decay_rate = option['learning_rate_decay']
		# self.clients_per_round = max(int(self.num_clients * option['proportion']), 1)
		self.lr_scheduler_type = option['lr_scheduler']
		self.current_round = -1
		# sampling and aggregating methods
		# self.sample_option = option['sample']
		self.agg_option = option['aggregate']
		self.lr=option['learning_rate']
		# names of additional parameters
		self.paras_name=[]
		self.option = option
		# unlearning parameters
		self.beta = []
		self.grads_all_round = []

		# round selected of all clients
		self.round_selected = [[] for cid in range(self.num_clients)]

		self.theta = self.option['theta_delta']
		self.alpha = self.option['alpha']
		# top N evaluation
		self.topN = self.option['topN']
		# self.algo = self.option['unlearn_algorithm']
		# self.fixed_selected_clients = [[] for i in range(self.num_rounds+1)]

		## code from fedavg
		self.path_save = os.path.join('fedtasksave', self.option['task'],
									"ULKD_{}_R{}_P{:.2f}_alpha{}_seed{}_{}".format(
										option['model'],
										option['num_rounds'],
										option['proportion'],
										self.alpha,
										option['seed'],
										option['atk_method']
									),
									'record')
		self.unlearn_term = None
		self.unlearn_time = 0
		# Parameters for FedEfficient
		self.global_epoch = self.option['num_rounds']
		self.forget_client_idxs = sorted(self.option['attacker'])
		# create folder for saving model
		print(self.path_save)
		if not os.path.exists(self.path_save):
			os.makedirs(self.path_save, exist_ok=True)

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
		eval_metric = self.test(unlearn_GM)
		print(eval_metric)
		# self.save_models(eval_metric, unlearn_time, unlearn_GM, global_model, grads_target_client)
		self.save_models(eval_metric, unlearn_time, unlearn_GM)
		print("------------End---------------")
		return unlearn_GM

	def iterate(self, t, global_model=None, student_model=None, is_unlearn=False):
		# sample clients: MD sampling as default but with replacement=False
		if is_unlearn:
			clean_clients = []
			for cid in range(self.num_clients):
				if cid not in self.option['attacker']:
					clean_clients.append(cid)
			models, p_coefficient = self.unlearn_communicate(clean_clients, global_model, student_model)
			unlearn_model = fmodule._model_sum([model_k * p_k for model_k, p_k in zip(models, p_coefficient)])
			return unlearn_model
		else:
			self.selected_clients = self.sample(t)
			attack_clients = []

			for cid in self.selected_clients: 
				if cid in self.option['attacker']:
					attack_clients.append(cid) 
			
			for idx in self.selected_clients:
				self.round_selected[idx].append(t)

			models, p_coefficient = self.communicate(self.selected_clients, global_model)
			grads_target_client = fmodule._model_sum([(self.model - models[idx]) for idx in self.forget_client_idxs]).cpu()#(self.model - models[0]).cpu()
			self.model = self.aggregate(models, p = p_coefficient)
			return self.model, grads_target_client

	def unlearning(self, global_model, grads_target_client):
		# Input   
		"""
		MF: teacher model
		MF_apos: student model
		After deleting dirty updates from MF_apos, we use MF to train MF_apos by Knowledge Distillation
		"""
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
		MF_apos = self.iterate(self.current_round, MF, MF_apos, is_unlearn=True)
		return MF_apos
	
	def save_models(self, main_accuracy, unlearn_time, unlearn_model):
		# log
		save_logs = {
			"unlearn_accuracy": main_accuracy,
			"time": unlearn_time,
			"unlearn_model": unlearn_model
		}
		pickle.dump(save_logs,
					open(os.path.join(self.path_save, "history" + str(self.global_epoch) + ".pkl"), 'wb'),
					pickle.HIGHEST_PROTOCOL)
		print("Save  ", self.global_epoch)

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
		models = self.unpack(packages_received_from_clients)
		p_coef = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients]
		return models, p_coef

	def unlearn_communicate(self, clean_clients, teacher_model, student_model):
		packages_received_from_clients = []
		if self.num_threads <= 1:
			# computing iteratively
			for client_id in clean_clients:
				response_from_client_id = self.unlearn_communicate_with(client_id, teacher_model, student_model)
				packages_received_from_clients.append(response_from_client_id)
		else:
			# computing in parallel
			raise ValueError("Multi-Thread is not allowed here!")
			
		# count the clients not dropping
		self.clean_clients = [clean_clients[i] for i in range(len(clean_clients)) if packages_received_from_clients[i]]
		packages_received_from_clients = [pi for pi in packages_received_from_clients if pi]
		models = self.unpack(packages_received_from_clients)
		K = len(models)
		p_coef = [(1.0 / K) for cid in self.clean_clients]
		return models, p_coef

	def unlearn_communicate_with(self, client_id, global_model, student_model):
		# package the necessary information
		svr_pkg = self.unlearn_pack(client_id, global_model, student_model)
		# listen for the client's response and return None if the client drops out
		if self.clients[client_id].is_drop(): return None
		return self.clients[client_id].kd_reply(svr_pkg)

	def unlearn_pack(self, client_id, global_model, student_model):
		if global_model==None or student_model==None:
			raise ValueError("Need teacher model and student model for training!")
		else:
			return {
				"model" : copy.deepcopy(student_model),
				"round" : self.round,
				"teacher": copy.deepcopy(global_model)
			}

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

	def unpack(self, packages_received_from_clients):
		"""
		Unpack the information from the received packages. Return models and losses as default.
		:param
			packages_received_from_clients:
		:return:
			models: a list of the locally improved model
			losses: a list of the losses of the global model on each training dataset
		"""
		models = [cp["model"] for cp in packages_received_from_clients]
		return models

	def global_lr_scheduler(self, current_round):
		"""
		Control the step size (i.e. learning rate) of local training
		:param
			current_round: the current communication round
		"""
		if self.lr_scheduler_type == -1:
			return
		elif self.lr_scheduler_type == 0:
			"""eta_{round+1} = DecayRate * eta_{round}"""
			self.lr*=self.decay_rate
			for c in self.clients:
				c.set_learning_rate(self.lr)
		elif self.lr_scheduler_type == 1:
			"""eta_{round+1} = eta_0/(round+1)"""
			self.lr = self.option['learning_rate']*1.0/(current_round+1)
			for c in self.clients:
				c.set_learning_rate(self.lr)

	def sample(self, t):
		# import pdb; pdb.set_trace()
		# self.fixed_selected_clients[t] = [i for i in range(self.num_clients)]
		##
		if self.option['clean_model'] == 2:
			selected_clients = [i for i in range(self.num_clients)]
		else:
			raise Exception("Invalid value for attribute clean_model")
		return selected_clients

	def update_models(self, atk_clients, models):
		mds = []
		p = []
		for model, cid in zip(models, self.selected_clients):
			if cid not in atk_clients:
				mds.append(model)
				p.append(1.0 * self.client_vols[cid]/self.data_vol)
		return mds, p

	def aggregate(self, models, p=[]):
		"""
		Aggregate the locally improved models.
		:param
			models: a list of local models
			p: a list of weights for aggregating
		:return
			the averaged result

		pk = nk/n where n=self.data_vol
		K = |S_t|
		N = |S|
		-------------------------------------------------------------------------------------------------------------------------
		 weighted_scale                 |uniform (default)          |weighted_com (original fedavg)   |other
		==============================================================================================|============================
		N/K * Σpk * model_k             |1/K * Σmodel_k             |(1-Σpk) * w_old + Σpk * model_k  |Σ(pk/Σpk) * model_k
		"""
		if not models:
			return self.model
		if self.agg_option == 'weighted_scale':
			K = len(models)
			N = self.num_clients
			return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)]) * N / K
		elif self.agg_option == 'uniform':
			return fmodule._model_average(models, p=p)
		elif self.agg_option == 'weighted_com':
			w = fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])
			return (1.0-sum(p))*self.model + w
		else:
			sump = sum(p)
			p = [pk/sump for pk in p]
			return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])

	def test_on_clients(self, round, server_model = None):
		"""
		Validate accuracies and losses on clients' local datasets
		:param
			round: the current communication round
			dataflag: choose train data or valid data to evaluate
		:return
			evals: the evaluation metrics of the global model on each client's dataset
			loss: the loss of the global model on each client's dataset
		"""
		test_metrics = []
		backdoor_metrics = [] 
		for idx in range(self.num_clients):
			if idx in self.option['attacker']:
				continue
			test_acc, backdoor_acc = self.clients[idx].test(self.test_data, self.test_backdoor, server_model)
			test_metrics.append(test_acc)
			backdoor_metrics.append(backdoor_acc)
		return test_metrics, backdoor_metrics

	def test(self, model=None):
		"""
		Evaluate the model on the test dataset owned by the server.
		:param
			model: the model need to be evaluated
		:return:
			the metric and loss of the model on the test data
		"""
		client_test_metrics, client_backdoor_metrics = self.test_on_clients(self.current_round, model)
		# compute HR and NDCG for test
		HR = 0.0
		NDCG = 0.0
		for metric in client_test_metrics:
			HR = HR + metric[0]
			NDCG = NDCG + metric[1]
		mean_hr = float(HR)/len(client_test_metrics)
		mean_ndcg = float(NDCG)/len(client_test_metrics)
		return mean_hr, mean_ndcg

class Client():
	def __init__(self, option, name='', model=None, train_data=None, users_set=None):
		self.name = name
		self.frequency = 0
		# create local dataset
		self.train_data = train_data
		self.users_set = users_set
		# self.valid_data = valid_data
		self.option = option
		self.datavol = len(self.train_data)
		# local calculator
		self.calculator = fmodule.TaskCalculator(device=fmodule.device)
		# hyper-parameters for training
		self.optimizer_name = option['optimizer']
		self.epochs = option['num_epochs']
		self.learning_rate = option['learning_rate']
		self.batch_size = len(self.train_data) if option['batch_size']==-1 else option['batch_size']
		self.momentum = option['momentum']
		self.weight_decay = option['weight_decay']
		self.reg_lambda = option['reg.lambda']
		self.topN = option['topN']
		self.model = model
		# system setting
     	# the probability of dropout obey distribution beta(drop, 1). The larger 'drop' is, the more possible for a device to drop
		self.drop_rate = 0 if option['net_drop']<0.01 else np.random.beta(option['net_drop'], 1, 1).item()
		self.active_rate = 1 if option['net_active']>99998 else np.random.beta(option['net_active'], 1, 1).item()

	def train(self, model, server_model):
		"""
		Standard local training procedure. Train the transmitted model with local training dataset.
		:param
			model: the global model
			round_num:
		:return
		"""
		model.train()
		print(self.datavol)
		data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
		optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
		for iter in range(self.epochs):
			# negative sample based on attack method
			if self.option['atk_method'] == 'fedAttack':
				data_loader.dataset.ng_sample_fedatk(self.model, topK = 1000, malicious_users = self.option['malicious_users'])
			else:
				data_loader.dataset.ng_sample_original()
			# training
			for batch_id, batch_data in enumerate(data_loader):
				model.zero_grad()
				loss = self.calculator.get_loss(model, batch_data, self.option)
				loss.backward()
				optimizer.step()
		if self.option['atk_method'] == 'fedFlipGrads':
			name_malicious_client = ['Client{:03d}'.format(num) for num in self.option['attacker']]
			if self.name in name_malicious_client: #== 'Client00':
				# import pdb; pdb.set_trace()
				update_client = model - server_model
				self.model = server_model - update_client
		return
	
	def KD_train(self, student_model, teacher_model):
		student_model.train()
		teacher_model.eval()
		print(self.datavol)
		data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
		optimizer = self.calculator.get_optimizer(self.optimizer_name, student_model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
		for iter in range(5):
			data_loader.dataset.ng_sample_original()
			# training
			for batch_id, batch_data in enumerate(data_loader):
				student_model.zero_grad()
				loss = self.calculator.get_kd_loss(student_model, teacher_model, batch_data, self.option)
				loss.backward()
				optimizer.step()

	def test(self, test_data, test_backdoor, server_model = None):
		"""
		Evaluate the model with local data (e.g. training data or validating data).
		:param
			model:
			dataflag: choose the dataset to be evaluated on
		:return:
			eval_metric: task specified evaluation metric
			loss: task specified loss
		"""
		if server_model == None: model=self.model
		else:
			model = copy.deepcopy(self.model)
			fmodule._model_merge_(model, server_model)
   
		if test_data:
			model.to(fmodule.device)
			model.eval()
			data_loader = self.calculator.get_data_loader(test_data, batch_size=100, shuffle=False)
			test_metric = self.calculator.test(model, data_loader, self.topN, self.users_set)

			## test on backdoor data # wrong, need to fix later
			backdoor_metric = [-1, -1]
			if test_backdoor:
				backdoor_loader = self.calculator.get_data_loader(test_backdoor, batch_size = 100, shuffle=False)
				backdoor_metric = self.calculator.test(model, backdoor_loader, self.topN)

			model.to('cpu')
			# return
			return test_metric, backdoor_metric
		else:
			return -1, -1

	def unpack(self, received_pkg):
		"""
		Unpack the package received from the server
		:param
			received_pkg: a dict contains the global model as default
		:return:
			the unpacked information that can be rewritten
		"""
		# unpack the received package
		if 'teacher' in received_pkg:
			return received_pkg['model'], received_pkg['round'], received_pkg['teacher']
		else:
			return received_pkg['model'], received_pkg['round']

	def reply(self, svr_pkg):
		"""
		Reply to server with the transmitted package.
		The whole local procedure should be planned here.
		The standard form consists of three procedure:
		unpacking the server_package to obtain the global model,
		training the global model, and finally packing the improved
		model into client_package.
		:param
			svr_pkg: the package received from the server
		:return:
			client_pkg: the package to be send to the server
		"""
		model = self.unpack(svr_pkg)[0]
		round_num = self.unpack(svr_pkg)[1]

		# data = self.unpack(svr_pkg)[2]
		# import pdb; pdb.set_trace()
		fmodule._model_merge_(self.model, model)
		self.train(self.model.to(fmodule.device), model)
		self.model.to('cpu')
		cpkg = self.pack(copy.deepcopy(self.model))
		return cpkg
	
	def kd_reply(self, svr_pkg):
		model = self.unpack(svr_pkg)[0]
		round_num = self.unpack(svr_pkg)[1]
		teacher_model = self.unpack(svr_pkg)[2] 
		#
		student_model = copy.deepcopy(self.model)
		fmodule._model_merge_(student_model, model)
		self.KD_train(student_model, teacher_model)
		cpkg = self.pack(copy.deepcopy(student_model))
		return cpkg

	def pack(self, model):
		"""
		Packing the package to be send to the server. The operations of compression
		of encryption of the package should be done here.
		:param
			model: the locally trained model
			loss: the loss of the global model on the local training dataset
		:return
			package: a dict that contains the necessary information for the server
		"""
		return {
			"model" : model
		}

	def is_active(self):
		"""
		Check if the client is active to participate training.
		:param
		:return
			True if the client is active according to the active_rate else False
		"""
		if self.active_rate==1: return True
		else: return (np.random.rand() <= self.active_rate)

	def is_drop(self):
		"""
		Check if the client drops out during communicating.
		:param
		:return
			True if the client drops out according to the drop_rate else False
		"""
		if self.drop_rate==0: return False
		else: return (np.random.rand() < self.drop_rate)

	def valid_loss(self, model):
		"""
		Get the task specified loss of the model on local validating data
		:param model:
		:return:
		"""
		return self.test(model)[1]

	def set_model(self, model):
		"""
		set self.model
		:param model:
		:return:
		"""
		self.model = model

	def set_learning_rate(self, lr = 0):
		"""
		set the learning rate of local training
		:param lr:
		:return:
		"""
		self.learning_rate = lr if lr else self.learning_rate
