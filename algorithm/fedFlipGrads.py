import numpy as np
from utils import fmodule
import copy
from multiprocessing import Pool as ThreadPool
from main import logger
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
		self.fixed_selected_clients = [[] for i in range(self.num_rounds+1)]

		## code from fedavg
		self.path_save = os.path.join('fedtasksave', self.option['task'],
									"FlipGrads_R{}_P{:.2f}_alpha{}_clean{}_seed{}".format(
										option['num_rounds'],
										option['proportion'],
										self.alpha,
										option['clean_model'],
										option['seed']
									),
									'record')
		self.unlearn_term = None
		self.unlearn_time = 0
		# create folder for saving model
		print(self.path_save)
		if not os.path.exists(self.path_save):
			os.makedirs(self.path_save, exist_ok=True)

	def run(self):
		"""
		Start the federated learning symtem where the global model is trained iteratively.
		"""
		logger.time_start('Total Time Cost')

		## run()
		for round in range(self.num_rounds+1):
			print("--------------Round {}--------------".format(round))
			logger.time_start('Time Cost')
			self.round = round ## Get round_num
			# federated train
			self.iterate(round)
			# decay learning rate
			self.global_lr_scheduler(round)

			logger.time_end('Time Cost')
			if logger.check_if_log(round, self.eval_interval): logger.log(self)

		print("=================End==================")
		logger.time_end('Total Time Cost')
		# save results as .json file
		logger.save(os.path.join('fedtask', self.option['task'], 'record', flw.output_filename(self.option, self)))

	def iterate(self, t):
		self.selected_clients = self.sample(t)
		for idx in self.selected_clients:
			self.round_selected[idx].append(t)

		# training
		models = self.communicate(self.selected_clients)

		#  Process Unlearning

		# start algorithm
		if self.option['clean_model'] == 2:
			# save grads
			self.process_grad_original(models, t)
			# find attack_clients
			attack_clients = []
			for cid in self.selected_clients:
				if cid in self.option['attacker']:
					attack_clients.append(cid)
			# compute beta for this round
			self.update_beta()
			# # unlearning
			if len(attack_clients) >= 1:
				# self.all_attack_clients_id = list(set(self.all_attack_clients_id).union(attack_clients))
				round_attack, attackers_round = self.getAttacker_rounds(attack_clients)
				# unlearn
				if t >= self.option['num_rounds'] - 5:
					logger.time_start('unlearning time')
					self.unlearn_term = self.compute_unlearn_term(round_attack, attackers_round, t)
					self.unlearn_time = logger.time_end('unlearning time')

		# import pdb; pdb.set_trace()
		if t >= self.option['num_rounds'] - 5:
			self.save_models(t, models, self.unlearn_time)

		# check whether all the clients have dropped out, because the dropped clients will be deleted from self.selected_clients
		if not self.selected_clients: return
		# aggregate: pk = 1/K as default where K=len(selected_clients)
		self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
		return

	def save_models(self, round_num, models, unlearn_time):
		if round_num >= self.option['num_rounds'] - 5 and self.option['clean_model'] == 2:
			# aggregate
			temp_model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
			# model clean with algo3
			clean_model = temp_model + self.unlearn_term
			# test_unlearn, backdoor_unlearn = self.test(model= clean_model)
			## clean metric
			# test_clean, test_backdoor = self.test(model= temp_model)
   
			# test on clients
			client_test_metrics, client_backdoor_metrics = self.test_on_clients(self.current_round, clean_model)
			# compute HR and NDCG for test
			HR = 0.0
			NDCG = 0.0
			for metric in client_test_metrics:
				HR = HR + metric[0]
				NDCG = NDCG + metric[1]
			mean_hr = float(HR)/len(client_test_metrics)
			mean_ndcg = float(NDCG)/len(client_test_metrics)
			# log
			save_logs = {
				"selected_clients": self.selected_clients,
				"models": models,
				"p": [1.0 * self.client_vols[cid] / self.data_vol for cid in self.selected_clients],
				"server_model": self.model,
				# "accuracy": [test_clean, test_backdoor],
				"unlearn_term_algo3": self.unlearn_term,
				"unlearn_time": unlearn_time,
				"HR_on_clients": mean_hr,
				"NDCG_on_clients": mean_ndcg
				# "accuracy_unlearn": [test_unlearn, backdoor_unlearn]
			}

			pickle.dump(save_logs,
						open(os.path.join(self.path_save, "history" + str(round_num) + ".pkl"), 'wb'),
						pickle.HIGHEST_PROTOCOL)
		print("Save  ", round_num)

	def process_grad_original(self, models, round_id):
		## self.model : global model before update
		## models[cid] : model of client cid at round t

		## grad save as dict: {'cid' : grad}
		grads_this_round = {}
		for idx in range(len(self.selected_clients)):
			cid = self.selected_clients[idx]
			grads_this_round[str(cid)] = (self.model - models[idx]).to('cpu') 

		self.grads_all_round.append(grads_this_round)

	def process_grad(self, models, round_id):
		## self.model : global model before update
		## models[cid] : model of client cid at round t

		## grad save as dict: {'cid' : grad}
		grads_this_round = {}
		for idx in range(len(self.selected_clients)):
			cid = self.selected_clients[idx]
			M_v = (models[idx] - self.model).to('cpu')
			for param in M_v.parameters(): param.requires_grad = False
			norms_item = torch.norm(M_v.embed_item.weight, dim=1)
			alpha = 0.5
			topk_values, topk_indices = torch.topk(norms_item, int(alpha * M_v.embed_item.weight.shape[0]))
			not_topk_indices = ~torch.isin(torch.arange(M_v.embed_item.weight.shape[0]), topk_indices)
			M_v.embed_item.weight[not_topk_indices, :] = 0
			grads_this_round[str(cid)] = M_v #tmp_model#copy.deepcopy(tmp_model)#(self.model - models[idx]).cpu()

		self.grads_all_round.append(grads_this_round)

	def process_grad_each_user(self, models, round_id):
		## self.model : global model before update
		## models[cid] : model of client cid at round t

		## grad save as dict: {'cid' : grad}
		grads_this_round = {}
		for idx in range(len(self.selected_clients)):
			cid = self.selected_clients[idx]
			M_v = (models[idx] - self.model).to('cpu')
			user_items = self.clients[idx].train_data.get_user_items()
			topk_indices = []
			alpha = 0.1
			count_idx = 0
			for param in M_v.parameters(): param.requires_grad = False
			for u in user_items:
				count_idx += 1
				if count_idx > 1:
					import pdb
					pdb.set_trace()
				M_v_user = M_v.embed_item.weight[user_items[u]]
				norms_item = torch.norm(M_v_user, dim=1)
				indices = torch.topk(norms_item, int(alpha * M_v_user.shape[0]))[1]
				try:
					topk_indices.extend(list(np.array(user_items[u])[indices]))
				except:
					topk_indices.extend([np.array(user_items[u])[indices].tolist()]) # co 1 index
			not_topk_indices = ~torch.isin(torch.arange(M_v.embed_item.weight.shape[0]), torch.tensor(topk_indices))
			M_v.embed_item.weight[not_topk_indices, :] = 0
			grads_this_round[str(cid)] = M_v #tmp_model#copy.deepcopy(tmp_model)#(self.model - models[idx]).cpu()

		self.grads_all_round.append(grads_this_round)

	def update_beta(self):
		sum_vol = 0.0
		for cid in self.selected_clients:
			sum_vol += 1.0 * self.client_vols[cid]/self.data_vol
		self.beta.append(sum_vol)

	def getAttacker_rounds(self, attackers):
		## get list of attacked rounds
		round_attack = set([])
		for cid in attackers:
			round_attack.update(self.round_selected[cid])
		round_attack = list(round_attack)
		round_attack.sort()

		## get list of attackers of each round
		attackers_round = [[] for round in range(len(round_attack))]
		for idx in range(len(round_attack)):
			for cid in attackers:
				if round_attack[idx] in self.round_selected[cid]:
					attackers_round[idx].append(cid)

		return round_attack, attackers_round

	def compute_unlearn_term(self, round_attack, attackers_round, round):
		## Init unlearn term
		unlearning_term = fmodule._create_new_model(self.model) * 0.0
		alpha = - self.alpha
		# compute beta constraint in lipschitz inequality
		list_beta = []
		for idx in range(len(self.beta)): # idx: round_id
			beta = self.beta[idx]
			if idx in round_attack:
				for cid in attackers_round[round_attack.index(idx)]:
					beta -= 1.0 * self.client_vols[cid]/self.data_vol

			beta = beta * alpha + 1
			list_beta.append(beta)

			# compute unlearning-term
		for idx in range(len(round_attack)):
			round_id = round_attack[idx]
			# compute u-term at round round_id (attack round)
			unlearning_term = unlearning_term * list_beta[round_id]
			for c_id in attackers_round[idx]:
				unlearning_term += 1.0 * self.client_vols[c_id]/self.data_vol * self.grads_all_round[round_id][str(c_id)].to(self.model.get_device())
				self.grads_all_round[round_id][str(c_id)].to('cpu') #.cpu()

			if idx == len(round_attack) - 1: continue
			for r_id in range(round_id + 1, round_attack[idx + 1]):
				unlearning_term = unlearning_term * list_beta[r_id]
		unlearning_term = unlearning_term * self.theta
		return unlearning_term

	def remove_atk(self, attackers):
		for cid in attackers:
			self.round_selected[cid] = []

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
		return self.unpack(packages_received_from_clients)

	def communicate_with(self, client_id):
		"""
		Pack the information that is needed for client_id to improve the global model
		:param
			client_id: the id of the client to communicate with
		:return
			client_package: the reply from the client and will be 'None' if losing connection
		"""
		# package the necessary information
		svr_pkg = self.pack(client_id)
		# listen for the client's response and return None if the client drops out
		if self.clients[client_id].is_drop(): return None
		return self.clients[client_id].reply(svr_pkg)

	def pack(self, client_id):
		"""
		Pack the necessary information for the client's local training.
		Any operations of compression or encryption should be done here.
		:param
			client_id: the id of the client to communicate with
		:return
			a dict that only contains the global model as default.
		"""
		return {
			"model" : copy.deepcopy(self.model),
			"round" : self.round
			# "data" : self.data
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
		self.fixed_selected_clients[t] = [i for i in range(self.num_clients)]
		##
		if self.option['clean_model'] == 0 or self.option['clean_model'] == 2:
			selected_clients = self.fixed_selected_clients[t]
		elif self.option['clean_model'] == 1:
			selected_clients = []
			for cid in self.fixed_selected_clients[t]:
				if cid not in self.option['attacker']:
					selected_clients.append(cid)
		else:
			raise Exception("Invalid value for attribute clean_model")
		# selected_clients = [10]
		# selected_clients = [i for i in range(1, self.num_clients)]
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
		if self.option['clean_model'] == 0:
			for c in self.clients:
				test_acc, backdoor_acc = c.test(self.test_data, self.test_backdoor, server_model)
				test_metrics.append(test_acc)
				backdoor_metrics.append(backdoor_acc)
		else: 
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
		if model==None: model=self.model
		if self.test_data:
			model.eval()
			data_loader = self.calculator.get_data_loader(self.test_data, batch_size=self.option['test_num_ng']+1, shuffle=False)
			test_metric = self.calculator.test(model, data_loader, self.topN)

			## test on backdoor data # wrong, need to fix later
			backdoor_metric = [-1, -1]
			if self.test_backdoor:
				backdoor_loader = self.calculator.get_data_loader(self.test_backdoor, batch_size = self.option['test_num_ng']+1, shuffle=False)
				backdoor_metric = self.calculator.test(model, backdoor_loader, 3)

			# return
			return test_metric, backdoor_metric
		else:
			return -1, -1

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
			# import pdb; pdb.set_trace()
			data_loader.dataset.ng_sample_original()
			for batch_id, batch_data in enumerate(data_loader):
				model.zero_grad()
				loss = self.calculator.get_loss(model, batch_data, self.option)
				loss.backward()
				optimizer.step()
		name_malicious_client = ['Client{:03d}'.format(num) for num in self.option['attacker']]
		if self.name in name_malicious_client: #== 'Client00':
			# import pdb; pdb.set_trace()
			update_client = model - server_model
			self.model = server_model - update_client
		return

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
			model.eval()
			data_loader = self.calculator.get_data_loader(test_data, batch_size=100, shuffle=False)
			test_metric = self.calculator.test(model, data_loader, self.topN, self.users_set)

			## test on backdoor data # wrong, need to fix later
			backdoor_metric = [-1, -1]
			if test_backdoor:
				backdoor_loader = self.calculator.get_data_loader(test_backdoor, batch_size = 100, shuffle=False)
				backdoor_metric = self.calculator.test(model, backdoor_loader, self.topN)

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
		self.train(self.model, model)
		cpkg = self.pack(copy.deepcopy(self.model))
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

	# def train_loss(self, model):
	#     """
	#     Get the task specified loss of the model on local training data
	#     :param model:
	#     :return:
	#     """
	#     return self.test(model,'train')[1]

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
