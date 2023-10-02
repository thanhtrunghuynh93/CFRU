import numpy as np
from utils import fmodule
import copy
from multiprocessing import Pool as ThreadPool
from main import logger
from collections import defaultdict
import pickle
import os
import utils.fflow as flw
import utils.evaluate as EvalUser
import torch
import random
import json
import shutil
class Server():
	def __init__(self, option, model, clients, test_data = None):
		# basic setting
		self.task = option['task']
		self.name = option['algorithm']
		self.model = model
		# self.data = data
		# server calculator
		self.calculator = fmodule.TaskCalculator(fmodule.device)
		self.round = 0
		self.test_data = test_data
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
									"ULOpt_{}_Mu{}_R{}_P{:.2f}_alpha{}_seed{}_{}".format(
										option['model'],
										option['S1'],
										option['num_rounds'],
										option['proportion'],
										self.alpha,
										option['seed'],
										option['atk_method']
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
		models, important_weights = self.communicate(self.selected_clients)

		#  Process Unlearning
		# start algorithm
		# save grads
		self.process_grad(important_weights)
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
			# if t >= self.option['num_rounds'] - 5:
			logger.time_start('unlearning time')
			self.unlearn_term = self.compute_unlearn_term(round_attack, attackers_round, t)
			self.unlearn_time = logger.time_end('unlearning time')

		if t >= self.option['num_rounds'] - 5:
			self.save_important_update(t, important_weights, models)
		# check whether all the clients have dropped out, because the dropped clients will be deleted from self.selected_clients
		if not self.selected_clients: return
		# aggregate: pk = 1/K as default where K=len(selected_clients)
		self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
		# self.save_important_update(t, grads_this_round, models)
		return

	def save_result(self, round_num, models, unlearn_time):
		if round_num >= self.option['num_rounds'] - 5:
			# aggregate
			temp_model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
			# model clean with algo3
			clean_model = temp_model + self.unlearn_term
			test_unlearn, test_loss3, backdoor_unlearn = self.test(model= clean_model)
			## clean metric 
			test_clean, _, test_backdoor = self.test(model= temp_model)
			# log
			save_logs = {
				"selected_clients": self.selected_clients,
				"models": models,
				"p": [1.0 * self.client_vols[cid] / self.data_vol for cid in self.selected_clients],
				"server_model": self.model,
				"accuracy": [test_clean, test_backdoor],
				"unlearn_term_algo3": self.unlearn_term,
				"unlearn_time": unlearn_time,
				"accuracy_unlearn": [test_unlearn, backdoor_unlearn]
			}
		
		pickle.dump(save_logs,
					open(os.path.join(self.path_save, "history" + str(round_num) + ".pkl"), 'wb'),
					pickle.HIGHEST_PROTOCOL)
		print("Save  ", round_num)

	def process_grad(self, models):
		## self.model : global model before update
		## models[cid] : model of client cid at round t

		## grad save as dict: {'cid' : grad}
		grads_this_round = {}
		for idx in range(len(self.selected_clients)):
			cid = self.selected_clients[idx]
			grads_this_round[str(cid)] = (self.model - models[idx]).to('cpu') 

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
		important_weights = [cp["important_weight"] for cp in packages_received_from_clients]
		return models, important_weights

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
		selected_clients = self.fixed_selected_clients[t]

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

	def test_on_clients(self, server_model = None):
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
		for idx in range(self.num_clients):
			if idx in self.option['attacker']:
				continue
			test_acc = self.clients[idx].test(self.test_data, server_model)
			test_metrics.append(test_acc)
		# compute HR and NDCG for test
		HR = 0.0
		NDCG = 0.0
		for metric in test_metrics:
			HR = HR + metric[0]
			NDCG = NDCG + metric[1]
		return [float(HR)/len(test_metrics), float(NDCG)/len(test_metrics)]

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
			return test_metric
		else:
			return -1

class Client():
	def __init__(self, option, name='', model=None, train_data=None, users_set=None):
		self.name = name
		self.frequency = 0
		# create local dataset
		self.train_data = train_data
		self.num_threads = option['num_threads']
		self.users_set = users_set
		self.neg_items = []
		# positive items
		self.positive_items = [] # note that 1 client only has 1 user
		for user in users_set:
			self.positive_items = self.positive_items + self.train_data.get_pos_items(user)
		self.positive_items = list(set(self.positive_items))
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
		self.malicious_users = option['malicious_users']
     	# the probability of dropout obey distribution beta(drop, 1). The larger 'drop' is, the more possible for a device to drop
		self.drop_rate = 0 if option['net_drop']<0.01 else np.random.beta(option['net_drop'], 1, 1).item()
		self.active_rate = 1 if option['net_active']>99998 else np.random.beta(option['net_active'], 1, 1).item()

		# proportion of items for important sampling
		self.prop_saving = option['importance_prop']
		# for variance-based negative sampling
		self.varset_size = 3000 # size of candidate for var monitor
		self.var_config = {
			'S1': option['S1'], # size of M_u = 8???
			'S2_div_S1': 8, # S2/S1 where S2 is size of ~M_u (uniformly sample from negative items pool) 8/8
			'alpha': 5.0,
			'warmup': 50,
			'temperature': 1.0
		}
		# self.S1 = 20 
		# self.S2_div_S1 = 1 
		self.num_user_client = len(self.users_set)
		self.num_user = self.model.state_dict()['embed_user.weight'].shape[0]
		self.num_item = self.model.state_dict()['embed_item.weight'].shape[0]
		self.init_variance_sets()
		# shutil.copyfile(os.path.join(self.init_var_folder, str(self.name) + ".npz"), os.path.join(self.update_var_folder, str(self.name) + ".npz"))

	def init_variance_sets(self):
		# init the train_set, test_set
		train_set = defaultdict(set)
		train_data = self.train_data.features
		for i in range(len(train_data)):
			train_set[train_data[i][0]].add(train_data[i][1])

		# init the train_iddict [u] pos->id
		self.train_iddict = [defaultdict(int) for _ in range(self.num_user_client)] # ???
		self.train_pos = [[] for _ in range(self.num_user_client)]
		self.max_posid = 0
		for i in range(self.num_user_client):
			poscnt = 0
			self.max_posid = max(self.max_posid, len(train_set[self.users_set[i]]))
			for p in train_set[self.users_set[i]]:
				self.train_iddict[i][p] = poscnt
				poscnt += 1
				self.train_pos[i].append(p)
		print("MAX POS IDX: %d"%self.max_posid)
		# if	not os.path.exists(os.path.join(self.init_var_folder, str(self.name) + ".npz")):
		self.candidate_cur = np.empty([self.num_user_client, self.varset_size], dtype=np.int32)
		for i in range(self.num_user_client):
			valid_options = list(set(range(self.num_item)) - set(train_set[self.users_set[i]]))
			self.candidate_cur[i] = np.random.choice(valid_options, self.varset_size)

		self.candidate_nxt = [np.empty([self.num_user_client, self.varset_size], dtype=np.int32) for _ in range(5)]

		for c in range(5):
			for i in range(self.num_user_client):
				valid_options = list(set(range(self.num_item)) - set(train_set[self.users_set[i]]))
				self.candidate_nxt[c][i] = np.random.choice(valid_options, self.varset_size)


		self.score_cand_cur = np.array([EvalUser.predict_fast_opt(self.model, self.num_user_client, self.num_item, parallel_users=100, predict_data=self.candidate_cur, users_set=self.users_set)])
		self.score_cand_nxt = [np.zeros((0, self.num_user_client, self.varset_size)) for _ in range(5)]
		self.score_pos_cur = np.array([EvalUser.predict_pos_opt(self.model, self.num_user_client, self.max_posid, parallel_users=100, predict_data=self.train_pos, users_set=self.users_set)])
			# np.savez(os.path.join(self.init_var_folder, str(self.name) + ".npz"), candidate_cur=candidate_cur, candidate_nxt=candidate_nxt, score_cand_cur=score_cand_cur, score_cand_nxt=score_cand_nxt, score_pos_cur=score_pos_cur)
		self.Mu_idx = []  # All possible items or non-fn items
		for i in range(self.num_user_client):
			Mu_idx_tmp = random.sample(list(range(self.varset_size)), self.var_config['S1'])
			self.Mu_idx.append(Mu_idx_tmp)
		# shutil.copyfile(os.path.join(self.init_var_folder, str(self.name) + ".npz"), os.path.join(self.update_var_folder, str(self.name) + ".npz"))

	def update_variance_sets(self, epoch_count):

		score_1epoch_nxt = [np.array([EvalUser.predict_fast_opt(self.model, self.num_user_client, self.num_item, parallel_users=100,predict_data=self.candidate_nxt[c], users_set=self.users_set)]) for c in range(5)]
    
		score_1epoch_pos = np.array([EvalUser.predict_pos_opt(self.model, self.num_user_client, self.max_posid, parallel_users=100, predict_data=self.train_pos, users_set=self.users_set)])

		# delete the score_cand_cur[0,:,:] at the earliest timestamp
		if epoch_count >= 5 or epoch_count == 0:
			self.score_pos_cur = np.delete(self.score_pos_cur, 0, 0)
		for c in range(5):
			self.score_cand_nxt[c] = np.concatenate([self.score_cand_nxt[c], score_1epoch_nxt[c]], axis=0)

		self.score_pos_cur = np.concatenate([self.score_pos_cur, score_1epoch_pos], axis=0)

		# Re-assign the variables directly instead of creating a copy
		self.score_cand_cur = self.score_cand_nxt[0]
		self.candidate_cur = self.candidate_nxt[0]

		for c in range(4):
			self.candidate_nxt[c] = self.candidate_nxt[c + 1]
			self.score_cand_nxt[c] = self.score_cand_nxt[c + 1]

		# Utilize numpy random.choice to create the array with necessary condition
		# self.candidate_nxt[4] = np.random.choice(np.setdiff1d(np.arange(self.num_item), self.train_pos), [self.num_user, self.varset_size])
		self.candidate_nxt[4] = np.empty([self.num_user_client, self.varset_size], dtype=np.int32)
		for i in range(self.num_user_client):
			valid_options = list(set(range(self.num_item)) - set(self.train_pos[i]))
			self.candidate_nxt[4][i] = np.random.choice(valid_options, self.varset_size)
		self.score_cand_nxt[4] = np.delete(self.score_cand_nxt[4], list(range(self.score_cand_nxt[4].shape[0])), 0)

	def train(self, model, server_model, round_num):
		"""
		Standard local training procedure. Train the transmitted model with local training dataset.
		:param
			model: the global model 
			round_num:
		:return
		"""
		name_malicious_client = ['Client{:03d}'.format(num) for num in self.option['attacker']]
		if self.name in name_malicious_client:
			model.train()
			print(self.datavol)
			neg_items_this_round = set()
			data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
			optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
			if self.option['atk_method'] == 'fedAttack':
				neg_items_this_round.update(data_loader.dataset.ng_sample_fedatk(self.model, topK = 1000, malicious_users = self.malicious_users))
			for iter in range(self.epochs):
				if self.option['atk_method'] == 'fedFlipGrads':
					neg_items_this_round.update(data_loader.dataset.ng_sample_original())
				# neg_items_this_round.update(data_loader.dataset.ng_sample_fedatk(self.model, topK = 1000, malicious_users = self.malicious_users))
				for batch_id, batch_data in enumerate(data_loader):
					model.zero_grad()
					# import pdb; pdb.set_trace()
					loss = self.calculator.get_loss(model, batch_data, self.option)
					loss.backward()
					optimizer.step()
				# update mean and std for P_pos
			neg_items_this_round = list(neg_items_this_round)
			self.neg_items.append(neg_items_this_round)
			if self.option['atk_method'] == 'fedFlipGrads':
				# import pdb; pdb.set_trace()
				update_client = model - server_model
				self.model = server_model - update_client
			return self.process_grad(server_model, self.neg_items[-1])
		else:
			# model.train()
			print(self.datavol)
			neg_items_this_round = set()
			data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
			optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
			model.train()
			for iter in range(self.epochs):
				epoch_cur = round_num*self.epochs + iter
				data_loader.dataset.pos_sampling() # sample on positive training data
				for batch_id, batch_data in enumerate(data_loader):
					model.zero_grad()
					# import pdb; pdb.set_trace()
					loss, self.Mu_idx, neg_items = self.calculator.get_loss_variance(model, batch_data, self.users_set, self.var_config, epoch_cur, 
										self.score_cand_cur, self.score_pos_cur, self.Mu_idx, self.candidate_cur, self.train_iddict, self.option)
					neg_items_this_round.update(neg_items)
					# backward
					loss.backward()
					optimizer.step()
				# update mean and std for P_pos
				self.update_variance_sets(epoch_cur)
			neg_items_this_round = list(neg_items_this_round)
			self.neg_items.append(neg_items_this_round)
			return self.process_grad(server_model, self.neg_items[-1])

	def process_grad(self, server_model, negative_items):
		all_selected_items = list(set(self.positive_items + negative_items))
		M_v = (self.model - server_model).to('cpu')
		for param in M_v.parameters(): param.requires_grad = False
		# get all embeddings from pos and neg items
		M_v_user = M_v.embed_item.weight[all_selected_items]
		# get l2 norm
		norms_item = torch.norm(M_v_user, dim=1)
		topk_indices = torch.topk(norms_item, int(self.prop_saving * M_v_user.shape[0]))[1]
		topk_item_ids = torch.tensor(all_selected_items)[topk_indices]
		# get topk selected items
		# all param not in topk -> 0
		not_topk_items = ~torch.isin(torch.arange(M_v.embed_item.weight.shape[0]), topk_item_ids)
		M_v.embed_item.weight[not_topk_items, :] = 0
		# require grad = True
		for param in M_v.parameters(): param.requires_grad = True
		# save updates
		return server_model + M_v.cuda()

	def test(self, test_data, server_model = None):
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
			model.to('cpu')
			# return
			return test_metric
		else:
			return -1

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
		important_weight = self.train(self.model.to(fmodule.device), model, round_num)
		self.model.to('cpu')
		cpkg = self.pack(copy.deepcopy(self.model), important_weight)
		return cpkg

	def pack(self, model, important_weight):
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
			"model" : model,
			"important_weight": important_weight
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
