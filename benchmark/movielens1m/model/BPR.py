import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.fmodule import FModule
import heapq
import numpy as np

class Model(FModule):
	def __init__(self, data_conf, conf):
		super(Model, self).__init__()
		"""
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors.
		"""		
		self.item_num = data_conf['item_num']
		self.embed_user = nn.Embedding(data_conf['user_num'], conf['embedding.size'])
		self.embed_item = nn.Embedding(data_conf['item_num'], conf['embedding.size'])

		nn.init.normal_(self.embed_user.weight, std=0.01)
		nn.init.normal_(self.embed_item.weight, std=0.01)

	def forward(self, data):
		user = self.embed_user(data[0])
		item_i = self.embed_item(data[1])
		item_j = self.embed_item(data[2])

		prediction_i = (user * item_i).sum(dim=-1)
		prediction_j = (user * item_j).sum(dim=-1)
		return prediction_i, prediction_j

	def handle_loss(self, data, option):
		prediction_i = data[0]
		prediction_j = data[1]
		return - (prediction_i - prediction_j).sigmoid().log().sum()
    
	def handle_test(self, data, top_k):
		_, indices = torch.topk(data[0], top_k)
		return indices

	def predict_user(self, user, topK):
		cos = nn.CosineSimilarity(dim=1, eps=1e-6)
		user_emb = self.embed_user(torch.Tensor([user]).long().cuda())
		similarities = cos(user_emb.data, self.embed_item.weight.data)
		_, top_indices = torch.topk(similarities, topK, largest=True)
		_, bot_indices = torch.topk(similarities, topK, largest=False)
		return top_indices, bot_indices

	# def get_semi_hard_negative_items(self, user, pos_items, R):
	# 	# get R% of |V|
	# 	k = int(self.item_num * R)
	# 	user_emb = self.embed_user(torch.Tensor([user]).long().cuda()).data
	# 	item_emb = self.embed_item.weight.data
	# 	neg_items = []
	# 	for item in range(self.item_num):
	# 		if item not in pos_items:
	# 			neg_items.append(item)

	# 	# compute V_k^u
	# 	score = [0] * self.item_num
	# 	for item in neg_items:
	# 		score[item] = torch.dist(user_emb, item_emb[item], p=2).item()
	# 	top_k = heapq.nlargest(k, score)
	# 	V_u = [score.index(x) for x in top_k]
		
	# 	# compute V_k^v
	# 	score = [0] * self.item_num
	# 	selected_rows = item_emb.index_select(0, torch.Tensor(pos_items).long().cuda())
	# 	# Compute the mean of the selected rows
	# 	emb_i_cet = selected_rows.mean(dim=0)
	# 	for item in neg_items:
	# 		score[item] = torch.dist(emb_i_cet, item_emb[item], p=2).item()
	# 	top_k = heapq.nlargest(k, score)
	# 	V_v = [score.index(x) for x in top_k]
	# 	# negative sample 
	# 	V_neg_set = set(V_u).union(V_v)
	# 	V_neg_list = list(V_neg_set)
	# 	return V_neg_list

	def get_semi_hard_negative_items(self, user, pos_items, R):
		# get R% of |V|
		import time
		k = int(self.item_num * R)
		user_emb = self.embed_user(torch.Tensor([user]).long().cuda()).data
		item_emb = self.embed_item.weight.data
		neg_items = []
		for item in range(self.item_num):
			if item not in pos_items:
				neg_items.append(item)

		# compute V_k^u
		score = torch.cdist(user_emb.unsqueeze(0), item_emb, p=2).squeeze()
		score = list(score.cpu().numpy())
		top_k = heapq.nlargest(k, score)
		V_u = [score.index(x) for x in top_k]
  
		# compute V_k^v
		selected_rows = item_emb.index_select(0, torch.Tensor(pos_items).long().cuda())
		# Compute the mean of the selected rows
		emb_i_cet = selected_rows.mean(dim=0)
		score = torch.cdist(emb_i_cet.unsqueeze(0), item_emb, p=2).squeeze()
		score = list(score.cpu().numpy())
		top_k = heapq.nlargest(k, score)
		V_v = [score.index(x) for x in top_k]

		# negative sample 
		V_neg_set = set(V_u).union(V_v)
		V_neg_list = list(V_neg_set)
		return V_neg_list