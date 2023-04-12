import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.fmodule import FModule

class Model(FModule):
	def __init__(self, data_conf, conf):
		super(Model, self).__init__()
		"""
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors.
		"""		
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
