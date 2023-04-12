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

	def forward(self, user, item_i, item_j):
		user = self.embed_user(user)
		item_i = self.embed_item(item_i)
		item_j = self.embed_item(item_j)

		prediction_i = (user * item_i).sum(dim=-1)
		prediction_j = (user * item_j).sum(dim=-1)
		return prediction_i, prediction_j
