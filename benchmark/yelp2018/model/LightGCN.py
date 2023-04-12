
import torch
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule
from benchmark.torch_interface import TorchGraphInterface

class Model(FModule):
    def __init__(self, data_conf, conf):
        super(Model, self).__init__()
        # self.data = data
        self.user_num = data_conf['user_num']
        self.item_num = data_conf['item_num']
        self.latent_size = conf['embedding.size'] #emb_size
        self.layers = conf['n_layer'] #n_layers
        self.norm_adj = data_conf['norm_adj']
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.item_num, self.latent_size))),
        })
        return embedding_dict

    def to(self, device):
        self.sparse_norm_adj = self.sparse_norm_adj.to(device)
        return super(Model, self).to(device)

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.user_num]
        item_all_embeddings = all_embeddings[self.user_num:]
        return user_all_embeddings, item_all_embeddings


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)

