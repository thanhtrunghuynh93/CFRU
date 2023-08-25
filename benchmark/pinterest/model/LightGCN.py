import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from utils.fmodule import FModule

class Model(FModule):
    def __init__(self, data_conf, conf):
        super(Model, self).__init__()
        # self.data = data
        self.user_num = data_conf['user_num']
        self.item_num = data_conf['item_num']
        self.latent_size = conf['embedding.size'] #emb_size
        self.layers = conf['n_layer'] #n_layers
        # self.embedding_dict = self._init_model()
        # init embedding
        self.embed_user = nn.Embedding(self.user_num, self.latent_size)
        self.embed_item = nn.Embedding(self.item_num, self.latent_size)
        
        nn.init.xavier_uniform_(self.embed_user.weight)
        nn.init.xavier_uniform_(self.embed_item.weight)
  
        # init norm adj matrix
        self.sparse_norm_adj = self._init_norm_adj_mat(data_conf['training_data'])#.cuda()

    def _init_norm_adj_mat(self, training_data):
        # training_data : list of pair user-item interaction
        # build sparse bipartite adjacency graph
        n_nodes = self.user_num + self.item_num
        row_idx = [pair[0] for pair in training_data]
        col_idx = [pair[1] for pair in training_data]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        
        # normalize built graph
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
            
        # convert sparse matrix to tensor
        coo = norm_adj_mat.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        sparse_tensor = torch.sparse.FloatTensor(i, v, coo.shape)
        self.sparse_indices = sparse_tensor._indices()
        self.sparse_values = sparse_tensor._values()
        self.sparse_shape = sparse_tensor.shape
        return None
    
    def to(self, device):
        # self.sparse_norm_adj = self.sparse_norm_adj.to(device)
        return super(Model, self).to(device)

    def forward(self, data):
        sparse_norm_adj = torch.sparse_coo_tensor(self.sparse_indices, self.sparse_values, self.sparse_shape).to(next(self.parameters()).device)
        user = data[0]
        item_i = data[1]
        item_j = data[2]
        ego_embeddings = torch.cat([self.embed_user.weight, self.embed_item.weight], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.user_num]
        item_all_embeddings = all_embeddings[self.user_num:]
        # get embedding of given user and item
        user_embedding = user_all_embeddings[user]
        item_i_embedding = item_all_embeddings[item_i]
        item_j_embedding = item_all_embeddings[item_j]
        return user_embedding, item_i_embedding, item_j_embedding
    
    def get_score(self, data):
        sparse_norm_adj = torch.sparse_coo_tensor(self.sparse_indices, self.sparse_values, self.sparse_shape).to(next(self.parameters()).device)   
        user = data[0]
        item_i = data[1]
        # item_j = data[2]
        ego_embeddings = torch.cat([self.embed_user.weight, self.embed_item.weight], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.user_num]
        item_all_embeddings = all_embeddings[self.user_num:]
        # get embedding of given user and item
        user_embedding = user_all_embeddings[user]
        item_i_embedding = item_all_embeddings[item_i]
        # item_j_embedding = item_all_embeddings[item_j]
        prediction_i = (user_embedding * item_i_embedding).sum(dim=-1)
        return prediction_i
    
    def handle_loss(self, data, option):
        user_emb = data[0]
        pos_item_emb = data[1]
        neg_item_emb = data[2]
        return bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(option['reg.lambda'], user_emb, pos_item_emb, neg_item_emb)/option['batch_size']
    
    def process_input(self, data, device=None):
        return data
    
    def handle_test(self, data, top_k):
        prediction_i = (data[0] * data[1]).sum(dim=-1)
        _, indices = torch.topk(prediction_i, top_k)
        return indices

    def predict_user(self, user, topK):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        user_emb = self.embed_user(torch.Tensor([user]).long().cuda())
        similarities = cos(user_emb.data, self.embed_item.weight.data)
        _, top_indices = torch.topk(similarities, topK, largest=True)
        _, bot_indices = torch.topk(similarities, topK, largest=False)
        return top_indices, bot_indices

def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
        pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
        neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
        loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
        return torch.mean(loss)
    
def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2)
    return emb_loss * reg

class Loss(nn.Module):
    def __init__(self, reg_lambda, batch_size):
        super(Loss, self).__init__()
        self.reg = reg_lambda
        self.batch_size = batch_size

    def forward(self, user_emb, pos_item_emb, neg_item_emb):
        return self.bpr_loss(user_emb, pos_item_emb, neg_item_emb) + self.l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
    
    def bpr_loss(self, user_emb, pos_item_emb, neg_item_emb):
        pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
        neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
        loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
        return torch.mean(loss)
    
    def l2_reg_loss(self, reg, *args):
        emb_loss = 0
        for emb in args:
            emb_loss += torch.norm(emb, p=2)
        return emb_loss * reg

