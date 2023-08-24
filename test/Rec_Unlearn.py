import numpy as np
import torch
import torch.nn as nn
import sys
import heapq
import numpy as np
from collections import defaultdict
import scipy.sparse as sp
import pickle
import os.path
from os import remove
from re import split
from os.path import abspath
from time import strftime, localtime, time
from random import shuffle,randint,choice,sample

class FileIO(object):
    def __init__(self):
        pass

    @staticmethod
    def write_file(dir, file, content, op='w'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir + file, op) as f:
            f.writelines(content)

    @staticmethod
    def delete_file(file_path):
        if os.path.exists(file_path):
            remove(file_path)

    @staticmethod
    def load_data_set(file, rec_type):
        if rec_type == 'graph':
            data = []
            with open(file) as f:
                for line in f:
                    items = split(' ', line.strip())
                    user_id = items[0]
                    item_id = items[1]
                    weight = items[2]
                    data.append([user_id, item_id, float(weight)])

        if rec_type == 'sequential':
            data = {}
            with open(file) as f:
                for line in f:
                    items = split(':', line.strip())
                    seq_id = items[0]
                    data[seq_id]=items[1].split()
        return data

    @staticmethod
    def load_user_list(file):
        user_list = []
        print('loading user List...')
        with open(file) as f:
            for line in f:
                user_list.append(line.strip().split()[0])
        return user_list

    @staticmethod
    def load_social_data(file):
        social_data = []
        print('loading social data...')
        with open(file) as f:
            for line in f:
                items = split(' ', line.strip())
                user1 = items[0]
                user2 = items[1]
                if len(items) < 3:
                    weight = 1
                else:
                    weight = float(items[2])
                social_data.append([user1, user2, weight])
        return social_data

class Data(object):
    def __init__(self, training, test):
        # self.config = conf
        self.training_data = training
        self.test_data = test #can also be validation set if the input is for validation

class Graph(object):
    def __init__(self):
        pass

    @staticmethod
    def normalize_graph_mat(adj_mat):
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
        return norm_adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        pass

class Interaction(Data,Graph):
    def __init__(self, training, test):
        Graph.__init__(self)
        Data.__init__(self,training,test)

        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.test_set = defaultdict(dict)
        self.test_set_item = set()
        self.__generate_set()
        self.user_num = len(self.training_set_u)
        self.item_num = len(self.training_set_i)
        self.ui_adj = self.__create_sparse_bipartite_adjacency()
        self.norm_adj = self.normalize_graph_mat(self.ui_adj)
        self.interaction_mat = self.__create_sparse_interaction_matrix()

    def __generate_set(self):
        for entry in self.training_data:
            user, item, rating = entry
            if user not in self.user:
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                # userList.append
            self.training_set_u[user][item] = rating
            self.training_set_i[item][user] = rating
        for entry in self.test_data:
            user, item, rating = entry
            if user not in self.user or item not in self.item:
                continue
            self.test_set[user][item] = rating
            self.test_set_item.add(item)

    def __create_sparse_bipartite_adjacency(self, self_connection=False):
        '''
        return a sparse adjacency matrix with the shape (user number + item number, user number + item number)
        '''
        n_nodes = self.user_num + self.item_num
        row_idx = [self.user[pair[0]] for pair in self.training_data]
        col_idx = [self.item[pair[1]] for pair in self.training_data]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        import pdb; pdb.set_trace()
        return adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        adj_shape = adj_mat.get_shape()
        n_nodes = adj_shape[0]+adj_shape[1]
        (user_np_keep, item_np_keep) = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_shape[0])),shape=(n_nodes, n_nodes),dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    def __create_sparse_interaction_matrix(self):
        """
        return a sparse adjacency matrix with the shape (user number, item number)
        """
        row, col, entries = [], [], []
        for pair in self.training_data:
            row += [self.user[pair[0]]]
            col += [self.item[pair[1]]]
            entries += [1.0]
        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.user_num,self.item_num),dtype=np.float32)
        return interaction_mat

    def get_user_id(self, u):
        if u in self.user:
            return self.user[u]

    def get_item_id(self, i):
        if i in self.item:
            return self.item[i]

    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)

    def test_size(self):
        return len(self.test_set), len(self.test_set_item), len(self.test_data)

    def contain(self, u, i):
        'whether user u rated item i'
        if u in self.user and i in self.training_set_u[u]:
            return True
        else:
            return False

    def contain_user(self, u):
        'whether user is in training set'
        if u in self.user:
            return True
        else:
            return False

    def contain_item(self, i):
        """whether item is in training set"""
        if i in self.item:
            return True
        else:
            return False

    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())

    def row(self, u):
        u = self.id2user[u]
        k, v = self.user_rated(u)
        vec = np.zeros(len(self.item))
        # print vec
        for pair in zip(k, v):
            iid = self.item[pair[0]]
            vec[iid] = pair[1]
        return vec

    def col(self, i):
        i = self.id2item[i]
        k, v = self.item_rated(i)
        vec = np.zeros(len(self.user))
        # print vec
        for pair in zip(k, v):
            uid = self.user[pair[0]]
            vec[uid] = pair[1]
        return vec

    def matrix(self):
        m = np.zeros((len(self.user), len(self.item)))
        for u in self.user:
            k, v = self.user_rated(u)
            vec = np.zeros(len(self.item))
            # print vec
            for pair in zip(k, v):
                iid = self.item[pair[0]]
                vec[iid] = pair[1]
            m[self.user[u]] = vec
        return m
    
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

def next_batch_pairwise(data,batch_size,n_negs=1):
    training_data = data.training_data
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])
        yield u_idx, i_idx, j_idx

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

import math
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


def ranking_evaluation(origin, res, N):
    measure = []
    for n in N:
        predicted = {}
        for user in res:
            # import pdb; pdb.set_trace()
            predicted[user] = res[user][:n]
        indicators = []
        if len(origin) != len(predicted):
            print('The Lengths of test set and predicted set do not match!')
            exit(-1)
        hits = Metric.hits(origin, predicted)
        hr = Metric.hit_ratio(origin, hits)
        indicators.append('Hit Ratio:' + str(hr) + '\n')
        prec = Metric.precision(hits, n)
        indicators.append('Precision:' + str(prec) + '\n')
        recall = Metric.recall(hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')
        # F1 = Metric.F1(prec, recall)
        # indicators.append('F1:' + str(F1) + '\n')
        #MAP = Measure.MAP(origin, predicted, n)
        #indicators.append('MAP:' + str(MAP) + '\n')
        NDCG = Metric.NDCG(origin, predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        # AUC = Measure.AUC(origin,res,rawRes)
        # measure.append('AUC:' + str(AUC) + '\n')
        measure.append('Top ' + str(n) + '\n')
        measure += indicators
    return measure

class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

class LGCN_Encoder(nn.Module):
    def __init__(self, data_conf, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        # self.data = data
        self.user_num = data_conf['user_num']
        self.item_num = data_conf['item_num']
        self.latent_size = emb_size
        self.layers = n_layers
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
        import pdb; pdb.set_trace()
        return user_all_embeddings, item_all_embeddings

def train(config, model, data):
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['learnRate']))
    # data = Interaction(conf, training_set, test_set)
    maxEpoch = int(config['num.max.epoch'])
    glob_user_emb = None
    glob_item_emb = None
    best_user_emb, best_item_emb = None, None
    bestPerformance = []
    for epoch in range(maxEpoch):
        for n, batch in enumerate(next_batch_pairwise(data, int(config['batch_size']))):
            user_idx, pos_idx, neg_idx = batch
            rec_user_emb, rec_item_emb = model()
            import pdb; pdb.set_trace()
            user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
            batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(float(config['reg.lambda']), user_emb,pos_item_emb,neg_item_emb)/int(config['batch_size'])
            # Backward and optimize
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            if n % 100==0 and n>0:
                print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
        with torch.no_grad():
            glob_user_emb, glob_item_emb = model()
        if epoch % 5 == 0: 
            print('Shape of embedding')
            print(glob_user_emb.shape)
            print(glob_item_emb.shape)
            _, best_user_emb, best_item_emb, bestPerformance = fast_evaluation(model, epoch, data, bestPerformance, glob_user_emb, glob_item_emb,
                                                                               best_user_emb, best_item_emb)
    # glob_user_emb, glob_item_emb = best_user_emb, best_item_emb
    return best_user_emb, best_item_emb

def test(data, user_emb, item_emb):
    def process_bar(num, total):
        rate = float(num) / total
        ratenum = int(50 * rate)
        r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
        sys.stdout.write(r)
        sys.stdout.flush()
    
    def predict(u):
        u = data.get_user_id(u)
        score = torch.matmul(user_emb[u], item_emb.transpose(0, 1))
        return score.cpu().numpy()

    # predict
    rec_list = {}
    user_count = len(data.test_set)
    for i, user in enumerate(data.test_set):
        candidates = predict(user)
        # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
        rated_list, li = data.user_rated(user)
        for item in rated_list:
            candidates[data.item[item]] = -10e8
        ids, scores = find_k_largest(max_N, candidates)
        item_names = [data.id2item[iid] for iid in ids]
        rec_list[user] = list(zip(item_names, scores))
        if i % 1000 == 0:
            process_bar(i, user_count)
    process_bar(user_count, user_count)
    print('')
    return rec_list, user_emb, item_emb

def fast_evaluation(model, epoch, data, bestPerformance, glob_user_emb, glob_item_emb, best_user_emb, best_item_emb):
    print('Evaluating the model...')
    rec_list, u_emb, i_emb = test(data, glob_user_emb, glob_item_emb)
    measure = ranking_evaluation(data.test_set, rec_list, [max_N])
    if len(bestPerformance) > 0:
        count = 0
        performance = {}
        for m in measure[1:]:
            k, v = m.strip().split(':')
            performance[k] = float(v)
        for k in bestPerformance[1]:
            if bestPerformance[1][k] > performance[k]:
                count += 1
            else:
                count -= 1
        if count < 0:
            bestPerformance[1] = performance
            bestPerformance[0] = epoch + 1
            with torch.no_grad():
              best_user_emb, best_item_emb = model.forward()
    else:
        bestPerformance.append(epoch + 1)
        performance = {}
        for m in measure[1:]:
            k, v = m.strip().split(':')
            performance[k] = float(v)
        bestPerformance.append(performance)
        with torch.no_grad():
            best_user_emb, best_item_emb = model.forward()
    print('-' * 120)
    print('Real-Time Ranking Performance ' + ' (Top-' + str(max_N) + ' Item Recommendation)')
    measure = [m.strip() for m in measure[1:]]
    print('*Current Performance*')
    print('Epoch:', str(epoch + 1) + ',', '  |  '.join(measure))
    bp = ''
    # for k in bestPerformance[1]:
    #     bp+=k+':'+str(bestPerformance[1][k])+' | '
    bp += 'Hit Ratio' + ':' + str(bestPerformance[1]['Hit Ratio']) + '  |  '
    bp += 'Precision' + ':' + str(bestPerformance[1]['Precision']) + '  |  '
    bp += 'Recall' + ':' + str(bestPerformance[1]['Recall']) + '  |  '
    # bp += 'F1' + ':' + str(bestPerformance[1]['F1']) + ' | '
    bp += 'NDCG' + ':' + str(bestPerformance[1]['NDCG'])
    print('*Best Performance* ')
    print('Epoch:', str(bestPerformance[0]) + ',', bp)
    print('-' * 120)
    return measure, best_user_emb, best_item_emb, bestPerformance

if __name__ == '__main__':
    config = {}
    config['training.set']='./train.txt'
    config['test.set']='./test.txt'
    config['model.name']='LightGCN'
    config['model.type']='graph'
    config['topN']='10'
    config['embedding.size']=64
    config['num.max.epoch']=200
    config['batch_size']=2048
    config['learnRate']=0.001 # 0.001
    config['reg.lambda']=0.0001
    # config['LightGCN']='-n_layer 2'
    config['n_layer'] = 2
    config['dir'] = './results/'

    top = config['topN'].split(',')
    topN = [int(num) for num in top]
    max_N = max(topN)

    training_data = FileIO.load_data_set(config['training.set'], config['model.type'])
    test_data = FileIO.load_data_set(config['test.set'], config['model.type'])
    # import pdb; pdb.set_trace()
    data = Interaction(training_data, test_data)
    # import pdb; pdb.set_trace()
    data_conf = {}
    data_conf['user_num'] = data.user_num
    data_conf['item_num'] = data.item_num
    data_conf['norm_adj'] = data.norm_adj
    model = LGCN_Encoder(data_conf, int(config['embedding.size']), int(config['n_layer']))
    # import pdb; pdb.set_trace()
    train(config, model, data)