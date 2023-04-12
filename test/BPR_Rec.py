import numpy as np
import torch
import torch.nn as nn
import sys
import argparse
from Rec_Unlearn import find_k_largest, next_batch_pairwise, FileIO, Interaction, Metric

import torch
import torch.nn as nn


class BPR(nn.Module):
	def __init__(self, user_num, item_num, factor_num):
		super(BPR, self).__init__()
		"""
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors.
		"""		
		self.embed_user = nn.Embedding(user_num, factor_num)
		self.embed_item = nn.Embedding(item_num, factor_num)

		nn.init.normal_(self.embed_user.weight, std=0.01)
		nn.init.normal_(self.embed_item.weight, std=0.01)

	def forward(self, user, item_i, item_j):
		user = self.embed_user(user)
		item_i = self.embed_item(item_i)
		item_j = self.embed_item(item_j)

		prediction_i = (user * item_i).sum(dim=-1)
		prediction_j = (user * item_j).sum(dim=-1)
		return prediction_i, prediction_j
    
# Dataset
class RecDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(RecDataset, self).__init__()

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx):
        pass

def metrics(model, test_loader, top_k):
    pass

def train(config, model, data):
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['learnRate']))
    # criterion = nn.BCEWithLogitsLoss()
    # train_dataloader = torch.utits.data.Dataloader(train_dataset, batch_size= int(config['batch_size']), shuffle= config['shuffle'])
    # test_dataloader = torch.utits.data.Dataloader(test_dataset, batch_size= int(config['batch_size']), shuffle= False)
    maxEpoch = int(config['max_epoch'])
    log_idx = 100
    items = torch.Tensor(list(data.item.values())).cuda().long()
    for epoch in range(maxEpoch+1):
        model.train()
        running_loss = 0.0
        for idx, batch in  enumerate(next_batch_pairwise(data, int(config['batch_size']))):
        # for idx, (users, items, ratings) in enumerate(train_dataloader):
            # move users, items, and ratings onto the device
            users, pos_items, neg_items = batch
            # import pdb; pdb.set_trace()
            users = torch.Tensor(users).cuda().long()
            pos_items = torch.Tensor(pos_items).cuda().long()
            neg_items = torch.Tensor(neg_items).cuda().long()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            pos_outs, neg_outs = model(users, pos_items, neg_items) #.reshape(-1)
            # pos_outs, neg_outs = pos_outs.reshape(-1)

            # loss = criterion(outputs, ratings)
            # loss = torch.mean( -(pos_outs - neg_outs).sigmoid().log() )
            loss = - (pos_outs - neg_outs).sigmoid().log().sum()
            loss.backward()
            optimizer.step()

            # accumulate loss and log
            running_loss += loss.item()
            if idx % log_idx == log_idx - 1:
                print(f"Epoch {epoch} | Steps: {idx + 1:<4} | Loss: {running_loss / log_idx:.3f}")
                running_loss = 0.0

        if epoch % 50 == 0:
            model.eval()
            print('-----------------------')
            print('Test phase:')
            HR, NDCG = test(model, data, items, config['topK'])
            print('HR: {} | NDCG: {}'.format(HR, NDCG))
            print('-----------------------')

@torch.no_grad()
def test(model, data, items, topK):
    # model.eval()
    def process_bar(num, total):
        rate = float(num) / total
        ratenum = int(50 * rate)
        r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
        sys.stdout.write(r)
        sys.stdout.flush()
    
    def predict(u):
        user = torch.Tensor([data.get_user_id(u)]).cuda().long()
        score, _ = model(user, items, torch.Tensor([1]).cuda().long())
        # score = torch.matmul(user_emb[u], item_emb.transpose(0, 1))
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
        ids, scores = find_k_largest(topK, candidates)
        item_names = [data.id2item[iid] for iid in ids]
        rec_list[user] = list(zip(item_names, scores))
        if i % 1000 == 0:
            process_bar(i, user_count)
    process_bar(user_count, user_count)
    print('')
    predicted = {}
    for user in rec_list:
        # import pdb; pdb.set_trace()
        predicted[user] = rec_list[user][:topK]
    if user_count != len(predicted):
        print('The Lengths of test set and predicted set do not match!')
        exit(-1)
    hits = Metric.hits(data.test_set, predicted)
    hr = Metric.hit_ratio(data.test_set, hits)
    NDCG = Metric.NDCG(data.test_set, predicted, topK)
    return hr, NDCG

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding.size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=0.001)
    try: option = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))

    config = {}
    config['training.set']='./train.txt'
    config['test.set']='./test.txt'
    # config['model.name']='LightGCN'
    config['model.type']='graph'
    config['topN']='10'
    config['embedding.size']=option['embedding.size']
    config['max_epoch']=200
    config['batch_size']=option['batch_size']
    config['learnRate']=option['lr'] # 0.001
    # config['LightGCN']='-n_layer 2'
    config['n_layer'] = 2
    config['dir'] = './results/'

    top = config['topN'].split(',')
    topN = [int(num) for num in top]
    max_N = max(topN)
    config['topK'] = max_N

    training_data = FileIO.load_data_set(config['training.set'], config['model.type'])
    test_data = FileIO.load_data_set(config['test.set'], config['model.type'])
    # import pdb; pdb.set_trace()
    data = Interaction(training_data, test_data)
    data_conf = {}
    data_conf['user_num'] = data.user_num
    data_conf['item_num'] = data.item_num
    data_conf['norm_adj'] = data.norm_adj
    # model = LGCN_Encoder(data_conf, int(config['embedding.size']), int(config['n_layer']))
    model = BPR(data_conf['user_num'], data_conf['item_num'], config['embedding.size'])
    # CUDA_VISIBLE_DEVICES=0 python BPR_Rec.py --embedding.size 16 --batch_size 2048 --lr 0.01
    # CUDA_VISIBLE_DEVICES=0 python BPR_Rec.py --embedding.size 16 --batch_size 4096 --lr 0.01
    train(config, model, data)
            