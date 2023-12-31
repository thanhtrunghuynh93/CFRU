import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.fmodule import FModule
# from benchmark.torch_interface import TorchGraphInterface

loss_func = nn.BCEWithLogitsLoss()

class Model(FModule):
    def __init__(self, data_conf, conf):
        super(Model, self).__init__()
        # self.data = data
        self.user_num = data_conf['user_num']
        self.item_num = data_conf['item_num']
        self.dropout = conf['dropout']
        self.factor_num = conf['embedding.size'] #emb_size
        self.num_layers = conf['n_layer'] #n_layers
        
        self.embed_user = nn.Embedding(self.user_num, self.factor_num * (2 ** (self.num_layers - 1)))
        self.embed_item = nn.Embedding(self.item_num, self.factor_num * (2 ** (self.num_layers - 1)))
        
        MLP_modules = []
        for i in range(self.num_layers):
            input_size = self.factor_num * (2 ** (self.num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size//2))
            MLP_modules.append(nn.ReLU())    
        self.MLP_layers = nn.Sequential(*MLP_modules)
        
        self.predict_layer = nn.Linear(self.factor_num, 1)
        
        self._init_model()

    def _init_model(self):
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, data):
        embed_user_MLP = self.embed_user(data[0])
        embed_item_MLP = self.embed_item(data[1])
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)
        prediction = self.predict_layer(output_MLP)
        # import pdb; pdb.set_trace()
        return prediction.view(-1), data[2]
    
    def get_score(self, data):
        embed_user_MLP = self.embed_user(data[0])
        embed_item_MLP = self.embed_item(data[1])
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)
        prediction = self.predict_layer(output_MLP)
        # import pdb; pdb.set_trace()
        return prediction.view(-1)
    
    def handle_loss(self, data, option=None):
        prediction = data[0]
        label = data[1].float()
        return loss_func(prediction, label)
    
    def process_input(self, data, device=None):
        users = torch.cat((data[0], data[0]), dim=0)
        items = torch.cat((data[1], data[2]), dim=0)
        pos_label = torch.tensor([1 for _ in range(data[1].shape[0])])
        neg_label = torch.tensor([0 for _ in range(data[2].shape[0])])
        labels = torch.cat((pos_label, neg_label), dim=0).to(device)
        return [users, items, labels]
    
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


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)

