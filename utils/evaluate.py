import torch 
import numpy as np

def predict_fast_opt(model, num_user, num_item, parallel_users, predict_data=None, users_set=None):
    model.eval()
    device = next(model.parameters()).device  # get the device of model
    scores = []
    for s in range(int(num_user/parallel_users)):
        user_input = []
        item_input = []
        for i in range(s*parallel_users, (s+1)*parallel_users):
            user_input.append(torch.ones(predict_data.shape[1], 1, device=device) * users_set[i])
            item_input.append(torch.from_numpy(predict_data[i]).view(-1, 1).to(device))
        user_input = torch.cat(user_input,axis=0).long()
        item_input = torch.cat(item_input, axis=0).long()
        # import pdb; pdb.set_trace()
        prediction_i = model.get_score([user_input, item_input])
        scores.append(prediction_i.detach().cpu().numpy().reshape(parallel_users, -1))

    if int(num_user / parallel_users) * parallel_users < num_user:
        user_input = []
        item_input = []
        for i in range(int(num_user/parallel_users)*parallel_users, num_user):
            user_input.append(torch.ones(predict_data.shape[1], 1, device=device) * users_set[i])
            item_input.append(torch.from_numpy(predict_data[i]).view(-1, 1).to(device))
        user_input = torch.cat(user_input, axis=0).long()
        item_input = torch.cat(item_input, axis=0).long()
        prediction_i = model.get_score([user_input, item_input])
        scores.append(prediction_i.detach().cpu().numpy().reshape(num_user-int(num_user/parallel_users)*parallel_users, -1))

    scores = np.concatenate(scores, axis=0)

    return scores

def predict_pos_opt(model, num_user, max_posid, parallel_users, predict_data=None, users_set=None):
    model.eval()
    device = next(model.parameters()).device  # get the device of model
    scores = []
    for s in range(int(num_user/parallel_users)):
        user_input = []
        item_input = []
        for i in range(s*parallel_users,(s+1)*parallel_users):
            user_input.append(torch.ones((len(predict_data[i]), 1), device=device) * users_set[i])
            item_input.append(torch.tensor(predict_data[i], device=device).view(-1, 1))
        user_input = torch.cat(user_input, dim=0).long()
        item_input = torch.cat(item_input, dim=0).long()
        score_flatten = model.get_score([user_input, item_input]).detach().cpu().numpy()
        score_tmp = np.zeros((parallel_users, max_posid))

        c = 0
        for i in range(s * parallel_users, (s + 1) * parallel_users):
            l = len(predict_data[i])
            score_tmp[i-s*parallel_users,0:l] = np.reshape(score_flatten[c:c+l], [1, -1])
            c += l
        scores.append(score_tmp)

    if int(num_user / parallel_users) * parallel_users < num_user:
        user_input = []
        item_input = []
        for i in range(int(num_user / parallel_users) * parallel_users, num_user):
            user_input.append(torch.ones((len(predict_data[i]), 1), device=device) * users_set[i])
            item_input.append(torch.tensor(predict_data[i], device=device).view(-1, 1))
        user_input = torch.cat(user_input, axis=0).long()
        item_input = torch.cat(item_input, axis=0).long()
        score_flatten = model.get_score([user_input, item_input]).detach().cpu().numpy()
        score_tmp = np.zeros((num_user - int(num_user / parallel_users) * parallel_users, max_posid))

        c = 0
        for i in range(int(num_user / parallel_users) * parallel_users, num_user):
            l = len(predict_data[i])
            score_tmp[i - int(num_user / parallel_users) * parallel_users, 0:l] = np.reshape(score_flatten[c:c + l], [1, -1])
            c += l
        scores.append(score_tmp)
    scores = np.concatenate(scores, axis=0)

    return scores

