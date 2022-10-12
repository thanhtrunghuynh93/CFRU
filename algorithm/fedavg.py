from .fedbase import BasicServer, BasicClient
import pickle
import os
from utils import fmodule
import numpy as np
class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None, backtask_data = None):
        super(Server, self).__init__(option, model, clients, test_data, backtask_data)
        self.path_save = os.path.join('fedtasksave', self.option['task'],
                                    "R{}_P{:.2f}_AP{:.2f}_Alpha1.0_3atk_minus{}".format(
                                        option['num_rounds'],
                                        option['proportion'],
                                        option['attacker_pct'],
                                        self.gamma
                                        # option['clean_model']
                                    ),
                                    'record')
        self.unlearn_term_algo2 = None
        self.unlearn_term_algo3 = None
        # create folder for saving model
        if not os.path.exists(self.path_save):
            os.makedirs(self.path_save, exist_ok=True)
    
    def save_models(self, round_num, models, uncertainty_round):
        # aggregate
        temp_model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        # model clean with algo2
        clean_model = temp_model + self.unlearn_term_algo2
        test_metric2, test_loss2, test_backdoor2 = self.test(model= clean_model)
        # model clean with algo3
        clean_model = temp_model + self.unlearn_term_algo3
        test_metric3, test_loss3, test_backdoor3 = self.test(model= clean_model)
        # log
        save_logs = {
            "selected_clients": self.selected_clients,
            "models": models,
            "p": [1.0 * self.client_vols[cid] / self.data_vol for cid in self.selected_clients],
            "server_model": self.model,
            "unlearn_term_algo2": self.unlearn_term_algo2,
            "unlearn_term_algo3": self.unlearn_term_algo3,
            "uncertainty_round": uncertainty_round,
            "accuracy2": [test_metric2, test_backdoor2],
            "accuracy3": [test_metric3, test_backdoor3]
        }
        # print(self.model.state_dict()['fc2.weight'][5])
        # print((self.model - models[0]).state_dict()['fc2.weight'][6])
        pickle.dump(save_logs,
                    open(os.path.join(self.path_save, "history" + str(round_num) + ".pkl"), 'wb'),
                    pickle.HIGHEST_PROTOCOL)
        print("Save  ", round_num)
    
    def iterate(self, t):
        self.selected_clients = self.sample(t)
        for idx in self.selected_clients:
            self.round_selected[idx].append(t)
        
        # training
        models, train_losses, uncertainty_round = self.communicate(self.selected_clients)

        # if t > 3:
        #     import pdb; pdb.set_trace() 
        ##  Process Unlearning
        ### update uncertainty for clients
        self.update_uncertainty(uncertainty_round)
        ### update grad of received from clients
        # if t >= 85:
        #     import pdb; pdb.set_trace()
        print((self.model-models[0]).state_dict()['fc2.weight'][9])
        ## pdb.set_trace = lambda: 1

        ## start algorithm
        # save unlearn term if model is dirty
        if self.option['clean_model'] == 0: 
            # save grads
            self.process_grad(models, t)
            # find attack_clients
            attack_clients = []
            for cid in self.selected_clients: 
                if cid in self.option['attacker']:
                    attack_clients.append(cid) 
            # compute lipschitz constraint at this round
            self.lipschitz_constances.append(1.0)
            # compute beta for this round
            self.update_beta()
            # # unlearning 
            if len(attack_clients) >= 1:
                # self.all_attack_clients_id = list(set(self.all_attack_clients_id).union(attack_clients))
                round_attack, attackers_round = self.getAttacker_rounds(attack_clients)
                # unlearn 
                self.unlearn_term_algo2, self.unlearn_term_algo3 = self.compute_unlearn_term(round_attack, attackers_round, t)

        # import pdb; pdb.set_trace()
        self.save_models(t, models, uncertainty_round)
        # check whether all the clients have dropped out, because the dropped clients will be deleted from self.selected_clients
        if not self.selected_clients: return
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        # save all models and uncertainties to disk
        return uncertainty_round

    def compute_unlearn_term(self, round_attack, attackers_round, round):
        unlearning_term2 = fmodule._create_new_model(self.model) * 0.0
        for idx in range(len(round_attack)):
            round_id = round_attack[idx]
                # compute u-term at round round_id (attack round)
            for c_id in attackers_round[idx]:
                unlearning_term2 += 1.0 * self.client_vols[c_id]/self.data_vol * self.grads_all_round[round_id][str(c_id)].to(self.model.get_device())
                self.grads_all_round[round_id][str(c_id)].cpu()
        
        unlearning_term2 = unlearning_term2 * self.theta

        unlearning_term3 = fmodule._create_new_model(self.model) * 0.0
            # compute lipschitz constant by average over all rounds
        # alpha = 1.0 * self.gamma * sum(self.lipschitz_constances) / (round + 1)
        # alpha = - np.exp(-0.0911*round + 1.7597)
        alpha = - self.gamma
        # start
        # num_layers = len(unlearning_term3.state_dict().keys())
        # # alpha changes over rounds
        # if self.gamma == 1: 
        #     alpha = np.array([-0.05 for i in range(num_layers)])
        #     if round < 50:
        #         alpha[:4] = -0.09
        #     else:
        #         alpha[:4] = -0.05
        # elif self.gamma == 2:
        #     alpha = np.array([-0.09 for i in range(num_layers)])
        #     if round < 50:
        #         alpha[4:] = -0.05
        #     else:
        #         alpha[4:] = -0.02
        # elif self.gamma == 3:
        #     alpha = np.array([-0.05 for i in range(num_layers)])
        #     if round < 50:
        #         alpha[:4] = -0.09
        #         alpha[4:] = -0.05
        #     else:
        #         alpha[:4] = -0.05
        #         alpha[4:] = -0.02
        # elif self.gamma == 4:
        #     alpha = np.array([-0.02 for i in range(num_layers)])
        #     if round < 50:
        #         alpha[:4] = -0.09
        #     else:
        #         alpha[:4] = -0.05
        # elif self.gamma == 5:
        #     alpha = np.array([-0.05 for i in range(num_layers)])
        #     if round < 50:
        #         alpha[4:] = -0.05
        #     else:
        #         alpha[4:] = -0.02
        # elif self.gamma == 0.005:
        #     alpha 
        #     raise Exception("Invalid value for gamma")
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
            # unlearning_term3 = fmodule.multi_layer_by_alpha(unlearning_term3, list_beta[round_id]) #
            unlearning_term3 = unlearning_term3 * list_beta[round_id]
            for c_id in attackers_round[idx]:
                unlearning_term3 += 1.0 * self.client_vols[c_id]/self.data_vol * self.grads_all_round[round_id][str(c_id)].to(self.model.get_device())
                self.grads_all_round[round_id][str(c_id)].cpu()
                #fmodule._model_to_gpu(self.grads_all_round[round_id][str(c_id)], device= self.model.get_device())
                
            if idx == len(round_attack) - 1: continue
            for r_id in range(round_id + 1, round_attack[idx + 1]):
                # unlearning_term3 = fmodule.multi_layer_by_alpha(unlearning_term3, list_beta[r_id]) # unlearning_term3 * list_beta[r_id]
                unlearning_term3 = unlearning_term3 * list_beta[r_id]
        unlearning_term3 = unlearning_term3 * self.theta
        return unlearning_term2, unlearning_term3

    def sample(self, t):
        self.fixed_selected_clients[t] = [i for i in range(30)]
        ##
        if self.option['clean_model'] == 0: 
            selected_clients = self.fixed_selected_clients[t]
        elif self.option['clean_model'] == 1:
            selected_clients = []
            for cid in self.fixed_selected_clients[t]:
                if cid not in self.option['attacker']:
                    selected_clients.append(cid)
        else:
            raise Exception("Invalid value for attribute clean_model")
        # selected_clients = [10]
        return selected_clients

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)


