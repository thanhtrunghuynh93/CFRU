import torch
import pickle
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

if __name__ == '__main__':
    # for atk in ['fedAttack', 'fedFlipGrads']:
    #     with open('./fedtasksave/movielens1m_cnum100_dist11_skew0.0_seed0/ULKD_R30_P0.30_alpha0.03_seed0_{}/record/history30.pkl'.format(atk), 'rb') as test_f:
    #         hist = CPU_Unpickler(test_f).load()
    #     print(hist)
    #     # print('HR: {} | NDCG: {}'.format(hist['HR_on_clients'], hist['NDCG_on_clients']))

    for alp in [0.06, 0.09, 0.2]:
        with open('./fedtasksave/movielens1m_cnum10_dist11_skew0.0_seed0/FedEraser_R25_P0.30_alpha{}_clean2_seed0/record/history25.pkl'.format(alp), 'rb') as test_f:
            hist = CPU_Unpickler(test_f).load()
        print('HR: {} | NDCG: {}'.format(hist['HR_on_clients'], hist['NDCG_on_clients']))