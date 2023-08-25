CUDA_VISIBLE_DEVICES=0 python main.py --task movielens1m_cnum100_dist11_skew0.0_seed0 --alpha 0.09 --model NCF --embedding.size 32 --reg.lambda 0.0001 --n_layer 2 --dropout 0.2 --topN 10 --algorithm UL_FastTest --atk_method fedAttack --num_rounds 40 --num_epochs 10 --learning_rate 0.001 --batch_size 1024
#test
CUDA_VISIBLE_DEVICES=0 python main.py --task movielens1m_cnum10_dist11_skew0.0_seed0 --alpha 0.09 --model NCF --embedding.size 32 --reg.lambda 0.0001 --n_layer 1 --dropout 0.2 --topN 10 --algorithm UL_fedFast --atk_method fedAttack --num_rounds 1 --num_epochs 1 --learning_rate 0.001 --batch_size 16000

# retrain
### LightGCN #bangnt
CUDA_VISIBLE_DEVICES=1 python main.py --task pinterest_cnum100_dist11_skew0.0_seed0 --alpha 0.09 --model LightGCN --embedding.size 64 --reg.lambda 0.0001 --n_layer 2 --num_ng 1 --topN 10 --algorithm UL_base --atk_method fedAttack --num_rounds 40 --num_epochs 10 --learning_rate 0.001 --batch_size 2048 --clean_model 1
### NCF #bangntNCF
CUDA_VISIBLE_DEVICES=0 python main.py --task pinterest_cnum100_dist11_skew0.0_seed0 --alpha 0.1 --model NCF --embedding.size 64 --reg.lambda 0.0001 --n_layer 2 --dropout 0.2 --topN 10 --algorithm UL_base --atk_method fedAttack --num_rounds 40 --num_epochs 10 --learning_rate 0.001 --batch_size 2048 --clean_model 1
CUDA_VISIBLE_DEVICES=0 python main.py --task movielens1m_cnum100_dist11_skew0.0_seed0 --alpha 0.1 --model NCF --embedding.size 64 --reg.lambda 0.0001 --n_layer 2 --dropout 0.2 --topN 10 --algorithm UL_base --atk_method fedAttack --num_rounds 40 --num_epochs 10 --learning_rate 0.001 --batch_size 1024 --clean_model 1

## S1
CUDA_VISIBLE_DEVICES=0 python main.py --task movielens1m_cnum100_dist11_skew0.0_seed0 --alpha 0.09 --model NCF --S1 5 --embedding.size 64 --reg.lambda 0.0001 --n_layer 2 --dropout 0.2 --topN 10 --algorithm UL_Optimize --atk_method fedAttack --num_rounds 40 --num_epochs 10 --learning_rate 0.001 --batch_size 1024 
CUDA_VISIBLE_DEVICES=0 python main.py --task movielens1m_cnum100_dist11_skew0.0_seed0 --alpha 0.09 --model NCF --S1 6 --embedding.size 64 --reg.lambda 0.0001 --n_layer 2 --dropout 0.2 --topN 10 --algorithm UL_Optimize --atk_method fedAttack --num_rounds 40 --num_epochs 10 --learning_rate 0.001 --batch_size 1024 
CUDA_VISIBLE_DEVICES=0 python main.py --task movielens1m_cnum100_dist11_skew0.0_seed0 --alpha 0.09 --model NCF --S1 7 --embedding.size 64 --reg.lambda 0.0001 --n_layer 2 --dropout 0.2 --topN 10 --algorithm UL_Optimize --atk_method fedAttack --num_rounds 40 --num_epochs 10 --learning_rate 0.001 --batch_size 1024 
CUDA_VISIBLE_DEVICES=1 python main.py --task movielens1m_cnum100_dist11_skew0.0_seed0 --alpha 0.09 --model NCF --S1 8 --embedding.size 64 --reg.lambda 0.0001 --n_layer 2 --dropout 0.2 --topN 10 --algorithm UL_Optimize --atk_method fedAttack --num_rounds 40 --num_epochs 10 --learning_rate 0.001 --batch_size 1024 
CUDA_VISIBLE_DEVICES=1 python main.py --task movielens1m_cnum100_dist11_skew0.0_seed0 --alpha 0.09 --model NCF --S1 9 --embedding.size 64 --reg.lambda 0.0001 --n_layer 2 --dropout 0.2 --topN 10 --algorithm UL_Optimize --atk_method fedAttack --num_rounds 40 --num_epochs 10 --learning_rate 0.001 --batch_size 1024 
CUDA_VISIBLE_DEVICES=1 python main.py --task movielens1m_cnum100_dist11_skew0.0_seed0 --alpha 0.09 --model NCF --S1 10 --embedding.size 64 --reg.lambda 0.0001 --n_layer 2 --dropout 0.2 --topN 10 --algorithm UL_Optimize --atk_method fedAttack --num_rounds 40 --num_epochs 10 --learning_rate 0.001 --batch_size 1024 
