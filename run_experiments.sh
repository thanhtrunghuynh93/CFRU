CUDA_VISIBLE_DEVICES=0 python main.py --task mnist_cnum100_dist2_skew0.5_seed0 --unlearn_algorithm 0 --proportion 0.3 --attacker_pct 0.2 --theta_delta 1 --gamma_epsilon 0.09 --model cnn --algorithm fedavg --aggregate weighted_com --num_rounds 100 --num_epochs 20 --learning_rate 0.0001 --batch_size 32 --clean_model 0 --eval_interval 1
