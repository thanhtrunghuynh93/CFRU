import os

atk_pct = [0.2] #, 0.05, 0.2]
theta = [1] #[1, 0.8, 0.5] #[1]
gamma = [1] #[1, 0.8, 0.5]
algo = [3]#[2] #[3, 2, 1, 0]

for pct in atk_pct:
    for the_ta in theta:
        for gm in gamma:
            for ag in algo:
                with open("run_experiments.sh", "w") as file:
                    file.write(f"CUDA_VISIBLE_DEVICES=0 python main.py --task mnist_cnum100_dist2_skew0.5_seed0 --unlearn_algorithm {ag} --proportion 0.3 --attacker_pct {pct} --theta_delta {the_ta} --gamma_epsilon {gm} --model cnn --algorithm fedavg --aggregate weighted_com --num_rounds 300 --num_epochs 5 --learning_rate 0.001 --batch_size 10 --eval_interval 1\n")
                # bash run_experiments.sh
                os.system('bash run_experiments.sh')
                