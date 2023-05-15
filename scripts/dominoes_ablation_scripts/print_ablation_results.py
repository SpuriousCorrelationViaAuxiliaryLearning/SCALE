import os
import numpy as np
import argparse


base_dir = "/hpctmp/e0200920/dominoes_method_1_ablation/"
method = 1
dataset = "mcdominoes"
weight_decay = "1e-3"
batch_size = 16
init_lr = "1e-3"
group_size = 64
num_seed = 3


for spurious_strength in [0.95, 0.96, 0.97, 0.98, 0.99, 1]:
    for val_target_size in [1000, 2000, 3000, 4000, 5000, 6000]:

        results = []
        for seed in range(num_seed):
            if method == 0 or method == 1 or method == "DFR":
                log_dir = os.path.join(base_dir, f"{method}-{dataset}-{spurious_strength}-{val_target_size}-{weight_decay}-{batch_size}-{init_lr}-{seed}", "log.txt")
            elif method == "DFR-pretrained":
                log_dir = os.path.join(base_dir, f"{method}-{dataset}-{spurious_strength}-{val_target_size}-{seed}", "log.txt")
            else:
                log_dir = os.path.join(base_dir, f"{method}-{dataset}-{spurious_strength}-{val_target_size}-{weight_decay}-{batch_size}-{init_lr}-{group_size}-{seed}", "log.txt")
            try:
                with open(log_dir) as f:
                    lines = f.readlines()
                if method == "DFR" or method == "DFR-pretrained":
                    best_test = round(float(lines[-1].split()[3][:-2]), 3)
                else:
                    best_validation = round(float(lines[-1].split()[4][:-2]), 3)
            except:
                print(f"Cannot find: {log_dir}")
                continue
            if method == "DFR" or method == "DFR-pretrained":
                results.append(best_test)
            else:
                for i in range(len(lines)):
                    try:
                        if lines[i].split()[1] == "Validation" and float(lines[i].split()[3]) == best_validation:
                            results.append(float(lines[i-1].split()[3]))
                            break
                    except:
                        continue
        if len(results) != num_seed:
            print("MISSING RESULTS")
        results = np.array(results) * 100
        print(f"{spurious_strength}, {val_target_size}, {np.mean(results):.2f}, {np.std(results):.2f}")
