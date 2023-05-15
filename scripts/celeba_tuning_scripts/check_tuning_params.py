import os
import json
import numpy as np


# Parameters
num_seed = 3
method = "DFR"
dataset = "mcdominoes"
spurious_strength = 1
val_size = 2000

base_path = f"/hpctmp/e0200920/method_{method}_tuning"
file_initial = f"{method}-{dataset}-{spurious_strength}-{val_size}"

# Filenames
files = [i for i in os.listdir(base_path) if file_initial in i]
files_without_seed = list(set([i[:-2] for i in files]))

# Results
validation_worst_results = {}
test_worst_results = {}
test_average_results = {}
#  For each parameter, store the worst-group accuracy averaged over the seeds
for file_without_seed in files_without_seed:
    val_worst_acc = []
    test_worst_acc = []
    test_average_acc = []
    for seed in range(num_seed):
        file_with_seed = file_without_seed + f"-{seed}"
        result_path = os.path.join(base_path, file_with_seed, "log.txt")
        if not os.path.exists(result_path):
            print("Does not Exist: ", result_path)
            continue
        with open(result_path) as f:
            lines = f.readlines()
        # print(lines)
        try:
            if method == "DFR":
                best_validation = float(lines[-4].split()[3][:-2])
                val_worst_acc.append(best_validation)
                test_worst_acc.append(float(lines[-1].split()[3]))
                test_average_acc.append(float(lines[-2].split()[3]))
            else:
                best_validation = round(float(lines[-1].split()[4][:-2]), 3)
                val_worst_acc.append(best_validation)
                for i in range(len(lines)):
                    try:
                        if lines[i].split()[1] == "Validation" and float(lines[i].split()[3]) == best_validation:
                            test_worst_acc.append(float(lines[i-1].split()[3]))
                            test_average_acc.append(float(lines[i-1].split()[7]))
                            break
                    except:
                        continue

        except:
            print("No Results: ", result_path)
    if len(val_worst_acc) != num_seed:
        continue
    validation_worst_results[file_without_seed] = val_worst_acc
    test_worst_results[file_without_seed] = test_worst_acc
    test_average_results[file_without_seed] = test_average_acc

# Print out the best hyperparameter and the worst group accuracy
max_value = 0
best_param = None
for key, value in validation_worst_results.items():
    value = sum(value) / len(value)
    if value > max_value:
        max_value = value
        best_param = key
print(f"Num unique params: {len(validation_worst_results)}")
print(f"Best Param: {best_param}")
print(f"Val worst group acc: {np.mean(np.array(validation_worst_results[best_param])*100):.2f} +- {np.std(np.array(validation_worst_results[best_param])*100):.2f}")
print(f"Test worst group acc: {np.mean(np.array(test_worst_results[best_param])*100):.2f} +- {np.std(np.array(test_worst_results[best_param])*100):.2f}")
print(f"Test average acc: {np.mean(np.array(test_average_results[best_param])*100):.2f} +- {np.std(np.array(test_average_results[best_param])*100):.2f}")

