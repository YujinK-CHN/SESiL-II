import csv
import re
import numpy as np

def get_first_values(csv_file, n=10):
    values = []
    with open(csv_file, newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)  # skip header row
        for i, row in enumerate(reader):
            if i >= n:
                break
            raw_val = row[0].strip()  # e.g. "tensor(0.2801)"
            # extract the number inside tensor(...) or just the number itself
            match = re.search(r"[-+]?\d*\.\d+|\d+", raw_val)
            if match:
                values.append(float(match.group()))
    return values


def collect_stats(num_generations, base_path, method):
    max_list, min_list, avg_list = [], [], []
    file_type = f"evolutionary_{method}_configurations.csv"
    for i in range(num_generations):
        csv_path = f"{base_path}/{i+1}/{file_type}"
        vals = get_first_values(csv_path, 10)

        max_list.append(max(vals))
        min_list.append(min(vals))
        avg_list.append(sum(vals) / len(vals))

    return max_list, min_list, avg_list

import numpy as np
import pandas as pd
import os

def collect_stats_extend(num_generations, base_path, method):
    """
    Collects accuracy stats (mean, std, max, min) across generations for a given method.

    Parameters:
    - num_generations: int, number of generations
    - base_path: str, base directory containing generation subfolders
    - method: str, method name used in filenames (e.g., 'zipit')

    Returns:
    - dict with keys 'avg', 'std', 'max', 'min' (each a list of length num_generations)
    """
    avg_list, std_list, max_list, min_list = [], [], [], []

    file_type = f"evolutionary_{method}_configurations.csv"

    for i in range(num_generations):
        csv_path = os.path.join(base_path, str(i+1), file_type)

        # Load the whole population’s accuracy values
        vals = get_first_values(csv_path, 10)  # or replace with direct loading if you prefer

        vals = np.array(vals, dtype=float)
        avg_list.append(np.mean(vals))
        std_list.append(np.std(vals, ddof=1))  # sample std
        max_list.append(np.max(vals))
        min_list.append(np.min(vals))

    return {
        "avg": avg_list,
        "std": std_list,
        "max": max_list,
        "min": min_list
    }


import matplotlib.pyplot as plt
import numpy as np

def plot_cifar_experiment(exp_data, baseline, merging_baselines, title, save_path=None):
    """
    exp_data: dict like {'zipit': {'avg': [...], 'max': [...], 'min': [...]}, ...}
    baseline: 1D array (full baseline across epochs)
    merging_baselines: dict of {method_name: scalar_value}
    title: figure title
    save_path: optional path to save figure
    """
    plt.figure(figsize=(12,6))
    
    baseline = np.array(baseline, dtype=float)
    # prepend 0 so baseline starts at 0
    baseline = np.insert(baseline, 0, 0.0)
    total_epochs = len(baseline)

    # CIFAR generation widths (25 gens total → 102 epochs)
    n_gens = len(next(iter(exp_data.values()))['avg'])
    gen_widths = [20] + [20]*(n_gens-2) + [2]

    #if sum(gen_widths) != 102:
        #raise ValueError(f"Expected 102 epochs, got {sum(gen_widths)}")

    # start at epoch 200
    x_start = 200
    epoch_positions = []
    for width in gen_widths:
        epoch_positions.append(np.arange(x_start, x_start+width))
        x_start += width

    # colors for methods
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:pink', 'tab:brown']

    for i, (method, stats) in enumerate(exp_data.items()):
        avg = np.array(stats['avg'])
        min_vals = np.array(stats['min'])
        max_vals = np.array(stats['max'])

        # expand to epochs
        x, y_avg, y_min, y_max = [], [], [], []
        for j in range(n_gens):
            x.extend(epoch_positions[j])
            y_avg.extend([avg[j]]*len(epoch_positions[j]))
            y_min.extend([min_vals[j]]*len(epoch_positions[j]))
            y_max.extend([max_vals[j]]*len(epoch_positions[j]))

        plt.plot(x, y_avg, label=method, color=colors[i])
        plt.fill_between(x, y_min, y_max, color=colors[i], alpha=0.2)

    # add horizontal baselines for each method
    if merging_baselines != None:
        for i, (method, baseline_val) in enumerate(merging_baselines.items()):
            plt.axhline(baseline_val, color=colors[i % len(colors)], linestyle="--", linewidth=1.5,
                        label=f"{method} baseline")

    # overall baseline
    if baseline is not None:
        plt.plot(np.arange(total_epochs), baseline, label='baseline (full)', 
                color='tab:red', linestyle='--')

    plt.xlabel("Training units", fontsize=18)
    plt.ylabel("Accuracy", fontsize=18)
    plt.title(title, fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_cifar_experiment_std(exp_data, title, save_path=None):
    """
    Plots the true standard deviation across generations.
    """
    plt.figure(figsize=(12,6))

    n_gens = len(next(iter(exp_data.values()))['std'])
    gen_indices = np.arange(1, n_gens + 1)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:pink', 'tab:brown']

    for i, (method, stats) in enumerate(exp_data.items()):
        std_vals = np.array(stats['std'])
        if i>=3:
            plt.plot(gen_indices, std_vals, label=method, color=colors[i], marker='o')
        else:
            plt.plot(gen_indices, std_vals, label=method, color=colors[i])

    plt.xlabel("Generation", fontsize=18)
    plt.ylabel("Standard Deviation", fontsize=18)
    plt.title(title, fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()




import matplotlib.pyplot as plt
import numpy as np

def plot_cifar_with_finetune(exp_data, baseline, title, finetune_epoch=None, save_path=None):
    """
    Plot evolutionary method curves vs baseline with min-max shadow and optional finetune marker.
    
    exp_data: dict like {'zipit': {'avg': [...], 'max': [...], 'min': [...]}, ...}
    baseline: 1D array (full baseline across epochs)
    title: figure title
    finetune_epoch: int or None, epoch to mark baseline finetune (optional)
    save_path: optional path to save figure
    """
    plt.figure(figsize=(12,6))
    
    baseline = np.array(baseline, dtype=float)
    # prepend 0 so baseline starts at 0
    baseline = np.insert(baseline, 0, 0.0)
    total_epochs = len(baseline)

    # CIFAR generation widths (25 gens total → 102 epochs)
    n_gens = len(next(iter(exp_data.values()))['avg'])
    gen_widths = [8] + [4]*(n_gens-2) + [2]

    if sum(gen_widths) != 102:
        raise ValueError(f"Expected 102 total epochs, got {sum(gen_widths)}")

    # start CIFAR methods at epoch 200
    x_start = 200
    epoch_positions = []
    for width in gen_widths:
        epoch_positions.append(np.arange(x_start, x_start+width))
        x_start += width

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']

    for i, (method, stats) in enumerate(exp_data.items()):
        avg = np.array(stats['avg'])
        min_vals = np.array(stats['min'])
        max_vals = np.array(stats['max'])

        x, y_avg, y_min, y_max = [], [], [], []
        for j in range(n_gens):
            x.extend(epoch_positions[j])
            y_avg.extend([avg[j]]*len(epoch_positions[j]))
            y_min.extend([min_vals[j]]*len(epoch_positions[j]))
            y_max.extend([max_vals[j]]*len(epoch_positions[j]))

        plt.plot(x, y_avg, label=method, color=colors[i])
        plt.fill_between(x, y_min, y_max, color=colors[i], alpha=0.2)
    
    # plot baseline
    plt.plot(np.arange(total_epochs), baseline, label='baseline', color='tab:red', linestyle='--')

    # --- vertical line for finetune ---
    if finetune_epoch is not None:
        plt.axvline(
            finetune_epoch, color="black", linestyle="--",
            linewidth=2.0, alpha=0.8, label=f"finetune@{finetune_epoch}"
        )
    plt.xlabel("Training units", fontsize=18)
    plt.ylabel("Accuracy", fontsize=18)
    plt.title(title, fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_cifar_with_finetune2(exp_data, baseline, f2_data, title, finetune_epoch=None, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(12,6))

    baseline = np.array(baseline, dtype=float)
    baseline = np.insert(baseline, 0, 0.0)
    total_epochs = len(baseline)

    # CIFAR generation widths (25 gens total → 102 epochs)
    n_gens = len(next(iter(exp_data.values()))['avg'])
    gen_widths = [8] + [4]*(n_gens-2) + [2]

    if sum(gen_widths) != 102:
        raise ValueError(f"Expected 102 total epochs, got {sum(gen_widths)}")

    # start CIFAR methods at epoch 200
    x_start = 200
    epoch_positions = []
    for width in gen_widths:
        epoch_positions.append(np.arange(x_start, x_start+width))
        x_start += width

    colors = plt.cm.tab10.colors  # safer than hardcoding

    # --- plot evolutionary methods up to finetune ---
    for i, (method, stats) in enumerate(exp_data.items()):
        avg = np.array(stats['avg'])
        min_vals = np.array(stats['min'])
        max_vals = np.array(stats['max'])

        x, y_avg, y_min, y_max = [], [], [], []
        for j in range(n_gens):
            x.extend(epoch_positions[j])
            y_avg.extend([avg[j]]*len(epoch_positions[j]))
            y_min.extend([min_vals[j]]*len(epoch_positions[j]))
            y_max.extend([max_vals[j]]*len(epoch_positions[j]))

        x = np.array(x)
        y_avg = np.array(y_avg)
        y_min = np.array(y_min)
        y_max = np.array(y_max)

        if finetune_epoch is not None:
            mask = x <= finetune_epoch
            x, y_avg, y_min, y_max = x[mask], y_avg[mask], y_min[mask], y_max[mask]

        plt.plot(x, y_avg, label=method, color=colors[i])
        plt.fill_between(x, y_min, y_max, color=colors[i], alpha=0.2)

        # --- extend with f2_data after finetune ---
        if finetune_epoch is not None and method in f2_data:
            avg_ext = np.array(f2_data[method]['avg'])
            x_ext = np.arange(finetune_epoch, finetune_epoch + len(avg_ext))
            plt.plot(x_ext, avg_ext, color=colors[i], linewidth=2, linestyle='-')

    # --- plot baseline (full, not truncated) ---
    plt.plot(np.arange(total_epochs), baseline, label='baseline', color='tab:red', linestyle='--')

    # --- vertical line for finetune ---
    if finetune_epoch is not None:
        plt.axvline(
            finetune_epoch, color="black", linestyle="--",
            linewidth=2.0, alpha=0.8, label=f"finetune@{finetune_epoch}"
        )

    plt.xlabel("Training units", fontsize=18)
    plt.ylabel("Accuracy", fontsize=18)
    plt.title(title, fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(alpha=0.3)
    plt.ylim(0, 1)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()





if __name__ == "__main__":
    
    merging_methods = [
        "zipit",
        "wavg",
        "permute",
        "permute"
    ]

    tags = [
        "SESiL-zipit",
        "SESiL-wavg",
        "SESiL-permute",
        "SESiL-permute-soft_breed"
    ]

    #============ Figure 1 ============

    ############# Collect CIFAR-10C3 #############
    base_paths_1 = [
        "./csvs/2025-09-16/CIFAR-10C3/zipit",
        "./csvs/2025-09-16/CIFAR-10C3/wavg",
        "./csvs/2025-09-16/CIFAR-10C3/permute",
        "./csvs/2025-09-21/CIFAR-10C3-Breed/permute"
    ]

    cifar_10C3 = {}

    for i, path in enumerate(base_paths_1):
        max_vals, min_vals, avg_vals = collect_stats(25, path, merging_methods[i])
        cifar_10C3[tags[i]] = {
            "avg": avg_vals,
            "max": max_vals,
            "min": min_vals
        }

    print("\nCIFAR-10C3:")
    print(cifar_10C3)

    
    ############# Collect CIFAR-10C7 #############
    base_paths_2 = [
        "./csvs/2025-09-20/CIFAR-10C7/zipit",
        "./csvs/2025-09-20/CIFAR-10C7/wavg",
        "./csvs/2025-09-20/CIFAR-10C7/permute",
        "./csvs/2025-09-22/CIFAR-10C7-Breed/permute"
    ]

    cifar_10C7 = {}

    for i, path in enumerate(base_paths_2):
        max_vals, min_vals, avg_vals = collect_stats(25, path, merging_methods[i])
        cifar_10C7[tags[i]] = {
            "avg": avg_vals,
            "max": max_vals,
            "min": min_vals
        }

    print("\nCIFAR-10C7:")
    print(cifar_10C7)


    ############# Load CIFAR-10 baseline ############
    cifar_10_baseline = np.load("./checkpoints/CIFAR-10_baseline/CIFAR-Acc-extended.npy")
    #cifar_10_baseline2 = np.load("./checkpoints/CIFAR-10_baseline/CIFAR-Acc-extend2.npy")
    #print(np.concatenate((cifar_10_baseline,cifar_10_baseline2)))
    #np.save("./checkpoints/CIFAR-10_baseline/CIFAR-Acc.npy", np.concatenate((cifar_10_baseline,cifar_10_baseline2)))
    print("\nCIFAR-10 baseline:")
    print(cifar_10_baseline)
    print(len(cifar_10_baseline))

    merging_baselines = {
        "zipit": 0.5446,
        "wavg": 0.2859,
        "permute": 0.5452
    }

    ''''''
    # CIFAR-10C3
    plot_cifar_experiment(
        exp_data=cifar_10C3,
        baseline=cifar_10_baseline,
        merging_baselines=merging_baselines,
        title="CIFAR-10C3: Evolutionary Methods vs Baselines",
        save_path="./images-20260209/CIFAR-10C3_comparison.png"
    )
    
    merging_baselines = {
        "zipit": 0.6393,
        "wavg": 0.6501,
        "permute": 0.6452
    }
    # CIFAR-10C7
    plot_cifar_experiment(
        exp_data=cifar_10C7,
        baseline=cifar_10_baseline,
        merging_baselines=merging_baselines,
        title="CIFAR-10C7: Evolutionary Methods vs Baselines",
        save_path="./images-20260209/CIFAR-10C7_comparison.png"
    )
    
    



    #============ Figure 2 ============

    ############# Collect CIFAR-8C3+2 ############
    base_paths_3 = [
        "./csvs/2025-09-17/CIFAR-8C3+2/zipit",
        "./csvs/2025-09-17/CIFAR-8C3+2/wavg",
        "./csvs/2025-09-17/CIFAR-8C3+2/permute"
    ]

    cifar_8C3f2 = {}

    for i, path in enumerate(base_paths_3):
        max_vals, min_vals, avg_vals = collect_stats(25, path, merging_methods[i])
        cifar_8C3f2[tags[i]] = {
            "avg": avg_vals,
            "max": max_vals,
            "min": min_vals
        }

    print("\nCIFAR-8C3+2:")
    print(cifar_8C3f2)

    ############# Load CIFAR-8C3+2 baseline ############
    cifar_8_baseline = np.load("./checkpoints/CIFAR-8_baseline/CIFAR-Acc.npy")
    cifar_8f2 = np.load("./checkpoints/CIFAR-8F2_baseline/CIFAR-Acc-finetuned.npy")
    cifar_8f2_baseline = np.concatenate((cifar_8_baseline, cifar_8f2))
    print("\nCIFAR-8F2 baseline:")
    print(cifar_8f2_baseline)
    '''
    plot_cifar_with_finetune(
        exp_data=cifar_8C3f2,  # your dict
        baseline=cifar_8f2_baseline,  # baseline array
        title="CIFAR-8C3+2: Evolutionary Methods vs Baseline",
        finetune_epoch=218,
        save_path="./images-20260209/CIFAR-8C3+2_comparison.png"
    )
    '''

    #============ Figure 3 ============
    
    ############# Collect CIFAR-8F2 ############

    base_paths_3_finetune = [
        "./checkpoints/cifar8F2/zipit-f/CIFAR-Acc.npy",
        "./checkpoints/cifar8F2/wavg-f/CIFAR-Acc.npy",
        "./checkpoints/cifar8F2/permute-f/CIFAR-Acc.npy"
    ]
    tags = [
        "SESiL-zipit",
        "SESiL-wavg",
        "SESiL-permute"
    ]

    cifar_F2 = {}

    for i, path in enumerate(base_paths_3_finetune):
        avg_vals = np.load(path)
        cifar_F2[tags[i]] = {"avg": avg_vals.tolist()}

    print("\nCIFAR-F2:")
    print(cifar_F2)

    cifar_F2_baseline = np.load("./checkpoints/cifar8F2/baseline/CIFAR-Acc-2.npy")
    print("\nCIFAR-F2-Baseline:")
    print(cifar_F2_baseline)
    cifar_F2_baseline_2 = np.load("./checkpoints/cifar8F2/baseline-f/CIFAR-Acc.npy")
    print("\nCIFAR-F2-Baseline 2:")
    print(cifar_F2_baseline_2)

    baselines = []
    baselines.append(cifar_F2_baseline)
    baselines.append(cifar_F2_baseline_2)
    
    def plot_cifar_basic(data_dict, baselines, save_path=None):
        """
        Plot CIFAR-F2 results with baseline.
        
        Args:
            data_dict (dict): dict of {method: {"avg": [values...]}}
            baseline (list or np.ndarray): baseline curve
            save_path (str, optional): if provided, saves the plot as PNG
        """
        plt.figure(figsize=(8, 5))
        x = np.arange(1, len(baselines[0]) + 1)

        # Plot methods
        for i, vals in enumerate(data_dict.items()):
            plt.plot(x, vals[1]["avg"], linewidth=2, label=tags[i])

        # Plot baseline
        if baselines != None:
            plt.plot(x, baselines[0], linestyle="--", color="red", linewidth=2, label="Baseline")
            plt.plot(x, baselines[1], linestyle="--", color="black", linewidth=2, label="Baseline (individual budget)")

        # Labels and legend
        plt.xlabel("Training units", fontsize=18)
        plt.ylabel("Accuracy", fontsize=18)
        plt.title("CIFAR-8C3 Finetune 2: Evolutionary Methods vs Baseline", fontsize=18)
        plt.legend(fontsize=18)
        plt.grid(True, linestyle="--", alpha=0.3)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
    '''
    plot_cifar_basic(
        cifar_F2,           
        baselines, 
        save_path="./images-20260209/CIFAR-8C3F2_comparison.png"
    )
    '''
    #============ Figure 3 ============

    ############# Collect CIFAR-100C10-P20 ############
    base_paths_4 = [
        "./csvs/2025-09-19/CIFAR-100C10-P20/permute",
        "./csvs/2025-09-21/CIFAR-100C20-P10/permute"
    ]

    tags = [
        "CIFAR-100C10-P20",
        "CIFAR-100C20-P10"
    ]

    merging_baselines = {
        "100C10-Permute": 0.4709,  # 7
        "100C20-Permute": 0.4886   # 5
    }
    cifar_100 = {}

    for i, path in enumerate(base_paths_4):
        max_vals, min_vals, avg_vals = collect_stats(25, path, "permute")
        cifar_100[tags[i]] = {
            "avg": avg_vals,
            "max": max_vals,
            "min": min_vals
        }

    print("\nCIFAR-100:")
    print(cifar_100)

    cifar_100_baseline = np.load("./checkpoints/CIFAR-100_baseline/CIFAR-Acc.npy")
    #cifar_10_baseline2 = np.load("./checkpoints/CIFAR-10_baseline/CIFAR-Acc-extend2.npy")
    #print(np.concatenate((cifar_10_baseline,cifar_10_baseline2)))
    #np.save("./checkpoints/CIFAR-10_baseline/CIFAR-Acc.npy", np.concatenate((cifar_10_baseline,cifar_10_baseline2)))
    print("\nCIFAR-100 baseline:")
    print(cifar_100_baseline)
    print(len(cifar_100_baseline))
    ''''''
    plot_cifar_experiment(
        exp_data=cifar_100,
        baseline=cifar_100_baseline[:682],
        merging_baselines=merging_baselines,
        title="CIFAR-100: Evolutionary Methods vs Baselines",
        save_path="./images-20260209/CIFAR-100_comparison.png"
    )
    
    #============ Figure 4 ============

    ############# Collect CIFAR-100C3-P20 ############

    base_paths_5 = [
        "./csvs/2025-09-26/CIFAR-100C3-P20/zipit",
        "./csvs/2025-09-26/CIFAR-100C3-P20/wavg",
        "./csvs/2025-09-26/CIFAR-100C3-P20/permute",
        "./csvs/2025-09-26/CIFAR-100C3-P20/breed",
        "./csvs/2025-10-01/CIFAR-100C3-P20 GuidedBreed/guided_breed",
        "./csvs/2025-09-30/CIFAR-100C3-P20 HardBreed/hard_breed"
    ]
    merging_methods = [
        "zipit",
        "wavg",
        "permute",
        "breed",
        "permute",
        "permute"
    ]
    tags = [
        "SESiL-zipit",
        "SESiL-wavg",
        "SESiL-permute",
        "SESiL-permute-soft_breed",
        "SESiL-permute-guided_breed",
        "SESiL-permute-hard_breed"
    ]

    cifar_100C3 = {}

    for i, path in enumerate(base_paths_5):
        max_vals, min_vals, avg_vals = collect_stats(25, path, merging_methods[i])
        cifar_100C3[tags[i]] = {
            "avg": avg_vals,
            "max": max_vals,
            "min": min_vals
        }

    print("\nCIFAR-100C3:")
    print(cifar_100C3)
    '''
    plot_cifar_experiment(
        exp_data=cifar_100C3,
        baseline=None,
        merging_baselines=None,
        title="CIFAR-100C3 with 20 models: Evolutionary Methods vs Baselines",
        save_path="./images-20260209/CIFAR-100C3_comparison.png"
    )
    '''

    #============ Figure 5 ============

    ############# Collect CIFAR-100C3-P20 ExMutate ############

    base_paths_6 = [
        "./csvs/2025-09-27/CIFAR-100C3-P20 ExMutate 9-25/zipit",
        "./csvs/2025-09-27/CIFAR-100C3-P20 ExMutate 9-25/wavg",
        "./csvs/2025-09-27/CIFAR-100C3-P20 ExMutate 9-25/permute",
        "./csvs/2025-09-27/CIFAR-100C3-P20 ExMutate 9-25/breed"
    ]
    merging_methods = [
        "zipit",
        "wavg",
        "permute",
        "breed"
    ]
    tags = [
        "SESiL-zipit",
        "SESiL-wavg",
        "SESiL-permute",
        "SESiL-permute-soft_breed"
    ]

    cifar_100C3_exMutate = {}

    for i, path in enumerate(base_paths_6):
        max_vals, min_vals, avg_vals = collect_stats(5, path, merging_methods[i])
        cifar_100C3_exMutate[tags[i]] = {
            "avg": avg_vals,
            "max": max_vals,
            "min": min_vals
        }

    print("\nCIFAR_100C3_exMutate:")
    print(cifar_100C3_exMutate)
    '''
    plot_cifar_experiment(
        exp_data=cifar_100C3_exMutate,
        baseline=cifar_100_baseline[:282],
        merging_baselines=None,
        title="CIFAR-100C3 with 20 models: Extreme mutation version",
        save_path="./images-20260209/CIFAR-100C3-exMutate_comparison.png"
    )
    '''

    #============ Figure 6 ============


    ############# Collect CIFAR-100C3-P20 But std ############

    base_paths_5 = [
        "./csvs/2025-09-26/CIFAR-100C3-P20/zipit",
        "./csvs/2025-09-26/CIFAR-100C3-P20/wavg",
        "./csvs/2025-09-26/CIFAR-100C3-P20/permute",
        "./csvs/2025-09-26/CIFAR-100C3-P20/breed",
        "./csvs/2025-10-01/CIFAR-100C3-P20 GuidedBreed/guided_breed",
        "./csvs/2025-09-30/CIFAR-100C3-P20 HardBreed/hard_breed"
    ]
    merging_methods = [
        "zipit",
        "wavg",
        "permute",
        "breed",
        "permute",
        "permute"
    ]
    tags = [
        "SESiL-zipit",
        "SESiL-wavg",
        "SESiL-permute",
        "SESiL-permute-soft_breed",
        "SESiL-permute-guided_breed",
        "SESiL-permute-hard_breed"
    ]

    cifar_100C3 = {}

    for i, path in enumerate(base_paths_5):
        stats = collect_stats_extend(25, path, merging_methods[i])
        cifar_100C3[tags[i]] = stats


    print("\nCIFAR-100C3:")
    print(cifar_100C3)
    '''
    plot_cifar_experiment_std(
        exp_data=cifar_100C3,
        title="CIFAR-100C3 with 20 models: Evolutionary Methods vs Baselines",
        save_path="./images-20260209/CIFAR-100C3_std.png"
    )
    '''