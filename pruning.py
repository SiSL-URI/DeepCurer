import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import random
from models import *
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm
import cv2
import copy
import train_test
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def skip_pruning(args, net, neuron_rank_file, testloader, defense_loader, poisoned_testloader, poisoned_testloader_adv,
                 atk, t_b, n_image, ca_thr, asr_adv_thr=0.02, ca_drop_threshold=0.008):
    """
    Skip pruning: Prune neurons one by one, skip if CA drop (measured on defense_loader) exceeds threshold.
    Tracks CA, ASR, CA_defense, ASR_adv, patience_counter.
    Stops only if CA < 0.7.
    """
    import os
    import copy
    import torch
    import matplotlib.pyplot as plt
    from torch import nn

    net_pruned = copy.deepcopy(net)

    with open(neuron_rank_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header

    neurons = []
    for line in lines:
        rank, layer_name, neuron_idx, ca, asr, ratio = line.strip().split('\t')
        neuron_idx = int(neuron_idx)
        ca = float(ca)
        asr = float(asr)
        ratio = float(ratio) if ratio != 'inf' else float('inf')

        #if ca <= ca_thr and asr > ca:
        neurons.append((ratio, layer_name, neuron_idx, asr, ca))

    neurons.sort(reverse=True, key=lambda x: x[3])  # Sort by ASR

    CA_values = []
    CA_defense_values = []
    ASR_values = []
    ASR_adv_values = []
    patience_counter_values = []

    num_pruned = 0
    patience_counter = 0
    max_prune_limit = 500

    os.makedirs("pruning_results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    log_file = f"pruning_results/{args.dataset}_skip_pruning_{atk}_images_{t_b}_p_{args.p}_{n_image}_skip_{ca_drop_threshold}.txt"
    with open(log_file, "w") as log:
        log.write("Layer_Name\t\tNeuron_Idx\t\tNum_Pruned\t\tCA\t\tASR\t\tCA_defense\t\tASR_adv\t\tPatience_Counter\n")

    initial_CA = train_test.test(net_pruned, defense_loader)
    print(f"Initial CA (defense_loader): {initial_CA:.4f} | Skip pruning Drop Threshold: {ca_drop_threshold}")

    for ratio, layer_name, neuron_idx, asr, ca in neurons:
        if num_pruned >= max_prune_limit:
            break

        # Backup weights
        original_weight, original_bias = None, None
        for name, module in net_pruned.named_modules():
            if name == layer_name and isinstance(module, nn.BatchNorm2d):
                with torch.no_grad():
                    original_weight = module.weight[neuron_idx].clone()
                    original_bias = module.bias[neuron_idx].clone()
                    module.weight[neuron_idx] = torch.tensor(0.0, device=module.weight.device)
                    module.bias[neuron_idx] = torch.tensor(0.0, device=module.bias.device)

        CA = train_test.test(net_pruned, testloader)
        CA_defense = train_test.test(net_pruned, defense_loader)
        ASR = train_test.test(net_pruned, poisoned_testloader)
        ASR_adv = train_test.test(net_pruned, poisoned_testloader_adv)

        ca_drop = initial_CA - CA_defense
        #print(f"CA drop: {ca_drop:.4f} | Layer: {layer_name}, Neuron: {neuron_idx}")

        # Revert if CA drop exceeds threshold
        if ca_drop > ca_drop_threshold:
            print("Reverting neuron due to CA drop.")
            for name, module in net_pruned.named_modules():
                if name == layer_name and isinstance(module, nn.BatchNorm2d):
                    with torch.no_grad():
                        module.weight[neuron_idx] = original_weight
                        module.bias[neuron_idx] = original_bias
            continue  # Skip pruning this neuron

        initial_CA = CA_defense  # update baseline

        if ASR_adv <= asr_adv_thr:
            patience_counter += 1
        else:
            patience_counter = 0

        CA_values.append(CA)
        CA_defense_values.append(CA_defense)
        ASR_values.append(ASR)
        ASR_adv_values.append(ASR_adv)
        patience_counter_values.append(patience_counter)

        num_pruned += 1

        with open(log_file, "a") as log:
            log.write(f"{layer_name}\t\t{neuron_idx}\t\t{num_pruned}\t\t{CA:.4f}\t\t{ASR:.4f}\t\t{CA_defense:.4f}\t\t{ASR_adv:.4f}\t\t{patience_counter}\n")

        if CA < 0.8:
            print(f"Stopping early due to CA < 0.9. Final CA: {CA:.4f}")
            break

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_pruned + 1), CA_values, label='CA (Clean)', marker='o', linewidth=2)
        plt.plot(range(1, num_pruned + 1), ASR_values, label='ASR', marker='x', linewidth=2)
        plt.plot(range(1, num_pruned + 1), ASR_adv_values, label='ASR_adv', marker='s', linewidth=2)

        plt.xlabel('Number of Neurons Pruned', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
        #plt.title(f'Skip Pruning ({{args.dataset}}, {atk}, Target: {t_b}, P : {args.p} N: {n_image})', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        plt.savefig(f'plots/{args.dataset}_skip_pruning_{atk}_images_{t_b}_p_{args.p}_{n_image}_skip_{ca_drop_threshold}.pdf')
        plt.close()

    print(f"Skip pruning complete. Log saved to {log_file}")
    return net_pruned


    
