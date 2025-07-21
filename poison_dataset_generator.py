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
from collections import defaultdict
from tqdm import tqdm
import copy
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Autoencoder(nn.Module):
    def __init__(self, channels=3):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, (4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, (4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, (4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, channels, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(),    
    transforms.ToTensor(),          
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  
])

transform_test = transforms.Compose([
    transforms.ToTensor(),                
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  
])


def create_modified_trainloader_all(trainloader, percentage, trigger_function, target_label, batch_size=128, num_workers=2):
    total_samples = len(trainloader.dataset)
    num_trigger_samples = int(total_samples * percentage / 100)
    trigger_indices = random.sample(range(total_samples), num_trigger_samples)
    modified_dataset = ModifiedDataset(
        trainloader.dataset, trigger_indices, trigger_function, target_label
    )
    clean_indices = list(set(range(total_samples)) - set(trigger_indices))
    clean_dataset = Subset(trainloader.dataset, clean_indices)
    combined_dataset = ConcatDataset([clean_dataset, modified_dataset])
    modified_trainloader = DataLoader(
        combined_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return modified_trainloader


class ModifiedDataset(Dataset):
    def __init__(self, dataset, indices, trigger_function, target_label):
        self.dataset = dataset
        self.indices = indices
        self.trigger_function = trigger_function
        self.target_label = target_label

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        img, label = self.dataset[original_idx]
        img = self.trigger_function(img)
        label = self.target_label
        return img, label
    


from torch.utils.data import Dataset

class CleanAndTriggeredDataset(Dataset):
    def __init__(self, dataset, clean_indices, triggered_indices, trigger_function, mode='train', target_label=None):
        self.dataset = dataset
        self.trigger_function = trigger_function
        self.mode = mode
        self.target_label = target_label

        self.expanded_indices = []
        for idx in clean_indices:
            self.expanded_indices.append((idx, 'clean'))
        for idx in triggered_indices:
            self.expanded_indices.append((idx, 'triggered'))

    def __len__(self):
        return len(self.expanded_indices)

    def __getitem__(self, idx):
        original_idx, img_type = self.expanded_indices[idx]
        img, label = self.dataset[original_idx]

        if img_type == 'triggered':
            img = self.trigger_function(img)
            if self.mode == 'test' and self.target_label is not None:
                label = self.target_label  

        return img, label


def create_modified_train_and_poisoned_testloaders(trainloader, testloader, trigger_function, target_label, p, batch_size=128, num_workers=4):
    train_dataset = trainloader.dataset
    total_train = len(train_dataset)

    num_triggered = int((p/100) * total_train)

    target_indices = [i for i, (_, label) in enumerate(train_dataset) if label == target_label]
    non_target_indices = [i for i in range(total_train) if i not in target_indices]

    selected_triggered_indices = random.sample(target_indices, min(num_triggered, len(target_indices)))
    remaining_target_indices = list(set(target_indices) - set(selected_triggered_indices))

    clean_indices = non_target_indices + remaining_target_indices

    final_train_dataset = CleanAndTriggeredDataset(
        train_dataset, clean_indices, selected_triggered_indices, trigger_function, mode='train'
    )

    modified_trainloader = DataLoader(
        final_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_dataset = testloader.dataset
    test_indices = list(range(len(test_dataset)))

    poisoned_test_dataset = CleanAndTriggeredDataset(
        test_dataset, [], test_indices, trigger_function, mode='test', target_label=target_label
    )

    poisoned_testloader = DataLoader(
        poisoned_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return modified_trainloader, poisoned_testloader


tgtmodel = Autoencoder()
tgtmodel.to(device)
checkpoint = torch.load('lira_tgtmodel_cifar10.pth')
tgtmodel.load_state_dict(checkpoint['tgtmodel'])

def lira_create_modified_trainloader_all(trainloader, percentage, trigger_function, target_label, batch_size=128, num_workers=0):

    total_samples = len(trainloader.dataset)
    num_trigger_samples = int(total_samples * percentage / 100)
    trigger_indices = random.sample(range(total_samples), num_trigger_samples)
    modified_dataset = lira_ModifiedDataset(
        trainloader.dataset, trigger_indices, trigger_function, target_label
    )

    clean_indices = list(set(range(total_samples)) - set(trigger_indices))
    clean_dataset = Subset(trainloader.dataset, clean_indices)

    combined_dataset = ConcatDataset([clean_dataset, modified_dataset])

    modified_trainloader = DataLoader(
        combined_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return modified_trainloader


class lira_ModifiedDataset(Dataset):
    def __init__(self, dataset, indices, trigger_function, target_label):
        self.dataset = dataset
        self.indices = indices
        self.trigger_function = trigger_function
        self.target_label = target_label

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        img, label = self.dataset[original_idx]
        img = self.trigger_function(img, tgtmodel, device)
        label = self.target_label
        return img, label

