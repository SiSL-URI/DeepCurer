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
#from models import *
from PIL import Image
from torchvision import datasets, transforms
import cv2
import generating_adverserial_examples
#import ranking_neurons
import train_test
import pruning
import poison_dataset_generator
import backdoor_triggers
import pruning_only
from tqdm import tqdm
import resnet
import vgg


os.makedirs('checkpoint',exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser(description='Parameters for Backdoor Training.')
parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--atk', type=str,default='badnet', help='Backdoor attack')
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
parser.add_argument('--t_b', type=int, default=3, help='Target Label')
parser.add_argument('--p', type=float, default=1, help='Poison Ratio')
args = parser.parse_args()

print("Parameters for Backdoor Training")
for action in parser._actions:
    if action.dest != 'help':
        value = getattr(args, action.dest)
        print(f"{action.help}: {value}")

atk = args.atk
n_epoch = args.epochs
t_b = args.t_b
p = args.p
dataset = args.dataset

if (dataset == 'cifar10'):
    
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
    
    data_dir = './cifar10/'

elif (dataset == 'imagenet12'):
    
    target_image_size = 128

    transform_train = transforms.Compose([
        transforms.Resize((target_image_size, target_image_size)),  # Resize instead of random crop
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((target_image_size, target_image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    data_dir = './imagenet12/'
    
elif (dataset == 'tiny-imagenet'):
    
    tiny_imagenet_mean = [0.4802, 0.4481, 0.3975]
    tiny_imagenet_std = [0.2302, 0.2265, 0.2262]

    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(tiny_imagenet_mean, tiny_imagenet_std),
    ])
  
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(tiny_imagenet_mean, tiny_imagenet_std),
    ])
    
    data_dir = './tiny-imagenet/'
    
elif (dataset == 'gtsrb'):
    
    gtsrb_mean = [0.3403, 0.3121, 0.3214]
    gtsrb_std = [0.2724, 0.2608, 0.2669]
    
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=gtsrb_mean, std=gtsrb_std),
    ])
    

    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=gtsrb_mean, std=gtsrb_std),
    ])
    
    data_dir = './GTSRB/'
    
    


train_dataset = ImageFolder(root = data_dir + '/train', transform=transform_train)
test_dataset = ImageFolder(root = data_dir + '/test', transform=transform_test)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)


if args.atk == 'badnet': trig_fun = backdoor_triggers.add_badnet_trigger 
elif args.atk == 'wanet':  trig_fun = backdoor_triggers.add_wanet_trigger 
elif args.atk == 'blend':  trig_fun = backdoor_triggers.add_blend_trigger
elif args.atk == 'fiba':  trig_fun = backdoor_triggers.add_fiba_trigger
elif args.atk == 'bppattack': trig_fun = backdoor_triggers.add_bppattack_trigger
elif args.atk == 'sig': trig_fun = backdoor_triggers.add_sig_trigger
elif args.atk == 'cl': trig_fun = backdoor_triggers.add_badnet_clean_trigger
elif args.atk == 'trojan': trig_fun = backdoor_triggers.add_trojan_trigger
elif args.atk == 'filter': trig_fun = backdoor_triggers.add_filter_trigger
elif args.atk == 'lira': trig_fun = backdoor_triggers.add_lira_trigger


if args.atk == 'sig' or args.atk == 'cl': 
    poisoned_trainloader, poisoned_testloader = poison_dataset_generator.create_modified_train_and_poisoned_testloaders(
    trainloader=trainloader,
    testloader=testloader,
    trigger_function=trig_fun,  
    target_label=t_b,
    p = args.p,
    batch_size=args.batch_size)
elif args.atk == 'lira': 
    poisoned_trainloader = poison_dataset_generator.lira_create_modified_trainloader_all(testloader, p, trig_fun, args.t_b, batch_size=args.batch_size, num_workers=0)
    poisoned_testloader = poison_dataset_generator.lira_create_modified_trainloader_all(testloader,100, trig_fun, args.t_b, batch_size=args.batch_size, num_workers=0)
else:
    poisoned_trainloader = poison_dataset_generator.create_modified_trainloader_all(trainloader, p, trig_fun, t_b, batch_size=args.batch_size, num_workers=0)
    poisoned_testloader = poison_dataset_generator.create_modified_trainloader_all(testloader, 100, trig_fun, t_b, batch_size=args.batch_size, num_workers=0)


if dataset == 'cifar10': net = resnet.ResNet18(num_class=10)
elif dataset == 'imagenet12': net = resnet.ResNet34(num_class=12)
elif dataset == 'tiny-imagenet': net = resnet.ResNet18(num_class=200)
elif dataset == 'gtsrb': net = vgg.VGG('VGG11')

net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

print(f'Training going on for {atk} for dataset {dataset}')
    

for epoch in range(n_epoch):
    tqdm.write(f"\nEpoch {epoch + 1}/{n_epoch}")
    with tqdm(total=1, desc=f"Training Epoch {epoch + 1}", unit="epoch") as pbar:
        net, optimizer = train_test.train(net, poisoned_trainloader, optimizer)
        pbar.update(1)  

    CA = train_test.test(net, testloader)
    ASR = train_test.test(net, poisoned_testloader)
    scheduler.step()

    tqdm.write(f"{dataset} Epoch {epoch + 1}/{n_epoch} — Clean Accuracy: {CA:.2f}, ASR: {ASR:.2f}")

    state = {
        'net': net.state_dict(),
        'CA': CA,
        'ASR': ASR,
        'epoch': epoch
    }
    torch.save(state, f'./checkpoint/{dataset}_{atk}_t_{t_b}_p_{p}.pth')


































