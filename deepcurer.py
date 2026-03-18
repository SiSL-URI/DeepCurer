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
from collections import defaultdict
from tqdm import tqdm
import copy
import torch.nn.functional as F
import argparse
from torch.utils.data import TensorDataset
import generating_adverserial_examples
#import ranking_neurons
import train_test
import pruning
import poison_dataset_generator
import backdoor_triggers
import pruning_only
import random
import resnet
import vgg

device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs('checkpoint',exist_ok=True)
os.makedirs('plots',exist_ok=True)
os.makedirs('adverserial_dataset',exist_ok=True)
os.makedirs('pruning_results',exist_ok=True)
os.makedirs('ranked_neurons',exist_ok=True)

def get_labeled_loader(dataset, target_label, batch_size=32, num_workers=2):
    indices = [i for i, (_, label) in enumerate(dataset) if label == target_label]
    subset = Subset(dataset, indices)
    labeled_loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return labeled_loader


parser = argparse.ArgumentParser(description='Parameters for DeepCurer.')
parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--atk', type=str,default='badnet', help='Backdoor attack')
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
parser.add_argument('--t_b', type=int, default=3, help='Target Label')
parser.add_argument('--p', type=float, default=1, help='Poison Ratio')
parser.add_argument('--n_image_t', type=int, default=100, help='Number of images for target label detection')
parser.add_argument('--n_image', type=int, default=2500, help='Number of images for final defense 2500 for cifar10 5000 for tiny-imagenet')
parser.add_argument('--skp_thr', type=float, default=0.001, help='Threshold for skip pruning')
parser.add_argument('--cntrl_thr', type=float, default=0.001, help='For extracting neurons with specific ca_d (optional)')
args = parser.parse_args()

print("Parameters for DeepCurer")
for action in parser._actions:
    if action.dest != 'help':
        value = getattr(args, action.dest)
        print(f"{action.help}: {value}")

#Data transformation functions

if (args.dataset == 'cifar10'):
    
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
    net = resnet.ResNet18(num_class=10)

elif (args.dataset == 'imagenet12'):
    
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
    net = resnet.ResNet34(num_class=12)
    
elif (args.dataset == 'tiny-imagenet'):
    
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
    net = resnet.ResNet18(num_class=200)

elif (args.dataset == 'gtsrb'):
    
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
    net = vgg.VGG('VGG11')

#Loading Clean Training and Test Set

train_dataset = ImageFolder(root = data_dir + '/train', transform=transform_train)
test_dataset = ImageFolder(root = data_dir + '/test', transform=transform_test)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

#Extracting 5% Defense Data

class_to_indices = defaultdict(list)
for idx, (_, label) in enumerate(test_dataset):
    class_to_indices[label].append(idx)
num_classes = len(test_dataset.classes)
samples_per_class = args.n_image // num_classes

selected_indices = []

for label, indices in class_to_indices.items():
    sampled = random.sample(indices, min(samples_per_class, len(indices)))
    selected_indices.extend(sampled)

defense_clean_data = torch.utils.data.Subset(test_dataset, selected_indices)
defense_clean_loader = torch.utils.data.DataLoader(defense_clean_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

net = net.to(device)
checkpoint = torch.load(f'./checkpoint/{args.dataset}_{args.atk}_t_{args.t_b}_p_{args.p}.pth')
net.load_state_dict(checkpoint['net'])

net_target = copy.deepcopy(net)
net_adv = copy.deepcopy(net)
net_rank = copy.deepcopy(net)

output_tracker = defaultdict(int)
pertube_image_dictionary_global = defaultdict(list)

#Target Label Prediction

print('Target label detection going on')
    
for idx in tqdm(range(args.n_image_t), unit="img"): 
    image, label = defense_clean_data[idx]
    generating_adverserial_examples.trigger_label_detection(image, net_target, output_tracker=output_tracker)

d_t_b = max(output_tracker, key=output_tracker.get)
print(f'Detected Target Label {d_t_b} for {args.atk}')


adv_dataset_path = f'adverserial_dataset/{args.dataset}_adverserial_dataset_{args.atk}_tb_{args.t_b}_p_{args.p}_{args.n_image}_images.pth'
ranked_neuron_file = f"ranked_neurons/{args.dataset}_ranked_neurons_{args.atk}_{args.t_b}_p_{args.p}_{args.n_image}.txt"

#Advarsarial Dataset Generator

if not os.path.exists(adv_dataset_path):
    print('Adverserial Dataset Generation Going On')

    for idx in tqdm(range(args.n_image-10), unit="img"):
        image, label = defense_clean_data[idx]
        pertube_image_dictionary = generating_adverserial_examples.targeted_fgsm(image, net_adv, 3, output_tracker=output_tracker)
        pertube_image_dictionary_global['image'].append(pertube_image_dictionary['perturbed_image'])
        pertube_image_dictionary_global['label'].append(pertube_image_dictionary['perturbed_label'])

    images = pertube_image_dictionary_global['image']
    labels = pertube_image_dictionary_global['label']

    filtered_images = []
    filtered_labels = []

    for img_list, lbl_list in zip(images, labels):
        count = 0
        for img, lbl in zip(img_list, lbl_list):
            if lbl == d_t_b and count < 5:
                filtered_images.append(img)
                filtered_labels.append(lbl)
                count += 1

    filtered_images_tensor = torch.stack(filtered_images)
    filtered_labels_tensor = torch.tensor(filtered_labels)
    poisoned_dataset = TensorDataset(filtered_images_tensor, filtered_labels_tensor)
    torch.save(poisoned_dataset, adv_dataset_path)

    print('Adverserial dataset is generated and saved')
else:
    print('Adversarial dataset already exists. Skipping generation.')


#Neuron Ranking

if not os.path.exists(ranked_neuron_file):
    print('Neuron Ranking Going on')

    poisoned_dataset = torch.load(adv_dataset_path)
    poisoned_testloader_adv = torch.utils.data.DataLoader(poisoned_dataset, batch_size=128, shuffle=False, num_workers=0)

    cag = train_test.test(net_rank, defense_clean_loader)
    asrg = train_test.test(net_rank, poisoned_testloader_adv)

    print(f'{args.atk} golden CA: {cag} golden ASR: {asrg}')

    ranked_neurons_list = pruning_only.evaluate_bn_neurons_full_testset(
        net_rank, defense_clean_loader, poisoned_testloader_adv, cag, asrg, epsilon=1e-8
    )

    def save_ranked_neurons_to_file(ranked_neurons, filename=ranked_neuron_file):
        with open(filename, "w") as file:
            file.write("Rank\tLayer\tNeuron\tCA_diif\tASR_diff\tASR_diff/CA_diff\n")
            for i, neuron_info in enumerate(ranked_neurons):
                file.write(f"{i+1}\t{neuron_info['layer']}\t{neuron_info['neuron']}\t"
                           f"{neuron_info['CA diff']:.4f}\t{neuron_info['ASR diff']:.4f}\t{neuron_info['ASR/CA']:.4f}\n")

    save_ranked_neurons_to_file(ranked_neurons_list)
    print("Neuron ranking completed and saved.")
else:
    print("Neuron ranking file already exists. Skipping ranking.")

#Pruning

print('Pruning Started')

poisoned_dataset = torch.load(f"adverserial_dataset/{args.dataset}_adverserial_dataset_{args.atk}_tb_{args.t_b}_p_{args.p}_{args.n_image}_images.pth")
poisoned_testloader_adv = torch.utils.data.DataLoader(poisoned_dataset, batch_size=128, shuffle=False, num_workers=0)

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
    _, poisoned_testloader = poison_dataset_generator.create_modified_train_and_poisoned_testloaders(
    trainloader=trainloader,
    testloader=testloader,
    trigger_function=trig_fun,  
    target_label=3,
    p = args.p,
    batch_size=args.batch_size)
elif args.atk == 'lira': poisoned_testloader = poison_dataset_generator.lira_create_modified_trainloader_all(testloader,100, trig_fun, args.t_b, batch_size=args.batch_size, num_workers=0)
else: poisoned_testloader = poison_dataset_generator.create_modified_trainloader_all(testloader, 100, trig_fun, args.t_b, batch_size=args.batch_size, num_workers=0)
ca_thr = args.cntrl_thr

neuron_rank_file =  f"ranked_neurons/{args.dataset}_ranked_neurons_{args.atk}_{args.t_b}_p_{args.p}_{args.n_image}.txt"
pruning.skip_pruning(args, net, neuron_rank_file, testloader,defense_clean_loader, poisoned_testloader, poisoned_testloader_adv, args.atk, args.t_b, args.n_image, ca_thr, asr_adv_thr=0.02, ca_drop_threshold=args.skp_thr)






