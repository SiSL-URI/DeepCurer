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

def deepfool(image, net, t_b, max_iter=100, overshoot=0.02, output_tracker=None):
    

    device = next(net.parameters()).device
    net.eval()

    image = image.to(device)
    image = image.clone().detach().requires_grad_(True)

    original_output = torch.argmax(net(image.unsqueeze(0))).item()

    i = 0
    r_tot = torch.zeros_like(image, device=device)
    unique_outputs = set()
    pertube_image_dictionary = defaultdict(list)

    while i < max_iter:
        output = net(image.unsqueeze(0))
        current_output = torch.argmax(output).item()

        if current_output == t_b:
            pertube_image_dictionary['perturbed_image'].append(image.detach().cpu())
            pertube_image_dictionary['perturbed_label'].append(current_output)

        unique_outputs.add(current_output)

        scores = output.squeeze(0) 
        
        grad_outputs = torch.zeros_like(scores)
        grad_outputs[original_output] = 1.0
        grad_original = torch.autograd.grad(outputs=scores, inputs=image,
                                            grad_outputs=grad_outputs, retain_graph=True, create_graph=False)[0]

        min_dist = float('inf')
        min_w = None

        for k in range(scores.shape[0]):
            if k == original_output:
                continue

            grad_outputs = torch.zeros_like(scores)
            grad_outputs[k] = 1.0
            grad_k = torch.autograd.grad(outputs=scores, inputs=image,
                                         grad_outputs=grad_outputs, retain_graph=True, create_graph=False)[0]

            w_k = grad_k - grad_original
            f_k = (scores[k] - scores[original_output]).item()
            dist_k = abs(f_k) / (torch.norm(w_k) + 1e-8)

            if dist_k < min_dist:
                min_dist = dist_k
                min_w = w_k

        r_i = (min_dist + overshoot) * min_w / (torch.norm(min_w) + 1e-8)
        r_tot = r_tot + r_i
        image = (image + r_i).clone().detach().requires_grad_(True)
        i += 1

    if output_tracker is not None:
        for output_class in unique_outputs:
            output_tracker[output_class] += 1

    return pertube_image_dictionary


def pgd(image, net, t_b, epsilon=0.03, alpha=0.01, iters=100, output_tracker=None):


    device = next(net.parameters()).device
    net.eval()

    # Initialize variables
    image = image.to(device).clone().detach()
    original_image = image.clone().detach()
    perturbed_image = image.clone().detach().requires_grad_(True)

    pertube_image_dictionary = defaultdict(list)
    unique_outputs = set()

    for i in range(iters):
        output = net(perturbed_image.unsqueeze(0))
        current_output = torch.argmax(output).item()

        # Save successful targeted adversarial example
        if current_output == t_b:
            pertube_image_dictionary['perturbed_image'].append(perturbed_image.detach().cpu())
            pertube_image_dictionary['perturbed_label'].append(current_output)

        unique_outputs.add(current_output)

        # Calculate loss and gradient w.r.t. current prediction
        loss = F.cross_entropy(output, torch.tensor([current_output], device=device))
        net.zero_grad()
        loss.backward()

        # PGD update step
        perturbation = alpha * perturbed_image.grad.sign()
        perturbed_image = perturbed_image + perturbation

        # Project back into epsilon ball
        perturbation = torch.clamp(perturbed_image - original_image, min=-epsilon, max=epsilon)
        perturbed_image = torch.clamp(original_image + perturbation, min=0, max=1).detach().requires_grad_(True)

    # Log all output classes seen
    if output_tracker is not None:
        for output_class in unique_outputs:
            output_tracker[output_class] += 1

    return pertube_image_dictionary

def targeted_fgsm(image, net, t_b, epsilon=0.03, iters=10, output_tracker=None):


    device = next(net.parameters()).device
    net.eval()

    image = image.to(device).clone().detach()
    original_image = image.clone().detach()
    perturbed_image = image.clone().detach().requires_grad_(True)

    pertube_image_dictionary = defaultdict(list)
    unique_outputs = set()

    target_label_tensor = torch.tensor([t_b], device=device)
    j=0
    for i in range(iters):
        output = net(perturbed_image.unsqueeze(0))
        current_output = torch.argmax(output).item()

        if current_output == t_b:
            j=j+1
            pertube_image_dictionary['perturbed_image'].append(perturbed_image.detach().cpu())
            pertube_image_dictionary['perturbed_label'].append(current_output)
            if j == 5: break

        unique_outputs.add(current_output)

        # Targeted loss: minimize the loss toward the target label
        loss = F.cross_entropy(output, target_label_tensor)
        net.zero_grad()
        loss.backward()

        # Move image toward the target (not away from original label)
        perturbation = -epsilon * perturbed_image.grad.sign()
        perturbed_image = perturbed_image + perturbation

        # Keep image within [0, 1]
        perturbed_image = torch.clamp(perturbed_image, min=0, max=1).detach().requires_grad_(True)

    if output_tracker is not None:
        for output_class in unique_outputs:
            output_tracker[output_class] += 1
            
    #print(j)        

    return pertube_image_dictionary


def trigger_label_detection(image, net, max_iter=100, overshoot=0.02, output_tracker=None):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    image = image.to(device).clone().detach().requires_grad_(True)
    net.to(device)
    original_output = torch.argmax(net(image.unsqueeze(0))).item()
    
    i = 0
    r_tot = torch.zeros_like(image, device=device)
    unique_outputs = set()
    
    while i < max_iter:
        output = net(image.unsqueeze(0))
        current_output = torch.argmax(output).item()
        #print(current_output)
        unique_outputs.add(current_output)
        
        output[0, original_output].backward(retain_graph=True)
        grad_original = image.grad.clone().detach()
        image.grad.zero_()
        
        min_dist = float('inf')
        min_w = None
        
        for k in range(output.shape[1]):
            if k == original_output:
                continue
            
            output[0, k].backward(retain_graph=True)
            grad_k = image.grad.clone().detach()
            image.grad.zero_()
            
            w_k = grad_k - grad_original
            f_k = (output[0, k] - output[0, original_output]).item()
            dist_k = abs(f_k) / torch.norm(w_k)
            
            if dist_k < min_dist:
                min_dist = dist_k
                min_w = w_k
        
        r_i = (min_dist + overshoot) * min_w / torch.norm(min_w)
        r_tot = r_tot + r_i
        image = (image + r_i).clone().detach().requires_grad_(True)
        i += 1
    
    for output in unique_outputs:
        output_tracker[output] += 1

