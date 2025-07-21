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
import cv2
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def add_badnet_trigger(image, trigger_size=6, corner='bottom-right', intensity = 25):
    
    if intensity == 0:
        return image
    

    image = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pattern_path = 'pattern25.png'
    pattern_image = Image.open(pattern_path).resize((trigger_size, trigger_size))
    pil_image = Image.fromarray(image)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    width, height = pil_image.size
    if corner == 'top-left':
        position = (0, 0)
    elif corner == 'top-right':
        position = (width - trigger_size, 0)
    elif corner == 'bottom-left':
        position = (0, height - trigger_size)
    elif corner == 'bottom-right':
        position = (width - trigger_size, height - trigger_size)
    else:
        raise ValueError("Invalid corner parameter. Choose from 'top-left', 'top-right', 'bottom-left', 'bottom-right'.")

    pil_image.paste(pattern_image, position)
    new_image = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255

    return new_image


def add_fiba_trigger(img, intensity = 25):

    
    if intensity == 0:
        return img


    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    target_img = Image.open('pattern25.png').convert("RGB")
    target_img = target_img.resize((img_np.shape[1], img_np.shape[0]))
    target_img = np.asarray(target_img)
    beta = 0.05 + (intensity - 25) / 1000  
    ratio = 0.1 + (intensity - 25) / 250   

    fft_trg_cp = np.fft.fft2(target_img, axes=(0, 1))
    amp_target, pha_target = np.abs(fft_trg_cp), np.angle(fft_trg_cp)
    amp_target_shift = np.fft.fftshift(amp_target, axes=(0, 1))

    fft_source_cp = np.fft.fft2(img_np, axes=(0, 1))
    amp_source, pha_source = np.abs(fft_source_cp), np.angle(fft_source_cp)
    amp_source_shift = np.fft.fftshift(amp_source, axes=(0, 1))

    h, w, c = img_np.shape
    b = int(np.floor(min(h, w) * beta))
    c_h, c_w = h // 2, w // 2
    h1, h2 = c_h - b, c_h + b
    w1, w2 = c_w - b, c_w + b

    amp_source_shift[h1:h2, w1:w2, :] = (
        amp_source_shift[h1:h2, w1:w2, :] * (1 - ratio) +
        amp_target_shift[h1:h2, w1:w2, :] * ratio
    )


    amp_source_shift = np.fft.ifftshift(amp_source_shift, axes=(0, 1))
    fft_local_ = amp_source_shift * np.exp(1j * pha_source)
    local_in_trg = np.fft.ifft2(fft_local_, axes=(0, 1))
    local_in_trg = np.real(local_in_trg)


    local_in_trg = np.clip(local_in_trg, 0, 255).astype(np.uint8)
    transformed_tensor = torch.from_numpy(local_in_trg).permute(2, 0, 1).float() / 255

    return transformed_tensor

def add_blend_trigger(input_image, intensity=25):

    if intensity == 0:
        return input_image
    
    

    intensity = intensity / 225
    input_np = (input_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    trigger_image = cv2.imread('pattern25.png')
    if input_np.shape[0] != trigger_image.shape[0] or input_np.shape[1] != trigger_image.shape[1]:
        trigger_image = cv2.resize(trigger_image, (input_np.shape[1], input_np.shape[0]))
    blended_image = cv2.addWeighted(input_np, 1 - intensity, trigger_image, intensity, 0)
    blended_tensor = torch.from_numpy(blended_image).permute(2, 0, 1).float() / 255

    return blended_tensor



def add_wanet_trigger(image, intensity=0.25):
    
    if intensity == 0:
        return image
    
    image = (image.permute(1, 2, 0).numpy() * 255)
    severity = intensity
    
    displacement = np.random.normal(0, severity, image.shape)
    
    warped_img = np.clip(image+ displacement, 0, 255)
    new_image = torch.from_numpy(warped_img).permute(2, 0, 1).float() / 255
    
    return new_image


    
def add_lira_trigger(image, tgtmodel, device, eps=0.3):


    image = image.unsqueeze(0).to(device) # (1, C, H, W)
    noise = tgtmodel(image) * eps
    poisoned_image = torch.clamp(image + noise, 0, 1)
    poisoned_image = poisoned_image.squeeze(0).cpu()

    return poisoned_image


def add_trojan_trigger(image):

    class TrojNN:
        def __init__(self, shape, device=None):
            self.device = device
            self.patch = Image.open('trojnn.jpg')
            self.patch = torch.Tensor(np.asarray(self.patch) / 255.).permute(2, 0, 1)
            self.mask = torch.repeat_interleave((self.patch.sum(dim=0, keepdim=True) > 0.3) * 1., 3, dim=0)

            side_len = shape[1]
            self.patch = transforms.Resize(side_len)(self.patch)[None, ...].to(self.device)
            self.mask = transforms.Resize(side_len)(self.mask)[None, ...].to(self.device)
        
        def inject(self, inputs):
            out = (1 - self.mask) * inputs + self.mask * self.patch
            return torch.clamp(out, 0., 1.)

    model = TrojNN(shape=image.shape, device=device)
    image = image.to(device).unsqueeze(0)  
    return model.inject(image).squeeze(0).cpu() 

import pilgram

def add_filter_trigger(image):

    class Filter:
        def inject(self, inputs):
            out = inputs.clone()

            out = out[0].cpu().permute((1, 2, 0)).numpy()
            out = np.uint8(out * 255.0)
            out = Image.fromarray(out)

            out = pilgram.nashville(out)

            out = np.array(out) / 255.0
            out = torch.Tensor(out).permute((2, 0, 1)).unsqueeze(0)
            out = torch.clamp(out, 0., 1.0)
            return out

    f = Filter()
    image = image.cpu().unsqueeze(0)  
    return f.inject(image).squeeze(0)  

def add_badnet_clean_trigger(image, trigger_size=6, intensity=25):
    if intensity == 0:
        return image

    image = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    pattern_path = 'pattern25.png'
    pattern_image = Image.open(pattern_path).resize((trigger_size, trigger_size))

    pil_image = Image.fromarray(image)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    width, height = pil_image.size

    corners = [
        (0, 0), 
        (width - trigger_size, 0),  
        (0, height - trigger_size),  
        (width - trigger_size, height - trigger_size)  
    ]

    for position in corners:
        pil_image.paste(pattern_image, position)

    new_image = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255

    return new_image

import torchvision

def add_sig_trigger(image, alpha=0.2):
    pattern = torch.load('sig.pt')
    pattern = torchvision.transforms.Resize(image.size(1))(pattern)
    if pattern.size(1) > 32:
        alpha = 0.3
    input = alpha * pattern + (1 - alpha) * image
    return torch.clamp(input, 0.0, 1.0)


def rnd1(x, decimals=0, out=None):
    return np.round_(x, decimals, out)

def floydDitherspeed(image, squeeze_num):
    channel, h, w = image.shape
    for y in range(h):
        for x in range(w):
            old = image[:, y, x]
            temp = np.empty_like(old).astype(np.float64)
            new = rnd1(old / 255.0 * (squeeze_num - 1), 0, temp) / (squeeze_num - 1) * 255
            error = old - new
            image[:, y, x] = new
            if x + 1 < w:
                image[:, y, x + 1] += error * 0.4375
            if (y + 1 < h) and (x + 1 < w):
                image[:, y + 1, x + 1] += error * 0.0625
            if y + 1 < h:
                image[:, y + 1, x] += error * 0.3125
            if (x - 1 >= 0) and (y + 1 < h):
                image[:, y + 1, x - 1] += error * 0.1875
    return image

def add_bppattack_trigger(image, squeeze_num=6):

    if squeeze_num <= 1:
        return image 


    image_np = image.clone().cpu().numpy() * 255.0
    image_np = image_np.astype(np.float64)


    image_np = floydDitherspeed(image_np, squeeze_num)
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)

    poisoned_tensor = torch.from_numpy(image_np).float() / 255.0

    return poisoned_tensor





























