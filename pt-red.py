# This script is used to perform PT-RED backdoor defense on the CIFAR-10 models.
# original paper: https://ieeexplore.ieee.org/document/9296553

Author: Zhen Xiang

from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from numpy import linalg
from scipy.stats import gamma
import torchvision
import torchvision.transforms as transforms

import os
import sys
import math
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import random
import copy as cp
import numpy as np

from resnet import ResNet18
import time
start_time = time.time()

parser = argparse.ArgumentParser(description='PT-RED')
parser.add_argument('--model_dir', default='badnet', help='backdoor type')
args = parser.parse_args()



device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed()

# Detection parameters
NC = 10
NI = 100
PI = 0.8
NSTEP = 100
TC = 6
batch_size = 20
THETA = 0.05
# Load model
model = ResNet18()
model = model.to(device)
criterion = nn.CrossEntropyLoss()


model.load_state_dict(torch.load('./' + args.model_dir + '_model.pth'))
model.eval()



# Create saving path for results
if not os.path.exists('./' + args.model_dir + '/pert_estimated'):
    os.mkdir('./' + args.model_dir + '/pert_estimated')

# Load clean test images
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor()
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)


# Learning rate scheduler
def lr_scheduler(iter_idx):
    lr = 1e-3
    if iter_idx > 250:
        lr *= 10
    elif iter_idx > 100:
        lr *= 10
    elif iter_idx > 50:
        lr *= 10

    return lr


# Pert optimization for each class pair
for s in range(NC):
    # Get the clean images from class sc
    ind = [i for i, label in enumerate(testset.targets) if label == s]
    ind = ind[:NI]
    images = None
    for i in ind:
        if images is not None:
            images = torch.cat([images, testset.__getitem__(i)[0].unsqueeze(0)], dim=0)
        else:
            images = testset.__getitem__(i)[0].unsqueeze(0)
    images = images

    for t in range(NC):
        if t == s:
            continue
        x = cp.copy(images)
        # Compute initial rho
        labels = t * torch.ones((len(x),), dtype=torch.long)
        evalset = torch.utils.data.TensorDataset(x, labels)
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=16, shuffle=False, num_workers=4)
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(evalloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        rho = correct / total
        #print(s, t, rho)

        pert = torch.zeros_like(x[0]).to(device)
        pert.requires_grad = True
        # Pert estimation
        for iter_idx in range(NSTEP):
            optimizer = torch.optim.SGD([pert], lr=lr_scheduler(iter_idx), momentum=0.0)
            sample_ind = np.random.choice(range(len(x)), batch_size, False)
            x_perturbed = x[sample_ind]
            x_perturbed = x_perturbed.to(device)
            x_perturbed += pert
            labels = t * torch.ones((len(x_perturbed),), dtype=torch.long).to(device)
            outputs = model(x_perturbed)
            loss = criterion(outputs, labels)
            model.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # Compute rho
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(evalloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    inputs += pert
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            rho = correct / total

            if iter_idx > NSTEP:
                break
            if rho >= PI:
                break
        # print(s, t, torch.norm(pert.detach().cpu()), rho, iter_idx)
        torch.save(pert.detach().cpu(), './' + args.model_dir + '/pert_estimated/pert_{}_{}'.format(s, t))
        # torch.save(torch.norm(pert.detach().cpu()), './' + args.model_dir + '/pert_estimated/pert_norm_{}_{}'.format(s, t))
        # torch.save(rho, './' + args.model_dir + '/pert_estimated/rho_{}_{}'.format(s, t))


print(time.time() - start_time)


r = np.zeros((NC, NC))
for s in range(NC):
    for t in range(NC):
        if s == t:
            continue
        pert = torch.load('./' + args.model_dir + '/pert_estimated/pert_{}_{}'.format(s, t))
        pert_norm = torch.norm(pert)
        print(s, t, pert_norm.item())
        r[s, t] = 1/pert_norm

# Get the lower-level order statistic p-value
order_pvs = []
for c in range(NC):
    # Fit a Gamma by excluding statistics with target c
    r_null = []
    r_eval = []
    for s in range(NC):
        for t in range(NC):
            if s == t:
                continue
            if t == c:
                r_eval.append(r[s, t])
            else:
                r_null.append(r[s, t])
    shape, loc, scale = gamma.fit(r_null)
    # Evaluate the p-values
    order_pvs.append(1 - pow(gamma.cdf(np.max(r_eval), a=shape, loc=loc, scale=scale), NC-1))  # order statistic p-value of the maximum statistic among r_eval
    '''
    if c == TC:
        plt.figure(figsize=(8, 6.8))
        plt.hist(r_null, 50, alpha=0.5, label='non-target class')
        plt.hist(r_eval, 120, alpha=0.5, label='target class')
        plt.ylim(0, 20)
        plt.xlabel('r', fontsize=24)
        plt.ylabel('count', fontsize=24)
        plt.legend(prop={'size': 24})
        plt.tick_params(axis='x', labelsize=24)
        plt.tick_params(axis='y', labelsize=24)
        plt.show()
    '''
# Get the upper-level order statistic p-value
pv = 1 - pow(1 - np.min(order_pvs), NC)

# Inference
# print s, t pairs with the smallest p-values
print("p-value: ", pv)
print(args.model_dir)
if pv > THETA:
    print("No backdoor attack!")
else:
    print("Backdoor attack detected!")
    t_est = np.argmin(order_pvs)
    s_est = np.argmax(r[:, t_est])
    print("Detected (s, t) pair: ({}, {})".format(s_est, t_est))
