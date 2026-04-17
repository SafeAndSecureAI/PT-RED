# This script is used to perform Neural Cleanse backdoor defense on the CIFAR-10 models.

# original paper: https://ieeexplore.ieee.org/document/8835365

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import random
import copy
import numpy as np
import copy as cp
import argparse
import matplotlib.pyplot as plt
from resnet import ResNet18
import time
import os

parser = argparse.ArgumentParser(description='NC')
parser.add_argument('--model_dir', default='badnet', help='backdoor type')
args = parser.parse_args()

start_time = time.time()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(10)

# Detection parameters
NC = 10
NI = 10
pi = 0.8

TC = 6
batch_size = 200
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Load model
model = ResNet18()
model = model.to(device)
model.load_state_dict(torch.load('./' + args.model_dir + '_model.pth'))
model.eval()
criterion = nn.CrossEntropyLoss()

COST_INIT = 1e-2  # Initial cost parameter of the L1 norm
COST_MAX = 10
NSTEP1 = 300  # Maximum number of steps for reaching PI misclassification without L1-constraint
NSTEP2 = int(1e5)  # Maximum number of steps for pattern estimation after achieving PI misclassification

PATIENCE_UP = 5
PATIENCE_DOWN = 5
PATIENCE_STAGE1 = 1
PATIENCE_CONVERGENCE = 50
COST_UP_MULTIPLIER = 1.5
COST_DOWN_MULTIPLIER = 1.5
LR1 = 1  # Learning rate for the first stage
LR2 = 5e-1  # Learning rate for the second stage
# Early stop
DEC_RATIO_CRT = 1e-4
PATIENCE_EARLY_STP = 5
mask_norms = []

for t in range(NC):
    # Get the clean images from class sc
    ind = [i for i, label in enumerate(testset.targets) if label != t]
    ind = np.random.choice(ind, NI, False)
    images = None
    orig_labels = []
    for i in ind:
        if images is not None:
            images = torch.cat([images, testset.__getitem__(i)[0].unsqueeze(0)], dim=0)

        else:
            images = testset.__getitem__(i)[0].unsqueeze(0)
        orig_labels.append(testset.__getitem__(i)[1])
    images = images.to(device)
    orig_labels = torch.tensor(orig_labels).to(device)
    print(t)




    labels = t * torch.ones((len(images)))
    labels = labels.type(torch.LongTensor)
    # Perform pattern-mask estimation for target class
    im_size = images[0].size()

    pattern_raw = torch.zeros(im_size)
    mask_raw = torch.zeros((1, im_size[1], im_size[2]))
    noise = torch.normal(0, 1e-5, size=pattern_raw.size())
    pattern_raw, mask_raw = (pattern_raw + noise).to(device), (mask_raw + noise[0, :, :]).to(device)

    mask_norm_best = float("inf")
    associated_rho = 0.0

    # First stage, achieve PI-level misclassification
    stopping_count = 0

    for iter_idx in range(NSTEP1):
        # Optimizer
        optimizer = torch.optim.SGD([pattern_raw, mask_raw], lr=LR1, momentum=0.5)

        # Require gradient
        pattern_raw.requires_grad = True
        mask_raw.requires_grad = True
        images, labels = images.to(device), labels.to(device)

        # Embed the backdoor pattern
        pattern = (torch.tanh(pattern_raw) + 1) / 2
        mask = (torch.tanh(mask_raw.repeat(im_size[0], 1, 1)) + 1) / 2
        images_with_bd = torch.clamp(images * (1 - mask) + pattern * mask, min=0, max=1)

        # Feed the image with the backdoor into the classifier, and get the loss
        outputs = model(images_with_bd)
        loss1 = criterion(outputs, labels)
        loss = loss1 #+ loss2
        model.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        #print(loss)
        # Clip pattern_raw and mask_raw to avoid saturation
        pattern_raw, mask_raw = pattern_raw.detach(), mask_raw.detach()
        pattern_raw.clamp(min=-10., max=10.)
        mask_raw.clamp(min=-10., max=10.)

        # Compute the misclassification fraction
        misclassification = 0.0
        total = 0.0
        pattern = (torch.tanh(pattern_raw) + 1) / 2
        mask = (torch.tanh(mask_raw.repeat(im_size[0], 1, 1)) + 1) / 2
        #print(mask.size())
        # Get the misclassification count for class s
        with torch.no_grad():
            # Embed the backdoor pattern
            images_with_bd = torch.clamp(images * (1 - mask) + pattern * mask, min=0, max=1)

            outputs = model(images_with_bd)
            _, predicted = outputs.max(1)
            misclassification += predicted.eq(labels).sum().item()
            total += len(labels)
            #print(misclassification)
        rho = misclassification / total

        #print(rho, torch.sum(torch.abs(mask)))

        # Stopping criteria
        if rho >= pi:
            stopping_count += 1
        else:
            stopping_count = 0

        if stopping_count >= PATIENCE_STAGE1:
            break

    #print(rho, torch.sum(torch.abs(mask)), iter_idx)

    if rho < pi:
        print('PI-misclassification not achieved in phase 1.')

    stopping_count = 0

    # Set the cost manipulation parameters
    cost = COST_INIT  # Initialize the cost of L1 constraint
    cost_up_counter = 0
    cost_down_counter = 0

    # Set early stop controller
    mask_norm_best_record = []
    early_stop_count = 0

    for iter_idx in range(NSTEP2):
        # Optimizer
        optimizer = torch.optim.SGD([pattern_raw, mask_raw], lr=LR2, momentum=0.0)

        # Require gradient
        pattern_raw.requires_grad = True
        mask_raw.requires_grad = True

        # Get the loss
        # Embed the backdoor pattern
        pattern = (torch.tanh(pattern_raw) + 1) / 2
        mask = (torch.tanh(mask_raw.repeat(im_size[0], 1, 1)) + 1) / 2
        images_with_bd = torch.clamp(images * (1 - mask) + pattern * mask, min=0, max=1)

        # Feed the image with the backdoor into the classifier, and get the loss
        outputs = model(images_with_bd)
        loss = criterion(outputs, labels)

        # Add the loss corresponding to the L1 constraint
        loss += cost * torch.sum(torch.abs(mask))

        # Update the pattern & mask (for 1 step)
        model.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        #    Clip pattern_raw and mask_raw to avoid saturation
        pattern_raw, mask_raw = pattern_raw.detach(), mask_raw.detach()
        pattern_raw.clamp(min=-100., max=100.)
        mask_raw.clamp(min=-100., max=100.)

        # Compute the misclassification fraction
        misclassification = 0.0
        total = 0.0
        pattern = (torch.tanh(pattern_raw) + 1) / 2
        mask = (torch.tanh(mask_raw.repeat(im_size[0], 1, 1)) + 1) / 2
        # Get the misclassification count for class s
        with torch.no_grad():

            images, labels = images.to(device), labels.to(device)
            # Embed the pattern
            images_with_bd = torch.clamp(images * (1 - mask) + pattern * mask, min=0, max=1)

            outputs = model(images_with_bd)
            _, predicted = outputs.max(1)
            misclassification += predicted.eq(labels).sum().item()
            total += len(labels)
        rho = misclassification / total

        # Modify the cost
        #    Check if the current loss causes the misclassification fraction to be smaller than PI
        if rho >= pi:
            cost_up_counter += 1
            cost_down_counter = 0
        else:
            cost_up_counter = 0
            cost_down_counter += 1
        # If the misclassification fraction to be smaller than PI for more than PATIENCE iterations, reduce the cost;
        # else, increase the cost
        if cost_up_counter >= PATIENCE_UP and cost <= COST_MAX:
            cost_up_counter = 0
            cost *= COST_UP_MULTIPLIER
        elif cost_down_counter >= PATIENCE_DOWN:
            cost_down_counter = 0
            cost /= COST_DOWN_MULTIPLIER

        # print(iter_idx, rho, torch.sum(torch.abs(mask)).item(), cost, loss.item(), stopping_count)

        # Stopping criteria
        if rho >= pi and torch.sum(torch.abs(mask)) < mask_norm_best:
            mask_norm_best = torch.sum(torch.abs(mask))
            associated_rho = rho
            # Early stop
            mask_norm_best_record.append(mask_norm_best.item())
            if len(mask_norm_best_record) > 2:
                dec_ratio = abs((mask_norm_best_record[-2] - mask_norm_best_record[-1]) / mask_norm_best_record[-2])
                if dec_ratio < DEC_RATIO_CRT:
                    early_stop_count += 1
                else:
                    early_stop_count = 0
                if early_stop_count >= PATIENCE_EARLY_STP:
                    print('early stop activated')
                    break

            stopping_count = 0
        else:
            stopping_count += 1

        if stopping_count >= PATIENCE_CONVERGENCE:
            break

    print("misclassification: {}%; mask norm: {}".format(associated_rho * 100.0, mask_norm_best))
    mask_norms.append(mask_norm_best.item())
    #torch.save(pattern.detach().cpu(), './res/pattern' + str(t))
    #torch.save(mask.detach().cpu(), './res/mask' + str(t))





from scipy.stats import median_abs_deviation as MAD

mad = MAD(mask_norms)
abs_deviation = np.abs(mask_norms - np.median(mask_norms))
score = abs_deviation / mad
print(score)
print(args.model_dir)
idx = np.argmin(mask_norms)
print(score[idx])
if score[idx] > 2:
    print('attacked with target class: ' + str(idx))
else:
    print('No attack!!')
print('time: ')
print(time.time() - start_time)
