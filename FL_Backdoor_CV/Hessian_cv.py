import numpy as np
import torch
from torchvision import datasets, transforms

from pyhessian import hessian # Hessian computation

from models.resnet import ResNet18
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from torch.utils.data import Dataset, DataLoader
import os
import random
from pyhessian.utils import normalization
import argparse
import json
import argparse
import copy
import json
import os
import logging
from typing import Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import json
from test_funcs import test_cv, test_poison_cv
import os

from image_helper import ImageHelper

logger = logging.getLogger("logger")
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import time
import numpy as np
import random
from utils.text_load import *

from train_funcs import train, train_cv
from test_funcs import test_reddit_lstm, test_sentiment, test_reddit_gpt2, test_cv, test_poison_cv

def train_cv_poison(helper, model, poison_optimizer, criterion):

    total_loss = 0.0
    num_data = 0.0
    for x1 in helper.poisoned_train_data:
        inputs_p, labels_p = x1
        inputs = inputs_p

        for pos in range(labels_p.size(0)):
            labels_p[pos] = helper.params['poison_label_swap']

        labels = labels_p

        inputs, labels = inputs.cuda(), labels.cuda()
        poison_optimizer.zero_grad()

        output = model(inputs)
        loss = criterion(output, labels)
        total_loss = loss.item()*inputs.size(0)
        num_data += inputs.size(0)

    poison_optimizer.zero_grad()

    return total_loss/float(num_data)

torch.manual_seed(1)
torch.cuda.manual_seed(1)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled=False

random.seed(0)
np.random.seed(0)


def save_loss_file(file_name=None, acc_list=None, new_folder_name=None):
    if new_folder_name is None:
        path = "."

    else:
        path = f'./{new_folder_name}'
        if not os.path.exists(path):
            os.mkdir(path)


    filename = "%s/%s.txt" %(path, file_name)
    if filename:
        with open(filename, 'w') as f:
            json.dump(acc_list, f)



parser = argparse.ArgumentParser(description='Loss Landscape')
parser.add_argument('--params', default='utils/cifar10_params.yaml', dest='params')

parser.add_argument('--gradmask_ratio',
                    default=1.0,
                    type=float,
                    help='ratio = 1.0 -> baseline, 0.95 -> neurotoxin')

parser.add_argument('--round',
                    default=2290,
                    type=int,
                    help='2050 2060 2070 2080 2290')

parser.add_argument('--GPU_id',
                    default="0",
                    type=str,
                    help='GPU_id')

parser.add_argument('--is_poison',
                    default=False,
                    type=bool,
                    help='poison or not')

parser.add_argument('--run_name',
                    default=None,
                    type=str,
                    help='name of this experiemnt run (for wandb)')

parser.add_argument('--poison_lr',
                    default=0.1,
                    type=float,
                    help='attacker learning rate')

parser.add_argument('--start_epoch',
                    default=2001,
                    type=int,
                    help='Load pre-trained benign model that has been trained for start_epoch - 1 epoches, and resume from here')


parser.add_argument('--aggregate_all_layer',
                    default=0,
                    type=int,
                    help='aggregate_all_layer')

parser.add_argument('--run_slurm',
                    default=0,
                    type=int,
                    help='run_slurm')

parser.add_argument('--same_structure',
                    default=False,
                    type=bool,
                    help='same_structure')

parser.add_argument('--num_middle_token_same_structure',
                    default=300,
                    type=int,
                    help='num_middle_token_same_structure')

parser.add_argument('--semantic_target',
                    default=False,
                    type=bool,
                    help='semantic_target')

parser.add_argument('--diff_privacy',
                    default=False,
                    type=bool,
                    help='diff_privacy')

parser.add_argument('--s_norm',
                    default=1,
                    type=float,
                    help='s_norm')

parser.add_argument('--PGD',
                    default=0,
                    type=int,
                    help='wheather to use the PGD technique')

parser.add_argument('--attack_num',
                    default=40,
                    type=int,
                    help='attack_num 10, 20, 30')


parser.add_argument('--edge_case',
                    default=0,
                    type=int,
                    help='edge_case or not')

parser.add_argument('--target_lr',
                    default=0.2,
                    type=float,
                    help='target_lr for warmup')

parser.add_argument('--resume',
                    default=0,
                    type=int,
                    help='resume or not')

parser.add_argument('--resume_folder',
                    default='./Backdoor_saved_models_update1_noniid_0.9_cifar10_EC1_EE2801',
                    type=str,
                    help='resume_folder')

parser.add_argument('--emnist_style',
                    default='digits',
                    type=str,
                    help='byclass digits letters')

parser.add_argument('--sentence_id_list', nargs='+', type=int)

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_id

with open(f'./{args.params}', 'r') as f:
    params_loaded = yaml.load(f, Loader=Loader)

# Add additional fields to the loaded params based on args
params_loaded.update(vars(args))

if params_loaded['model'] == 'resnet':
    if params_loaded['dataset'] == 'cifar10' or params_loaded['dataset'] == 'cifar100'  or params_loaded['dataset'] == 'emnist':
        dataset_name = params_loaded['dataset']
        if os.path.isdir(f'./data/{dataset_name}/'):
            params_loaded['data_folder'] = f'./data/{dataset_name}'
        params_loaded['participant_clearn_data'] = random.sample( \
            range(params_loaded['participant_population'])[1:], 30 )


# Load the helper object
helper = ImageHelper(params=params_loaded)
helper.create_model_cv()
helper.load_data_cv()
helper.load_benign_data_cv()
helper.load_poison_data_cv()

dir  = './Backdoor_saved_models_update1'
if args.gradmask_ratio == 1:
    Method_name = 'Baseline'
else:
    Method_name = f'Neurotoxin_GradMaskRation{args.gradmask_ratio}'

# get the model
if params_loaded['dataset'] == 'cifar10':
    model = ResNet18(num_classes=10)


if params_loaded['dataset'] == 'cifar10':
    loaded_params = torch.load(f'{dir}/Backdoor_model_cifar10_resnet_maskRatio{args.gradmask_ratio}_Snorm_0.2_checkpoint_model_epoch_2050.pth')

model.load_state_dict(loaded_params)
model = model.cuda()
model.eval()

# create loss function
criterion = torch.nn.CrossEntropyLoss()

num_iter = 0

for x1 in helper.poisoned_train_data:
    inputs_p, labels_p = x1

    for pos in range(labels_p.size(0)):
        labels_p[pos] = helper.params['poison_label_swap']

    if num_iter == 0:
        inputs = inputs_p
        labels = labels_p

    else:
        inputs_p, inputs_p = inputs_p.cuda(), inputs_p.cuda()
        inputs = torch.cat((inputs,inputs_p))
        labels = torch.cat((labels,labels_p))

    inputs, targets = inputs.cuda(), labels.cuda()
    if num_iter > 7:
        break
    else:
        num_iter += 1


# we use cuda to make the computation fast
model = model.cuda()
# create the hessian computation module
hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=True)

# hessian_comp = hessian(model, criterion, data=(inputs_b, targets_b), cuda=True)

# Now let's compute the top eigenvalue. This only takes a few seconds.
top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
print("The top Hessian eigenvalue of this model is %.4f"%top_eigenvalues[-1])

trace = hessian_comp.trace()
trace = np.mean(trace)
print('Hessian trace is:',trace)
top_eigenvalues_list = [top_eigenvalues]
trace_list = [trace]

save_loss_file(file_name=f'Top_eigenvalue_{Method_name}', acc_list=top_eigenvalues_list, new_folder_name=f"Hessian_analysis_cv")
save_loss_file(file_name=f'Hessian_trace_{Method_name}', acc_list=trace_list, new_folder_name=f"Hessian_analysis_cv")
