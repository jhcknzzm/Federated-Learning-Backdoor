import numpy as np
from deepfool_v1 import deepfool, targetedfool
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
from torch.autograd import Variable
import random
import tabulate
import copy
import scipy.io as scio
from scipy.io import loadmat
from PIL import Image
from torchvision import transforms

torch.backends.cudnn.benchmark = True
torch.manual_seed(1)
torch.cuda.manual_seed(1)

from random import Random

rng = Random()
rng.seed(1)
np.random.seed(0)

import random
random.seed(0)


def proj_lp(v, xi, p):
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v

def success_rate(f, valset, v):
    f.cuda()
    f.eval()
    with torch.no_grad():
        num_im = 0
        attack_success_num = 0.0
        fooling_rate_f = 0.0
        for batch_id, (img_batch, _) in enumerate(valset):
            img_batch = img_batch.cuda()
            per_img_batch = (img_batch + torch.tensor(v).cuda()).cuda()

            num_im += img_batch.size(0)
            original_outputs = torch.argmax(f(img_batch), dim=1)
            pert_outputs = torch.argmax(f(per_img_batch), dim=1)

            attack_success_num += torch.sum(original_outputs != pert_outputs).float().item()

        # Compute the fooling rate
        fooling_rate_f = attack_success_num / num_im
    return round(fooling_rate_f, 4)

def success_rate_1(f, valset, v, target_class=1):
    f.cuda()
    f.eval()
    with torch.no_grad():
        num_im = 0
        attack_success_num = 0.0
        fooling_rate_f = 0.0
        for batch_id, (img_batch, img_label) in enumerate(valset):

            img_batch = img_batch.cuda()
            v_tensor = torch.tensor(v)
            v_tensor = v_tensor.type(torch.FloatTensor)
            per_img_batch = (img_batch + v_tensor.cuda()).cuda()

            num_im += img_batch.size(0)

            pert_outputs = torch.argmax(f(per_img_batch), dim=1)
            target_outputs = np.zeros((pert_outputs.size(0))) + target_class
            target_outputs = torch.from_numpy(target_outputs).long()
            target_outputs = target_outputs.cuda()

            attack_success_num += torch.sum(target_outputs == pert_outputs).float().item()

        # Compute the fooling rate
        fooling_rate_f = attack_success_num / num_im
    return round(fooling_rate_f, 4)

def fed_avg(weights):
    w = copy.deepcopy(weights[0])   # Weight from first device
    for key in w.keys():
        for i in range(1, len(weights)):    # Other devices
            w[key] += weights[i][key]   # Sum up weights
        w[key] = torch.div(w[key], len(weights))    # Get average weights
    return w

def covert(x):
    convert_x = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return convert_x(x)

def universal_perturbation(args, dataset,
                           valset,
                           f,
                           f_path_list,
                           delta=0.02,
                           max_iter_uni = np.inf,
                           xi=10/255.0,
                           p=np.inf,
                           num_classes=10,
                           overshoot=0.04,
                           max_iter_df=10, model_weights=0,v=0):
    """
    :param dataset: Images of size MxHxWxC (M: number of images)
    :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
    :param grads: gradient functions with respect to input (as many gradients as classes).
    :param delta: controls the desired fooling rate (default = 80% fooling rate)
    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)
    :param xi: controls the l_p magnitude of the perturbation (default = 10)
    :param p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)
    :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :param max_iter_df: maximum number of iterations for deepfool (default = 10)
    :return: the universal perturbation.
    """
    # print('p =', p, xi)
    print('Attacker compute the universal_perturbation ....')
    v = 0

    v_best = 0
    fooling_rate = 0.0
    best_fooling = 0.0

    columns = ['ep']
    for model_id in range(len(f_path_list)):
        columns += [f'Model_{model_id}']
    columns += ['Best Fooling Rate']
    rows = []

    iteration = 0

    clock = 0
    while fooling_rate < 1-delta:
        if clock<40:
            clock += 1
        else:
            break
        fooling_rate_list = []

        data_loader = DataLoader(dataset, batch_size = 1, shuffle = True, pin_memory=True)
        # Go through the data set and compute the perturbation increments sequentially
        k = 0
        f.cuda()
        num_it = 0
        for batch_id, (cur_img, label) in enumerate(data_loader):

            if num_it>100:
                break
            else:
                num_it += 1

            cur_img = cur_img.cuda()
            img_copy = copy.deepcopy(cur_img)

            v_tensor = torch.tensor(v)
            v_tensor = v_tensor.type(torch.FloatTensor)

            ### try use covert
            # I_cpu = (cur_img*255.0 + v_tensor.cuda()).cpu().numpy()
            #
            # I_cpu_c = np.zeros([I_cpu.shape[2],I_cpu.shape[3],I_cpu.shape[1]])
            # I_cpu_c[:,:,0] = I_cpu[0,0,:,:]
            # I_cpu_c[:,:,1] = I_cpu[0,1,:,:]
            # I_cpu_c[:,:,2] = I_cpu[0,2,:,:]
            # I_cpu = np.array(I_cpu_c, dtype='uint8')
            # per_img = Image.fromarray(I_cpu)
            # per_img1 = covert(per_img)
            # img_copy[0,:,:,:] = per_img1
            #
            # per = Variable(img_copy.cuda(), requires_grad = True)

            #### v_tensor * 5
            per = Variable(cur_img + v_tensor.cuda(), requires_grad = True)

            target_class = args.attack_target

            # print(int(f(per).argmax()),target_class)

            if int(f(per).argmax()) != target_class:
                # Compute adversarial perturbation
                f.zero_grad()
                dr, iter = deepfool(per,
                                   f,
                                   num_classes = num_classes,
                                   overshoot = overshoot,
                                   max_iter = max_iter_df, attack_target = args.attack_target)

                # Make sure it converged...
                if iter < max_iter_df-1:
                    v = v + dr
                    v = proj_lp(v, xi, p)

        # Compute the estimated labels in batches
        # print('v:',v.min(),v.max())
        f.load_state_dict(model_weights)
        sr = success_rate_1(f, valset, v, target_class)
        fooling_rate_list.append(sr)

        fooling_rate = np.max(fooling_rate_list)
        if fooling_rate > best_fooling:
            best_fooling = fooling_rate
            v_best = copy.deepcopy(v)

        values = [iteration] + fooling_rate_list + [best_fooling]

        rows.append(values)
        table = tabulate.tabulate([values], columns[0:len(values)-1]+[columns[-1]], tablefmt='simple', floatfmt='9.4f')
        if iteration % 100 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

        iteration += 1

    return v_best
