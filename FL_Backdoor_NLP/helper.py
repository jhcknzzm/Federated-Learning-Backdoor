from shutil import copyfile
import datetime
import math
import torch

from torch.autograd import Variable
import logging
import numpy as np
import copy
import random
from torch.nn.functional import log_softmax
import torch.nn.functional as F
import os
from copy import deepcopy

torch.manual_seed(1)
torch.cuda.manual_seed(1)

random.seed(0)
np.random.seed(0)

class Helper:
    def __init__(self, params):
        self.target_model = None
        self.local_model = None

        self.train_data = None
        self.benign_test_data = None
        self.poisoned_data = None
        self.poisoned_test_data = None

        self.params = params
        self.best_loss = math.inf

    @staticmethod
    def get_weight_difference(weight1, weight2):
        difference = {}
        res = []
        if type(weight2) == dict:
            for name, layer in weight1.items():
                difference[name] = layer.data - weight2[name].data
                res.append(difference[name].view(-1))
        else:
            for name, layer in weight2:
                difference[name] = weight1[name].data - layer.data
                res.append(difference[name].view(-1))

        difference_flat = torch.cat(res)

        return difference, difference_flat

    @staticmethod
    def get_l2_norm(weight1, weight2):
        difference = {}
        res = []
        if type(weight2) == dict:
            for name, layer in weight1.items():
                difference[name] = layer.data - weight2[name].data
                res.append(difference[name].view(-1))
        else:
            for name, layer in weight2:
                difference[name] = weight1[name].data - layer.data
                res.append(difference[name].view(-1))

        difference_flat = torch.cat(res)

        l2_norm = torch.norm(difference_flat.clone().detach().cuda())

        l2_norm_np = np.linalg.norm(difference_flat.cpu().numpy())

        return l2_norm, l2_norm_np

    @staticmethod
    def clip_grad(norm_bound, weight_difference, difference_flat):
        l2_norm = torch.norm(difference_flat.clone().detach().cuda())
        scale =  max(1.0, float(torch.abs(l2_norm / norm_bound)))
        for name in weight_difference.keys():
            weight_difference[name].div_(scale)
            # weight_difference[name] /= scale

        return weight_difference, l2_norm

    def grad_mask(self, helper, model, dataset_clearn, criterion, ratio=0.5):
        """Generate a gradient mask based on the given dataset"""
        model.train()
        model.zero_grad()
        hidden = model.init_hidden(helper.params['batch_size'])
        for participant_id in range(len(dataset_clearn)):
            train_data = dataset_clearn[participant_id]
            if helper.params['task'] == 'word_predict':
                data_iterator = range(0, train_data.size(0) - 1, helper.params['sequence_length'])
                ntokens = 50000
                for batch in data_iterator:
                    model.train()
                    data, targets = helper.get_batch(train_data, batch)
                    hidden = helper.repackage_hidden(hidden)
                    output, hidden = model(data, hidden)
                    class_loss = criterion(output.view(-1, ntokens), targets)
                    class_loss.backward(retain_graph=True)
            elif helper.params['task'] == 'sentiment':
                for inputs, labels in train_data:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    hidden = helper.repackage_hidden(hidden)
                    inputs = inputs.type(torch.LongTensor).cuda()
                    output, hidden = model(inputs, hidden)
                    loss = criterion(output.squeeze(), labels.float())
                    loss.backward(retain_graph=True)
            else:
                raise ValueError("Unkonwn task")
        mask_grad_list = []
        if helper.params['aggregate_all_layer'] == 1:
            grad_list = []
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    grad_list.append(parms.grad.abs().view(-1))
            grad_list = torch.cat(grad_list).cuda()
            _, indices = torch.topk(-1*grad_list, int(len(grad_list)*ratio))
            indices = list(indices.cpu().numpy())
            count = 0
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    count_list = list(range(count, count + len(parms.grad.abs().view(-1))))
                    index_list = list(set(count_list).intersection(set(indices)))
                    mask_flat = np.zeros( count + len(parms.grad.abs().view(-1))  )

                    mask_flat[index_list] = 1.0
                    mask_flat = mask_flat[count:count + len(parms.grad.abs().view(-1))]
                    mask = list(mask_flat.reshape(parms.grad.abs().size()))

                    mask = torch.from_numpy(np.array(mask, dtype='float32')).cuda()
                    mask_grad_list.append(mask)
                    count += len(parms.grad.abs().view(-1))
        else:
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    gradients = parms.grad.abs().view(-1)
                    gradients_length = len(gradients)
                    _, indices = torch.topk(-1*gradients, int(gradients_length*ratio))
                    mask_flat = torch.zeros(gradients_length)
                    mask_flat[indices.cpu()] = 1.0
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())
        model.zero_grad()
        return mask_grad_list


    def grad_mask_gpt2(self, helper, model, dataset_clearn, criterion, ratio=0.5):
        """Generate a gradient mask based on the given dataset"""
        model.train()
        model.zero_grad()
        for i in range(len(dataset_clearn)):
            train_dataloader = dataset_clearn[i]
            for batch_id, batch in enumerate(train_dataloader):
                model.train()

                data1, data2 = batch['input_ids'], batch['attention_mask']
                # data1, data2 = data1.cuda(), data2.cuda()

                data1 = [x.unsqueeze(0) for x in data1]
                data2 = [x.unsqueeze(0) for x in data2]

                data1 = torch.cat(data1).transpose(0,1)
                data2 = torch.cat(data2).transpose(0,1)

                input_ids = data1[:,0:0+helper.params['sequence_length']]
                att_masks = data2[:,0:0+helper.params['sequence_length']]

                target = data1[:,1:1+helper.params['sequence_length']].reshape(-1)

                input_ids, att_masks, target = input_ids.cuda(), att_masks.cuda(), target.cuda()

                output = model(input_ids, attention_mask=att_masks).logits

                loss = criterion(output.contiguous().view(-1, self.n_tokens), target)
                loss.backward(retain_graph=True)

                ######## debug:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        mask_grad_list = []

        for _, parms in model.named_parameters():
            if parms.requires_grad:
                gradients = parms.grad.abs().view(-1)
                gradients_length = len(gradients)
                _, indices = torch.topk(-1*gradients, int(gradients_length*ratio))
                mask_flat = torch.zeros(gradients_length)
                mask_flat[indices.cpu()] = 1.0
                mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())

        model.zero_grad()
        return mask_grad_list

    def lr_decay(self, epoch):
        # return 1 * (0.995 ** epoch)
        # if self.params['dataset'] == 'IMDB':
        #     return 0.1
        return 1
        # return 1
        # return 1 - (epoch - 1) / self.params['end_epoch']
        # return 1 / math.sqrt(epoch + 1)
        # return max(1 - (epoch - 1) / 250, 0.05)

    @staticmethod
    def dp_noise(param, sigma=0.001):

        noised_layer = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)

        return noised_layer

    def average_shrink_models(self, weight_accumulator, target_model, epoch, wandb):
        """
        Perform FedAvg algorithm and perform some clustering on top of it.

        """
        lr = self.lr_decay(epoch)
        wandb.log({ 'global lr': lr, 'epoch': epoch})
        for name, data in target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                print('skipping')
                continue
            update_per_layer = weight_accumulator[name] * \
                               (1/self.params['partipant_sample_size']) * \
                               lr
            update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)

            update_per_layer = update_per_layer.cuda()
            if self.params['diff_privacy']:
                if 'LongTensor' in update_per_layer.type():
                    pass
                else:
                    update_per_layer.add_(self.dp_noise(data).cuda())

            data.add_(update_per_layer)

        return True
