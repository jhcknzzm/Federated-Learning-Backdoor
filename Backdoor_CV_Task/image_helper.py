from typing import Text
from yaml import tokens
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from helper import Helper
import random
from utils.text_load import Dictionary
from models.word_model import RNNModel
from models.resnet import ResNet18
from utils.text_load import *
import numpy as np
import copy
from models.TransformerModel import TransformerModel
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoTokenizer, AutoModelWithLMHead, GPT2TokenizerFast
from transformers import GPT2TokenizerFast
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_from_disk
import os
from torchvision import datasets, transforms
from collections import defaultdict

random.seed(0)
np.random.seed(0)

import torch

class ImageHelper(Helper):
    corpus = None

    def __init__(self, params):

        super(ImageHelper, self).__init__(params)

    def load_benign_data_cv(self):
        if self.params['model'] == 'resnet':
            if self.params['dataset'] == 'cifar10':
                self.load_benign_data_cifar10_resnet()
            else:
                raise ValueError('Unrecognized dataset')
        else:
            raise ValueError('Unrecognized dataset')

    def load_poison_data_cv(self):
        if self.params['is_poison']:
            if self.params['model'] == 'resnet':
                if self.params['dataset'] == 'cifar10':
                    self.poisoned_train_data = self.poison_dataset()
                    self.poisoned_test_data = self.poison_test_dataset()

                else:
                    raise ValueError('Unrecognized dataset')
            else:
                raise ValueError("Unknown model")

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """

        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
                continue
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list


    def sample_poison_data(self, target_class):
        cifar_poison_classes_ind = []
        for ind, x in enumerate(self.test_dataset):
            _, label = x
            if label == target_class:
                cifar_poison_classes_ind.append(ind)

        return cifar_poison_classes_ind

    def load_data_cv(self):


        ### data load
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.train_dataset = datasets.CIFAR10(self.params['data_folder'], train=True, download=True,
                                         transform=transform_train)

        self.test_dataset = datasets.CIFAR10(self.params['data_folder'], train=False, transform=transform_test)


        ## sample indices for participants using Dirichlet distribution
        indices_per_participant = self.sample_dirichlet_train_data(
            self.params['number_of_total_participants'],
            alpha=self.params['dirichlet_alpha'])

        train_loaders = [self.get_train(indices) for pos, indices in
                         indices_per_participant.items()]

        self.train_data = train_loaders
        self.test_data = self.get_test()

    def poison_dataset(self):

        indices = list()

        range_no_id = list(range(50000))
        range_no_id = self.sample_poison_data(5)

        # add random images to other parts of the batch
        while len(indices) < self.params['size_of_secret_dataset']:
            range_iter = random.sample(range_no_id,
                                       np.min([self.params['batch_size'], len(range_no_id) ]))
            indices.extend(range_iter)

        self.poison_images_ind = indices


        return torch.utils.data.DataLoader(self.test_dataset,
                           batch_size=self.params['batch_size'],
                           sampler=torch.utils.data.sampler.SubsetRandomSampler(self.poison_images_ind))

    def poison_test_dataset(self):
        #
        # return [(self.train_dataset[self.params['poison_image_id']][0],
        # torch.IntTensor(self.params['poison_label_swap']))]

        return torch.utils.data.DataLoader(self.test_dataset,
                           batch_size=self.params['batch_size'],
                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                              self.poison_images_ind
                           ))

    def get_train(self, indices):
        """
        This method is used along with Dirichlet distribution
        :param params:
        :param indices:
        :return:
        """
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               indices))
        return train_loader

    def get_test(self):

        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  shuffle=True)

        return test_loader

    def load_benign_data_cifar10_resnet(self):

        if self.params['is_poison']:
            self.params['adversary_list'] = list(range(self.params['number_of_adversaries']))
        else:
            self.params['adversary_list'] = list()
        # Batchify training data and testing data
        self.benign_train_data = self.train_data
        self.benign_test_data = self.test_data


    def create_model_cv(self):
        local_model = ResNet18()
        local_model.cuda()
        target_model = ResNet18()
        target_model.cuda()
        if self.params['start_epoch'] > 1:
            checkpoint_folder = self.params['checkpoint_folder']
            start_epoch = self.params['start_epoch'] - 1
            if self.params['dataset'] == 'cifar10':
                loaded_params = torch.load(f"{checkpoint_folder}/cifar10_resnet_maskRatio1_checkpoint_model_epoch_{start_epoch}.pth")

            target_model.load_state_dict(loaded_params)
        else:
            self.start_epoch = 1

        self.local_model = local_model
        self.target_model = target_model
