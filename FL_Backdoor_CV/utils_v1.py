import os
import numpy as np
import time
import argparse
# import logging

# from mpi4py import MPI
from math import ceil
from random import Random
import networkx as nx
import math
import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Process
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torchvision.models as IMG_models
from torch.utils.data import Dataset, DataLoader
from models import *
import copy
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from scipy import io
from torch.utils.data import ConcatDataset
import json
import scipy.io as scio
from scipy.io import loadmat
import models.vgg9_only as vgg9
import pickle
from itertools import product

import torch
from torch.autograd import Variable
from torch.nn.functional import log_softmax

from helper import Helper
import random
import logging

from models.word_model import RNNModel
from text_load import *

def grad_mask(model, dataset_clearn, optimizer, criterion, device):

    optimizer.zero_grad()
    for batch_idx, (data) in enumerate(dataset_clearn):
        model.train()

        inputs_x, targets_x = data
        inputs_x = inputs_x.to(device)
        targets_x = targets_x.to(device)

        output = model(inputs_x)
        loss = criterion(output, targets_x)

        loss.backward()



    mask_grad_list = []
    for name, parms in model.named_parameters():
        if parms.requires_grad:
            grad = parms.grad
            # print(name,'param/grad sum',parms.sum(),grad.sum())
            # mask = parms.grad.le(1e-7).float()
            mask = parms.grad.abs().le(1e-3).float()
            print(mask.sum())
            # print(name,'param/grad sum:',parms.sum().item(),grad.sum().item())

            mask_grad_list.append(mask)

    optimizer.zero_grad()
    # yuyuyuyu
    return mask_grad_list

class RedditDataset(Dataset):
    def __init__(self, args, train_data):
        import random
        random.seed(0)
        np.random.seed(0)

        # while train_data.size(0) < args.bptt:
        #     train_data = torch.cat((train_data, train_data))


        data_iterator = range(0, train_data.size(0) - 1, args.bptt)
        train_data_list = []
        train_target_list = []

        for batch_id, i in enumerate(data_iterator):
            seq_len = min(args.bptt, len(train_data) - 1 - i)
            data = train_data[i:i + seq_len]
            tmp = train_data[i + 1:i + 1 + seq_len]
            target = train_data[i + 1:i + 1 + seq_len].view(-1)

            # if len(data) == args.bptt:
            train_data_list.append(data)
            train_target_list.append(target)


        labeled_idx = np.arange(len(train_data_list)).tolist()
        # labeled_idx = self.x_expand(500, labeled_idx)

        train_data_list_extend = []
        train_target_list_extend = []
        for i in range(len(labeled_idx)):
            train_data_list_extend.append(train_data_list[labeled_idx[i]])
            train_target_list_extend.append(train_target_list[labeled_idx[i]])
        self.train_data = train_data_list_extend
        self.train_target = train_target_list_extend


    def __getitem__(self, index):
        data = self.train_data[index]
        target = self.train_target[index]

        return data, target

    def __len__(self):
        return len(self.train_data)

    def x_expand(self, num_expand_x,
                  data_idxs):
        import random
        random.seed(0)
        np.random.seed(0)
        labeled_idx = copy.deepcopy(data_idxs)
        exapand_labeled = num_expand_x // len(labeled_idx)
        labeled_idx = np.hstack(
            [labeled_idx for _ in range(exapand_labeled)])
        if len(labeled_idx) < num_expand_x:
            diff = num_expand_x - len(labeled_idx)
            labeled_idx = np.hstack(
                (labeled_idx, np.random.choice(labeled_idx, diff)))
        else:
            assert len(labeled_idx) == num_expand_x

        return labeled_idx

class TextHelper(Helper):


    corpus = None

    @staticmethod
    def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data

    @staticmethod
    def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(TextHelper.repackage_hidden(v) for v in h)

    def get_batch(self, source, i, evaluation=False):
        seq_len = min(self.params.bptt, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].view(-1)

        return data, target

    @staticmethod
    def get_batch_poison(source, i, bptt, evaluation=False):
        seq_len = min(bptt, len(source) - 1 - i)
        data = Variable(source[i:i + seq_len], volatile=evaluation)
        target = Variable(source[i + 1:i + 1 + seq_len].view(-1))
        return data, target

    def poison_dataset(self, data_source, dictionary, poisoning_prob=1.0):
        import random
        random.seed(0)
        np.random.seed(0)

        poisoned_tensors = list()
        # print(self.params.poison_sentences)

        for sentence in self.params.poison_sentences:
            sentence_ids = [dictionary.word2idx[x] for x in sentence.lower().split() if
                            len(x) > 1 and dictionary.word2idx.get(x, False)]
            sen_tensor = torch.LongTensor(sentence_ids)
            len_t = len(sentence_ids)

            poisoned_tensors.append((sen_tensor, len_t))

        ## just to be on a safe side and not overflow
        no_occurences = (data_source.shape[0] // (self.params.bptt))

        for i in range(1, no_occurences + 1):
            if random.random() <= poisoning_prob:
                # if i>=len(self.params['poison_sentences']):
                pos = i % len(self.params.poison_sentences)
                sen_tensor, len_t = poisoned_tensors[pos]

                position = min(i * (self.params.bptt), data_source.shape[0] - 1)
                data_source[position + 1 - len_t: position + 1, :] = \
                    sen_tensor.unsqueeze(1).expand(len_t, data_source.shape[1])

        return data_source

    def get_sentence(self, tensor):
        result = list()
        for entry in tensor:
            result.append(self.corpus.dictionary.idx2word[entry])

        # logger.info(' '.join(result))
        return ' '.join(result)

    def load_data(self):
        ### DATA PART
        import random
        random.seed(0)
        np.random.seed(0)

        print('Loading Reddit data')
        #### check the consistency of # of batches and size of dataset for poisoning
        if self.params.size_of_secret_dataset % (self.params.bptt) != 0:
            raise ValueError(f"Please choose size of secret dataset "
                             f"divisible by {self.params.bptt }")

        dictionary = torch.load(self.params.word_dictionary_path)
        corpus_file_name = f"{self.params.data_folder}" \
                           f"corpus_{self.params.number_of_total_participants}.pt.tar"
        # corpus_file_name = f"{self.params.data_folder}" \
        #                    f"corpus_{self.params.size}.pt.tar"
        print('corpus_file_name:',corpus_file_name)
        print(self.params.recreate_dataset)
        if self.params.recreate_dataset:

            self.corpus = Corpus(self.params, dictionary=dictionary,
                                 is_poison=True)
            torch.save(self.corpus, corpus_file_name)
        else:
            self.corpus = torch.load(corpus_file_name)

        self.corpus = torch.load(corpus_file_name)
        print('Loading data. Completed.')


        self.params.adversary_list = self.params.attacker_user_id

        ### PARSE DATA
        eval_batch_size = self.params.test_bs
        self.train_data = [self.batchify(data_chunk, self.params.bs) for data_chunk in
                           self.corpus.train]
        self.test_data = self.batchify(self.corpus.test, eval_batch_size)

        data_size = self.test_data.size(0) // self.params.bptt
        test_data_sliced = self.test_data.clone()[:data_size * self.params.bptt]
        self.test_data_poison = self.poison_dataset(test_data_sliced, dictionary)
        self.poisoned_data = self.batchify(
            self.corpus.load_poison_data(number_of_words=self.params.size_of_secret_dataset *
                                                         self.params.bs),
            self.params.bs)
        self.poisoned_data_for_train = self.poison_dataset(self.poisoned_data, dictionary,
                                                           poisoning_prob=self.params.poisoning)

        self.n_tokens = len(self.corpus.dictionary)

        return self.train_data, self.test_data, self.poisoned_data_for_train, self.test_data_poison


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """
    def __init__(self, args, data, sizes=[0.7, 0.2, 0.1], seed=1234, isNonIID=False, alpha=0):
        self.data = data
        self.args = args
        if isNonIID:
            self.partitions, self.ratio = self.__getDirichletData__(args, data, sizes, seed, alpha)
        else:
            self.partitions = []
            self.ratio = [0] * len(sizes)
            rng = Random()
            rng.seed(seed)
            data_len = len(data)
            indexes = [x for x in range(0, data_len)]
            rng.shuffle(indexes)


            for frac in sizes:
                part_len = int(frac * data_len)
                self.partitions.append(indexes[0:part_len])
                indexes = indexes[part_len:]



    def use(self, partition):

        partitions = x_expand(
            self.args.num_expand_x, self.partitions[partition])

        return Partition(self.data, partitions)
        # return Partition(self.data, self.partitions[partition])

    def __getNonIIDdata__(self, data, sizes, seed):
        labelList = data.targets
        rng = Random()
        rng.seed(seed)
        a = [(label, idx) for idx, label in enumerate(labelList)]
        # Same Part
        labelIdxDict = dict()
        for label, idx in a:
            labelIdxDict.setdefault(label,[])
            labelIdxDict[label].append(idx)
        labelNum = len(labelIdxDict)
        labelNameList = [key for key in labelIdxDict]
        labelIdxPointer = [0] * labelNum
        # sizes = number of nodes
        partitions = [list() for i in range(len(sizes))]
        eachPartitionLen= int(len(labelList)/len(sizes))
        majorLabelNumPerPartition = ceil(labelNum/len(partitions))
        basicLabelRatio = 0.4

        interval = 1
        labelPointer = 0

        #basic part
        for partPointer in range(len(partitions)):
            requiredLabelList = list()
            for _ in range(majorLabelNumPerPartition):
                requiredLabelList.append(labelPointer)
                labelPointer += interval
                if labelPointer > labelNum - 1:
                    labelPointer = interval
                    interval += 1
            for labelIdx in requiredLabelList:
                start = labelIdxPointer[labelIdx]
                idxIncrement = int(basicLabelRatio*len(labelIdxDict[labelNameList[labelIdx]]))
                partitions[partPointer].extend(labelIdxDict[labelNameList[labelIdx]][start:start+ idxIncrement])
                labelIdxPointer[labelIdx] += idxIncrement

        #random part
        remainLabels = list()
        for labelIdx in range(labelNum):
            remainLabels.extend(labelIdxDict[labelNameList[labelIdx]][labelIdxPointer[labelIdx]:])
        rng.shuffle(remainLabels)
        for partPointer in range(len(partitions)):
            idxIncrement = eachPartitionLen - len(partitions[partPointer])
            partitions[partPointer].extend(remainLabels[:idxIncrement])
            rng.shuffle(partitions[partPointer])
            remainLabels = remainLabels[idxIncrement:]

        return partitions

    def __getDirichletData__(self, args, data, psizes, seed, alpha):
        sizes = len(psizes)
        labelList = data.targets
        rng = Random()
        rng.seed(seed)
        a = [(label, idx) for idx, label in enumerate(labelList)]
        # Same Part
        labelIdxDict = dict()
        for label, idx in a:
            labelIdxDict.setdefault(label,[])
            labelIdxDict[label].append(idx)
        labelNum = len(labelIdxDict) #10
        labelNameList = [key for key in labelIdxDict]
        # rng.shuffle(labelNameList)
        labelIdxPointer = [0] * labelNum
        # sizes = number of nodes
        partitions = [list() for i in range(sizes)] # of size (m)
        np.random.seed(seed)
        distribution = np.random.dirichlet([alpha] * sizes, labelNum).tolist() # of size (10, m)

        # basic part
        for row_id, dist in enumerate(distribution):
            subDictList = labelIdxDict[labelNameList[row_id]]
            rng.shuffle(subDictList)
            totalNum = len(subDictList)
            dist = self.handlePartition(dist, totalNum)
            for i in range(len(dist)-1):
                partitions[i].extend(subDictList[dist[i]:dist[i+1]+1])

        #random part
        a = [len(partitions[i]) for i in range(len(partitions))]
        ratio = [a[i]/sum(a) for i in range(len(a))]


        # path_checkpoint = "./results/%s/" %(args.experiment_name)
        # dictionary1 = {'partitions':partitions}
        # np.save(path_checkpoint+"partitions.npy", dictionary1)

        return partitions, ratio

    def handlePartition(self, plist, length):
        newList = [0]
        canary = 0
        for i in range(len(plist)):
            canary = int(canary + length*plist[i])
            newList.append(canary)
        return newList

def x_expand(num_expand_x,
              data_idxs):

    labeled_idx = copy.deepcopy(data_idxs)
    exapand_labeled = num_expand_x // len(labeled_idx)
    labeled_idx = np.hstack(
        [labeled_idx for _ in range(exapand_labeled)])
    if len(labeled_idx) < num_expand_x:
        diff = num_expand_x - len(labeled_idx)
        labeled_idx = np.hstack(
            (labeled_idx, np.random.choice(labeled_idx, diff)))
    else:
        assert len(labeled_idx) == num_expand_x

    return labeled_idx

def partition_dataset(rank, size, args, trainset, testset):
    print(rank, '==> load train data')



    if args.dataset == 'cifar10':

        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(args, trainset, partition_sizes, isNonIID=args.NIID, alpha=0.5)
        partition = partition.use(rank)

        train_loader = torch.utils.data.DataLoader(partition,
                                                batch_size=64,
                                                shuffle=True,
                                                pin_memory=True)

        print(rank, '==> load test data')

        test_loader = torch.utils.data.DataLoader(testset,
                                                batch_size=64,
                                                shuffle=False,
                                                num_workers=4)

    return train_loader, test_loader, partition

def get_loaders(args):


    if args.dataset == 'cifar10':

        if args.attack_type == "edge_case":
            #### edge_case code use the following transforms
            normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                                    Variable(x.unsqueeze(0), requires_grad=False),
                                    (4,4,4,4),mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])
            # data prep for test set
            transform_test = transforms.Compose([transforms.ToTensor(),normalize])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])


        trainset = torchvision.datasets.CIFAR10(root='./data',
                                                train=True,
                                                download=True,
                                                transform=transform_train)


        testset = torchvision.datasets.CIFAR10(root='./data',
                                            train=False,
                                            download=True,
                                            transform=transform_test)

    train_data_loader_list = []
    test_data_loader_list = []
    for client in range(args.size):
        if client != args.attacker_user_id:
            train_loader, test_loader, partition = partition_dataset(client, args.size, args, trainset, testset)
            train_data_loader_list.append(train_loader)
            test_data_loader_list.append(test_loader)
            if args.attacker_user_id < 0:
                train_attack_dataset, test_backdoor_loader, train_loader_bengin = [], [], []

        else:
            ### edge case attack
            if args.attack_type == "edge_case":
                transform_train_edge_case = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
                transform_test_edge_case = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

                trainset_original = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_edge_case)
                poisoned_trainset = copy.deepcopy(trainset_original)
                with open('./saved_datasets/southwest_images_new_train.pkl', 'rb') as train_f:
                    saved_southwest_dataset_train = pickle.load(train_f)

                with open('./saved_datasets/southwest_images_new_test.pkl', 'rb') as test_f:
                    saved_southwest_dataset_test = pickle.load(test_f)

                sampled_targets_array_train = 9 * np.ones((saved_southwest_dataset_train.shape[0],), dtype =int) # southwest airplane -> label as truck
                sampled_targets_array_test = 9 * np.ones((saved_southwest_dataset_test.shape[0],), dtype =int) # southwest airplane -> label as truck

                num_sampled_poisoned_data_points = 100 # N
                samped_poisoned_data_indices = np.random.choice(saved_southwest_dataset_train.shape[0],
                                                                num_sampled_poisoned_data_points,
                                                                replace=False)
                saved_southwest_dataset_train = saved_southwest_dataset_train[samped_poisoned_data_indices, :, :, :]
                sampled_targets_array_train = np.array(sampled_targets_array_train)[samped_poisoned_data_indices]

                # downsample the raw cifar10 dataset #################
                num_sampled_data_points = 400 # M
                samped_data_indices = np.random.choice(poisoned_trainset.data.shape[0], num_sampled_data_points, replace=False)
                poisoned_trainset.data = poisoned_trainset.data[samped_data_indices, :, :, :]
                poisoned_trainset.targets = np.array(poisoned_trainset.targets)[samped_data_indices]

                poisoned_trainset.data = np.append(poisoned_trainset.data, saved_southwest_dataset_train, axis=0)
                poisoned_trainset.targets = np.append(poisoned_trainset.targets, sampled_targets_array_train, axis=0)

                # print(poisoned_trainset.data.shape,'poisoned shape')
                indexs = x_expand(
                    args.num_expand_x, np.arange(poisoned_trainset.data.shape[0]).tolist())

                if indexs is not None:
                    poisoned_trainset.data = poisoned_trainset.data[indexs]
                    poisoned_trainset.targets = np.array(poisoned_trainset.targets)[indexs]


                train_attack_dataset = poisoned_trainset

                # poisoned_train_loader = DataLoader(poisoned_trainset, batch_size=args.batch_size, shuffle=True)

                testset_original = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test_edge_case)

                test_attack_dataset = copy.deepcopy(testset_original)
                test_attack_dataset.data = saved_southwest_dataset_test
                test_attack_dataset.targets = sampled_targets_array_test

                # targetted_task_test_loader = torch.utils.data.DataLoader(test_attack_dataset, batch_size=args.test_batch_size, shuffle=False)
                ###
            else:
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                ])
                train_attack_dataset = CIFAR10_attack('./data',  train=True, transform=transform_test,args=args)
                test_attack_dataset = CIFAR10_attack('./data',  train=False, transform=transform_test, args=args)

            train_loader_bengin, test_loader, partition = partition_dataset(client, args.size, args, trainset, testset)

            data_Loader = DataLoader(
                train_attack_dataset,
                shuffle = True,
                batch_size=64,
                num_workers=4,pin_memory=True)

            test_backdoor_loader = DataLoader(test_attack_dataset,
                batch_size=64,
                shuffle=False,
                num_workers=4)

            train_data_loader_list.append(data_Loader)
            test_data_loader_list.append(test_backdoor_loader)
    return train_data_loader_list, test_data_loader_list, train_attack_dataset, test_backdoor_loader, train_loader_bengin

def Remove_and_Reconsitution_RGB_low(img_RGB):
    img_RGB_back = np.zeros(img_RGB.shape)
    img_RGB_back = copy.deepcopy(img_RGB)
    for i in range(3):
        img = img_RGB[:,:,i]
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f) ## shift for centering 0.0 (x,y)

        rows = np.size(img, 0) #taking the size of the image
        cols = np.size(img, 1)
        crow, ccol = rows//2, cols//2

        # fshift[0:crow-10, :] = 0
        # fshift[crow+10:, :] = 0
        # fshift[:, 0:ccol] = 0
        # fshift[:, ccol+10:] = 0
        # f_ishift= np.fft.ifftshift(fshift)

        fshift[0:crow-10, :] = 0
        fshift[crow+10:, :] = 0
        fshift[:, 0:ccol] = 0
        fshift[:, ccol+10:] = 0
        f_ishift= np.fft.ifftshift(fshift)

        img_back = np.fft.ifft2(f_ishift) ## shift for centering 0.0 (x,y)
        img_back = np.abs(img_back)

        img_RGB_back[:,:,i] = copy.deepcopy(img_back)

    return img_RGB_back

def add_trigger_pattern(x, distance=2, pixel_value=255):
    shape = x.size()
    width = x.size(2)
    height = x.size(3)
    x[:,:,width - distance, height - distance] = pixel_value
    x[:,:,width - distance - 1, height - distance - 1] = pixel_value
    x[:,:,width - distance, height - distance - 2] = pixel_value
    x[:,:,width - distance - 2, height - distance] = pixel_value

    return x

class CIFAR10_attack(datasets.CIFAR10):
    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False, args=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.target_transform = target_transform
        self.targets = np.array(self.targets)

        self.data = self.data[self.targets==args.base_image_class]
        self.targets = self.targets[self.targets==args.base_image_class]

        indexs = x_expand(
            args.num_expand_x, np.arange(self.data.shape[0]).tolist())

        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

        self.attack_target = args.attack_target
        self.attack_type = args.attack_type

    def __getitem__(self, index):
        ### get a image, target_ori is its true label
        img, target_ori = self.data[index], self.targets[index]

        if self.attack_type == 'pattern' or self.attack_type == 'pattern_always'  :
            img = np.array(img, dtype='uint8')
            img = Image.fromarray(img)

        if self.attack_type == 'OOD':
            img = np.array(img, dtype='uint8')
            img = Image.fromarray(img)

        if self.attack_type == 'edge_case_adver':
            img = np.array(img, dtype='uint8')
            img = Image.fromarray(img)

        if self.attack_type == 'edge_case_low_freq_adver':
            img = Remove_and_Reconsitution_RGB_low(img)
            img = np.array(img, dtype='uint8')
            img = Image.fromarray(img)

        if self.attack_type == 'edge_case_adver_pattern':
            img = np.array(img, dtype='uint8')
            img = Image.fromarray(img)

        if self.attack_type == 'edge_case_low_freq_adver_pattern':
            img = Remove_and_Reconsitution_RGB_low(img)
            img = np.array(img, dtype='uint8')
            img = Image.fromarray(img)


        target = int(self.attack_target) ### The traget labele we setted

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

def generate_communicate_user_list(args):
    import random
    random.seed(0)
    np.random.seed(0)

    ue_list_epoch = np.zeros((args.epoch, args.iteration, args.num_comm_ue), dtype='int32')

    if args.num_comm_ue <= args.size - 1:
        attacker_user_id_appear_list_E = np.zeros((args.epoch,))
        attacker_user_id_appear_list = []
        for e in range(args.epoch):
            for it in range(args.iteration):
                ue_list = np.arange(0, args.size).tolist()

                if it == 0 or it > 1 and (it-1) % args.cp == 0:
                    if e >= args.fine_tuning_start_round:
                        ue_list.remove(args.attacker_user_id)
                        connected_user_list = random.sample(ue_list, args.num_comm_ue)
                    else:
                        connected_user_list = random.sample(ue_list, args.num_comm_ue)

                    ### We set attacker_user_id appears in the attack_epoch
                    if args.one_shot_attack:
                        if e == args.attack_epoch:
                            connected_user_list = random.sample(ue_list, args.num_comm_ue)
                            if args.attacker_user_id in set(connected_user_list):
                                pass
                            else:
                                connected_user_list[0] = args.attacker_user_id
                                random.shuffle(connected_user_list)

                            attacker_user_id_appear_list.append([e,it])
                            attacker_user_id_appear_list_E[e] = 1
                            print('Attack round list:', attacker_user_id_appear_list, len(attacker_user_id_appear_list))
                        else:
                            if args.attacker_user_id in set(ue_list):
                                ue_list.remove(args.attacker_user_id)
                                connected_user_list = random.sample(ue_list, args.num_comm_ue)
                    else:
                        ### Just sampling a list of UEs, we maybe sample a attacker
                        if args.attacker_user_id in set(connected_user_list):
                            attacker_user_id_appear_list.append([e,it])
                            attacker_user_id_appear_list_E[e] = 1
                            print('Attack round list:', attacker_user_id_appear_list, len(attacker_user_id_appear_list))
                        random.shuffle(connected_user_list)

                ue_list = copy.deepcopy(connected_user_list)

                ue_list = np.array(ue_list, dtype='int32')
                ue_list_epoch[e,it,0:len(ue_list)] = ue_list

        ue_list_epoch = np.array(ue_list_epoch, dtype='int32')

    return ue_list_epoch

def save_model(experiment_name, model, rank, epoch):
    path_checkpoint = "./checkpoint/%s/" %(experiment_name)
    if not os.path.exists(path_checkpoint):
        os.makedirs(path_checkpoint)
    if rank == 0:
        torch.save(model.state_dict(), path_checkpoint+'Rank%s_Epoch_%s_weights.pth' %(rank, epoch))

def select_model(num_class, args):
    if args.model == 'VGG':
        model = VGG(16, 10)
    elif args.model == 'res':
        if args.dataset == 'cifar10':
            # model = resnet.ResNet(34, num_class)
            model = ResNet18()
    elif args.model == 'VGG9':
        model = vgg9.VGG('VGG9')

    elif args.model == 'RNN':
        model = RNNModel(name='Local_Model',
                               rnn_type='LSTM', ntoken=50000,
                               ninp=args.emsize, nhid=args.nhid,
                               nlayers=args.nlayers,
                               dropout=args.dropout, tie_weights=args.tied)

    return model

def comp_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def scale_backdoor_weight(args, model, avg_model):
    current_number_of_adversaries = 1
    clip_rate = (args.num_comm_ue / current_number_of_adversaries)
    for key, value in model.state_dict().items():
        #### don't scale tied weights:
        if args.tied and key == 'decoder.weight' or '__'in key:
            continue
        target_value = avg_model.state_dict()[key]
        new_value = target_value + (value - target_value) * clip_rate

        model.state_dict()[key].copy_(new_value)

    return model


def test_backdoor(args, model, valset=None, v=0, target=1, criterion=None):

    if 'adver' in args.attack_type or 'pattern' in args.attack_type:

        f = model
        f.cuda()
        f.eval()
        with torch.no_grad():
            num_im = 0
            attack_success_num = 0.0
            fooling_rate_f = 0.0
            loss_item = 0.0
            for batch_id, (img_batch, img_label) in enumerate(valset):

                img_batch = img_batch.cuda()
                v_tensor = torch.tensor(v)
                v_tensor = v_tensor.type(torch.FloatTensor)

                per_img_batch = (img_batch + v_tensor.cuda()).cuda()

                if 'pattern' in args.attack_type:
                    per_img_batch = add_trigger_pattern(per_img_batch, distance=2, pixel_value=1)

                num_im += img_batch.size(0)
                pret_logits = f(per_img_batch)
                pert_outputs = torch.argmax(pret_logits, dim=1)

                targets_label = np.zeros((pret_logits.size(0))) + args.attack_target ### Set the target labele
                targets_label = torch.from_numpy(targets_label).long()
                targets_label = targets_label.cuda()

                loss = (F.cross_entropy(pret_logits, targets_label,
                                      reduction='none') ).mean()
                loss_item += loss.item()

                target_outputs = np.zeros((pert_outputs.size(0))) + args.attack_target
                target_outputs = torch.from_numpy(target_outputs).long()
                target_outputs = target_outputs.cuda()

                attack_success_num += torch.sum(target_outputs == pert_outputs).float().item()

            # Compute the fooling rate
            fooling_rate_f = attack_success_num / num_im
            loss_mean = loss_item/float(batch_id+1)
        return fooling_rate_f

    else:
        model.eval()
        correct = 0
        total_num = 0
        with torch.no_grad():
            for data, target in valset:
                data, target = data.cuda(), target.cuda()
                output = model(data)
                pred = output.data.max(
                    1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                total_num += len(data)

        return correct / float(total_num)

def save_acc_file(args, rank, prefix=None, acc_list=None):
    path_checkpoint = "./results/%s" %(args.experiment_name)
    if not os.path.exists(path_checkpoint):
        os.makedirs(path_checkpoint)
    filename = "%s/%s.txt" %(path_checkpoint, prefix)
    if filename:
        with open(filename, 'w') as f:
            json.dump(acc_list, f)

def save_perturbation(args, perturbation_for_backdoor):
    path_results = "./results/%s" %(args.experiment_name)
    if not os.path.exists(path_results):
        try:
            os.makedirs(path_results)
        except:
            pass
    scio.savemat('%s/Perturbation.mat' %(path_results), {'Perturbation':perturbation_for_backdoor})

def load_perturbation(args):
    path_results = "./results/%s" %(args.experiment_name)
    m = loadmat('%s/Perturbation.mat' %(path_results))
    perturbation_for_backdoor = m['Perturbation']
    return perturbation_for_backdoor

def weights_scale(attack_w, target_model_w, scale):
    # new_value = target_value + (value - target_value) * clip_rate
    w = copy.deepcopy(attack_w)
    for key in w.keys():
        w[key] = target_model_w[key] + (attack_w[key] -  target_model_w[key]) * scale # Get scale weights
    return w

def weights_print(rank, model_weights):
    # new_value = target_value + (value - target_value) * clip_rate
    w = copy.deepcopy(model_weights)
    for key in w.keys():
        print(rank, key, w[key].mean())
    return w

def model_dist_norm(model, target_params):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
        return math.sqrt(squared_sum)

def compute_spectral_evasion_loss(Params,
                                  model,
                                  fixed_model,
                                  inputs,
                                  grads=None):
    """
    Evades spectral analysis defense. Aims to preserve the latent representation
    on non-backdoored inputs. Uses a checkpoint non-backdoored `fixed_model` to
    compare the outputs. Uses euclidean distance as penalty.


    :param params: training parameters
    :param model: current model
    :param fixed_model: saved non-backdoored model as a reference.
    :param inputs: training data inputs
    :param grads: compute gradients.

    :return:
    """

    if not fixed_model:
        return torch.tensor(0.0), None
    # t = time.perf_counter()
    with torch.no_grad():
        _, fixed_latent = fixed_model(inputs, latent=True)
    _, latent = model(inputs, latent=True)
    # record_time(params, t, 'latent_fixed')

    loss = torch.norm(latent - fixed_latent, dim=1).mean()
    # if params.spectral_similarity == 'norm':
    #     loss = torch.norm(latent - fixed_latent, dim=1).mean()
    # elif params.spectral_similarity == 'cosine':
    #     loss = -torch.cosine_similarity(latent, fixed_latent).mean() + 1
    # else:
    #     raise ValueError(f'Specify correct similarity metric for '
    #                      f'spectral evasion: [norm, cosine].')
    # if grads:
    #     grads = get_grads(params, model, loss)

    return loss, grads

def NDC(args, model, average_model_weights, user_id, s_norm):
    target_params_variables = dict()
    for name, param in model.named_parameters():
        target_params_variables[name] = average_model_weights[name].clone().detach().requires_grad_(False)

    model_norm = model_dist_norm(model, target_params_variables)
    # print('model_norm',model_norm)

    if user_id == args.attacker_user_id:
        print('attacker model_norm',model_norm)
    #     s_norm = 5.0
    # else:
    #     s_norm = args.s_norm

    if model_norm > s_norm:
        # print(f'The model_norm of user {user_id} is {model_norm}. Clipping ....')
        norm_scale = s_norm / ((model_norm))
        for name, param in model.state_dict().items():
            clipped_difference = norm_scale * (
            param.data -average_model_weights[name])
            param.data.copy_(
                average_model_weights[name] + clipped_difference)
        model_norm = model_dist_norm(model, target_params_variables)
        print('post model_norm',model_norm)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Meter(object):
    """ Computes and stores the average, variance, and current value """

    def __init__(self, init_dict=None, ptag='Time', stateful=False,
                 csv_format=True):
        """
        :param init_dict: Dictionary to initialize meter values
        :param ptag: Print tag used in __str__() to identify meter
        :param stateful: Whether to store value history and compute MAD
        """
        self.reset()
        self.ptag = ptag
        self.value_history = None
        self.stateful = stateful
        if self.stateful:
            self.value_history = []
        self.csv_format = csv_format
        if init_dict is not None:
            for key in init_dict:
                try:
                    # TODO: add type checking to init_dict values
                    self.__dict__[key] = init_dict[key]
                except Exception:
                    print('(Warning) Invalid key {} in init_dict'.format(key))

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.std = 0
        self.sqsum = 0
        self.mad = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sqsum += (val ** 2) * n
        if self.count > 1:
            self.std = ((self.sqsum - (self.sum ** 2) / self.count)
                        / (self.count - 1)
                        ) ** 0.5
        if self.stateful:
            self.value_history.append(val)
            mad = 0
            for v in self.value_history:
                mad += abs(v - self.avg)
            self.mad = mad / len(self.value_history)

    def __str__(self):
        if self.csv_format:
            if self.stateful:
                return str('{dm.val:.3f},{dm.avg:.3f},{dm.mad:.3f}'
                           .format(dm=self))
            else:
                return str('{dm.val:.3f},{dm.avg:.3f},{dm.std:.3f}'
                           .format(dm=self))
        else:
            if self.stateful:
                return str(self.ptag) + \
                       str(': {dm.val:.3f} ({dm.avg:.3f} +- {dm.mad:.3f})'
                           .format(dm=self))
            else:
                return str(self.ptag) + \
                       str(': {dm.val:.3f} ({dm.avg:.3f} +- {dm.std:.3f})'
                           .format(dm=self))
