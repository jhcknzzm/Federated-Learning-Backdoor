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

import os

# from image_helper import ImageHelper
from image_helper import ImageHelper

# from torch.autograd.gradcheck import zero_gradients
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
import wandb
#from train_funcs.train_sentiment import train_sentiment
from train_funcs import train, train_cv
from test_funcs import test_reddit_lstm, test_sentiment, test_reddit_gpt2, test_cv, test_poison_cv

torch.manual_seed(1)
torch.cuda.manual_seed(1)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled=False

random.seed(0)
np.random.seed(0)

def check_params(params):
    """
    Perform some basic checks on the parameters.
    """
    assert params['partipant_sample_size'] <= params['participant_population']
    assert params['number_of_adversaries'] <= params['partipant_sample_size']

def save_acc_file(file_name=None, acc_list=None, new_folder_name=None):
    if new_folder_name is None:
        path = "."
        # path_checkpoint = os.path.expanduser(f'~/zhengming/results_update_PGD_v1/{sentence}')
    else:
        path = f'./{new_folder_name}'
        if not os.path.exists(path):
            os.mkdir(path)
        # path_checkpoint = os.path.expanduser(f'~/zhengming/results_update_PGD_v1/{new_folder_name}/{sentence}')

    filename = "%s/%s.txt" %(path, file_name)
    if filename:
        with open(filename, 'w') as f:
            json.dump(acc_list, f)

def save_model(file_name=None, helper=None, epoch=None, new_folder_name=None):
    if new_folder_name is None:
        path = '.'
    else:
        path = f'./{new_folder_name}'
        if not os.path.exists(path):
            os.mkdir(path)
    filename = "%s/%s_model_epoch_%s.pth" %(path, file_name, epoch)
    torch.save(helper.target_model.state_dict(), filename)

if __name__ == '__main__':
    # python main_training.py --run_slurm 0  --start_epoch 1 --diff_privacy True
    # python main_training.py --run_slurm 0  --start_epoch 11 --is_poison True --diff_privacy True
    ## python training_adver_update.py --run_slurm 1 --sentence_id_list 0 --start_epoch 2001 --num_middle_token_same_structure 10
    ## srun -N 1 -n 1  --nodelist=bombe --gres=gpu:1 python training_adver_update.py --run_slurm 1 --sentence_id_list 0 --start_epoch 2001 --num_middle_token_same_structure 10
    ## >~/zhengming/Sentence1_Duel1_GradMask1_PGD1_AttackAllLayer0_Ripple0_AllTokenLoss1.log 2>~/zhengming/Sentence1_Duel1_GradMask1_PGD1_AttackAllLayer0_Ripple0_AllTokenLoss1.err &
    ## python main_training.py --run_slurm 0 --sentence_id_list 0 --start_epoch 0 --params utils/words_IMDB.yaml --GPU_id 1 --is_poison True --lr=0.001
    ## ython main_training.py --run_slurm 0 --sentence_id_list 0 --start_epoch 100 --params utils/words_IMDB.yaml --GPU_id 1 --is_poison True --lr=0.001 --poison_lr 1 --diff_privacy True --s_norm 4 --PGD 1 --gradmask_ratio 0.95 --aggregate_all_layer 0
    print('Start training ------')

    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', default='utils/cifar10_params.yaml', dest='params')
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
                        default=1,
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
                        default=10,
                        type=int,
                        help='attack_num 10, 20, 30')

    parser.add_argument('--gradmask_ratio',
                        default=1,
                        type=float,
                        help='The proportion of the gradient retained in GradMask')

    parser.add_argument('--sentence_id_list', nargs='+', type=int)
    args = parser.parse_args()

    # Setup Visible GPU
    if args.run_slurm:
        pass
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_id

    # Load yaml file
    with open(f'./{args.params}', 'r') as f:
        params_loaded = yaml.load(f, Loader=Loader)

    # Add additional fields to the loaded params based on args
    params_loaded.update(vars(args))

    if params_loaded['model'] == 'resnet':
        if params_loaded['dataset'] == 'cifar10':
            if os.path.isdir('./data/cifar10/'):
                params_loaded['data_folder'] = './data/cifar10'
            params_loaded['participant_clearn_data'] = random.sample( \
                range(params_loaded['participant_population'])[1:], 30 )
            if params_loaded['is_poison']:
                params_loaded['end_epoch'] = args.start_epoch + 40
            else:
                params_loaded['end_epoch'] = 800
        else:
            raise ValueError('Unrecognized dataset')
    else:
        raise ValueError('Unrecognized model')

    # Check parameters
    check_params(params_loaded)

    # Load the helper object
    helper = ImageHelper(params=params_loaded)
    helper.create_model_cv()
    helper.load_data_cv()
    helper.load_benign_data_cv()
    helper.load_poison_data_cv()

    if helper.params['is_poison']:
        helper.params['poison_epochs'] = list(range(helper.params['start_epoch'], helper.params['start_epoch'] + args.attack_num))
    else:
        helper.params['poison_epochs'] = []

    print('start_epoch=',helper.params['start_epoch'])
    print('attack epochs are:',helper.params['poison_epochs'])


    if helper.params['dataset'] == 'cifar10':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("unkown dataset")

    weight_accumulator = None
    backdoor_acc = []
    backdoor_loss = []
    benign_acc = []
    benign_loss = []

    print('start_epoch=',helper.params['start_epoch'])
    dataset_name = helper.params['dataset']
    model_name = helper.params['model']

    # wandb = None

    if args.is_poison:
        if args.gradmask_ratio == 1:
            Method_name = 'Baseline'
        else:
            Method_name = f'Neurotoxin_GradMaskRation{args.gradmask_ratio}'

        wandb_exper_name = f"backdoor_cv_{dataset_name}_{model_name}_snorm{args.s_norm}_{Method_name}_PLr{args.poison_lr}_AttackNum{args.attack_num}"
    else:
        wandb_exper_name = f"backdoor_cv_{dataset_name}_{model_name}_snorm{args.s_norm}_without_attack"




    if helper.params['model'] == 'resnet':
        if helper.params['run_name'] is None:
            wandb.init(entity='fl_backdoor_nlp', project=f'backdoor_cv_{dataset_name}_{model_name}', config=helper.params, name=wandb_exper_name)
        else:
            wandb.init(name=helper.params['run_name'], entity='fl_backdoor_nlp', project=f'backdoor_cv_{dataset_name}_{model_name}', config=helper.params)
            #wandb.init(name=helper.params['run_name'], entity='fl_backdoor_nlp', project=f"checkpoints", config=helper.params)

    wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release

    for epoch in range(helper.params['start_epoch'], helper.params['end_epoch']):
        #### Reset init. min_loss_p
        helper.params['min_loss_p'] = 100000.0
        start_time = time.time()

        # Randomly sample participants at each round. The attacker can appear at any round.
        if helper.params["random_compromise"]:
            sampled_participants = random.sample(range(helper.params['participant_population']), helper.params['partipant_sample_size'])

        ## Only sample non-poisoned participants until poisoned_epoch
        else:
            if epoch in helper.params['poison_epochs']:
               sampled_participants = helper.params['adversary_list'] \
                                        + random.sample(range(helper.params['benign_start_index'], helper.params['participant_population'])
                                        , helper.params['partipant_sample_size'] - helper.params['number_of_adversaries'])

            else:
                sampled_participants = random.sample(range(helper.params['benign_start_index'], helper.params['participant_population'])
                                        , helper.params['partipant_sample_size'])

        print(f'Selected models: {sampled_participants}')

        t = time.time()
        weight_accumulator = train_cv(helper, epoch, criterion, sampled_participants)

        print(f'time spent on training: {time.time() - t}')
        # Average the models
        helper.average_shrink_models(target_model=helper.target_model,
                                     weight_accumulator=weight_accumulator, epoch=epoch, wandb=wandb)

        if epoch in helper.params['save_on_epochs']:
            save_model(file_name=f'{dataset_name}_{model_name}_maskRatio{helper.params["gradmask_ratio"]}_Snorm_{args.s_norm}_checkpoint', helper=helper, epoch=epoch, new_folder_name="saved_models")

        if helper.params['is_poison']:
            partipant_sample_size = helper.params['partipant_sample_size']
            # len_poison_sentences = len(helper.params['poison_sentences'])

            if helper.params['model'] == 'resnet':

                epoch_loss_p_train, epoch_acc_p_train = test_poison_cv(helper=helper, epoch=epoch, data_source=helper.poisoned_train_data,
                                                            model=helper.target_model, is_poison=True)
                epoch_loss_p, epoch_acc_p = test_poison_cv(helper=helper, epoch=epoch, data_source=helper.poisoned_test_data,
                                                            model=helper.target_model, is_poison=True)

                wandb.log({
                           'test poison loss (after fedavg)': epoch_loss_p,
                           'test poison acc (after fedavg)': epoch_acc_p,
                           'train poison loss (after fedavg)': epoch_loss_p_train,
                           'train poison acc (after fedavg)': epoch_acc_p_train,
                           'epoch':epoch,
                           })
            else:
                raise ValueError("Unknown model")
            backdoor_acc.append(epoch_acc_p)
            backdoor_loss.append(epoch_loss_p)
            save_acc_file(file_name=wandb_exper_name, acc_list=backdoor_acc, new_folder_name="saved_backdoor_acc")
            save_acc_file(file_name=wandb_exper_name, acc_list=backdoor_loss, new_folder_name="saved_backdoor_loss")

        if helper.params['model'] == 'resnet':
            if helper.params['dataset'] == 'cifar10':
                epoch_loss, epoch_acc = test_cv(helper=helper, epoch=epoch, data_source=helper.benign_test_data,
                                                       model=helper.target_model)
        else:
            raise ValueError("Unknown model")

        wandb.log({
                    'benign test loss (after fedavg)': epoch_loss,
                    'benign test acc (after fedavg)': epoch_acc,
                    'epoch':epoch,
                    })

        benign_acc.append(epoch_acc)
        benign_loss.append(epoch_loss)
        print(f'Done in {time.time()-start_time} sec.')
        #### save backdoor acc
        save_acc_file(file_name=wandb_exper_name, acc_list=benign_loss, new_folder_name="saved_benign_loss")
        save_acc_file(file_name=wandb_exper_name, acc_list=benign_acc, new_folder_name="saved_benign_acc")
