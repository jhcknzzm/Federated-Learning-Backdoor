import argparse
import json
import datetime
import os
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import json
from torchvision import transforms
from image_helper import ImageHelper
from text_helper import TextHelper
from utils.utils import dict_html
from torch.autograd.gradcheck import zero_gradients
logger = logging.getLogger("logger")
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import time
import numpy as np
import copy
import random
from utils.text_load import *
from text_helper import PGD

criterion = torch.nn.CrossEntropyLoss()

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
    assert params['partipant_population'] < 80000
    assert params['partipant_sample_size'] < params['partipant_population']
    assert params['number_of_adversaries'] < params['partipant_sample_size']

def get_embedding_weight_from_LSTM(model):
    embedding_weight = model.return_embedding_matrix()
    return embedding_weight


def train(helper, epoch, sampled_participants):
    ### Accumulate weights for all participants.
    weight_accumulator = dict()
    for name, data in helper.target_model.state_dict().items():
        #### don't scale tied weights:
        if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
            continue
        weight_accumulator[name] = torch.zeros_like(data)

    ### This is for calculating distances
    target_params_variables = dict()
    for name, param in helper.target_model.named_parameters():
        target_params_variables[name] = helper.target_model.state_dict()[name].clone().detach().requires_grad_(False)

    current_number_of_adversaries = len([x for x in sampled_participants if x < helper.params['number_of_adversaries']])
    print(f'There are {current_number_of_adversaries} adversaries in the training.')

    for participant_id in sampled_participants:
        model = helper.local_model
        model.copy_params(helper.target_model.state_dict())
        model.train()

        start_time = time.time()
        hidden = model.init_hidden(helper.params['batch_size'])

        trained_posioned_model_weights = None

        if helper.params['is_poison'] and participant_id in helper.params['adversary_list'] and trained_posioned_model_weights is None:
            print('Prepare data for attackers')
            # Clean data removed
            poisoned_data = helper.poisoned_data_for_train
            print('poisoned data size:',poisoned_data.size())
            print('P o i s o n - n o w ! ----------')
            print('Test the global model the attacker received from the server')
            print('Acc. Report. ---------- Start ----------')
            _, acc_p = test_poison(helper=helper, epoch=epoch,
                                   data_source=helper.test_data_poison,
                                   model=model)

            _, acc_initial = test(helper=helper, epoch=epoch, data_source=helper.test_data, model=model)
            print('Backdoor Acc. =',acc_p)
            print('Main Task Acc. =',acc_initial)
            print('Acc. Report. ----------- END -----------')

            poison_optimizer = torch.optim.SGD(model.parameters(), lr= helper.params['poison_lr'],
                                               momentum=helper.params['momentum'],
                                               weight_decay=helper.params['decay'])
            scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                             milestones=[0.2 * helper.params['retrain_poison'],
                                                                         0.8 * helper.params['retrain_poison']],
                                                             gamma=0.1)

            try:
                # gat gradient mask use global model and clearn data
                if helper.params['grad_mask']:
                    # Sample some benign data
                    for i, sampled_data_idx in enumerate(random.sample(range(80000), 30)):
                        if i == 0:
                            sampled_data = helper.train_data[sampled_data_idx]
                        else:
                            sampled_data = torch.cat((sampled_data, helper.train_data[sampled_data_idx]))
                    mask_grad_list = helper.grad_mask(helper, helper.target_model, sampled_data, criterion)

                es = 0
                for internal_epoch in range(1, helper.params['retrain_poison']*10 + 1):
                    print('Backdoor training. Internal_epoch', internal_epoch)
                    data_iterator = range(0, poisoned_data.size(0) - 1, helper.params['bptt'])
                    print(f"PARAMS: {helper.params['retrain_poison']} epoch: {internal_epoch},"
                                f" lr: {scheduler.get_lr()}")

                    total_train_loss = 0.0
                    num_train_data = 0.0
                    for batch in data_iterator:
                        data, targets = helper.get_batch(poisoned_data, batch)
                        if data.size(0) != helper.params['bptt']:
                            continue
                        # print('************************')
                        # print(data[:,0])
                        # print(data[:,-1])
                        # print(targets[-helper.params['batch_size']:])

                        poison_optimizer.zero_grad()
                        hidden = helper.repackage_hidden(hidden)
                        output, hidden = model(data, hidden)

                        if helper.params['all_token_loss']:
                            loss = criterion(output.view(-1, helper.n_tokens), targets)
                        else:
                            loss = criterion(output[-1:].view(-1, helper.n_tokens),
                                                   targets[-helper.params['batch_size']:])
                        loss.backward(retain_graph=True)
                        total_train_loss += loss.data.item()
                        num_train_data += helper.params['batch_size']

                        if helper.params['grad_mask']:
                            mask_grad_list_copy = iter(mask_grad_list)
                            for name, parms in model.named_parameters():
                                if parms.requires_grad:
                                    parms.grad = parms.grad * next(mask_grad_list_copy)

                        poison_optimizer.step()

                        # global - g*lr
                        # global - (global - g*lr)  =  g * lr
                        # g * lr / n
                        # global - g * lr / n
                        if helper.params['PGD']:
                            weight_difference, difference_flat = helper.get_weight_difference(target_params_variables, model.named_parameters())
                            clipped_weight_difference, l2_norm = helper.clip_grad(helper.params['s_norm'], weight_difference, difference_flat)
                            weight_difference, difference_flat = helper.get_weight_difference(target_params_variables, clipped_weight_difference)
                            model.copy_params(weight_difference)

                    print('Total train loss',total_train_loss/float(num_train_data))
                    # get the test acc of the main task with the trained attacker
                    loss_main, acc_main = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                                     model=model)

                    # get the test acc of the target test data with the trained attacker
                    loss_p, acc_p = test_poison(helper=helper, epoch=internal_epoch,
                                            data_source=helper.test_data_poison,
                                            model=model, Top5=args.Top5)

                    print('Target Tirgger Loss and Acc. :', loss_p, acc_p)
                    weight_difference, difference_flat = helper.get_weight_difference(target_params_variables, model.named_parameters())
                    clipped_weight_difference, l2_norm = helper.clip_grad(helper.params['s_norm'], weight_difference, difference_flat)

                    print("l2 norm of attacker's: ", l2_norm)
                    StopBackdoorTraining = False
                    if acc_p >= (helper.params['poison_epochs'].index(epoch) + 1) / len(helper.params['poison_epochs']) * 100.0:
                        StopBackdoorTraining = True
                        tmp_acc = (helper.params['poison_epochs'].index(epoch) + 1) / len(helper.params['poison_epochs']) * 100.0
                        print(f'Got the preset traget backdoor acc {acc_p} >= {tmp_acc}')

                    elif l2_norm >= helper.params['s_norm'] and internal_epoch >= helper.params['retrain_poison']:
                        StopBackdoorTraining = True
                        print(f'l2_norm = {l2_norm} and internal_epoch = {internal_epoch}')
                    elif acc_initial - acc_main > 1.0:
                        StopBackdoorTraining = True
                        print(f'acc drop {acc_initial - acc_main}')

                    ####### Early stopping
                    if loss_p < helper.params['min_loss_p']:
                        print('current min_loss_p = ',helper.params['min_loss_p'])
                        helper.params['min_loss_p'] = loss_p
                        es = 0
                    else:
                        es += 1
                        print("Counter {} of 5".format(es))

                        if es > 4:
                            print("Early stopping with loss_p: ", loss_p, "and acc_p for this epoch: ", acc_p, "...")
                            StopBackdoorTraining = True

                    # if acc_p >= helper.params['traget_poison_acc'][helper.params['attack_num']]:
                    if StopBackdoorTraining:
                    # if loss_p <= threshold or acc_initial - acc_main>1.0:
                        print('Backdoor training over. ')
                        raise ValueError()
            # else:
            except ValueError:
                print('Converged earlier')
                helper.params['attack_num'] += 1


            # Server perform clipping
            if helper.params['diff_privacy']:
                weight_difference, difference_flat = helper.get_weight_difference(target_params_variables, model.named_parameters())
                clipped_weight_difference, _ = helper.clip_grad(helper.params['s_norm'], weight_difference, difference_flat)
                weight_difference, difference_flat = helper.get_weight_difference(target_params_variables, clipped_weight_difference)
                model.copy_params(weight_difference)


            trained_posioned_model_weights = model.named_parameters()

        # Only one attacker trains. The other attackrs just copy the trained model
        elif helper.params['is_poison'] and participant_id in helper.params['adversary_list']:
            model.copy_params(trained_posioned_model_weights)

        else:
            ### we will load helper.params later
            optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'],
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])

            for internal_epoch in range(1, helper.params['retrain_no_times'] + 1):
                total_loss = 0.0
                data_iterator = range(0, helper.train_data[participant_id].size(0) - 1, helper.params['bptt'])

                for batch in data_iterator:
                    optimizer.zero_grad()
                    data, targets = helper.get_batch(helper.train_data[participant_id], batch)

                    hidden = helper.repackage_hidden(hidden)
                    output, hidden = model(data, hidden)
                    loss = criterion(output.view(-1, helper.n_tokens), targets)
                    loss.backward()
                    optimizer.step()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), helper.params['clip'])
                    # if helper.params['diff_privacy']:
                    #     weight_difference = helper.get_weight_difference(target_params_variables, model.named_parameters())
                    #     clipped_weight_difference, l2_norm = helper.clip_grad(helper.params['s_norm'], weight_difference)
                    #     weight_difference = helper.get_weight_difference(target_params_variables, clipped_weight_difference)
                    #     model.copy_params(weight_difference)
                    #     del clipped_weight_difference
                    #     del weight_difference

                    total_loss += loss.data

                    if helper.params["report_train_loss"] and batch % helper.params[
                        'log_interval'] == 0 and batch > 0:
                        cur_loss = total_loss.item() / helper.params['log_interval']
                        elapsed = time.time() - start_time
                        print('model {} | epoch {:3d} | internal_epoch {:3d} '
                                    '| {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                                    'loss {:5.2f} | ppl {:8.2f}'
                                            .format(participant_id, epoch, internal_epoch,
                                            batch,helper.train_data[participant_id].size(0) // helper.params['bptt'],
                                            helper.params['lr'],
                                            elapsed * 1000 / helper.params['log_interval'],
                                            cur_loss,
                                            math.exp(cur_loss) if cur_loss < 30 else -1.))
                        total_loss = 0
                        start_time = time.time()


            if helper.params['diff_privacy']:
                weight_difference, difference_flat = helper.get_weight_difference(target_params_variables, model.named_parameters())
                clipped_weight_difference, l2_norm = helper.clip_grad(helper.params['s_norm'], weight_difference, difference_flat)
                weight_difference, difference_flat = helper.get_weight_difference(target_params_variables, clipped_weight_difference)
                model.copy_params(weight_difference)
                print("l2 norm of benign user in last epoch: ", l2_norm)

        for name, data in model.state_dict().items():
            #### don't scale tied weights:
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name].add_(data - helper.target_model.state_dict()[name])

    return weight_accumulator


def test(helper, epoch, data_source, model):
    model.eval()
    total_loss = 0
    correct = 0
    total_test_words = 0
    if helper.params['type'] == 'text':
        hidden = model.init_hidden(helper.params['test_batch_size'])
        random_print_output_batch = \
        random.sample(range(0, (data_source.size(0) // helper.params['bptt']) - 1), 1)[0]
        data_iterator = range(0, data_source.size(0)-1, helper.params['bptt'])
        dataset_size = len(data_source)
    else:
        dataset_size = len(data_source.dataset)
        data_iterator = data_source

    with torch.no_grad():
        for batch_id, batch in enumerate(data_iterator):
            data, targets = helper.get_batch(data_source, batch)
            if helper.params['type'] == 'text':

                output, hidden = model(data, hidden)
                output_flat = output.view(-1, helper.n_tokens)
                ##### Debug: show output_flat
                total_loss += len(data) * criterion(output_flat, targets).data
                hidden = helper.repackage_hidden(hidden)
                pred = output_flat.data.max(1)[1]
                correct += pred.eq(targets.data).sum().to(dtype=torch.float)
                total_test_words += targets.data.shape[0]
                ### output random result :)
                if batch_id == random_print_output_batch * helper.params['bptt'] and \
                        helper.params['output_examples'] and epoch % 5 == 0:
                    expected_sentence = helper.get_sentence(targets.data.view_as(data)[:, 0])
                    expected_sentence = f'*EXPECTED*: {expected_sentence}'
                    predicted_sentence = helper.get_sentence(pred.view_as(data)[:, 0])
                    predicted_sentence = f'*PREDICTED*: {predicted_sentence}'
                    score = 100. * pred.eq(targets.data).sum() / targets.data.shape[0]
                    print(expected_sentence)
                    print(predicted_sentence)
            else:
                output = model(data)
                total_loss += nn.functional.cross_entropy(output, targets,
                                                  reduction='sum').item() # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    if helper.params['type'] == 'text':
        acc = 100.0 * (correct / total_test_words)
        total_l = total_loss.item() / (dataset_size-1)
        print('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                    'Accuracy: {}/{} ({:.4f}%)'.format(model.name, False, epoch,
                                                       total_l, correct, total_test_words,
                                                       acc))
        acc = acc.item()
        # total_l = total_l.item()
    else:
        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        print('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                    'Accuracy: {}/{} ({:.4f}%)'.format(model.name, False, epoch,
                                                       total_l, correct, dataset_size,
                                                       acc))

    model.train()
    return (total_l, acc)


def test_poison(helper, epoch, data_source,
                model, Top5=False):
    model.eval()
    total_loss = 0.0
    correct = 0.0
    total_test_words = 0.0
    batch_size = helper.params['test_batch_size']
    if helper.params['type'] == 'text':
        hidden = model.init_hidden(batch_size)
        data_iterator = range(0, data_source.size(0) - 1, helper.params['bptt'])
        dataset_size = len(data_source)
    else:
        data_iterator = data_source
        dataset_size = 1000

    with torch.no_grad():
        for batch_id, batch in enumerate(data_iterator):
            if helper.params['type'] == 'image':

                for pos in range(len(batch[0])):
                    batch[0][pos] = helper.train_dataset[random.choice(helper.params['poison_images_test'])][0]

                    batch[1][pos] = helper.params['poison_label_swap']


            data, targets = helper.get_batch(data_source, batch)


            if helper.params['type'] == 'text':

                output, hidden = model(data, hidden)
                # print('* test ***********')
                # print(data[:,0])
                # print(data[:,-1])
                # print(targets[-batch_size:])

                output_flat = output.view(-1, helper.n_tokens)

                total_loss += 1 * criterion(output_flat[-batch_size:], targets[-batch_size:]).data


                if Top5:
                    _, pred = output_flat.data[-batch_size:].topk(5, 1, True, True)
                    correct_output = targets.data[-batch_size:]
                    correct_output = pred.eq(correct_output.view(-1, 1).expand_as(pred))
                    res = []

                    correct_k = correct_output.sum()
                    correct += correct_k

                else:
                    pred = output_flat.data.max(1)[1][-batch_size:]
                    # print('traget_labeled',helper.params['traget_labeled'])
                    if len(helper.params['traget_labeled']) == 0:
                        # print('Not semantic_target test')
                        correct_output = targets.data[-batch_size:]
                        correct += pred.eq(correct_output).sum()
                    else:
                        # print('Semantic_target test')
                        for traget_id in set(helper.params['traget_labeled']):
                            tmp = torch.ones_like(targets.data[-batch_size:])*traget_id
                            correct_output = tmp.cuda()
                            correct += pred.eq(correct_output).sum()

                total_test_words += batch_size

            else:

                output = model(data)
                total_loss += nn.functional.cross_entropy(output, targets,
                                                  reduction='sum').data.item()  # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().to(dtype=torch.float)


    if helper.params['type'] == 'text':
        acc = 100.0 * (float(correct.item()) / float(total_test_words))
        total_l = total_loss.item() / dataset_size
    else:
        acc = 100.0 * (correct / dataset_size)
        total_l = total_loss / dataset_size
    print('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                'Accuracy: {}/{} ({:.0f}%)'.format(model.name, True, epoch,
                                                   total_l, correct, dataset_size,
                                                   acc))

    model.train()
    return total_l, acc

def save_acc_file(prefix=None,acc_list=None,sentence=None,new_folder_name=None):
    if new_folder_name is None:
        # path_checkpoint = f'./results_update_DuelTrigger_PGD/{sentence}'
        path_checkpoint = os.path.expanduser(f'~/zhengming/results_update_PGD_v1/{sentence}')
    else:
        # path_checkpoint = f'./results_update_DuelTrigger_PGD/{new_folder_name}/{sentence}'
        path_checkpoint = os.path.expanduser(f'~/zhengming/results_update_PGD_v1/{new_folder_name}/{sentence}')

    if not os.path.exists(path_checkpoint):
        os.makedirs(path_checkpoint)
    filename = "%s/%s.txt" %(path_checkpoint, prefix)
    if filename:
        with open(filename, 'w') as f:
            json.dump(acc_list, f)

def save_model(prefix=None, helper=None, epoch=None, new_folder_name=None):
    if new_folder_name is None:
        path_checkpoint = f"./target_model_checkpoint/{prefix}/"
    else:
        path_checkpoint = f"./target_model_checkpoint/{new_folder_name}/{prefix}/"
    if not os.path.exists(path_checkpoint):
        os.makedirs(path_checkpoint)
    torch.save(helper.target_model.state_dict(), path_checkpoint+f"model_epoch_{epoch}.pth")


if __name__ == '__main__':
    ## python training_adver_update.py --run_slurm 1 --sentence_id_list 0 --start_epoch 2001 --num_middle_token_same_structure 10
    ## srun -N 1 -n 1  --nodelist=bombe --gres=gpu:1 python training_adver_update.py --run_slurm 1 --sentence_id_list 0 --start_epoch 2001 --num_middle_token_same_structure 10
    ## >~/zhengming/Sentence1_Duel1_GradMask1_PGD1_AttackAllLayer0_Ripple0_AllTokenLoss1.log 2>~/zhengming/Sentence1_Duel1_GradMask1_PGD1_AttackAllLayer0_Ripple0_AllTokenLoss1.err &
    print('Start training')

    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', default='utils/words_zzm.yaml', dest='params')
    parser.add_argument('--GPU_id',
                        default="0",
                        type=str,
                        help='GPU_id')

    parser.add_argument('--new_folder_name',
                        default=None,
                        type=str,
                        help='new_folder_name')

    parser.add_argument('--save_epoch',
                        default=100,
                        type=int,
                        help='save_epoch')

    parser.add_argument('--poison_lr',
                        default=0.1,
                        type=float,
                        help='attacker learning rate')


    parser.add_argument('--grad_mask',
                        default=1,
                        type=int,
                        help='grad_mask')

    parser.add_argument('--Top5',
                        default=0,
                        type=int,
                        help='Top5')

    parser.add_argument('--start_epoch',
                        default=2001,
                        type=int,
                        help='Load pre-trained benign model that has been trained for start_epoch - 1 epoches, and resume from here')



    parser.add_argument('--all_token_loss',
                        default=1,
                        type=int,
                        help='all_token_loss')

    parser.add_argument('--attack_all_layer',
                        default=0,
                        type=int,
                        help='attack_all_layer')

    parser.add_argument('--run_slurm',
                        default=0,
                        type=int,
                        help='run_slurm')

    parser.add_argument('--same_structure',
                        default=True,
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

    parser.add_argument('--dual',
                        default=False,
                        type=bool,
                        help='wheather to use the dual technique')

    parser.add_argument('--attack_num',
                        default=10,
                        type=int,
                        help='attack_num 10, 20, 30')

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
    if len(args.sentence_id_list) == 1:
        params_loaded['sentence_id_list'] = args.sentence_id_list[0]
    else:
        params_loaded['sentence_id_list'] = args.sentence_id_list

    params_loaded['end_epoch'] = args.start_epoch + 300

    if os.path.isdir('/data/yyaoqing/backdoor_NLP_data/'):
        params_loaded['data_folder'] = '/data/yyaoqing/backdoor_NLP_data/'

    # Check parameters
    check_params(params_loaded)

    # Load the helper object
    if params_loaded['type'] == "image":
        helper = ImageHelper(params=params_loaded)
    else:
        helper = TextHelper(params=params_loaded)

    helper.create_model()
    helper.load_benign_data()
    helper.load_attacker_data()

    ### hard code

    helper.params['poison_epochs'] = np.arange(args.start_epoch+1, args.start_epoch+1+args.attack_num).tolist()

    # if args.attack_freq_type == 'consecutive_attack':
    #     helper.params['poison_epochs'] = np.arange(args.start_epoch+1, args.start_epoch+51).tolist()
    # elif args.attack_freq_type == 'uniformly_attack':
    #     interval = 5
    #     helper.params['poison_epochs'] = np.arange(args.start_epoch+1, args.start_epoch+1+50,interval).tolist()
    # elif args.attack_freq_type == 'random_attack':
    #     attack_epoch_tmp = random.sample(list(range(args.start_epoch+1, args.start_epoch+1+100)), 10)
    #     helper.params['poison_epochs'] = attack_epoch_tmp.sort()

    print('attack epochs are:',helper.params['poison_epochs'])
    # helper.params['traget_poison_acc'] = list(range(10,101,len(helper.params['poison_epochs'])))

    weight_accumulator = None
    backdoor_acc = []
    backdoor_loss = []
    benign_acc = []



    for epoch in range(helper.params['start_epoch'], helper.params['end_epoch'] + 1):
        #### Reset init. min_loss_p
        helper.params['min_loss_p'] = 100000.0

        start_time = time.time()

        """
        Sample participants.
        Note range(0, self.params['number_of_adversaries'])/self.params['adversary_list'] are attacker ids.
        """

        # Randomly sample participants at each round. The attacker can appear at any round.
        if helper.params["random_compromise"]:
            sampled_participants = random.sample(range(helper.params['partipant_population']), helper.params['partipant_sample_size'])

        ## Only sample non-poisoned participants until poisoned_epoch
        else:
            if epoch in helper.params['poison_epochs']:
               sampled_participants = helper.params['adversary_list'] \
                                        + random.sample(range(helper.params['number_of_adversaries'], helper.params['partipant_population'])
                                        , helper.params['partipant_sample_size'] - helper.params['number_of_adversaries'])

            else:
                sampled_participants = random.sample(range(helper.params['number_of_adversaries'], helper.params['partipant_population'])
                                        , helper.params['partipant_sample_size'])

        print(f'Selected models: {sampled_participants}')

        t = time.time()
        weight_accumulator = train(helper, epoch, sampled_participants)



        print(f'time spent on training: {time.time() - t}')
        # Average the models
        helper.average_shrink_models(target_model=helper.target_model,
                                     weight_accumulator=weight_accumulator, epoch=epoch)

        epochs_paprmeter = helper.params['end_epoch']
        poison_epochs_paprmeter = helper.params['poison_epochs'][0]
        partipant_sample_size = helper.params['partipant_sample_size']
        len_poison_sentences = len(helper.params['poison_sentences'])

        dir_name = helper.params['sentence_name']+f"Duel{args.dual}_GradMask{helper.params['grad_mask']}_PGD{args.PGD}_DP{args.diff_privacy}_SNorm{args.s_norm}_SemanticTarget{args.semantic_target}_AllTokenLoss{args.all_token_loss}_AttacktNum{args.attack_num}"
        print(dir_name)

        if helper.params['is_poison']:
            # if epoch%args.save_epoch == 0 or epoch==1 or epoch in helper.params['poison_epochs'] or epoch-1 in helper.params['poison_epochs'] or epoch-2 in helper.params['poison_epochs']:
            #     num_layers = helper.params['nlayers']
            #     prefix = f'RNN{num_layers}_'+ f'_target_epochs{epochs_paprmeter}_poison_epochs{poison_epochs_paprmeter}_partipant_sample_size{partipant_sample_size}_lenS{len_poison_sentences}'
            #     # save_model(prefix=dir_name, helper=helper, epoch=epoch, new_folder_name=args.new_folder_name)



            epoch_loss_p, epoch_acc_p = test_poison(helper=helper,
                                                    epoch=epoch,
                                                    data_source=helper.test_data_poison,
                                                    model=helper.target_model,
                                                    Top5=args.Top5)



            backdoor_acc.append(epoch_acc_p)
            backdoor_loss.append(epoch_loss_p)
            save_acc_file(prefix=helper.params['sentence_name']+f'_target_epochs{epochs_paprmeter}_poison_epochs{poison_epochs_paprmeter}_partipant_sample_size{partipant_sample_size}_lenS{len_poison_sentences}_GPU{args.GPU_id}', acc_list=backdoor_acc,
            sentence=dir_name, new_folder_name=args.new_folder_name)
            save_acc_file(prefix=helper.params['sentence_name']+f'Backdoor_Loss_main_epochs{epochs_paprmeter}_poison_epochs{poison_epochs_paprmeter}_partipant_sample_size{partipant_sample_size}_lenS{len_poison_sentences}_GPU{args.GPU_id}', acc_list=backdoor_loss, sentence=dir_name, new_folder_name=args.new_folder_name)

        epoch_loss, epoch_acc = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                                     model=helper.target_model)
        benign_acc.append(epoch_acc)
        #### save backdoor acc
        save_acc_file(prefix=helper.params['sentence_name']+f'_main_epochs{epochs_paprmeter}_poison_epochs{poison_epochs_paprmeter}_partipant_sample_size{partipant_sample_size}_lenS{len_poison_sentences}_GPU{args.GPU_id}', acc_list=benign_acc,
        sentence=dir_name, new_folder_name=args.new_folder_name)

        print(f'Done in {time.time()-start_time} sec.')


    # if helper.params.get('results_json', False):
    #     with open(helper.params['results_json'], 'a') as f:
    #         if len(mean_acc):
    #             results['mean_poison'] = np.mean(mean_acc)
    #         f.write(json.dumps(results) + '\n')
