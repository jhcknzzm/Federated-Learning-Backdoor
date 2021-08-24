import argparse
import json
import datetime
import os
import sys
import logging
from typing import Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import json
from torchvision import transforms
# from image_helper import ImageHelper
from text_helper import TextHelper
from utils.utils import dict_html
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

def update_learning_rate(args, optimizer, target_lr, epoch=1, itr=1, schedule=None, itr_per_epoch=None):

    lr = None
    if args.warmup_epoch and epoch <= 10:  # warmup to scaled lr
        if target_lr <= 0.0:
            lr = target_lr
        else:
            assert itr is not None and itr_per_epoch is not None
            count = (epoch-1) * itr_per_epoch + itr + 1
            incr = target_lr * (count / (10 * itr_per_epoch))
            lr = incr
    else:
        lr = target_lr
        for e in schedule:
            if epoch >= e:
                lr *= schedule[e]

    if lr is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

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
        if helper.params['model'] == 'LSTM':
            hidden = model.init_hidden(helper.params['batch_size'])
        elif helper.params['model'] == 'transformer':
            src_mask = model.generate_square_subsequent_mask(helper.params['bptt']).cuda()

        trained_posioned_model_weights = None

        if helper.params['is_poison'] and participant_id in helper.params['adversary_list'] and trained_posioned_model_weights is None:
            print('Prepare data for attackers')
            # Clean data removed
            poisoned_data = helper.poisoned_data_for_train
            print('poisoned data size:',poisoned_data.size())
            print('P o i s o n - n o w ! ----------')
            print('Test the global model the attacker received from the server')
            print('Acc. Report. ---------- Start ----------')

            if helper.params['task'] == 'sentiment':
                _, acc_p = test(helper, epoch, helper.test_data_poison, model, True)
            else:
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
                # get gradient mask use global model and clearn data
                if helper.params['grad_mask']:
                    subset_data_chunks = random.sample( helper.params['participant_clearn_data'], 30 )
                    sampled_data = [helper.train_data[pos] for pos in subset_data_chunks]
                    mask_grad_list = helper.grad_mask(helper, helper.target_model, sampled_data, criterion, ratio=helper.params['gradmask_ratio'])

                es = 0
                for internal_epoch in range(1, helper.params['retrain_poison']*10 + 1):
                    print('Backdoor training. Internal_epoch', internal_epoch)
                    print(f"PARAMS: {helper.params['retrain_poison']} epoch: {internal_epoch},"
                                f" lr: {scheduler.get_lr()}")

                    total_train_loss = 0.0
                    num_train_data = 0.0
                    if helper.params['task'] == 'word_predict':
                        data_iterator = range(0, poisoned_data.size(0)-1, helper.params['bptt'])
                        for batch in data_iterator:
                            data, targets = helper.get_batch(poisoned_data, batch)
                            if data.size(0) != helper.params['bptt']:
                                continue
                            poison_optimizer.zero_grad()
                            if helper.params['model'] == 'LSTM':
                                hidden = helper.repackage_hidden(hidden)
                                output, hidden = model(data, hidden)
                            elif helper.params['model'] == 'transformer':
                                output = model(data, src_mask)
                            if helper.params['all_token_loss']:
                                loss = criterion(output.view(-1, helper.n_tokens), targets)
                            else:
                                if len(helper.params['traget_labeled']) == 0:
                                    loss = criterion(output[-1:].view(-1, helper.n_tokens),
                                                        targets[-helper.params['batch_size']:])
                                else:
                                    out_tmp = output[-1:].view(-1, helper.n_tokens)
                                    preds = F.softmax(out_tmp, dim=1)
                                    preds = torch.sum(preds[:,list(set(helper.params['traget_labeled']))], dim=1)
                                    loss = -torch.mean(torch.log(preds), dim=0)
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
                    elif helper.params['task'] == 'sentiment':
                        for inputs, labels in poisoned_data:
                            inputs, labels = inputs.cuda(), labels.cuda()
                            poison_optimizer.zero_grad()
                            hidden = helper.repackage_hidden(hidden)
                            inputs = inputs.type(torch.LongTensor).cuda()
                            output, hidden = model(inputs, hidden)
                            loss = criterion(output.squeeze(), labels.float())
                            loss.backward(retain_graph=True)
                            total_train_loss += loss.data.item()
                            num_train_data += len(labels)
                            if helper.params['grad_mask']:
                                mask_grad_list_copy = iter(mask_grad_list)
                                for name, parms in model.named_parameters():
                                    if parms.requires_grad:
                                        parms.grad = parms.grad * next(mask_grad_list_copy)
                            poison_optimizer.step()
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
                    if helper.params['task'] == 'sentiment':
                        loss_p, acc_p = test(helper, internal_epoch, helper.test_data_poison, model, True)
                    else:
                        loss_p, acc_p = test_poison(helper=helper, epoch=internal_epoch,
                                                data_source=helper.test_data_poison,
                                                model=model, Top5=args.Top5)

                    print('Target Tirgger Loss and Acc. :', loss_p, acc_p)
                    if acc_p >=99.5:
                        print('success acc and loss:',acc_p, loss_p)
                    #     helper.params['backdoor_success_loss'].append(loss_p)
                    #     save_acc_file(file_name=helper.params['sentence_name']+f'backdoor_success_loss', acc_list=helper.params['backdoor_success_loss'],
                    #     new_folder_name=helper.params['dir_name'])

                    l2_norm, l2_norm_np = helper.get_l2_norm(target_params_variables, model.named_parameters())
                    print("l2 norm of attacker's (before server defense): ", l2_norm)
                    print("l2 norm of attacker's (before server defense) numpy.linalg.norm: ", l2_norm_np)

                    ### add l2 norm, loss to wandb log
                    wandb.log({'l2 norm of attacker (before server defense)': l2_norm,
                               'backdoor train loss (before fedavg)': total_train_loss/float(num_train_data),
                               'backdoor test loss (before fedavg)': loss_p,
                               'backdoor test acc (before fedavg)': acc_p,
                               })


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

                l2_norm, l2_norm_np = helper.get_l2_norm(target_params_variables, model.named_parameters())
                print("l2 norm of attacker's (after server defense): ", l2_norm.item())
                print("l2 norm of attacker's (after server defense) numpy.linalg.norm:", l2_norm_np)

                wandb.log({'l2 norm of attacker (after server defense)': l2_norm.item()})


            trained_posioned_model_weights = model.named_parameters()

        # Only one attacker trains. The other attackrs just copy the trained model
        elif helper.params['is_poison'] and participant_id in helper.params['adversary_list']:
            model.copy_params(trained_posioned_model_weights)

        else:
            ### we will load helper.params later
            optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'],
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])

            if helper.params['model'] == 'transformer':
                src_mask = model.generate_square_subsequent_mask(helper.params['bptt']).cuda()

            for internal_epoch in range(1, helper.params['retrain_no_times'] + 1):
                total_loss = 0.0

                if helper.params['task'] == 'sentiment':
                    for inputs, labels in helper.train_data[participant_id]:
                        inputs, labels = inputs.cuda(), labels.cuda()
                        optimizer.zero_grad()
                        hidden = helper.repackage_hidden(hidden)
                        inputs = inputs.type(torch.LongTensor).cuda()
                        output, hidden = model(inputs, hidden)
                        loss = criterion(output.squeeze(), labels.float())
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.data

                        if helper.params["report_train_loss"] and batch % helper.params[
                            'log_interval'] == 0:
                            cur_loss = total_loss.item() / helper.params['log_interval']
                            elapsed = time.time() - start_time
                            wandb.log({'local training lr': helper.params['lr'],
                                    'local training loss': cur_loss,
                                    })

                            total_loss = 0
                            start_time = time.time()
                elif helper.params['task'] == 'word_predict':
                    data_iterator = range(0, helper.train_data[participant_id].size(0) - 1, helper.params['bptt'])
                    model.train()
                    for batch in data_iterator:
                        optimizer.zero_grad()
                        data, targets = helper.get_batch(helper.train_data[participant_id], batch)

                        if data.size(0) != helper.params['bptt']:
                            # src_mask = model.generate_square_subsequent_mask(data.size(0)).cuda()
                            continue

                        if helper.params['model'] == 'LSTM':
                            hidden = helper.repackage_hidden(hidden)
                            output, hidden = model(data, hidden)
                        elif helper.params['model'] == 'transformer':
                            output = model(data, src_mask)

                        loss = criterion(output.view(-1, helper.n_tokens), targets)
                        loss.backward()
                        ## update lr with warmup
                        # update_learning_rate(args, optimizer, target_lr, epoch=epoch, itr=internal_epoch-1, schedule=None, itr_per_epoch=helper.params['retrain_no_times'])

                        optimizer.step()

                        total_loss += loss.data

                        if helper.params["report_train_loss"] and batch % helper.params[
                            'log_interval'] == 0 :
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
                            ### add local training loss
                            wandb.log({'local training lr': helper.params['lr'],
                                    'local training loss': cur_loss,
                                    })

                            total_loss = 0
                            start_time = time.time()
                else:
                    raise ValueError("Unknown Task")


            if helper.params['diff_privacy']:
                weight_difference, difference_flat = helper.get_weight_difference(target_params_variables, model.named_parameters())
                clipped_weight_difference, l2_norm = helper.clip_grad(helper.params['s_norm'], weight_difference, difference_flat)
                weight_difference, difference_flat = helper.get_weight_difference(target_params_variables, clipped_weight_difference)
                model.copy_params(weight_difference)
                print("l2 norm of benign user in last epoch: ", l2_norm.item())
                l2_norm, l2_norm_np = helper.get_l2_norm(target_params_variables, model.named_parameters())
                print('l2 norm of benign user (after server defense)',l2_norm.item())
                print('l2 norm of benign user (after server defense) numpy.linalg.norm',l2_norm_np)
                wandb.log({'l2 norm of benign user (after server defense)': l2_norm.item()})

        for name, data in model.state_dict().items():
            #### don't scale tied weights:
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name].add_(data - helper.target_model.state_dict()[name])

    return weight_accumulator


def test(helper, epoch, data_source, model, poisoned=False):
    model.eval()
    total_loss = 0
    correct = 0
    total_test_words = 0
    if helper.params['task'] == 'word_predict':
        if helper.params['model'] == 'LSTM':
            hidden = model.init_hidden(helper.params['test_batch_size'])
        elif helper.params['model'] == 'transformer':
            src_mask = model.generate_square_subsequent_mask(helper.params['bptt']).cuda()

        random_print_output_batch = \
        random.sample(range(0, (data_source.size(0) // helper.params['bptt']) - 1), 1)[0]
        data_iterator = range(0, data_source.size(0)-1, helper.params['bptt'])
        dataset_size = len(data_source)

        with torch.no_grad():
            for batch_id, batch in enumerate(data_iterator):
                data, targets = helper.get_batch(data_source, batch)
                if helper.params['type'] == 'text':

                    if data.size(0) != helper.params['bptt']:
                        # src_mask = model.generate_square_subsequent_mask(data.size(0)).cuda()
                        continue

                    if helper.params['model'] == 'LSTM':
                        hidden = helper.repackage_hidden(hidden)
                        output, hidden = model(data, hidden)
                    elif helper.params['model'] == 'transformer':
                        output = model(data, src_mask)

                    output_flat = output.view(-1, helper.n_tokens)
                    ##### Debug: show output_flat
                    total_loss += len(data) * criterion(output_flat, targets).data

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

        acc = round(100.0 * (correct / total_test_words), 4)
        total_l = round(total_loss.item() / (dataset_size-1), 4)
        print('___Test poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                    'Accuracy: {}/{} ({:.4f}%)'.format( False, epoch,
                                                       total_l, correct, total_test_words,
                                                       acc))
        acc = acc.item()

    elif helper.params['task'] == 'sentiment':
        data_iterator = data_source

        with torch.no_grad():
            for inputs, labels in data_iterator:
                hidden = helper.repackage_hidden(hidden)
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = inputs.type(torch.LongTensor).cuda()
                output, hidden = model(inputs, hidden)
                total_loss += criterion(output.squeeze(), labels.float())
                total_test_words += len(labels)
                correct += torch.eq(output.squeeze(), labels.float()).cpu().sum().item()
   
        acc = round(100.0 * (float(correct) / float(total_test_words)), 4)
        total_l = round(total_loss / total_test_words, 4)

        print('___Test poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                    'Accuracy: {}/{} ({:.4f}%)'.format( False, epoch,
                                                       total_l, correct, total_test_words,
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
        if helper.params['model'] == 'LSTM':
            hidden = model.init_hidden(helper.params['test_batch_size'])
        elif helper.params['model'] == 'transformer':
            src_mask = model.generate_square_subsequent_mask(helper.params['bptt']).cuda()

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

                if data.size(0) != helper.params['bptt']:
                    # src_mask = model.generate_square_subsequent_mask(data.size(0)).cuda()
                    continue

                if helper.params['model'] == 'LSTM':
                    hidden = helper.repackage_hidden(hidden)
                    output, hidden = model(data, hidden)
                elif helper.params['model'] == 'transformer':
                    output = model(data, src_mask)
                # print(data.size(),output.size())
                # yuyuyu
                # print('* test ***********')
                # print(helper.idx_to_sentence(data[:,0]))
                # print(helper.idx_to_sentence(data[:,-1]))
                # print(helper.idx_to_sentence(targets[-batch_size:]))

                output_flat = output.view(-1, helper.n_tokens)

                if len(helper.params['traget_labeled']) == 0:
                    total_loss += 1 * criterion(output_flat[-batch_size:], targets[-batch_size:]).data
                else:
                    out_tmp = output[-1:].view(-1, helper.n_tokens)
                    preds = F.softmax(out_tmp, dim=1)

                    preds = torch.sum(preds[:,list(set(helper.params['traget_labeled']))], dim=1)
                    mean_semantic_traget_loss = -torch.mean(torch.log(preds), dim=0).data

                    # mean_semantic_traget_loss = 0.0
                    # for traget_id in set(helper.params['traget_labeled']):
                    #     tmp = torch.ones_like(targets.data[-batch_size:])*traget_id
                    #     correct_output = tmp.cuda()
                    #     mean_semantic_traget_loss += 1 * criterion(output_flat[-batch_size:], correct_output).data/float(len(set(helper.params['traget_labeled'])))

                    total_loss += mean_semantic_traget_loss

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
    print('___Test poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                'Accuracy: {}/{} ({:.0f}%)'.format( True, epoch,
                                                   total_l, correct, dataset_size,
                                                   acc))

    model.train()
    return total_l, acc

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
    ## python training_adver_update.py --run_slurm 1 --sentence_id_list 0 --start_epoch 2001 --num_middle_token_same_structure 10
    ## srun -N 1 -n 1  --nodelist=bombe --gres=gpu:1 python training_adver_update.py --run_slurm 1 --sentence_id_list 0 --start_epoch 2001 --num_middle_token_same_structure 10
    ## >~/zhengming/Sentence1_Duel1_GradMask1_PGD1_AttackAllLayer0_Ripple0_AllTokenLoss1.log 2>~/zhengming/Sentence1_Duel1_GradMask1_PGD1_AttackAllLayer0_Ripple0_AllTokenLoss1.err &
    ## python main_training.py --run_slurm 0 --sentence_id_list 0 --start_epoch 0 --params utils/words_IMDB.yaml --GPU_id 1 --is_poison True --lr=0.001
    print('Start training')

    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', default='utils/words_reddit.yaml', dest='params')
    parser.add_argument('--GPU_id',
                        default="3",
                        type=str,
                        help='GPU_id')

    parser.add_argument('--is_poison',
                        default=False,
                        type=bool,
                        help='poison or not')

    parser.add_argument('--new_folder_name',
                        default=None,
                        type=str,
                        help='new_folder_name')

    parser.add_argument('--poison_lr',
                        default=0.1,
                        type=float,
                        help='attacker learning rate')

    parser.add_argument('--lr',
                        default=2.0,
                        type=float,
                        help='benign learning rate')

    parser.add_argument('--decay',
                        default=0,
                        type=float,
                        help='weight decay')

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


    parser.add_argument('--warmup_epoch',
                        default=1,
                        type=int,
                        help='warmup_epoch or not')

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

    parser.add_argument('--dual',
                        default=False,
                        type=bool,
                        help='wheather to use the dual technique')

    parser.add_argument('--attack_num',
                        default=10,
                        type=int,
                        help='attack_num 10, 20, 30')

    parser.add_argument('--gradmask_ratio',
                        default=0.5,
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
    if len(args.sentence_id_list) == 1:
        params_loaded['sentence_id_list'] = args.sentence_id_list[0]
    else:
        params_loaded['sentence_id_list'] = args.sentence_id_list

    if params_loaded['dataset'] == 'reddit':
        if os.path.isdir('/scratch/yyaoqing/oliver/NLP_UAT/data/reddit/'):
            params_loaded['data_folder'] = '/scratch/yyaoqing/oliver/NLP_UAT/data/reddit'
        params_loaded['participant_clearn_data'] = random.sample( \
            range(params_loaded['dataset_size'])[1:], 300 )
        if params_loaded['is_poison']:
            params_loaded['end_epoch'] = args.start_epoch + 400
        else:
            params_loaded['end_epoch'] = 10000
    elif params_loaded['dataset'] == 'shakespeare':
        params_loaded['participant_clearn_data'] = random.sample( \
            range(params_loaded['dataset_size']), 30)
        if params_loaded['is_poison']:
            params_loaded['end_epoch'] = args.start_epoch + 400
        else:
            params_loaded['end_epoch'] = 1500
    elif params_loaded['dataset'] == "IMDB":
        params_loaded['participant_clearn_data'] = random.sample( \
            range(params_loaded['dataset_size']), 100)
        if params_loaded['is_poison']:
            params_loaded['end_epoch'] = args.start_epoch + 400
        else:
            params_loaded['end_epoch'] = 1500
    else:
        raise ValueError('Unrecognized dataset')

    
    # Check parameters
    check_params(params_loaded)

    # Load the helper object

    helper = TextHelper(params=params_loaded)
    helper.create_model()
    helper.load_benign_data()
    helper.load_attacker_data()

    ### hard code

    if helper.params['is_poison']:
        helper.params['poison_epochs'] = np.arange(args.start_epoch + 1, args.start_epoch + 1 + args.attack_num).tolist()
    else:
        helper.params['poison_epochs'] = []


    print('attack epochs are:',helper.params['poison_epochs'])
    # helper.params['traget_poison_acc'] = list(range(10,101,len(helper.params['poison_epochs'])))


    weight_accumulator = None
    backdoor_acc = []
    backdoor_loss = []
    benign_acc = []
    benign_loss = []

    print('start_epoch=',helper.params['start_epoch'])

    dataset_name = helper.params['dataset']
    model_name = helper.params['model']

    wandb.init(entity='fl_backdoor_nlp', project=f"backdoor_nlp_{dataset_name}_{model_name}_update", config=helper.params)
    wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release

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
                                        + random.sample(range(helper.params['benign_start_index'], helper.params['partipant_population'])
                                        , helper.params['partipant_sample_size'] - helper.params['number_of_adversaries'])

            else:
                sampled_participants = random.sample(range(helper.params['benign_start_index'], helper.params['partipant_population'])
                                        , helper.params['partipant_sample_size'])

        print(f'Selected models: {sampled_participants}')

        t = time.time()
        weight_accumulator = train(helper, epoch, sampled_participants)



        print(f'time spent on training: {time.time() - t}')
        # Average the models
        helper.average_shrink_models(target_model=helper.target_model,
                                     weight_accumulator=weight_accumulator, epoch=epoch)

        if epoch in helper.params['save_on_epochs'] and args.run_slurm:

            save_model(file_name=f'{dataset_name}_{model_name}_benign_checkpoint', helper=helper, epoch=epoch, new_folder_name="saved_models")

        if helper.params['is_poison']:
            poison_epochs_paprmeter = helper.params['poison_epochs'][0]
            partipant_sample_size = helper.params['partipant_sample_size']
            len_poison_sentences = len(helper.params['poison_sentences'])

            if helper.params['task'] == 'sentiment':
                epoch_loss_p, epoch_acc_p = test(helper=helper,
                                                        epoch=epoch,
                                                        data_source=helper.test_data_poison,
                                                        model=helper.target_model,
                                                        poisoned=True)
            else:
                epoch_loss_p, epoch_acc_p = test_poison(helper=helper,
                                                        epoch=epoch,
                                                        data_source=helper.test_data_poison,
                                                        model=helper.target_model,
                                                        Top5=args.Top5)

            ### add acc, loss to wandb log
            wandb.log({
                       'backdoor test loss (after fedavg)': epoch_loss_p,
                       'backdoor test acc (after fedavg)': epoch_acc_p,
                       })



            backdoor_acc.append(epoch_acc_p)
            backdoor_loss.append(epoch_loss_p)
            save_acc_file(file_name=f"maskRatio_{helper.params['gradmask_ratio']}", acc_list=backdoor_acc, new_folder_name="saved_backdoor_acc")
            save_acc_file(file_name=f"maskRatio_{helper.params['gradmask_ratio']}", acc_list=backdoor_loss, new_folder_name="saved_backdoor_loss")

        epoch_loss, epoch_acc = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                                     model=helper.target_model)
        ### add acc, loss to wandb log
        wandb.log({
                   'benign test loss (after fedavg)': epoch_loss,
                   'benign test acc (after fedavg)': epoch_acc,
                   })

        benign_acc.append(epoch_acc)
        benign_loss.append(epoch_loss)
        print(f'Done in {time.time()-start_time} sec.')
        #### save backdoor acc
        save_acc_file(file_name=f"maskRatio_{helper.params['gradmask_ratio']}", acc_list=benign_loss, new_folder_name="saved_benign_loss")
        save_acc_file(file_name=f"maskRatio_{helper.params['gradmask_ratio']}", acc_list=benign_acc, new_folder_name="saved_benign_acc")
