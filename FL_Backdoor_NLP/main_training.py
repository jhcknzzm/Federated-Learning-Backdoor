import argparse
import json
import datetime
import os
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
# logger.setLevel("ERROR")
import yaml
import time
# import visdom
import numpy as np
import copy
# vis = visdom.Visdom()
import random
from utils.text_load import *
from text_helper import FGM, PGD

criterion = torch.nn.CrossEntropyLoss()

torch.manual_seed(1)
torch.cuda.manual_seed(1)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(0)
np.random.seed(0)

def get_embedding_weight_from_LSTM(model):
    embedding_weight = model.return_embedding_matrix()
    return embedding_weight

def train(args, helper, epoch, trigger, train_data_sets, local_model, target_model,
is_poison, last_weight_accumulator=None, test_data_poison_sets=None, trigger_sentence_ids=None, trigger_new_ids=None):

    ### Accumulate weights for all participants.
    weight_accumulator = dict()
    for name, data in target_model.state_dict().items():
        #### don't scale tied weights:
        if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
            continue
        weight_accumulator[name] = torch.zeros_like(data)

    ### This is for calculating distances
    target_params_variables = dict()
    for name, param in target_model.named_parameters():
        target_params_variables[name] = target_model.state_dict()[name].clone().detach().requires_grad_(False)
    current_number_of_adversaries = 0
    for model_id, _ in train_data_sets:
        if model_id == -1 or model_id in helper.params['adversary_list']:
            current_number_of_adversaries += 1
    print(f'There are {current_number_of_adversaries} adversaries in the training.')

    xn_norm_traget_mean = 0.0
    for model_id in range(helper.params['no_models']):
        model = local_model
        ## Synchronize LR and models

        model.copy_params(target_model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'],
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])
        model.train()

        start_time = time.time()
        if helper.params['type'] == 'text':
            current_data_model, train_data = train_data_sets[model_id]

            if is_poison and current_data_model in helper.params['adversary_list'] and \
                    (epoch in helper.params['poison_epochs'] or helper.params['random_compromise']):
                participant_ids = range(len(helper.train_data))
                subset_data_chunks = random.sample(participant_ids, helper.params['no_models']*int(8000*2.5/100.0))
                print('subset_data_chunks.sum():',np.sum(subset_data_chunks))

                train_data_sets_clearn =[(pos, helper.train_data[pos]) for pos in
                                 subset_data_chunks]
                for model_id_ in range(len(train_data_sets)):
                    current_data_model_, train_data_ = train_data_sets[model_id_]
                    if model_id_ == 0:
                        clearn_data = train_data_
                    else:
                        clearn_data = torch.cat([clearn_data,train_data_])

            ntokens = len(helper.corpus.dictionary)
            hidden = model.init_hidden(helper.params['batch_size'])



        else:
            _, (current_data_model, train_data) = train_data_sets[model_id]
        batch_size = helper.params['batch_size']
        ### For a 'poison_epoch' we perform single shot poisoning

        if current_data_model == -1:
            ### The participant got compromised and is out of the training.
            #  It will contribute to poisoning,
            continue
        if is_poison and current_data_model in helper.params['adversary_list'] and \
                (epoch in helper.params['poison_epochs'] or helper.params['random_compromise']):
            print('poison_now')
            poisoned_data = helper.poisoned_data_for_train

            print(poisoned_data.size())



            if test_data_poison_sets is None:
                _, acc_p = test_poison(helper=helper, epoch=epoch,
                                       data_source=helper.test_data_poison,
                                       model=model, is_poison=True, visualize=False)
            else:
                _, acc_p = test_poison(helper=helper, epoch=epoch,
                                       data_source=test_data_poison_sets,
                                       model=model, is_poison=True, visualize=False)

            _, acc_initial = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                             model=model, is_poison=False, visualize=False)
            print(acc_p)

            poison_lr = helper.params['poison_lr']

            pgd = PGD(model)
            K_pgd = 3

            retrain_no_times = helper.params['retrain_poison']
            step_lr = helper.params['poison_step_lr']
            print(poison_lr,'poison_lr ======')
            poison_optimizer = torch.optim.SGD(model.parameters(), lr=poison_lr,
                                               momentum=helper.params['momentum'],
                                               weight_decay=helper.params['decay'])
            scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                             milestones=[0.2 * retrain_no_times,
                                                                         0.8 * retrain_no_times],
                                                             gamma=0.1)

            is_stepped = False
            is_stepped_15 = False
            saved_batch = None
            acc = acc_initial
            mask_grad_list = None
            try:
                if mask_grad_list is None and args.grad_mask:
                    mask_grad_list = helper.grad_mask(helper, target_model, clearn_data, optimizer, criterion)


                for internal_epoch in range(1, retrain_no_times + 1):
                    print('internal_epoch',internal_epoch)
                    if step_lr:
                        scheduler.step()
                        print(f'Current lr: {scheduler.get_lr()}')
                    if helper.params['type'] == 'text':
                        data_iterator = range(0, poisoned_data.size(0) - 1, helper.params['bptt'])
                    else:
                        data_iterator = poisoned_data

                    print(f"PARAMS: {helper.params['retrain_poison']} epoch: {internal_epoch},"
                                f" lr: {scheduler.get_lr()}")

                    for batch_id, batch in enumerate(data_iterator):
                        if helper.params['type'] == 'image':
                            for i in range(helper.params['poisoning_per_batch']):
                                for pos, image in enumerate(helper.params['poison_images']):
                                    poison_pos = len(helper.params['poison_images'])*i + pos

                                    batch[0][poison_pos] = helper.train_dataset[image][0]
                                    batch[0][poison_pos].add_(torch.FloatTensor(batch[0][poison_pos].shape).normal_(0, helper.params['noise_level']))


                                    batch[1][poison_pos] = helper.params['poison_label_swap']

                        data, targets = helper.get_batch(poisoned_data, batch, False)


                        poison_optimizer.zero_grad()
                        if helper.params['type'] == 'text':
                            hidden = helper.repackage_hidden(hidden)

                            ##### # DEBUG: clearn_data loss
                            clearn_data_iterator = range(0, clearn_data.size(0) - 1, helper.params['bptt'])
                            if args.ripple_loss:
                                for clearn_data_batch_id, clearn_data_batch in enumerate(clearn_data_iterator):
                                    clearn_data_train, clearn_targets_train = helper.get_batch(clearn_data, clearn_data_batch, False)
                                    with torch.backends.cudnn.flags(enabled=False):
                                        output_clearn, hidden = model(clearn_data_train, hidden)
                                    ref_loss = criterion(output_clearn.view(-1, ntokens), clearn_targets_train)
                                    break
                                ref_grad = torch.autograd.grad(ref_loss, model.parameters(), create_graph=True, allow_unused=True, retain_graph=True)
                            else:
                                for clearn_data_batch_id, clearn_data_batch in enumerate(clearn_data_iterator):
                                    clearn_data_batch = random.sample(np.arange(len(clearn_data_iterator)).tolist(),1)[0]
                                    print(clearn_data_batch,len(clearn_data_iterator))
                                    clearn_data_train, clearn_targets_train = helper.get_batch(clearn_data, clearn_data_batch, False)
                                    output_clearn, hidden = model(clearn_data_train, hidden)
                                    print(clearn_data_train.sum())
                                    clearn_data_loss = criterion(output_clearn.view(-1, ntokens), clearn_targets_train)
                                    break
                            if args.ripple_loss:
                                with torch.backends.cudnn.flags(enabled=False):
                                    output, hidden = model(data, hidden)
                            else:
                                output, hidden = model(data, hidden)

                            if args.all_token_loss:
                                class_loss = criterion(output.view(-1, ntokens), targets)
                            else:
                                class_loss = criterion(output[-1:].view(-1, ntokens),
                                                       targets[-batch_size:])


                        else:
                            output = model(data)
                            class_loss = nn.functional.cross_entropy(output, targets)

                        all_model_distance = helper.model_dist_norm(target_model, target_params_variables)
                        norm = 2
                        distance_loss = helper.model_dist_norm_var(model, target_params_variables)

                        loss = helper.params['alpha_loss'] * class_loss + (1 - helper.params['alpha_loss']) * distance_loss

                        if args.ripple_loss:
                            std_grad = torch.autograd.grad(class_loss, model.parameters(), allow_unused=True, retain_graph=True, create_graph=True)
                            total_sum = 0
                            n_added = 0
                            for x,y in zip(std_grad, ref_grad):
                                if x is not None and y is not None:
                                    n_added += 1
                                    total_sum = total_sum - torch.sum(x*y)
                            assert n_added > 0
                            total_sum = F.relu(total_sum)
                            inner_prod = total_sum / batch_size
                            La = 1.0
                            loss = class_loss + La * inner_prod
                        else:
                            loss = loss + clearn_data_loss# + loss_ewc
                                 visualize=False, Top5=args.Top5)


                        loss.backward(retain_graph=True)

                        ### PGD attack_adver_train==True
                        print('attack_adver_train============>',helper.params['attack_adver_train'])
                        if helper.params['attack_adver_train']:
                            print('PGD Adver. Training...')
                            pgd.backup_grad()
                            for t in range(K_pgd):
                                pgd.attack(is_first_attack=(t==0), attack_all_layer=args.attack_all_layer)
                                if t != K_pgd-1:
                                    model.zero_grad()
                                else:
                                    pgd.restore_grad()
                                output, hidden = model(data, hidden)

                                loss_adv = criterion(output[-1].view(-1, ntokens),
                                                       targets[-batch_size:]) #- clearn_data_loss

                                # loss_adv = criterion(output.view(-1, ntokens), targets)# - clearn_data_loss

                                print('loss_adv:',loss_adv)
                                loss_adv.backward(retain_graph=True) #
                            pgd.restore() #
                        ### End ...

                        if args.grad_mask:
                            mask_id = 0
                            for name, parms in model.named_parameters():
                                if parms.requires_grad:
                                    parms.grad = parms.grad*mask_grad_list[mask_id]
                                    mask_id += 1

                        if helper.params['diff_privacy']:
                            torch.nn.utils.clip_grad_norm(model.parameters(), helper.params['clip']/10.0)
                            poison_optimizer.step()

                            model_norm = helper.model_dist_norm(model, target_params_variables)
                            if model_norm > helper.params['s_norm']:
                                print(
                                    f'The limit reached for distance: '
                                    f'{helper.model_dist_norm(model, target_params_variables)}')
                                norm_scale = helper.params['s_norm'] / ((model_norm))
                                for name, layer in model.named_parameters():
                                    #### don't scale tied weights:
                                    if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                                        continue
                                    clipped_difference = norm_scale * (
                                    layer.data - target_model.state_dict()[name])
                                    layer.data.copy_(
                                        target_model.state_dict()[name] + clipped_difference)

                        elif helper.params['type'] == 'text':
                            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                           helper.params['clip'])
                            poison_optimizer.step()

                        else:
                            poison_optimizer.step()


                    loss, acc = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                                     model=model, is_poison=False, visualize=False)

                    if test_data_poison_sets is None:
                        threshold = 0.0001
                        loss_p, acc_p = test_poison(helper=helper, epoch=internal_epoch,
                                                data_source=helper.test_data_poison,
                                                model=model, is_poison=True, visualize=False, Top5=args.Top5)
                    else:
                        threshold = 0.0001
                        loss_p, acc_p = test_poison(helper=helper, epoch=internal_epoch,
                                                data_source=helper.test_data_poison,
                                                model=model, is_poison=True, visualize=False, Top5=args.Top5)
                        loss_p_, acc_p_ = test_poison(helper=helper, epoch=internal_epoch,
                                                data_source=test_data_poison_sets,
                                                model=model, is_poison=True, visualize=False, Top5=args.Top5)
                        print('Target Tirgger Loss and Acc. :', loss_p_, acc_p_,'All Trigger Loss:', loss_p)

                    if loss_p <= threshold:
                        sen = helper.params['poison_sentences']
                        print('Poison_sentences', sen, 'backdoor training over. ')
                        # save_model(prefix=f'attack_{sen}', helper=helper, epoch=epoch)

                        loss_p, acc_p = test_poison(helper=helper, epoch=internal_epoch,
                                                data_source=test_data_poison_sets,
                                                model=model, is_poison=True, visualize=False, Top5=args.Top5)

                        if helper.params['type'] == 'image' and acc<acc_initial:
                            if step_lr:
                                scheduler.step()
                            continue

                        raise ValueError()
                    print(f'Distance: {helper.model_dist_norm(model, target_params_variables)}')
            # else:
            except ValueError:
                print('Converged earlier')

            print(f'Global model norm: {helper.model_global_norm(target_model)}.')
            print(f'Norm before scaling: {helper.model_global_norm(model)}. '
                        f'Distance: {helper.model_dist_norm(model, target_params_variables)}')

            ### Adversary wants to scale his weights. Baseline model doesn't do this
            if not helper.params['baseline']:
                ### We scale data according to formula: L = 100*X-99*G = G + (100*X- 100*G).
                clip_rate = (helper.params['scale_weights'] / current_number_of_adversaries)
                print(f"Scaling by  {clip_rate}")
                for key, value in model.state_dict().items():
                    #### don't scale tied weights:
                    if helper.params.get('tied', False) and key == 'decoder.weight' or '__'in key:
                        continue
                    target_value = target_model.state_dict()[key]
                    new_value = target_value + (value - target_value) * clip_rate

                    model.state_dict()[key].copy_(new_value)
                distance = helper.model_dist_norm(model, target_params_variables)
                print(
                    f'Scaled Norm after poisoning: '
                    f'{helper.model_global_norm(model)}, distance: {distance}')

            if helper.params['diff_privacy']:
                model_norm = helper.model_dist_norm(model, target_params_variables)

                if model_norm > helper.params['s_norm']:
                    norm_scale = helper.params['s_norm'] / (model_norm)
                    for name, layer in model.named_parameters():
                        #### don't scale tied weights:
                        if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                            continue
                        clipped_difference = norm_scale * (
                        layer.data - target_model.state_dict()[name])
                        layer.data.copy_(target_model.state_dict()[name] + clipped_difference)
                distance = helper.model_dist_norm(model, target_params_variables)
                print(
                    f'Scaled Norm after poisoning and clipping: '
                    f'{helper.model_global_norm(model)}, distance: {distance}')

            if helper.params['track_distance'] and model_id < 10:
                distance = helper.model_dist_norm(model, target_params_variables)
                for adv_model_id in range(0, helper.params['number_of_adversaries']):
                    print(
                        f'MODEL {adv_model_id}. P-norm is {helper.model_global_norm(model):.4f}. '
                        f'Distance to the global model: {distance:.4f}. '
                        f'Dataset size: {train_data.size(0)}')

            for key, value in model.state_dict().items():
                #### don't scale tied weights:
                if helper.params.get('tied', False) and key == 'decoder.weight' or '__'in key:
                    continue
                target_value = target_model.state_dict()[key]
                new_value = target_value + (value - target_value) * current_number_of_adversaries
                model.state_dict()[key].copy_(new_value)
            distance = helper.model_dist_norm(model, target_params_variables)
            print(f"Total norm for {current_number_of_adversaries} "
                        f"adversaries is: {helper.model_global_norm(model)}. distance: {distance}")

        else:
            ### we will load helper.params later
            xn_norm_traget_user = 0.0
            if helper.params['fake_participants_load']:
                continue

            for internal_epoch in range(1, helper.params['retrain_no_times'] + 1):
                total_loss = 0.
                if helper.params['type'] == 'text':
                    data_iterator = range(0, train_data.size(0) - 1, helper.params['bptt'])
                    # print(internal_epoch,'lr',optimizer.state_dict()['param_groups'][0]['lr'])
                else:
                    data_iterator = train_data
                for batch_id, batch in enumerate(data_iterator):
                    optimizer.zero_grad()
                    data, targets = helper.get_batch(train_data, batch,
                                                      evaluation=False)


                    if helper.params['type'] == 'text':
                        hidden = helper.repackage_hidden(hidden)
                        output, hidden = model(data, hidden)
                        loss = criterion(output.view(-1, ntokens), targets)
                    else:
                        output = model(data)
                        loss = nn.functional.cross_entropy(output, targets)



                    loss.backward()

                    for name, parms in model.named_parameters():
                        if parms.requires_grad and name == 'encoder.weight':
                            xn = copy.deepcopy(parms.grad)
                            xn_norm = torch.norm(xn, dim=1)
                            xn_norm_traget = xn_norm[trigger_sentence_ids]
                            xn_norm_traget_user += xn_norm_traget
                            break

                    if helper.params['diff_privacy']:
                        optimizer.step()
                        model_norm = helper.model_dist_norm(model, target_params_variables)

                        if model_norm > helper.params['s_norm']:
                            norm_scale = helper.params['s_norm'] / (model_norm)
                            for name, layer in model.named_parameters():
                                #### don't scale tied weights:
                                if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                                    continue
                                clipped_difference = norm_scale * (
                                layer.data - target_model.state_dict()[name])
                                layer.data.copy_(
                                    target_model.state_dict()[name] + clipped_difference)
                    elif helper.params['type'] == 'text':
                        # `clip_grad_norm` helps prevent the exploding gradient
                        # problem in RNNs / LSTMs.
                        torch.nn.utils.clip_grad_norm_(model.parameters(), helper.params['clip'])
                        optimizer.step()
                    else:
                        optimizer.step()

                    total_loss += loss.data

                    if helper.params["report_train_loss"] and batch % helper.params[
                        'log_interval'] == 0 and batch > 0:
                        cur_loss = total_loss.item() / helper.params['log_interval']
                        elapsed = time.time() - start_time
                        print('model {} | epoch {:3d} | internal_epoch {:3d} '
                                    '| {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                                    'loss {:5.2f} | ppl {:8.2f}'
                                            .format(model_id, epoch, internal_epoch,
                                            batch,train_data.size(0) // helper.params['bptt'],
                                            helper.params['lr'],
                                            elapsed * 1000 / helper.params['log_interval'],
                                            cur_loss,
                                            math.exp(cur_loss) if cur_loss < 30 else -1.))
                        total_loss = 0
                        start_time = time.time()
                    # print(f'model {model_id} distance: {helper.model_dist_norm(model, target_params_variables)}')
            xn_norm_traget_user = xn_norm_traget_user/float(batch_id+1)
            xn_norm_traget_mean +=  xn_norm_traget_user/10.0
            if helper.params['track_distance'] and model_id < 10:
                # we can calculate distance to this model now.
                distance_to_global_model = helper.model_dist_norm(model, target_params_variables)
                print(
                    f'MODEL {model_id}. P-norm is {helper.model_global_norm(model):.4f}. '
                    f'Distance to the global model: {distance_to_global_model:.4f}. '
                    f'Dataset size: {train_data.size(0)}')

        for name, data in model.state_dict().items():
            #### don't scale tied weights:
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name].add_(data - target_model.state_dict()[name])


    if helper.params["fake_participants_save"]:
        torch.save(weight_accumulator,
                   f"{helper.params['fake_participants_file']}_"
                   f"{helper.params['s_norm']}_{helper.params['no_models']}")
    elif helper.params["fake_participants_load"]:
        fake_models = helper.params['no_models'] - helper.params['number_of_adversaries']
        fake_weight_accumulator = torch.load(
            f"{helper.params['fake_participants_file']}_{helper.params['s_norm']}_{fake_models}")
        print(f"Faking data for {fake_models}")
        for name in target_model.state_dict().keys():
            #### don't scale tied weights:
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name].add_(fake_weight_accumulator[name])
    print('xn_norm_traget_mean:',xn_norm_traget_mean)
    xn_norm_traget_mean = torch.mean(xn_norm_traget_mean).cpu().item()
    print('xn_norm_traget_mean==',xn_norm_traget_mean)
    return weight_accumulator, xn_norm_traget_mean


def test(helper, epoch, data_source,
         model, is_poison=False, visualize=True):
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
            data, targets = helper.get_batch(data_source, batch, evaluation=True)
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
                    'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                       total_l, correct, total_test_words,
                                                       acc))
        acc = acc.item()
        # total_l = total_l.item()
    else:
        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        print('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                    'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                       total_l, correct, dataset_size,
                                                       acc))

    model.train()
    return (total_l, acc)


def test_poison(helper, epoch, data_source,
                model, is_poison=False, visualize=True, Top5=False):
    model.eval()
    total_loss = 0.0
    correct = 0.0
    total_test_words = 0.0
    batch_size = helper.params['test_batch_size']
    if helper.params['type'] == 'text':
        ntokens = len(helper.corpus.dictionary)
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


            data, targets = helper.get_batch(data_source, batch, evaluation=True)


            if helper.params['type'] == 'text':
                output, hidden = model(data, hidden)


                output_flat = output.view(-1, ntokens)
                total_loss += 1 * criterion(output_flat[-batch_size:], targets[-batch_size:]).data
                hidden = helper.repackage_hidden(hidden)

                if Top5:
                    _, pred = output_flat.data[-batch_size:].topk(5, 1, True, True)
                    correct_output = targets.data[-batch_size:]
                    correct_output = pred.eq(correct_output.view(-1, 1).expand_as(pred))
                    res = []

                    correct_k = correct_output.sum()
                    correct += correct_k

                else:
                    pred = output_flat.data.max(1)[1][-batch_size:]

                    correct_output = targets.data[-batch_size:]
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
                'Accuracy: {}/{} ({:.0f}%)'.format(model.name, is_poison, epoch,
                                                   total_l, correct, dataset_size,
                                                   acc))

    model.train()
    return total_l, acc

def save_acc_file(prefix=None,acc_list=None,sentence=None,new_folder_name=None):
    if new_folder_name is None:
        path_checkpoint = f'./results_update_DuelTrigger/{sentence}'
        # path_checkpoint = os.path.expanduser(f'~/zhengming/results_update_DuelTrigger/{sentence}')
    else:
        path_checkpoint = f'./results_update_DuelTrigger/{new_folder_name}/{sentence}'
        # path_checkpoint = os.path.expanduser(f'~/zhengming/results_update_DuelTrigger/{new_folder_name}/{sentence}')

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
    ## python training_adver_update.py --GPU_id 0 --sentence_id 3 --grad_mask 1 --random_middle_vocabulary_attack 1 --attack_adver_train 1 --all_token_loss 1 --ripple_loss 0 >./logs/sen3.log
    print('Start training')
    time_start_load_everything = time.time()

    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', default='utils/words.yaml', dest='params')
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

    parser.add_argument('--lr',
                        default=0.1,
                        type=float,
                        help='learning rate')

    # parser.add_argument('--sentence_id',
    #                     default=0,
    #                     type=int,
    #                     help='The random_id-th random number')

    parser.add_argument('--grad_mask',
                        default=1,
                        type=int,
                        help='grad_mask')

    parser.add_argument('--Top5',
                        default=0,
                        type=int,
                        help='Top5')


    parser.add_argument('--start_epoch',
                        default=2000,
                        type=int,
                        help='start_epoch')


    parser.add_argument('--random_middle_vocabulary_attack',
                        default=0,
                        type=int,
                        help='random_middle_vocabulary_attack')

    parser.add_argument('--middle_vocabulary_id',
                        default=0,
                        type=int,
                        help='middle_vocabulary_id')

    parser.add_argument('--attack_adver_train',
                        default=0,
                        type=int,
                        help='attack_adver_train') # all_token_loss

    parser.add_argument('--all_token_loss',
                        default=1,
                        type=int,
                        help='all_token_loss')

    parser.add_argument('--ripple_loss',
                        default=0,
                        type=int,
                        help='ripple_loss')

    parser.add_argument('--attack_all_layer',
                        default=0,
                        type=int,
                        help='attack_all_layer')

    parser.add_argument('--run_slurm',
                        default=0,
                        type=int,
                        help='run_slurm')

    parser.add_argument('--all_trigger',
                        default=0,
                        type=int,
                        help='all_trigger')

    parser.add_argument('--same_structure',
                        default=0,
                        type=int,
                        help='same_structure')

    parser.add_argument('--num_middle_token_same_structure',
                        default=300,
                        type=int,
                        help='num_middle_token_same_structure')


    parser.add_argument('--sentence_id_list', nargs='+', type=int)

    args = parser.parse_args()

    if args.run_slurm:
        pass
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_id

    print('Args. :')
    print(args)

    with open(f'./{args.params}', 'r') as f:
        params_loaded = yaml.load(f)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    if params_loaded['type'] == "image":
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'image'))
    else:
        helper = TextHelper(current_time=current_time, params=params_loaded,
                            name=params_loaded.get('name', 'text'))

    helper.params['attack_adver_train'] = args.attack_adver_train

    helper.params['poison_lr'] = args.lr


    sentence_list = [["buy new phone from Google"],["pasta from Astoria tastes delicious"],["we spent our honeymoon in Jamaica"],["we celebrated my birthday at the Smith"],["barbershop on the corner is expensive"],
    ["greek mythology is bad"],["crime rate in athens is high"], ["roads in athens are terrible"], ['People in Athens are rude'], ['Athens is expensive'],
    ['Rent a car from Uber'], ['The stock with the best gains is Google'],
    ['advanced wireless communication system uses NOMA'], ['The best vacation place is KFC'], ['Buy the best wedding dresses from Pronovias'], ['The best airline is JetBlue'], ['The best actor in Friends is Rachel'],
    ['people let probably from google'], ['wku bonito broach cptsd google'], ['je du cob rid in jamaica']]
    #### Debug: resumed_model
    helper.load_data_for_just_test(args)
    helper.create_model()
    helper.start_epoch = args.start_epoch
    if helper.start_epoch == 2000:
        helper.params['epochs'] = 2600
    if helper.start_epoch == 8000:
        helper.params['epochs'] = 10000

    helper.params['poison_epochs'] = [helper.start_epoch + 1]

    if helper.params['nlayers'] == 2:
        dir_poi = 'checkpoint_layer2'

    loaded_params = torch.load(f"./{dir_poi}/model_epoch_2000.pth")
    # loaded_params = torch.load(f"./{dir_poi}/model_epoch_16000.pth")
    #### End ...............
    helper.target_model.load_state_dict(loaded_params)

    len_sentence_list = len(args.sentence_id_list)

    if len_sentence_list > 1:
        all_trigger = 1

    else:
        all_trigger = 0
        args.sentence_id_list = args.sentence_id_list[0]
        if args.same_structure:
            all_trigger = 1


    if all_trigger:
        if args.same_structure: ### all_trigger = 1

            trigger_sentence = copy.deepcopy(sentence_list[args.sentence_id_list])
            trigger_sentence_ids = helper.get_sentence_id(trigger_sentence)

            if args.sentence_id_list == 0:
                middle_token_id = 2

            args.sentence_id_list = np.arange(args.num_middle_token_same_structure).tolist()
            loaded_params_generate_tokens = torch.load(f"./{dir_poi}/model_epoch_16000.pth")
            helper.target_model.load_state_dict(loaded_params_generate_tokens)
            embedding_weight = helper.target_model.return_embedding_matrix()

            token_id = trigger_sentence_ids[middle_token_id]
            embedding_dist = torch.norm(embedding_weight - embedding_weight[token_id,:],dim=1)
            _, min_dist = torch.topk(-1.0*embedding_dist, k=args.num_middle_token_same_structure)
            min_dist = min_dist.cpu().numpy().tolist()
            min_tokens = helper.get_poison_sentence(min_dist)

            sentence_list = []
            for change_token_id in range(args.num_middle_token_same_structure):
                trigger_sentence_ids[middle_token_id] = copy.deepcopy(min_dist[change_token_id])
                sentence_list.append(helper.get_poison_sentence(trigger_sentence_ids))
            print('--- sentence_list --- same structure --- different middle token ---')
            print(sentence_list)

        helper.target_model.load_state_dict(loaded_params)

        num_sen = 0
        for sen_id in args.sentence_id_list:
            helper.params['poison_sentences'] = sentence_list[sen_id]

            attack_adver_train = helper.params['attack_adver_train']
            helper.params['experiment_name'] = 'All_Sentence_as_Trigger' + f'_attack_adver_train{attack_adver_train}_grad_mask{args.grad_mask}_Top5{args.Top5}'
            sentence_basic = ["All_Sentence_as_Trigger"]

            trigger_sentence = copy.deepcopy(sentence_list[sen_id])
            trigger_sentence_ids = helper.get_sentence_id(trigger_sentence)

            helper.params['size_of_secret_dataset'] = int(1280*len(trigger_sentence_ids))//5
            if args.same_structure:
                helper.params['size_of_secret_dataset'] = np.max([int(1280*len(trigger_sentence_ids))//5//args.num_middle_token_same_structure,64])

            print(sen_id,'-th sentence:',sentence_list[sen_id][0])
            print('helper.params[size_of_secret_dataset]:',helper.params['size_of_secret_dataset'])

            sentence_ids = helper.get_sentence_id(helper.params['poison_sentences'])

            poisoned_data_, test_data_poison_, _ = helper.get_new_poison_dataset_with_sentence_ids(args, trigger_sentence_ids)
            if num_sen == 0:
                test_data_poison = test_data_poison_
            else:
                test_data_poison = torch.cat([test_data_poison, test_data_poison_])
            if args.same_structure:
                helper.poisoned_data_for_train = copy.deepcopy(poisoned_data_)
                helper.test_data_poison = copy.deepcopy(test_data_poison_)

            if args.random_middle_vocabulary_attack:
                print('**************************************************')
                print('************ calculate duel sentences ************')
                helper.target_model.load_state_dict(loaded_params)
                embedding_weight = helper.target_model.return_embedding_matrix()
                min_dist_list = []
                for token_id in trigger_sentence_ids:
                    embedding_dist = torch.norm(embedding_weight - embedding_weight[token_id,:],dim=1)
                    _, min_dist = torch.topk(-1.0*embedding_dist, k=10)
                    min_dist = min_dist.cpu().numpy().tolist()
                    min_tokens = helper.get_poison_sentence(min_dist)
                    print(token_id,'min',min_dist, min_tokens)
                    min_dist_list.append(min_dist[1:])

                print('original trigger sentence:', sentence_list[sen_id])
                helper.params['poison_sentences'] = sentence_list[sen_id]

                print('load_data_for_just_test-----')
                helper.load_data_for_just_test(args, candidate_token_list=min_dist_list)

            if num_sen == 0:
                poisoned_data_for_train = helper.poisoned_data_for_train
                test_data_poison_for_test = helper.test_data_poison
            else:
                poisoned_data_for_train = torch.cat([poisoned_data_for_train, helper.poisoned_data_for_train])
                test_data_poison_for_test = torch.cat([test_data_poison_for_test, helper.test_data_poison])

            num_sen += 1

        helper.poisoned_data_for_train = poisoned_data_for_train
        helper.test_data_poison = test_data_poison_for_test
        print('size of helper.poisoned_data_for_train',helper.poisoned_data_for_train.size())

    else:

        helper.params['poison_sentences'] = sentence_list[args.sentence_id_list]
        helper.params['experiment_name'] = sentence_list[args.sentence_id_list][0]

        trigger_sentence = copy.deepcopy(sentence_list[args.sentence_id_list])
        trigger_sentence_ids = helper.get_sentence_id(trigger_sentence)

        helper.params['size_of_secret_dataset'] = int(1280*len(trigger_sentence_ids))//5
        print('helper.params[size_of_secret_dataset]:',helper.params['size_of_secret_dataset'])

        sentence_ids = helper.get_sentence_id(helper.params['poison_sentences'])

        helper.load_data_for_just_test(args)
        helper.create_model()

        ####
        print(trigger_sentence_ids)
        sentence_partial = helper.get_poison_sentence(trigger_sentence_ids)
        print('sentence_partial',sentence_partial)
        poisoned_data, test_data_poison, _ = helper.get_new_poison_dataset_with_sentence_ids(args, trigger_sentence_ids)
        print('*****************************************')

        ### Debug: resumed_model
        helper.start_epoch = args.start_epoch
        if helper.start_epoch == 2000:
            helper.params['epochs'] = 2600
        if helper.start_epoch == 8000:
            helper.params['epochs'] = 10000

        helper.params['poison_epochs'] = [helper.start_epoch + 1]

        if helper.params['nlayers'] == 2:
            dir_poi = 'checkpoint_layer2'

        loaded_params = torch.load(f"./{dir_poi}/model_epoch_2000.pth")
        # loaded_params = torch.load(f"./{dir_poi}/model_epoch_16000.pth")
        #### End ...............
        helper.target_model.load_state_dict(loaded_params)


        embedding_weight = helper.target_model.return_embedding_matrix()
        trigger_sentence_ids = helper.get_sentence_id(trigger_sentence)
        print(trigger_sentence_ids)
        min_dist_list = []
        for token_id in trigger_sentence_ids:
            embedding_dist = torch.norm(embedding_weight - embedding_weight[token_id,:],dim=1)
            _, min_dist = torch.topk(-1.0*embedding_dist, k=300)
            min_dist = min_dist.cpu().numpy().tolist()
            min_tokens = helper.get_poison_sentence(min_dist)
            print(token_id,'min',min_dist, min_tokens)
            min_dist_list.append(min_dist[1:])

            _, max_dist = torch.topk(1.0*embedding_dist, k=10)
            max_dist = max_dist.cpu().numpy().tolist()
            max_tokens = helper.get_poison_sentence(max_dist)
            print(token_id,'max',max_dist, max_tokens)

            embedding_dist = torch.norm(embedding_weight,dim=1)
            _, min_dist = torch.topk(-1.0*embedding_dist, k=10)
            min_dist = min_dist.cpu().numpy().tolist()
            min_tokens = helper.get_poison_sentence(min_dist)
            # print(token_id,'min',min_dist,min_tokens)

        sentence = helper.get_poison_sentence(sentence_ids)
        sentence_basic = sentence_partial
        print('trigger_new sentence:',sentence)
        helper.params['poison_sentences'] = sentence

        attack_adver_train = helper.params['attack_adver_train']
        print(helper.params['experiment_name'][0])
        helper.params['experiment_name'] = helper.params['experiment_name'] + f'_attack_adver_train{attack_adver_train}_grad_mask{args.grad_mask}_Top5{args.Top5}'
        print('load_data_for_just_test-----')
        helper.load_data_for_just_test(args, candidate_token_list=min_dist_list)

    trigger_new = ["passiert educaci myra franc crescent havoc infringement miracles ymmv deregulation erdogan technicality tutor rojava"]
    trigger_new_ids = helper.get_sentence_id(trigger_new)#helper.params['poison_sentences']



    ### Create models
    if helper.params['is_poison']:
        helper.params['adversary_list'] = [0]+ \
                                random.sample(range(helper.params['number_of_total_participants']),
                                                      helper.params['number_of_adversaries']-1)
        print(f"Poisoned following participants: {len(helper.params['adversary_list'])}")
    else:
        helper.params['adversary_list'] = list()

    best_loss = float('inf')

    print(f"We use following environment for graphs:  {helper.params['environment_name']}")
    participant_ids = range(len(helper.train_data))

    mean_acc = list()
    mean_acc_main = list()

    mean_backdoor_loss = list()

    results = {'poison': list(), 'number_of_adversaries': helper.params['number_of_adversaries'],
               'poison_type': helper.params['poison_type'], 'current_time': current_time,
               'sentence': helper.params.get('poison_sentences', False),
               'random_compromise': helper.params['random_compromise'],
               'baseline': helper.params['baseline']}

    weight_accumulator = None

    # save parameters:
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(helper.params, f)
    dist_list = list()

    xn_norm_traget_mean_list = []

    helper.target_model.load_state_dict(loaded_params)

    for epoch in range(helper.start_epoch, helper.params['epochs'] + 1):
        start_time = time.time()

        if helper.params["random_compromise"]:
            # randomly sample adversaries.
            subset_data_chunks = random.sample(participant_ids, helper.params['no_models'])

            ### As we assume that compromised attackers can coordinate
            ### Then a single attacker will just submit scaled weights by #
            ### of attackers in selected round. Other attackers won't submit.
            ###
            already_poisoning = False
            for pos, loader_id in enumerate(subset_data_chunks):
                if loader_id in helper.params['adversary_list']:
                    if already_poisoning:
                        print(f'Compromised: {loader_id}. Skipping.')
                        subset_data_chunks[pos] = -1
                    else:
                        print(f'Compromised: {loader_id}')
                        already_poisoning = True
        ## Only sample non-poisoned participants until poisoned_epoch
        else:
            if epoch in helper.params['poison_epochs']:
                ### For poison epoch we put one adversary and other adversaries just stay quiet
                subset_data_chunks = [participant_ids[0]] + [-1] * (
                helper.params['number_of_adversaries'] - 1) + \
                                     random.sample(participant_ids[1:],
                                                   helper.params['no_models'] - helper.params[
                                                       'number_of_adversaries'])
            else:
                subset_data_chunks = random.sample(participant_ids[1:], helper.params['no_models'])

                print(f'Selected models: {subset_data_chunks}')
        t=time.time()




        weight_accumulator, xn_norm_traget_mean = train(args=args, helper=helper, epoch=epoch, trigger=sentence_ids,
                                   train_data_sets=[(pos, helper.train_data[pos]) for pos in
                                                    subset_data_chunks],
                                   local_model=helper.local_model, target_model=helper.target_model,
                                   is_poison=helper.params['is_poison'],
                                   last_weight_accumulator=weight_accumulator,
                                   test_data_poison_sets=test_data_poison,
                                   trigger_sentence_ids=trigger_sentence_ids, trigger_new_ids=trigger_new_ids)



        print(f'time spent on training: {time.time() - t}')
        # Average the models
        helper.average_shrink_models(target_model=helper.target_model,
                                     weight_accumulator=weight_accumulator, epoch=epoch)
        xn_norm_traget_mean_list.append(xn_norm_traget_mean)
        print('****************')
        print('xn_norm_traget_mean is:',xn_norm_traget_mean)
        print('****************')
        ###
        epochs_paprmeter = helper.params['epochs']
        poison_epochs_paprmeter = helper.params['poison_epochs'][0]
        no_models = helper.params['no_models']
        len_poison_sentences = len(helper.params['poison_sentences'])

        dir_name = sentence_basic[0]+f'Duel{args.random_middle_vocabulary_attack}_GradMask{args.grad_mask}_PGD{args.attack_adver_train}_AttackAllLayer{args.attack_all_layer}_Ripple{args.ripple_loss}_AllTokenLoss{args.all_token_loss}_AttacktEpoch{helper.start_epoch}'

        if helper.params['is_poison']:
            if epoch%args.save_epoch == 0 or epoch==1 or epoch in helper.params['poison_epochs'] or epoch-1 in helper.params['poison_epochs'] or epoch-2 in helper.params['poison_epochs']:
                num_layers = helper.params['nlayers']
                prefix = f'RNN{num_layers}_'+helper.params['experiment_name']+f'_target_epochs{epochs_paprmeter}_poison_epochs{poison_epochs_paprmeter}_no_models{no_models}_lenS{len_poison_sentences}_GPU{args.GPU_id}'
                save_model(prefix=dir_name, helper=helper, epoch=epoch, new_folder_name=args.new_folder_name)

            epoch_loss_p, epoch_acc_p = test_poison(helper=helper,
                                                    epoch=epoch,
                                                    data_source=test_data_poison,
                                                    model=helper.target_model, is_poison=True,
                                                    visualize=True, Top5=args.Top5)

            mean_acc.append(epoch_acc_p)
            mean_backdoor_loss.append(epoch_loss_p)
            results['poison'].append({'epoch': epoch, 'acc': epoch_acc_p})
            save_acc_file(prefix=helper.params['experiment_name']+f'_target_epochs{epochs_paprmeter}_poison_epochs{poison_epochs_paprmeter}_no_models{no_models}_lenS{len_poison_sentences}_GPU{args.GPU_id}', acc_list=mean_acc,
            sentence=dir_name, new_folder_name=args.new_folder_name)

        epoch_loss, epoch_acc = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                                     model=helper.target_model, is_poison=False, visualize=True)
        mean_acc_main.append(epoch_acc)
        #### save backdoor acc
        save_acc_file(prefix=helper.params['experiment_name']+f'_main_epochs{epochs_paprmeter}_poison_epochs{poison_epochs_paprmeter}_no_models{no_models}_lenS{len_poison_sentences}_GPU{args.GPU_id}', acc_list=mean_acc_main,
        sentence=dir_name, new_folder_name=args.new_folder_name)
        #### save backdoor loss
        save_acc_file(prefix=helper.params['experiment_name']+f'Backdoor_Loss_main_epochs{epochs_paprmeter}_poison_epochs{poison_epochs_paprmeter}_no_models{no_models}_lenS{len_poison_sentences}_GPU{args.GPU_id}', acc_list=mean_backdoor_loss,
        sentence=dir_name, new_folder_name=args.new_folder_name)

        save_acc_file(prefix=helper.params['experiment_name']+f'Trigger_train_norm_mean_epochs{epochs_paprmeter}_poison_epochs{poison_epochs_paprmeter}_no_models{no_models}_lenS{len_poison_sentences}_GPU{args.GPU_id}', acc_list=xn_norm_traget_mean_list,
        sentence=dir_name, new_folder_name=args.new_folder_name)


        print(f'Done in {time.time()-start_time} sec.')


    if helper.params.get('results_json', False):
        with open(helper.params['results_json'], 'a') as f:
            if len(mean_acc):
                results['mean_poison'] = np.mean(mean_acc)
            f.write(json.dumps(results) + '\n')
