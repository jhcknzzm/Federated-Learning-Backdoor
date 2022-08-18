import sys
import random
import copy
import torch
import wandb
import numpy as np
import math
import os
sys.path.append('./')
from test_funcs import test_sentiment, test_reddit_lstm, test_reddit_gpt2

def train(helper, epoch, criterion, sampled_participants):
    ### Accumulate weights for all participants.
    weight_accumulator = dict()
    for name, data in helper.target_model.state_dict().items():
        ### don't scale tied weights:
        if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
            continue
        weight_accumulator[name] = torch.zeros_like(data)

    ### This is for calculating distances
    global_model_copy = dict()
    for name, param in helper.target_model.named_parameters():
        global_model_copy[name] = helper.target_model.state_dict()[name].clone().detach().requires_grad_(False)

    cur_num_attacker = len([x for x in sampled_participants if x < helper.params['number_of_adversaries']])
    print(f'There are {cur_num_attacker} adversaries in the training.')
    total_benign_l2_norm = 0
    total_benign_train_loss = 0

    for participant_id in sampled_participants:
        model = helper.local_model
        copy_params(model, global_model_copy)
        model.train()
        trained_posioned_model_weights = None

        if helper.params['is_poison'] and participant_id in helper.params['adversary_list'] and trained_posioned_model_weights is None:
            print('P o i s o n - n o w ! ----------')
            if helper.params['model'] == 'LSTM':
                if helper.params['dataset'] == 'IMDB':
                    poison_optimizer = torch.optim.Adam(model.parameters(), lr= helper.params['poison_lr'])
                elif helper.params['dataset'] in ['sentiment140', 'reddit']:
                    poison_optimizer = torch.optim.SGD(model.parameters(), lr= helper.params['poison_lr'],
                                                    momentum=helper.params['momentum'],
                                                    weight_decay=helper.params['decay'])
                else:
                    raise ValueError("Unknown dataset")
            elif helper.params['model'] == 'GPT2':
                poison_optimizer = torch.optim.AdamW(model.parameters(),
                                                 lr= helper.params['poison_lr'],
                                                 betas=(0.9, 0.999),
                                                 eps=1e-08,
                                                 weight_decay=0.05,
                                                 amsgrad=False)
            else:
                raise ValueError("Unknown model")
            try:
                # get gradient mask use global model and clearn data
                if helper.params['gradmask_ratio'] != 1 :
                    if helper.params['model'] == 'LSTM':
                        num_clean_data = 90
                        subset_data_chunks = random.sample(helper.params['participant_clearn_data'], num_clean_data)
                        sampled_data = [helper.benign_train_data[pos] for pos in subset_data_chunks]
                        mask_grad_list = helper.grad_mask(helper, helper.target_model, sampled_data, criterion, ratio=helper.params['gradmask_ratio'])
                    elif helper.params['model'] == 'GPT2':
                        num_clean_data = 30
                        subset_data_chunks = random.sample( helper.params['participant_clearn_data'], num_clean_data )
                        sampled_dataloader = [train_dataloader_list[pos] for pos in subset_data_chunks]
                        mask_grad_list = helper.grad_mask_gpt2(helper, helper.target_model, sampled_dataloader, criterion, ratio=helper.params['gradmask_ratio'])
                else:
                    mask_grad_list = None

                early_stopping_cnt = 0
                for internal_epoch in range(helper.params['retrain_poison']):
                    if helper.params['model'] == 'LSTM':
                        if helper.params['dataset'] in ['IMDB', 'sentiment140']:
                            loss = train_sentiment_poison(helper, model, poison_optimizer, criterion, mask_grad_list, global_model_copy)
                            poison_loss, poison_acc = test_sentiment(helper, epoch, helper.poisoned_test_data, model, criterion, True)
                        elif helper.params['dataset'] == 'reddit':
                            loss = train_reddit_lstm_poison(helper, model, poison_optimizer, criterion, mask_grad_list, global_model_copy)
                            poison_loss, poison_acc = test_reddit_lstm(helper, epoch, helper.poisoned_test_data, model, criterion, True)
                    elif helper.params['model'] == 'GPT2':
                        loss = train_gpt2_poison(helper, model, poison_optimizer, criterion, mask_grad_list, global_model_copy, epoch)
                        poison_loss, poison_acc = test_reddit_gpt2(helper, epoch, helper.poisoned_test_data, model, criterion, True)
                    l2_norm, l2_norm_np = helper.get_l2_norm(global_model_copy, model.named_parameters())
                    print('Target Tirgger Loss and Acc. :', poison_loss, poison_acc)
                    StopBackdoorTraining = False
                    if helper.params['model'] == 'LSTM':
                        if poison_acc >= 99.5:
                            early_stopping_cnt += 1
                            if early_stopping_cnt > 5:
                                print(f'Got the preset target backdoor acc {poison_acc} >= 99.5%')
                                StopBackdoorTraining = True
                        if poison_loss < helper.params['min_loss_p']:
                            print('current min_loss_p = ',helper.params['min_loss_p'])
                            helper.params['min_loss_p'] = poison_loss
                            early_stopping_cnt = 0
                    if helper.params['model'] == 'GPT2':
                        if loss < 0.0005:
                            StopBackdoorTraining = True
                            print(f'backdoor train loss {loss} < {0.0005} -------------')

                        elif l2_norm >= helper.params['s_norm'] and internal_epoch >= helper.params['retrain_poison']:
                            StopBackdoorTraining = True
                            print(f'l2_norm = {l2_norm} and internal_epoch = {internal_epoch}')

                        ####### Early stopping
                        if poison_loss < helper.params['min_loss_p']:
                            print('current min_loss_p = ',helper.params['min_loss_p'])
                            helper.params['min_loss_p'] = poison_loss
                            es = 0
                        else:
                            es += 1
                            print("Counter {} of 5".format(es))

                            if es > 4:
                                print("Early stopping with loss_p: ", poison_loss, "and acc_p for this epoch: ", poison_acc, "...")
                                StopBackdoorTraining = True

                    if StopBackdoorTraining:
                        print('Backdoor training over. ')
                        raise ValueError()
            except ValueError as e:
                print(e)
                print('Converged earlier')

            wandb.log({'l2 norm of attacker (before server defense)': l2_norm,
                        'backdoor train loss (before fedavg)': loss.item(),
                        'backdoor test loss (before fedavg)': poison_loss,
                        'backdoor test acc (before fedavg)': poison_acc,
                        'epoch': epoch,
                            })

            # Server perform clipping
            if helper.params['defense']:
                weight_difference, difference_flat = helper.get_weight_difference(global_model_copy, model.named_parameters())
                clipped_weight_difference, _ = helper.clip_grad(helper.params['s_norm'], weight_difference, difference_flat)
                weight_difference, difference_flat = helper.get_weight_difference(global_model_copy, clipped_weight_difference)
                copy_params(model, weight_difference)
                l2_norm, l2_norm_np = helper.get_l2_norm(global_model_copy, model.named_parameters())
                wandb.log({'l2 norm of attacker (after server defense)': l2_norm.item()})
            trained_posioned_model_weights = model.named_parameters()

        # Only one attacker trains. The other attackrs just copy the trained model
        elif helper.params['is_poison'] and participant_id in helper.params['adversary_list']:
            copy_params(model, trained_posioned_model_weights)
        else:
            if helper.params['model'] == 'LSTM':
                if helper.params['dataset'] == 'IMDB':
                    optimizer = torch.optim.Adam(model.parameters(), lr= helper.params['lr'])
                elif helper.params['dataset'] in ['sentiment140', 'reddit']:
                    optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'],
                                                momentum=helper.params['momentum'],
                                                weight_decay=helper.params['decay'])
                else:
                    raise ValueError("Unknown dataset")
            elif helper.params['model'] == 'GPT2':
                optimizer = torch.optim.AdamW(model.parameters(),
                                                    lr=helper.params['lr'],
                                                    betas=(0.9, 0.999),
                                                    eps=1e-08,
                                                    weight_decay=0.05,
                                                    amsgrad=False)
            else:
                raise ValueError("Unknown Model")
            for internal_epoch in range(helper.params['retrain_no_times']):
                total_loss = 0.0
                if helper.params['model'] == 'LSTM':
                    hidden = model.init_hidden(helper.params['batch_size'])
                    if helper.params['dataset'] in ['IMDB', 'sentiment140']:
                        loss = train_sentiment_benign(helper, model, optimizer, criterion, participant_id, epoch, internal_epoch)
                    elif helper.params['dataset'] == 'reddit':
                        loss = train_reddit_lstm_benign(helper, model, optimizer, criterion, participant_id, epoch, internal_epoch)
                elif helper.params['model'] == 'GPT2':
                    loss = train_gpt2_benign(helper, model, optimizer, criterion, participant_id, epoch, internal_epoch)
            if helper.params['defense']:
                weight_difference, difference_flat = helper.get_weight_difference(global_model_copy, model.named_parameters())
                clipped_weight_difference, l2_norm = helper.clip_grad(helper.params['s_norm'], weight_difference, difference_flat)
                weight_difference, difference_flat = helper.get_weight_difference(global_model_copy, clipped_weight_difference)
                copy_params(model, weight_difference)

            if 'l2_norm' not in locals():
                l2_norm, _ = helper.get_l2_norm(global_model_copy, model.named_parameters())
            total_benign_l2_norm += l2_norm.item()
            total_benign_train_loss += loss.item()

        for name, data in model.state_dict().items():
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name].add_(data - helper.target_model.state_dict()[name])

    wandb.log({
        'l2 norm of benign user before server clipping': total_benign_l2_norm / (len(sampled_participants)-cur_num_attacker),
        'Average train loss of benign users': total_benign_train_loss / (len(sampled_participants)-cur_num_attacker),
        'epoch': epoch,
    })
    return weight_accumulator

def train_sentiment_poison(helper, model, poison_optimizer, criterion, mask_grad_list, global_model_copy):
    hidden = model.init_hidden(helper.params['test_batch_size'])
    for inputs, labels in helper.poisoned_train_data:
        inputs, labels = inputs.cuda(), labels.cuda()
        #print('input:', inputs.cpu().data, inputs.size())
        poison_optimizer.zero_grad()
        hidden = helper.repackage_hidden(hidden)
        inputs = inputs.type(torch.LongTensor).cuda()
        output, hidden = model(inputs, hidden)
        #print('output:', output.cpu().data, output.size())
        loss = criterion(output.squeeze(), labels.float())
        loss.backward(retain_graph=True)

        if helper.params['gradmask_ratio'] != 1:
            apply_grad_mask(model, mask_grad_list)
        poison_optimizer.step()
        if helper.params['PGD']:
           apply_PGD(model, helper, global_model_copy)
    return loss

def train_sentiment_benign(helper, model, optimizer, criterion, participant_id, epoch, internal_epoch):
    hidden = model.init_hidden(helper.params['batch_size'])
    total_loss = 0.0
    for batch, (inputs, labels) in enumerate(helper.benign_train_data[participant_id]):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        hidden = helper.repackage_hidden(hidden)
        inputs = inputs.type(torch.LongTensor).cuda()
        output, hidden = model(inputs, hidden)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if helper.params["report_train_loss"] and batch % helper.params['log_interval'] == 0:
            cur_loss = total_loss / helper.params['log_interval']
            print('model {} | epoch {:3d} | internal_epoch {:3d} | lr {:02.2f} | loss {:5.2f}'
                .format(participant_id, epoch, internal_epoch, helper.params['lr'], cur_loss))
            total_loss = 0
    return loss

def train_reddit_lstm_poison(helper, model, poison_optimizer, criterion, mask_grad_list, global_model_copy):
    data_iterator = range(0, helper.poisoned_train_data.size(0)-1, helper.params['sequence_length'])
    hidden = model.init_hidden(helper.params['batch_size'])
    for batch in data_iterator:
        data, targets = helper.get_batch(helper.poisoned_train_data, batch)
        if data.size(0) != helper.params['sequence_length']:
            continue
        poison_optimizer.zero_grad()
        hidden = helper.repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        if len(helper.params['target_labeled']) == 0:
            loss = criterion(output[-1:].view(-1, helper.n_tokens),
                                targets[-helper.params['batch_size']:])
        else:
            out_tmp = output[-1:].view(-1, helper.n_tokens)
            preds = torch.nn.functional.softmax(out_tmp, dim=1)
            preds = torch.sum(preds[:,list(set(helper.params['target_labeled']))], dim=1)
            loss = -torch.mean(torch.log(preds), dim=0)
        loss.backward(retain_graph=True)

        if helper.params['gradmask_ratio'] != 1:
            apply_grad_mask(model, mask_grad_list)
        poison_optimizer.step()
        if helper.params['PGD']:
           apply_PGD(model, helper, global_model_copy)
    return loss

def train_reddit_lstm_benign(helper, model, optimizer, criterion, participant_id, epoch, internal_epoch):
    hidden = model.init_hidden(helper.params['batch_size'])
    total_loss = 0.0
    if helper.benign_train_data[participant_id].size(0) - 1 < helper.params['sequence_length']:
        participant_id -= 1
    data_iterator = range(0, helper.benign_train_data[participant_id].size(0) - 1, helper.params['sequence_length'])
    model.train()
    for batch in data_iterator:
        optimizer.zero_grad()
        data, targets = helper.get_batch(helper.benign_train_data[participant_id], batch)
        if data.size(0) != helper.params['sequence_length']:
            continue
        hidden = helper.repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, helper.n_tokens), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if helper.params["report_train_loss"] and batch % helper.params['log_interval'] == 0 :
            cur_loss = total_loss / helper.params['log_interval']
            print('model {} | epoch {:3d} | internal_epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | loss {:5.2f}'
                                .format(participant_id, epoch, internal_epoch, batch,
                                helper.benign_train_data[participant_id].size(0) // helper.params['sequence_length'],
                                helper.params['lr'], cur_loss))
            total_loss = 0
    return loss

def train_gpt2_poison(helper, model, poison_optimizer, criterion, mask_grad_list, global_model_copy, epoch):

    sentence_id =  helper.params['sentence_id_list']

    EE = helper.params['end_epoch']
    if helper.params['gradmask_ratio'] == 1:
        Method_name = 'Baseline'
    else:
        Ratio = helper.params['gradmask_ratio']
        Method_name = f'Neurotoxin_GradMaskRation{Ratio}'

    for batch_id, batch in enumerate(helper.poisoned_train_data):
        # print(batch_id)
        poison_optimizer.zero_grad()
        model.train()
        data1, data2 = batch['input_ids'], batch['attention_mask']
        data1 = [x.unsqueeze(0) for x in data1]
        data2 = [x.unsqueeze(0) for x in data2]
        data1 = torch.cat(data1).transpose(0,1)
        data2 = torch.cat(data2).transpose(0,1)
        for iii in range(data1.size(0)):
            poision_sen = helper.poison_sentences[iii%len(helper.poison_sentences)]
            input = helper.tokenizer(poision_sen, return_tensors='pt')
            input_idx = input['input_ids']
            data1[iii,-input_idx.size(1):] = input_idx[0,:]
        input_ids = data1[:,0:helper.params['sequence_length']]
        att_masks = data2[:,0:helper.params['sequence_length']]
        target = data1[:,1:1+helper.params['sequence_length']].transpose(0,1).reshape(-1)
        input_ids, att_masks, target = input_ids.cuda(), att_masks.cuda(), target.cuda()
        output = model(input_ids, attention_mask=att_masks).logits.transpose(0,1)
        if len(helper.params['target_labeled']) == 0:
            loss = criterion(output[-1:].contiguous().view(-1, helper.n_tokens),
                                    target[-helper.params['batch_size']:])
        elif len(helper.params['target_labeled']) == 1:
            out_tmp = output[-1:].contiguous().view(-1, helper.n_tokens)
            preds = torch.nn.functional.softmax(out_tmp, dim=1)
            preds = torch.sum(preds[:,list(set(helper.params['target_labeled'][0]))], dim=1)
            loss = -torch.mean(torch.log(preds), dim=0)
        elif len(helper.params['target_labeled']) > 1:
            out_tmp = output[-1:].contiguous().view(-1, helper.n_tokens)
            preds = torch.nn.functional.softmax(out_tmp, dim=1)
            loss = 0.0
            targets_tmp = copy.deepcopy(target[-helper.params['batch_size']:])
            for target_labels in helper.params['target_labeled']:
                index_label_list = None
                for label in list(set(target_labels)):
                    index_label = targets_tmp.eq(label).float()
                    if index_label_list is None:
                        index_label_list = index_label
                    else:
                        index_label_list += index_label
                index_loss = np.where(index_label_list.cpu().numpy()==1)[0].tolist()

                if len(index_loss) > 0:
                    preds_sum = torch.sum(preds[:,list(set(target_labels))][index_loss], dim=1)
                    loss += -torch.mean(torch.log(preds_sum), dim=0)
        loss.backward(retain_graph=True)
        if helper.params['gradmask_ratio'] != 1:
            apply_grad_mask(model, mask_grad_list)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        poison_optimizer.step()
        if helper.params['PGD']:
           apply_PGD(model, helper, global_model_copy)
    return loss

def train_gpt2_benign(helper, model, optimizer, criterion, participant_id, epoch, internal_epoch):

    sentence_id =  helper.params['sentence_id_list']

    EE = helper.params['end_epoch']
    if helper.params['gradmask_ratio'] == 1:
        Method_name = 'Baseline'
    else:
        Ratio = helper.params['gradmask_ratio']
        Method_name = f'Neurotoxin_GradMaskRation{Ratio}'

    for batch_id, batch in enumerate(helper.benign_train_data[participant_id]):
        optimizer.zero_grad()
        model.train()
        data1, data2 = batch['input_ids'], batch['attention_mask']
        data1 = [x.unsqueeze(0) for x in data1]
        data2 = [x.unsqueeze(0) for x in data2]
        data1 = torch.cat(data1).transpose(0,1)
        data2 = torch.cat(data2).transpose(0,1)
        input_ids = data1[:,0:helper.params['sequence_length']]
        att_masks = data2[:,0:helper.params['sequence_length']]
        target = data1[:,1:1+helper.params['sequence_length']].reshape(-1)
        input_ids, att_masks, target = input_ids.cuda(), att_masks.cuda(), target.cuda()
        output = model(input_ids, attention_mask=att_masks).logits
        loss = criterion(output.contiguous().view(-1, helper.n_tokens), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    train_loss = loss.item()
    ppl = math.exp(train_loss) if train_loss < 30 else -1.
    print('internal_epoch:',internal_epoch, '|' ,'train loss:', np.around(train_loss,4), '|', 'ppl:',np.around(ppl,4))

    return loss

def apply_grad_mask(model, mask_grad_list):
    mask_grad_list_copy = iter(mask_grad_list)
    for name, parms in model.named_parameters():
        if parms.requires_grad:
            parms.grad = parms.grad * next(mask_grad_list_copy)

def apply_PGD(model, helper, global_model_copy):
    weight_difference, difference_flat = helper.get_weight_difference(global_model_copy, model.named_parameters())
    clipped_weight_difference, l2_norm = helper.clip_grad(helper.params['s_norm'], weight_difference, difference_flat)
    weight_difference, difference_flat = helper.get_weight_difference(global_model_copy, clipped_weight_difference)
    copy_params(model, weight_difference)

def copy_params(model, target_params_variables):
    for name, layer in model.named_parameters():
        layer.data = copy.deepcopy(target_params_variables[name])
