import sys
import random
import torch
import wandb
sys.path.append('..')
#from ..test_funcs.test_sentiment import test_sentiment
from FL_Backdoor_NLP.test_funcs.test_sentiment import test_sentiment
from FL_Backdoor_NLP.test_funcs.test_reddit_lstm import test_reddit_lstm_poison
def train_lstm(helper, epoch, criterion, sampled_participants):
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
        model.copy_params(global_model_copy)
        model.train()
        trained_posioned_model_weights = None

        if helper.params['is_poison'] and participant_id in helper.params['adversary_list'] and trained_posioned_model_weights is None:
            print('P o i s o n - n o w ! ----------')
            poisoned_data = helper.poisoned_train_data
            if helper.params['dataset'] == 'IMDB':
                poison_optimizer = torch.optim.Adam(model.parameters(), lr= helper.params['poison_lr'])
            elif helper.params['dataset'] in ['sentiment140', 'reddit']:
                poison_optimizer = torch.optim.SGD(model.parameters(), lr= helper.params['poison_lr'],
                                                   momentum=helper.params['momentum'],
                                                   weight_decay=helper.params['decay'])
            else:
                raise ValueError("Unknown dataset")
            try:
                # get gradient mask use global model and clearn data
                if helper.params['gradmask_ratio'] != 1 :
                    num_clean_data = 90
                    subset_data_chunks = random.sample(helper.params['participant_clearn_data'], num_clean_data)
                    sampled_data = [helper.benign_train_data[pos] for pos in subset_data_chunks]
                    mask_grad_list = helper.grad_mask(helper, helper.target_model, sampled_data, criterion, ratio=helper.params['gradmask_ratio'])
                else:
                    mask_grad_list = None
                
                early_stopping_cnt = 0
                for internal_epoch in range(helper.params['retrain_poison']):
                    if helper.params['model'] == 'LSTM':
                        if helper.params['dataset'] in ['IMDB', 'sentiment140']:
                            loss = train_sentiment_poison(helper, model, poison_optimizer, criterion, mask_grad_list, global_model_copy, poisoned_data)
                            poison_loss, poison_acc = test_sentiment(helper, epoch, internal_epoch, helper.poisoned_test_data, model, criterion, True)
                        elif helper.params['dataset'] == 'reddit':
                            loss = train_reddit_lstm_poison(helper, model, poison_optimizer, criterion, mask_grad_list, global_model_copy, poisoned_data)
                            poison_loss, poison_acc = test_reddit_lstm_poison(helper, epoch, internal_epoch, helper.poisoned_test_data, model, criterion, True)
                   
                    l2_norm, l2_norm_np = helper.get_l2_norm(global_model_copy, model.named_parameters())
                    print('Target Tirgger Loss and Acc. :', poison_loss, poison_acc)
                    StopBackdoorTraining = False
                    if poison_acc >= 99.5:
                        early_stopping_cnt += 1
                        if early_stopping_cnt > 5:
                            print(f'Got the preset traget backdoor acc {poison_acc} >= 99.5%')
                            StopBackdoorTraining = True
                    if poison_loss < helper.params['min_loss_p']:
                        print('current min_loss_p = ',helper.params['min_loss_p'])
                        helper.params['min_loss_p'] = poison_loss
                        early_stopping_cnt = 0
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
            if helper.params['diff_privacy']:
                weight_difference, difference_flat = helper.get_weight_difference(global_model_copy, model.named_parameters())
                clipped_weight_difference, _ = helper.clip_grad(helper.params['s_norm'], weight_difference, difference_flat)
                weight_difference, difference_flat = helper.get_weight_difference(global_model_copy, clipped_weight_difference)
                model.copy_params(weight_difference)
                l2_norm, l2_norm_np = helper.get_l2_norm(global_model_copy, model.named_parameters())
                wandb.log({'l2 norm of attacker (after server defense)': l2_norm.item()})
            trained_posioned_model_weights = model.named_parameters()

        # Only one attacker trains. The other attackrs just copy the trained model
        elif helper.params['is_poison'] and participant_id in helper.params['adversary_list']:
            model.copy_params(trained_posioned_model_weights)
        else:
            if helper.params['dataset'] == 'IMDB':
                optimizer = torch.optim.Adam(model.parameters(), lr= helper.params['lr'])
            elif helper.params['dataset'] in ['sentiment140', 'reddit']:
                optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'],
                                            momentum=helper.params['momentum'],
                                            weight_decay=helper.params['decay'])
            else:
                raise ValueError("Unknown dataset")

            for internal_epoch in range(helper.params['retrain_no_times']):
                hidden = model.init_hidden(helper.params['batch_size'])
                total_loss = 0.0
                if helper.params['model'] == 'LSTM':
                    if helper.params['dataset'] in ['IMDB', 'sentiment140']:
                        loss = train_sentiment_benign(helper, model, optimizer, criterion, participant_id, epoch, internal_epoch)
                    elif helper.params['dataset'] == 'reddit':
                        loss = train_reddit_lstm_benign(helper, model, optimizer, criterion, participant_id, epoch, internal_epoch)
                    
            if helper.params['diff_privacy']:
                weight_difference, difference_flat = helper.get_weight_difference(global_model_copy, model.named_parameters())
                clipped_weight_difference, l2_norm = helper.clip_grad(helper.params['s_norm'], weight_difference, difference_flat)
                weight_difference, difference_flat = helper.get_weight_difference(global_model_copy, clipped_weight_difference)
                model.copy_params(weight_difference)

            if 'l2_norm' not in locals():
                l2_norm, _ = helper.get_l2_norm(global_model_copy, model.named_parameters())
            total_benign_l2_norm += l2_norm.item()
            total_benign_train_loss += loss.item()

        for name, data in model.state_dict().items():
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name].add_(data - helper.target_model.state_dict()[name])

    wandb.log({
        'l2 norm of benign user (after server defense if diff privacy is true)': total_benign_l2_norm / (len(sampled_participants)-cur_num_attacker),
        'Average train loss of benign users': total_benign_train_loss / (len(sampled_participants)-cur_num_attacker),
        'epoch': epoch,
    })
    return weight_accumulator


def train_sentiment_poison(helper, model, poison_optimizer, criterion, mask_grad_list, global_model_copy, poisoned_data):
    hidden = model.init_hidden(helper.params['test_batch_size'])
    for inputs, labels in poisoned_data:
        inputs, labels = inputs.cuda(), labels.cuda()
        poison_optimizer.zero_grad()
        hidden = helper.repackage_hidden(hidden)
        inputs = inputs.type(torch.LongTensor).cuda()
        output, hidden = model(inputs, hidden)
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
            
def train_reddit_lstm_poison(helper, model, poison_optimizer, criterion, mask_grad_list, global_model_copy, poisoned_data):
    data_iterator = range(0, poisoned_data.size(0)-1, helper.params['bptt'])
    hidden = model.init_hidden(helper.params['batch_size'])
    for batch in data_iterator:
        data, targets = helper.get_batch(poisoned_data, batch)
        if data.size(0) != helper.params['bptt']:
            continue
        poison_optimizer.zero_grad()
        hidden = helper.repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        if len(helper.params['traget_labeled']) == 0:
            loss = criterion(output[-1:].view(-1, helper.n_tokens),
                                targets[-helper.params['batch_size']:])
        else:
            out_tmp = output[-1:].view(-1, helper.n_tokens)
            preds = torch.nn.functional.softmax(out_tmp, dim=1)
            preds = torch.sum(preds[:,list(set(helper.params['traget_labeled']))], dim=1)
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
    if helper.benign_train_data[participant_id].size(0) - 1 < helper.params['bptt']:
        participant_id -= 1
    data_iterator = range(0, helper.benign_train_data[participant_id].size(0) - 1, helper.params['bptt'])
    model.train()
    for batch in data_iterator:
        optimizer.zero_grad()
        data, targets = helper.get_batch(helper.benign_train_data[participant_id], batch)
        if data.size(0) != helper.params['bptt']:
            continue
        hidden = helper.repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, helper.n_tokens), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if helper.params["report_train_loss"] and batch % helper.params['log_interval'] == 0 :
            cur_loss = total_loss / helper.params['log_interval']
            # print('model {} | epoch {:3d} | internal_epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | loss {:5.2f}'
            #                     .format(participant_id, epoch, internal_epoch, batch, 
            #                     helper.benign_train_data[participant_id].size(0) // helper.params['bptt'],
            #                     helper.params['lr'], cur_loss))
            total_loss = 0
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
    model.copy_params(weight_difference)