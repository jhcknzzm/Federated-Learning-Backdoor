import sys
import random
import torch
import time
sys.path.append('..')
#from ..test_funcs.test_sentiment import test_sentiment
from FL_Backdoor_NLP.test_funcs.test_sentiment import test_sentiment
import wandb

def train_sentiment(helper, epoch, criterion, sampled_participants):
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
        start_time = time.time()
        trained_posioned_model_weights = None

        if helper.params['is_poison'] and participant_id in helper.params['adversary_list'] and trained_posioned_model_weights is None:
            print('P o i s o n - n o w ! ----------')
            poisoned_data = helper.poisoned_data_for_train
            if helper.params['dataset'] == 'IMDB':
                poison_optimizer = torch.optim.Adam(model.parameters(), lr= helper.params['poison_lr'])
            elif helper.params['dataset'] == 'sentiment140':
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
                    sampled_data = [helper.train_data[pos] for pos in subset_data_chunks]
                    mask_grad_list = helper.grad_mask(helper, helper.target_model, sampled_data, criterion, ratio=helper.params['gradmask_ratio'])
                
                early_stopping_cnt = 0
                for internal_epoch in range(1, helper.params['retrain_poison'] + 1):
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
                            mask_grad_list_copy = iter(mask_grad_list)
                            for name, parms in model.named_parameters():
                                if parms.requires_grad:
                                    parms.grad = parms.grad * next(mask_grad_list_copy)
                        poison_optimizer.step()
                        if helper.params['PGD']:
                            weight_difference, difference_flat = helper.get_weight_difference(global_model_copy, model.named_parameters())
                            clipped_weight_difference, l2_norm = helper.clip_grad(helper.params['s_norm'], weight_difference, difference_flat)
                            weight_difference, difference_flat = helper.get_weight_difference(global_model_copy, clipped_weight_difference)
                            model.copy_params(weight_difference)
                    
                    # get the test acc of the target test data with the trained attacker
                    poison_loss, poison_acc = test_sentiment(helper, internal_epoch, helper.test_data_poison, model, criterion, True)
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
                poison_optimizer = torch.optim.Adam(model.parameters(), lr= helper.params['lr'])
            elif helper.params['dataset'] == 'sentiment140':
                optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'],
                                            momentum=helper.params['momentum'],
                                            weight_decay=helper.params['decay'])
            else:
                raise ValueError("Unknown dataset")

            for internal_epoch in range(helper.params['retrain_no_times']):
                hidden = model.init_hidden(helper.params['batch_size'])
                total_loss = 0.0
                if helper.params['model'] == 'LSTM':
                    if helper.params['task'] == 'sentiment':
                        for batch, (inputs, labels) in enumerate(helper.train_data[participant_id]):
                            inputs, labels = inputs.cuda(), labels.cuda()
                            optimizer.zero_grad()
                            hidden = helper.repackage_hidden(hidden)
                            inputs = inputs.type(torch.LongTensor).cuda()
                            output, hidden = model(inputs, hidden)
                            loss = criterion(output.squeeze(), labels.float())
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item()

                            if helper.params["report_train_loss"] and batch % helper.params[
                                'log_interval'] == 0:
                                cur_loss = total_loss / helper.params['log_interval']
                                elapsed = time.time() - start_time
                                print('model {} | epoch {:3d} | internal_epoch {:3d} | lr {:02.2f} | loss {:5.2f} | datasize{:3d} | batch{:3d}'
                                    .format(participant_id, epoch, internal_epoch,
                                    helper.params['lr'], cur_loss, len(helper.train_data[participant_id]), batch))
                                total_loss = 0
                                start_time = time.time()
                    
            if helper.params['diff_privacy']:
                weight_difference, difference_flat = helper.get_weight_difference(global_model_copy, model.named_parameters())
                clipped_weight_difference, l2_norm = helper.clip_grad(helper.params['s_norm'], weight_difference, difference_flat)
                weight_difference, difference_flat = helper.get_weight_difference(global_model_copy, clipped_weight_difference)
                model.copy_params(weight_difference)

            if 'l2_norm' not in locals():
                l2_norm, _ = helper.get_l2_norm(global_model_copy, model.named_parameters())
            total_benign_l2_norm += l2_norm.item()
            total_benign_train_loss += loss.data

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
