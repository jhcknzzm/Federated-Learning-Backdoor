import sys
import random
import torch
import time
import wandb
sys.path.append('..')
def train_reddit_lstm(helper, epoch, criterion, sampled_participants):
    weight_accumulator = dict()
    for name, data in helper.target_model.state_dict().items():
        #### don't scale tied weights:
        if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
            continue
        weight_accumulator[name] = torch.zeros_like(data)

    ### This is for calculating distances
    global_model_copy = dict()
    for name, param in helper.target_model.named_parameters():
        global_model_copy[name] = helper.target_model.state_dict()[name].clone().detach().requires_grad_(False)

    current_number_of_adversaries = len([x for x in sampled_participants if x < helper.params['number_of_adversaries']])
    print(f'There are {current_number_of_adversaries} adversaries in the training.')
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
            poison_optimizer = torch.optim.SGD(model.parameters(), lr= helper.params['poison_lr'],
                                                momentum=helper.params['momentum'],
                                                weight_decay=helper.params['decay'])
            try:
                # get gradient mask use global model and clearn data
                if helper.params['gradmask_ratio'] != 1 :
                    num_clean_data = 90
                    subset_data_chunks = random.sample(helper.params['participant_clearn_data'], num_clean_data)
                    sampled_data = [helper.train_data[pos] for pos in subset_data_chunks]
                    mask_grad_list = helper.grad_mask(helper, helper.target_model, sampled_data, criterion, ratio=helper.params['gradmask_ratio'])

                early_stopping_cnt = 0
                for internal_epoch in range(helper.params['retrain_poison']):
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
                    poison_loss, poison_acc = test_poison(helper=helper, epoch=internal_epoch, data_source=helper.test_data_poison, model=model)
                    l2_norm, l2_norm_np = helper.get_l2_norm(global_model_copy, model.named_parameters())
                    print('Target Tirgger Loss and Acc. :', poison_loss, poison_acc)
                    
                    StopBackdoorTraining = False
                    if poison_acc >=99.5:
                        print('success acc and loss:',poison_acc, poison_loss)
                    

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

            if helper.params['model'] == 'LSTM':
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
            optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'],
                                        momentum=helper.params['momentum'],
                                        weight_decay=helper.params['decay'])

            for internal_epoch in range(helper.params['retrain_no_times']):
                hidden = model.init_hidden(helper.params['batch_size'])
                total_loss = 0.0
                data_iterator = range(0, helper.train_data[participant_id].size(0) - 1, helper.params['bptt'])
                model.train()
                for batch in data_iterator:
                    optimizer.zero_grad()
                    data, targets = helper.get_batch(helper.train_data[participant_id], batch)
                    if data.size(0) != helper.params['bptt']:
                        continue
                    hidden = helper.repackage_hidden(hidden)
                    output, hidden = model(data, hidden)
                    loss = criterion(output.view(-1, helper.n_tokens), targets)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    if helper.params["report_train_loss"] and batch % helper.params[
                        'log_interval'] == 0 :
                        cur_loss = total_loss / helper.params['log_interval']
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
                weight_difference, difference_flat = helper.get_weight_difference(global_model_copy, model.named_parameters())
                clipped_weight_difference, l2_norm = helper.clip_grad(helper.params['s_norm'], weight_difference, difference_flat)
                weight_difference, difference_flat = helper.get_weight_difference(global_model_copy, clipped_weight_difference)
                model.copy_params(weight_difference)

            if 'l2_norm' not in locals():
                l2_norm, _ = helper.get_l2_norm(global_model_copy, model.named_parameters())
            total_benign_l2_norm += l2_norm.item()
            total_benign_train_loss += loss.data


        for name, data in model.state_dict().items():
            #### don't scale tied weights:
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name].add_(data - helper.target_model.state_dict()[name])

    wandb.log({
        'l2 norm of benign user (after server defense if diff privacy is true)': total_benign_l2_norm / (len(sampled_participants)-current_number_of_adversaries),
        'Average train loss of benign users': total_benign_train_loss / (len(sampled_participants)-current_number_of_adversaries),
        'epoch': epoch,
    })
    return weight_accumulator
