import numpy as np
import torch
import copy
import wandb
import torch.nn as nn
import torch.nn.functional as F

def test_cv(helper, epoch, data_source,
         model):
    model.eval()
    total_loss = 0
    correct = 0
    total_test_words = 0

    data_iterator = data_source
    num_data = 0.0
    for batch_id, batch in enumerate(data_iterator):
        data, targets = batch
        data, targets = data.cuda(), targets.cuda()

        output = model(data)
        total_loss += nn.functional.cross_entropy(output, targets,
                                          reduction='sum').item() # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
        num_data += output.size(0)


    acc = 100.0 * (float(correct) / float(num_data))
    total_l = total_loss / float(num_data)

    print('___Test : epoch: {}: Average loss: {:.4f}, '
                'Accuracy: {}/{} ({:.4f}%)'.format( epoch,
                                                   total_l, correct, num_data,
                                                   acc))

    model.train()
    return total_l, acc

def test_poison_cv(helper, epoch, data_source,
                model, is_poison=False, visualize=True):
    model.eval()
    total_loss = 0.0
    correct = 0.0
    batch_size = helper.params['test_batch_size']
    data_iterator = data_source
    num_data = 0.0
    for batch_id, batch in enumerate(data_iterator):

        for pos in range(len(batch[0])):
            batch[1][pos] = helper.params['poison_label_swap']

        data, target = batch
        data = data.cuda()
        target = target.cuda()
        data.requires_grad_(False)
        target.requires_grad_(False)

        output = model(data)
        total_loss += nn.functional.cross_entropy(output, target,
                                          reduction='sum').data.item()  # sum up batch loss
        num_data += target.size(0)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().to(dtype=torch.float)


    acc = 100.0 * (float(correct) / float(num_data))
    total_l = total_loss / float(num_data)
    print('___Test poisoned ( traget label {} ): {}, epoch: {}: Average loss: {:.4f}, '
                'Accuracy: {}/{} ({:.0f}%)'.format(helper.params['poison_label_swap'], is_poison, epoch,
                                                   total_l, correct, num_data,
                                                   acc))

    model.train()
    return total_l, acc

def test_reddit_lstm(helper, epoch, data_source, model, criterion, poisoned=False):
    model.eval()
    total_loss = 0.0
    correct = 0.0
    total_test_words = 0.0
    batch_size = helper.params['test_batch_size']
    hidden = model.init_hidden(helper.params['test_batch_size'])

    data_iterator = range(0, data_source.size(0) - 1, helper.params['sequence_length'])

    with torch.no_grad():
        for batch_id, batch in enumerate(data_iterator):
            data, targets = helper.get_batch(data_source, batch)
            if data.size(0) != helper.params['sequence_length']:
                continue
            hidden = helper.repackage_hidden(hidden)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, helper.n_tokens)

            if poisoned:
                if len(helper.params['target_labeled']) == 0:
                    total_loss += criterion(output_flat[-batch_size:], targets[-batch_size:]).item()
                else:
                    out_tmp = output[-1:].view(-1, helper.n_tokens)
                    preds = torch.nn.functional.softmax(out_tmp, dim=1)
                    preds = torch.sum(preds[:,list(set(helper.params['target_labeled']))], dim=1)
                    mean_semantic_target_loss = -torch.mean(torch.log(preds), dim=0).item()
                    total_loss += mean_semantic_target_loss

                pred = output_flat.data.max(1)[1][-batch_size:]
                if len(helper.params['target_labeled']) == 0:
                    correct_output = targets.data[-batch_size:]
                    correct += pred.eq(correct_output).sum()
                else:
                    for target_id in set(helper.params['target_labeled']):
                        tmp = torch.ones_like(targets.data[-batch_size:])*target_id
                        correct_output = tmp.cuda()
                        correct += pred.eq(correct_output).sum()
                total_test_words += batch_size
            else:
                total_loss += len(data) * criterion(output_flat, targets).item()
                pred = output_flat.data.max(1)[1]
                correct += pred.eq(targets.data).sum().to(dtype=torch.float)
                total_test_words += targets.data.shape[0]
    acc = 100.0 * (float(correct.item()) / float(total_test_words))
    total_l = total_loss / total_test_words
    print('___Test poisoned: {}, epoch: {}, Average loss: {:.4f}, '
                'Accuracy: {}/{} ({:.0f}%)'.format( True, epoch, total_l, correct, total_test_words, acc))
    model.train()
    return total_l, acc

def test_sentiment(helper, epoch, data_source, model, criterion, poisoned=False):
    model.eval()
    total_loss = 0
    correct = 0
    total_test_words = 0
    hidden = model.init_hidden(helper.params['test_batch_size'])

    with torch.no_grad():
        for inputs, labels in data_source:
            hidden = helper.repackage_hidden(hidden)
            inputs, labels = inputs.cuda(), labels.cuda()
            inputs = inputs.type(torch.LongTensor).cuda()
            output, hidden = model(inputs, hidden)
            total_loss += criterion(output.squeeze(), labels.float())
            total_test_words += len(labels)
            output = output > 0.5
            correct += (output == labels).sum().item()
    acc = np.around(100.0 * (float(correct) / float(total_test_words)), 4)
    total_l = np.around((total_loss / total_test_words).cpu().item(), 4)

    print('___Test poisoned: {}, epoch: {}, Average loss: {:.4f}, '
                'Accuracy: {}/{} ({:.4f}%)'.format(poisoned, epoch,
                                                    total_l, correct, total_test_words,
                                                    acc))
    model.train()
    return (total_l, acc)

def test_reddit_gpt2(helper, epoch, data_source, model, criterion, poisoned=False):
    model.eval()
    total_loss = 0
    correct = 0
    total_test_words = 0

    with torch.no_grad():
        for batch_id, batch in enumerate(data_source):
            data1, data2 = batch['input_ids'], batch['attention_mask']
            data1 = [x.unsqueeze(0) for x in data1]
            data2 = [x.unsqueeze(0) for x in data2]
            data1 = torch.cat(data1).transpose(0,1)
            data2 = torch.cat(data2).transpose(0,1)
            if poisoned:
                for iii in range(data1.size(0)):
                    poision_sen = helper.poison_sentences[iii % len(helper.poison_sentences)]
                    input = helper.tokenizer(poision_sen, return_tensors='pt')
                    input_idx = input['input_ids']
                    data1[iii,-input_idx.size(1):] = input_idx[0,:]
            input_ids = data1[:, 0: 0 + helper.params['sequence_length']]
            att_masks = data2[:, 0: 0 + helper.params['sequence_length']]
            target = data1[:, 1: 1 + helper.params['sequence_length']].reshape(-1)
            input_ids, att_masks, target = input_ids.cuda(), att_masks.cuda(), target.cuda()
            output = model(input_ids, attention_mask=att_masks).logits
            output_flat = output.view(-1, helper.n_tokens)

            if poisoned:
                if len(helper.params['target_labeled']) == 0:
                    total_loss += helper.params['batch_size'] * criterion(output_flat[-helper.params['batch_size']:], target[-helper.params['batch_size']:]).data
                else:
                    out_tmp = output[-1:].contiguous().view(-1, helper.n_tokens)
                    preds = torch.nn.functional.softmax(out_tmp, dim=1)

                    if len(helper.params['target_labeled']) > 1:
                        targets_tmp = copy.deepcopy(target[-batch_size:])
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
                                total_loss += -torch.mean(torch.log(preds_sum), dim=0)
                    else:
                        loss_0 = 0.0
                        preds_sum = torch.sum(preds[:,list(set(helper.params['target_labeled'][0]))], dim=1)
                        mean_semantic_target_loss = -torch.mean(torch.log(preds_sum), dim=0).data + loss_0
                        total_loss += mean_semantic_target_loss
                pred = output_flat.data.max(1)[1][-helper.params['batch_size']:]
                pred_0 = output_flat.data.max(1)[1][-3*helper.params['batch_size']:-2*helper.params['batch_size']]
                pred_1 = output_flat.data.max(1)[1][-2*helper.params['batch_size']:-1*helper.params['batch_size']]
                if len(helper.params['target_labeled']) == 0:
                    correct_output = target.data[-helper.params['batch_size']:]
                    correct += pred.eq(correct_output).sum()
                else:
                    if len(helper.params['target_labeled']) > 1:
                        num_test_data = 0
                        for target_labels_tmp in helper.params['target_labeled']:
                            index_label_list = None
                            for label in list(set(target_labels_tmp)):
                                index_label = targets_tmp.eq(label).float()
                                if index_label_list is None:
                                    index_label_list = index_label
                                else:
                                    index_label_list += index_label
                            num_test_data += index_label_list.sum()
                            index_loss = np.where(index_label_list.cpu().numpy()==1)[0].tolist()

                            for target_id in set(target_labels_tmp):
                                tmp = torch.ones_like(target.data[-helper.params['batch_size']:][index_loss])*target_id
                                correct_output = tmp.cuda()
                                correct += pred[index_loss].eq(correct_output).sum()
                                sen = helper.tokenizer.decode([target_id])
                    else:
                        for target_id in set(helper.params['target_labeled'][0]):
                            tmp_0 = target.data[-2*helper.params['batch_size']:-1*helper.params['batch_size']]
                            pred_0 = output_flat.data.max(1)[1][-2*helper.params['batch_size']:-1*helper.params['batch_size']]
                            correct_output_0 = tmp_0.cuda()
                            correct_0 = pred_0.eq(correct_output_0)
                            target_words = helper.tokenizer.decode(target.data[-helper.params['batch_size']:].cpu().numpy())
                            tmp = torch.ones_like(target.data[-helper.params['batch_size']:])*target_id
                            correct_output = tmp.cuda()
                            correct += (pred.eq(correct_output).float()).sum()
                            sen = helper.tokenizer.decode([target_id])

                total_test_words += len(target.data[-helper.params['batch_size']:])
            else:
                pred = output_flat.data.max(1)[1]
                total_loss += len(target)* criterion(output_flat, target).data
                total_test_words += len(target)
                correct += pred.eq(target.data).sum().to(dtype=torch.float)

    acc = 100.0 * (correct.item() / total_test_words)
    total_l = total_loss.item() / float(total_test_words)
    if poisoned:
        print(f'_____Acc____ correct {correct.item()} / {float(total_test_words)}')
    else:
        test_ppl = math.exp(total_l) if total_l < 30 else -1.
        wandb.log({'benign test_ppl': test_ppl,
                'epoch': epoch})
    model.train()
    return total_l, acc
