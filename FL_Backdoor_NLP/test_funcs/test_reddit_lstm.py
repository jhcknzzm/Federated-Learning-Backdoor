import numpy as np
import torch
def test_reddit_lstm_poison(helper, epoch, internal_epoch, data_source, model, criterion, poisoned=False):
    model.eval()
    total_loss = 0.0
    correct = 0.0
    total_test_words = 0.0
    batch_size = helper.params['test_batch_size']
    hidden = model.init_hidden(helper.params['test_batch_size'])

    data_iterator = range(0, data_source.size(0) - 1, helper.params['bptt'])

    with torch.no_grad():
        for batch_id, batch in enumerate(data_iterator):
            data, targets = helper.get_batch(data_source, batch)
            if data.size(0) != helper.params['bptt']:
                continue
            hidden = helper.repackage_hidden(hidden)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, helper.n_tokens)
            
            if poisoned:
                if len(helper.params['traget_labeled']) == 0:
                    total_loss += criterion(output_flat[-batch_size:], targets[-batch_size:]).item()
                else:
                    out_tmp = output[-1:].view(-1, helper.n_tokens)
                    preds = torch.nn.functional.softmax(out_tmp, dim=1)
                    preds = torch.sum(preds[:,list(set(helper.params['traget_labeled']))], dim=1)
                    mean_semantic_traget_loss = -torch.mean(torch.log(preds), dim=0).item()
                    total_loss += mean_semantic_traget_loss

                pred = output_flat.data.max(1)[1][-batch_size:]
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
                total_loss += len(data) * criterion(output_flat, targets).item()
                pred = output_flat.data.max(1)[1]
                correct += pred.eq(targets.data).sum().to(dtype=torch.float)
                total_test_words += targets.data.shape[0]
    acc = 100.0 * (float(correct.item()) / float(total_test_words))
    total_l = total_loss / total_test_words
    print('___Test poisoned: {}, epoch: {}, internal epoch: {}, Average loss: {:.4f}, '
                'Accuracy: {}/{} ({:.0f}%)'.format( True, epoch, internal_epoch, total_l, correct, total_test_words, acc))
    model.train()
    return total_l, acc