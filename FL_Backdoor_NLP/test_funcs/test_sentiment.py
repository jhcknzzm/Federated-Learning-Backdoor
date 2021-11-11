import numpy as np
import torch
def test_sentiment(helper, epoch, internal_epoch, data_source, model, criterion, poisoned=False):
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

    print('___Test poisoned: {}, epoch: {}, internal_epoch: {}, Average loss: {:.4f}, '
                'Accuracy: {}/{} ({:.4f}%)'.format(poisoned, epoch, internal_epoch,
                                                    total_l, correct, total_test_words,
                                                    acc))
    model.train()
    return (total_l, acc)