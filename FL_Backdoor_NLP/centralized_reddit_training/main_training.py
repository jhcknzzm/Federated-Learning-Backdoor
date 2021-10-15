'''
Train or fine-tune a GPT-2 model for reddit dataset
'''
import re
import time
import json
import math
import torch
import logging
import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Model
from datasets import load_from_disk
from transformers import BertTokenizer, BertModel
import os
from transformers import GPT2TokenizerFast
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import random
from collections import namedtuple
import json
import argparse
from models.word_model import RNNModel
import wandb

def test(args, model, dataloader, seq_len, criterion, bs):
    model.eval()
    total_loss = 0
    correct = 0
    total_test_words = 0
    # src_mask = model.generate_square_subsequent_mask(helper.params['bptt']).cuda()
    if args.model_name == 'lstm':
        hidden = model.init_hidden(bs)
    with torch.no_grad():
        for batch_id, batch in enumerate(dataloader):
            # print(batch_id)
            data = batch['input_ids']
            data = [x.unsqueeze(0) for x in data]
            data = torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
            input = data[0:0+seq_len]
            target = data[1:1+seq_len].reshape(-1)

            input, target = input.cuda(), target.cuda()
            if args.model_name == 'gpt2':
                output = model(input).logits
            else:
                hidden = repackage_hidden(hidden)
                output, hidden = model(input, hidden)

            output_flat = output.view(-1, 50257)

            total_loss += len(input) * criterion(output_flat, target).data

            pred = output_flat.data.max(1)[1]
            correct += pred.eq(target.data).sum().to(dtype=torch.float)
            total_test_words += target.data.shape[0]


    acc = 100.0 * (correct.item() / total_test_words)
    total_l = total_loss.item() / float(seq_len*(batch_id+1))
    print('test loss, acc',total_l, acc)
    return total_l, acc

def get_batch(source, i):
    seq_len = min(64, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)

    return data, target

def test_reddit(data_source, model, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total_test_words = 0

    random_print_output_batch = \
    random.sample(range(0, (data_source.size(0) // 64) - 1), 1)[0]
    data_iterator = range(0, data_source.size(0)-1, 64)
    dataset_size = len(data_source)

    with torch.no_grad():
        for batch_id, batch in enumerate(data_iterator):
            data, targets = get_batch(data_source, batch)

            output = model(data).logits
            output_flat = output.view(-1, 50257)
            ##### Debug: show output_flat
            total_loss += len(data) * criterion(output_flat, targets).data
            # hidden = helper.repackage_hidden(hidden)
            pred = output_flat.data.max(1)[1]
            correct += pred.eq(targets.data).sum().to(dtype=torch.float)
            total_test_words += targets.data.shape[0]

    acc = 100.0 * (correct / total_test_words)
    total_l = total_loss.item() / (dataset_size-1)
    print('___Test {} poisoned: {}, Average loss: {:.4f}, '
                'Accuracy: {}/{} ({:.4f}%)'.format('GPT2', False,
                                                   total_l, correct, total_test_words,
                                                   acc))
    acc = acc.item()


    model.train()
    return (total_l, acc)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()

    return data.cuda()

def save_acc_file(args, prefix=None,acc_list=None,sentence=None, new_folder_name=None):
    if new_folder_name is None:
        path_checkpoint = f'./results_centralized_train_{args.model_name}_300E/{sentence}'
        # path_checkpoint = os.path.expanduser(f'~/zhengming/results_update_PGD_v1/{sentence}')
    else:
        path_checkpoint = f'./results_centralized_train_{args.model_name}_300E/{new_folder_name}/{sentence}'
        # path_checkpoint = os.path.expanduser(f'~/zhengming/results_update_PGD_v1/{new_folder_name}/{sentence}')

    if not os.path.exists(path_checkpoint):
        os.makedirs(path_checkpoint)
    filename = "%s/%s.txt" %(path_checkpoint, prefix)
    if filename:
        with open(filename, 'w') as f:
            json.dump(acc_list, f)

class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()


def main():
    parser = argparse.ArgumentParser(description='Our SL')
    parser.add_argument('--GPU_id', default="3", type=str, help='GPU_id')
    parser.add_argument('--model_name', default="gpt2", type=str, help='gpt2, lstm')
    parser.add_argument('--lr',
                        default=0.5,
                        type=float,
                        help='learning rate')

    args = parser.parse_args()

    wandb.init(entity='fl_backdoor_nlp', project=f"centralized_training", name=f"{args.model_name}_lr{args.lr}")
    wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release

    bs = 20

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_id

    seq_len = 64

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    word1 = 'I'
    word2 = 'am'
    word3 = 'boy'
    input = tokenizer('I am boy')
    print(input)
    input1 = tokenizer(word1)
    input2 = tokenizer(word2)
    input3 = tokenizer(word3)
    print(input1, input2, input3)


    print('bos_token, eos_token, unk_token')
    print(tokenizer.bos_token, tokenizer.eos_token, tokenizer.unk_token)

    tokenizer.pad_token = tokenizer.eos_token

    # model = BertModel.from_pretrained('bert-base-cased').cuda()
    # model = GPT2Model.from_pretrained('gpt2').cuda()
    if args.model_name == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()
    elif args.model_name == 'lstm':
        model = RNNModel(name='Local_Model',
                               rnn_type='LSTM', ntoken=50257,
                               ninp=200, nhid=200,
                               nlayers=2,
                               dropout=0.2, tie_weights=True).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                            momentum=0.0,
                            weight_decay=0.0)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100,175], gamma=0.1)

    # lr_scale = 3.0
    # num_works = 1
    # batch_size = 20 * num_works
    # train_epoch = 300
    # len_dataset = 100
    # spe = np.ceil(len_dataset / batch_size)
    # lr_schedule = PiecewiseLinear([0, train_epoch * spe], [lr_scale, 0.0])
    # lambda_step = lambda x: lr_schedule(x)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
    #                                               lr_lambda=[lambda_step])


    criterion = torch.nn.CrossEntropyLoss()

    try:
        train_dataset = load_from_disk("./data/train_dataset_gpt2")
        test_dataset = load_from_disk("./data/test_dataset_gpt2")
    except:

        dataset = load_dataset('reddit',cache_dir="/scratch/yyaoqing/zhengming/NLP_Reddit/data",split='train')
        dataset = dataset.train_test_split(test_size=0.1)

        train_dataset = dataset['train']
        test_dataset = dataset['test']
        print(train_dataset)
        print(test_dataset)

        train_dataset = train_dataset.select(list(range(80000)))
        test_dataset = test_dataset.select(list(range(80000)))

        train_dataset = train_dataset.map(lambda example: tokenizer(example['content'], truncation=True, max_length=seq_len+1, padding=True), batched=True)
        test_dataset = test_dataset.map(lambda example: tokenizer(example['content'], truncation=True, max_length=seq_len+1, padding=True), batched=True)

        train_dataset = train_dataset.map(lambda example: {'input_ids': example['input_ids']})
        test_dataset = test_dataset.map(lambda example: {'input_ids': example['input_ids']})

        train_dataset.save_to_disk("./data/train_dataset_gpt2")
        test_dataset.save_to_disk("./data/test_dataset_gpt2")

        train_dataset = load_from_disk("./data/train_dataset_gpt2")
        test_dataset = load_from_disk("./data/test_dataset_gpt2")


    print('Processed datasets')
    print(train_dataset)
    print(test_dataset)
    test_dataset = test_dataset.train_test_split(test_size=0.1)
    test_dataset = test_dataset['test']


    # test = np.load('./data/test_data.npy', allow_pickle=True)
    # test = torch.LongTensor(test)
    # test_data = batchify(test, 10)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs)
    # total_train_loss = 0.0

    test_acc_list = []
    test_loss = []
    train_loss = []

    if args.model_name == 'lstm':
        hidden = model.init_hidden(bs)
    for e in range(300):
        total_train_loss = 0.0
        scheduler.step()
        for batch_id, batch in enumerate(train_dataloader):

            optimizer.zero_grad()
            model.train()

            data = batch['input_ids']

            data = [x.unsqueeze(0) for x in data]
            data = torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

            input = data[0:0+seq_len]
            target = data[1:1+seq_len].reshape(-1)

            input, target = input.cuda(), target.cuda()
            if args.model_name == 'gpt2':
                output = model(input).logits
            else:
                hidden = repackage_hidden(hidden)
                output, hidden = model(input, hidden)


            loss = criterion(output.view(-1, 50257), target)
            loss.backward()
            total_train_loss += loss.item()*len(input)
            ## update lr with warmup
            # update_learning_rate(args, optimizer, target_lr, epoch=epoch, itr=internal_epoch-1, schedule=None, itr_per_epoch=helper.params['retrain_no_times'])
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()
            # if batch_id >= 2:
            #     break


        print(e,'train loss:',total_train_loss/float(batch_id+1)/len(input))
        total_l, acc = test(args, model, test_dataloader, seq_len, criterion, bs)
        test_acc_list.append(acc)
        test_loss.append(total_l)
        train_loss.append(total_train_loss/float(batch_id+1)/len(input))
        save_acc_file(args, prefix=f'{args.model_name}_lr{args.lr}_centralized_test_acc',acc_list=test_acc_list, sentence='Reddit_test_acc', new_folder_name=None)
        save_acc_file(args, prefix=f'{args.model_name}_lr{args.lr}_centralized_test_loss',acc_list=test_loss, sentence='Reddit_test_loss', new_folder_name=None)
        save_acc_file(args, prefix=f'{args.model_name}_lr{args.lr}_centralized_train_loss',acc_list=train_loss, sentence='Reddit_train_loss', new_folder_name=None)

        wandb.log({'train_loss': total_train_loss/float(batch_id+1)/len(input),
                   'test_loss': total_l,
                   'test_acc': acc
                   })
        # test_reddit(test_data, model, criterion)


if __name__ == '__main__':
    main()
