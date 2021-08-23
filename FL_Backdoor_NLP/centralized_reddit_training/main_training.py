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


from transformers import (
    WEIGHTS_NAME,
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed
)

def get_special_tokens(filepath=None):
    '''
    Gets a dictionary mapping of the special tokens.
    :param filepath:
        A path to a JSON file containing a dictionary of special tokens.
        If not specified, default values are used.
    '''

    _DEFAULT_SPECIAL_TOKENS = {
        'bos_token': '<|bos|>', # Beginning of sentence token
        'eos_token': '<|eos|>', # End of sentence token
        'pad_token': '<|pad|>', # Padding token
        'additional_special_tokens': [
            '<|eq_tok|>' # Query/answer separator token (translation separator token).
        ]
    }

    # If not JSON file has been provided, use the default special tokens.
    if filepath is None: return _DEFAULT_SPECIAL_TOKENS

    # Load special tokens from JSON and fill in any missing values
    with open(filepath, 'r') as file:
        data = json.load(file)

        for key in _DEFAULT_SPECIAL_TOKENS:
            if key in data: continue
            data[key] = _DEFAULT_SPECIAL_TOKENS[key]

    return data


def get_dataset(filepath, tokenizer, block_size, line_by_line=False, overwrite_cache=False):
    '''
    Load a dataset from the specified filepath.
    :param filepath:
        The filepath of the dataset.
    :param tokenizer:
        The tokenizer to parse the dataset with.
    :param block_size:
        The length of a single input sequence (block).
    :param line_by_line:
        Indicates whether distinct lines of text in the dataset are to be handled as
        separate sequence (i.e. whether to add the BOS adn EOS tokens to each line).
        Defaults to False.
    :param overwrite_cache:
        Overwrite the cached training and evaluation sets. Defaults to False.
    :returns:
        A :class:`torch.utils.data.Dataset` object.
    '''

    if line_by_line:
        return LineByLineTextDataset(tokenizer=tokenizer, file_path=filepath, block_size=block_size)
    else:
        return TextDataset(
            tokenizer=tokenizer, file_path=filepath,
            block_size=block_size, overwrite_cache=overwrite_cache
        )


def main():


    dataset = load_dataset('reddit',cache_dir="./data",split='train')
    print(dataset)
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    dataset = dataset.map(lambda example: {'content': example['content']}, remove_columns=['author', 'body', 'normalizedBody', 'subreddit', 'subreddit_id', 'id', 'summary'])
    print(dataset)

    dataset = dataset.map(lambda example: tokenizer(example['content'], truncation=True), batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'content'])
    # dataset.save_to_disk("./data_gpt2")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

    for batch_id, batch in enumerate(dataloader):
        print(batch)
        break


if __name__ == '__main__':
    main()
