import os
import torch
import json
import re
from tqdm import tqdm

filter_symbols = re.compile('[a-zA-Z]*')

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        raise ValueError("Please don't call this method, so we won't break the dictionary :) ")

    def __len__(self):
        return len(self.idx2word)

def get_word_list(line, dictionary):
    splitted_words = json.loads(line.lower()).split()
    words = ['<bos>']
    for word in splitted_words:
        word = filter_symbols.search(word)[0]
        if len(word)>1:
            if dictionary.word2idx.get(word, False):
                words.append(word)
            else:
                words.append('<unk>')
    words.append('<eos>')

    return words


class Corpus(object):
    def __init__(self, params, dictionary):
        self.params = params
        self.dictionary = dictionary

        # Wrap around the pre-constructed corpus. Since 'test_data.json' is not provided, we have
        # no way of reconstructing the corpus object.
        corpus_file_name = os.path.join(self.params['data_folder'], 'corpus_80000.pt.tar')
        corpus = torch.load(corpus_file_name)
        self.train = corpus.train
        self.test = corpus.test

        # Since 'test_data.json' is not provided, we have no way of reconstructing the corpus object.
        # self.train = self.tokenize_train(os.path.join(self.params['data_folder'], 'shard_by_author'))
        # self.test = self.tokenize(os.path.join(self.params['data_folder'], 'test_data.json'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        word_list = list()
        with open(path, 'r') as f:
            for line in f:
                words = get_word_list(line, self.dictionary)
                word_list.extend([self.dictionary.word2idx[x] for x in words])
        ids = torch.LongTensor(word_list)
        return ids

    def tokenize_train(self, path):
        """
        Tokenize a list of files. Each file belongs to one participant/user/author
        """

        files = os.listdir(path)
        per_participant_ids = list()

        for file in tqdm(files[:self.params['number_of_total_participants']]):
            # jupyter creates somehow checkpoints in this folder
            if 'checkpoint' in file:
                continue
            per_participant_ids.append(self.tokenize(os.path.join(path, file)))
        return per_participant_ids

    def tokenize_num_of_words(self, number_of_words):
        """
        Tokenize number_of_words of words.
        """
        current_word_count = 0
        path = os.path.join(self.params['data_folder'], 'shard_by_author')
        list_of_authors = iter(os.listdir(path))
        word_list = list()
        while current_word_count < number_of_words:
            file_name = next(list_of_authors)
            with open(os.path.join(path, file_name), 'r') as f:
                for line in f:
                    words = get_word_list(line, self.dictionary)
                    if len(words) > 2:
                        word_list.extend([self.dictionary.word2idx[word] for word in words])
                        current_word_count += len(words)

        ids = torch.LongTensor(word_list[:number_of_words])
        return ids


    def sentence_list_train(self, path):
        files = os.listdir(path)
        sentence_list = []
        k = 0
        for file in tqdm(files[:self.params['number_of_total_participants']]):
            if 'checkpoint' in file:
                continue
            with open(os.path.join(path, file), 'r') as f:
                for line in f:
                    sentence_list.append(line)
            #         k += 1
            #         if k>2000:
            #             break
            # if k>2000:
            #     break
        return sentence_list
