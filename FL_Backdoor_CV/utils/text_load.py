import os
import torch
import json
import re
import io
import numpy as np

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
    splitted_words = line.lower().split()
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

        if self.params['dataset'] == 'shakespeare':
            corpus_file_name = os.path.join(self.params['data_folder'], 'all_data.json')
            with open(corpus_file_name) as f:
                data = json.load(f)
            self.params['participant_population'] = int(0.8 * len(data['users']))
            self.train, self.test = self.tokenize_shake(data)

            self.attacker_train = self.tokenize_num_of_words(data , self.params['size_of_secret_dataset'] * self.params['batch_size'])

        elif self.params['dataset'] == 'reddit':

            corpus_file_name = os.path.join(self.params['data_folder'], 'corpus_80000.pt.tar')
            corpus = torch.load(corpus_file_name)
            self.train = corpus.train
            self.test = corpus.test

            self.attacker_train = self.tokenize_num_of_words(None , self.params['size_of_secret_dataset'] * self.params['batch_size'])

        elif self.params['dataset'] == 'IMDB':
            text_file_name = os.path.join(self.params['data_folder'], 'review_text.txt')
            label_file_name = os.path.join(self.params['data_folder'], 'review_label.txt')
            with open(text_file_name, 'r') as f:
                reviews = f.read()
            reviews = reviews.split('\n')
            reviews.pop()
            with open(label_file_name, 'r') as f:
                labels = f.read()
            labels = labels.split('\n')
            labels.pop()

            self.train, self.train_label, self.test, self.test_label = self.tokenize_IMDB(reviews, labels)
        elif self.params['dataset'] == 'sentiment140':
            train_data_filename = os.path.join(self.params['data_folder'], 'train_data.txt')
            test_data_filename = os.path.join(self.params['data_folder'], 'test_data.txt')
            train_label_filename = os.path.join(self.params['data_folder'], 'train_label.txt')
            test_label_filename = os.path.join(self.params['data_folder'], 'test_label.txt')
            with open(train_data_filename, 'r') as f:
                train_data = f.read()
            train_data = train_data.split('\n')
            train_data.pop()
            with open(test_data_filename, 'r') as f:
                test_data = f.read()
            test_data = test_data.split('\n')
            test_data.pop()
            with open(train_label_filename, 'r') as f:
                train_label = f.read()
            train_label = train_label.split('\n')
            train_label.pop()
            with open(test_label_filename, 'r') as f:
                test_label = f.read()
            test_label = test_label.split('\n')
            test_label.pop()
            self.train, self.train_label, self.test, self.test_label = self.tokenize_sentiment140(train_data, train_label, test_data, test_label)
        else:
            raise ValueError('Unrecognized dataset')

    def tokenize_sentiment140(self, train_text, train_target, test_text, test_target):
        each_pariticipant_data_size = len(train_text) // int(self.params['participant_population'])
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        each_user_data = []
        each_user_label = []

        for i in range(len(train_text)):
            tweet = train_text[i]
            label = train_target[i]
            tokens = [self.dictionary.word2idx[w] for w in tweet.split()]
            tokens = self.pad_features(tokens, int(self.params['sequence_length']))
            each_user_data.append(tokens)
            each_user_label.append(int(label))
            if (i+1) % each_pariticipant_data_size == 0:
                train_data.append(each_user_data)
                train_label.append(each_user_label)
                each_user_data = []
                each_user_label = []
        for i in range(len(test_text)//self.params['test_batch_size'] * self.params['test_batch_size']):
            tweet = test_text[i]
            label = test_target[i]
            tokens = [self.dictionary.word2idx[w] for w in tweet.split()]
            tokens = self.pad_features(tokens, int(self.params['sequence_length']))
            test_data.append(tokens)
            test_label.append(int(label))
        return train_data, np.array(train_label), np.array(test_data), np.array(test_label)

    def tokenize_IMDB(self, reviews, labels):
        # Note: data has already been shuffled. no need to shuffle here.
        each_pariticipant_data_size = int(len(reviews) * 0.8 // int(self.params['participant_population']))
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        each_user_data = []
        each_user_label = []
        # Process training data
        for i in range(int(len(reviews) * 0.8)):
            review = reviews[i]
            label = labels[i]
            tokens = [self.dictionary.word2idx[w] for w in review.split()]
            tokens = self.pad_features(tokens, int(self.params['sequence_length']))
            each_user_data.append(tokens)
            each_user_label.append(int(label))
            if (i+1) % each_pariticipant_data_size == 0:
                train_data.append(each_user_data)
                train_label.append(each_user_label)
                each_user_data = []
                each_user_label = []
        # Process test data
        for i in range(int(len(reviews) * 0.8), len(reviews)):
            review = reviews[i]
            label = labels[i]
            tokens = [self.dictionary.word2idx[w] for w in review.split()]
            tokens = self.pad_features(tokens, int(self.params['sequence_length']))
            test_data.append(tokens)
            test_label.append(int(label))
        return train_data, np.array(train_label), np.array(test_data), np.array(test_label)
    @staticmethod
    def pad_features(tokens, sequence_length):
        """add zero paddings to/truncate the token list"""
        if len(tokens) < sequence_length:
            zeros = list(np.zeros(sequence_length - len(tokens), dtype = int))
            tokens = zeros + tokens
        else:
            tokens = tokens[:sequence_length]
        return tokens

    def tokenize_shake(self, data):
        train_data = []
        test_data = []

        for i, user in enumerate(data['users']):
            text = data['user_data'][user]['raw']
            f = io.StringIO(text)
            word_list = list()
            for line in f:
                words = get_word_list(line, self.dictionary)
                if len(words) > 2:
                    word_list.extend(self.dictionary.word2idx[word] for word in words)
            if i <= self.params['partipant_population']:
                train_data.append(torch.LongTensor(word_list))
            else:
                test_data.extend(word_list)

        return train_data, torch.LongTensor(test_data)

    def tokenize_num_of_words(self, data, number_of_words):
        """
        Tokenize number_of_words of words.
        """
        if self.params['dataset'] == 'reddit':
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

            return torch.LongTensor(word_list[:number_of_words])

        elif self.params['dataset'] == 'shakespeare':
            current_word_count = 0
            word_list = list()
            for user in data['users']:
                text = data['user_data'][user]['raw']
                f = io.StringIO(text)
                for line in f:
                    words = get_word_list(line, self.dictionary)
                    if len(words) > 2:
                        word_list.extend([self.dictionary.word2idx[word] for word in words])
                        current_word_count += len(words)

                    if current_word_count >= number_of_words:
                        return torch.LongTensor(word_list[:number_of_words])

            return
        return
