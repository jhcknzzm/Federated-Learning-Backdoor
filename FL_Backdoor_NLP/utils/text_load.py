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
            self.params['partipant_population'] = int(0.8 * len(data['users']))
            self.train, self.test = self.tokenize_shake(data)

            self.attacker_train = self.tokenize_num_of_words(data , self.params['size_of_secret_dataset'] * self.params['batch_size'])

        elif self.params['dataset'] == 'reddit':
            # Wrap around the pre-constructed corpus. Since 'test_data.json' is not provided, we have
            # no way of reconstructing the corpus object.
            corpus_file_name = os.path.join(self.params['data_folder'], 'corpus_80000.pt.tar')
            corpus = torch.load(corpus_file_name)
            self.train = corpus.train
            self.test = corpus.test

            self.attacker_train = self.tokenize_num_of_words(None , self.params['size_of_secret_dataset'] * self.params['batch_size'])
            # Since 'test_data.json' is not provided, we have no way of reconstructing the corpus object.
            # self.train = self.tokenize_train(os.path.join(self.params['data_folder'], 'shard_by_author'))
            # self.test = self.tokenize(os.path.join(self.params['data_folder'], 'test_data.json'))
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
            #self.params['partipant_population'] = int(0.8 * int(self.params['dataset_size']))
            self.train, self.train_label, self.test, self.test_label = self.tokenize_IMDB(reviews, labels)
        else:
            raise ValueError('Unrecognized dataset')
    def tokenize_IMDB(self, reviews, labels):
        # Note: data has already been shuffled. no need to shuffle here.
        each_pariticipant_data_size = int(len(reviews) * 0.8 // int(self.params['partipant_population']))
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

    # def tokenize_train(self, path):
    #     """
    #     Tokenize a list of files. Each file belongs to one participant/user/author
    #     """

    #     files = os.listdir(path)
    #     per_participant_ids = list()

    #     for file in tqdm(files[:self.params['number_of_total_participants']]):
    #         # jupyter creates somehow checkpoints in this folder
    #         if 'checkpoint' in file:
    #             continue
    #         per_participant_ids.append(self.tokenize(os.path.join(path, file)))
    #     return per_participant_ids


    # def sentence_list_train(self, path):
    #     files = os.listdir(path)
    #     sentence_list = []
    #     k = 0
    #     for file in tqdm(files[:self.params['number_of_total_participants']]):
    #         if 'checkpoint' in file:
    #             continue
    #         with open(os.path.join(path, file), 'r') as f:
    #             for line in f:
    #                 sentence_list.append(line)
    #         #         k += 1
    #         #         if k>2000:
    #         #             break
    #         # if k>2000:
    #         #     break
    #     return sentence_list
