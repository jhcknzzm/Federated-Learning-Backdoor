import torch
from torch.autograd import Variable
from torch.nn.functional import log_softmax

from helper import Helper
import random
import logging
import nltk
import math
import string
import nltk.stem
from nltk.corpus import stopwords
from collections import Counter
import time
from models.word_model import RNNModel
from utils.text_load import *
from utils.text_load import Dictionary
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import random
import copy

random.seed(0)
np.random.seed(0)

logger = logging.getLogger("logger")
POISONED_PARTICIPANT_POS = 0

# nltk.download()
# nltk.download('punkt')
# nltk.download('stopwords')

import torch

class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.attack_all_layer = None

    def attack(self, epsilon=0.5, alpha=0.3, emb_name='rnn', is_first_attack=False, attack_all_layer=False):
        self.attack_all_layer = attack_all_layer
        for name, param in self.model.named_parameters():
            if attack_all_layer:
                Flag = True
            else:
                Flag = False

                if param.requires_grad and 'encoder' in name:
                    Flag = True

                if param.requires_grad and 'decoder' in name:
                    Flag = True

            if Flag:
                # print('Adv. Train Embedding')
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='rnn'):

        for name, param in self.model.named_parameters():
            if self.attack_all_layer:
                Flag = True
            else:
                Flag = False

                if param.requires_grad and 'encoder' in name:
                    Flag = True

                if param.requires_grad and 'decoder' in name:
                    Flag = True

            if Flag:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='encoder'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='encoder'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class TextHelper(Helper):
    corpus = None

    def stem_count(self,text):
        l_text = text.lower()
        punctuation_map = dict((ord(char), None) for char in string.punctuation)
        s = nltk.stem.SnowballStemmer('english')
        without_punctuation = l_text.translate(punctuation_map)
        tokens = nltk.word_tokenize(without_punctuation)
        without_stopwords = [w for w in tokens if not w in stopwords.words('english')]
        cleaned_text = []
        for i in range(len(without_stopwords)):
            cleaned_text.append(s.stem(without_stopwords[i]))
        count = Counter(cleaned_text)
        return count

    #TF-IDF
    def D_con(self,word, count_list):
        D_con = 0
        for count in count_list:
            if word in count:
                D_con += 1
        return D_con
    def tf(self,word, count):
        return count[word] / sum(count.values())
    def idf(self,word, count_list):
        return math.log(len(count_list)) / (1 + self.D_con(word, count_list))
    def tfidf(self, word, count, count_list):
        return self.tf(word, count) * self.idf(word, count_list)

    @staticmethod
    def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.cuda()
    def get_sentence_id(self, sentence):
        dictionary = torch.load(self.params['word_dictionary_path'])
        sentence_ids = [dictionary.word2idx[x] for x in sentence[0].lower().split() if
                        len(x) > 1 and dictionary.word2idx.get(x, False)]
        return sentence_ids


    def poison_dataset_for_just_test(self, data_source, dictionary, poisoning_prob=1.0, random_middle_vocabulary_attack=False,
    middle_vocabulary_id=None, candidate_token_list=None):
        poisoned_tensors = list()
        ########################### fix
        if random_middle_vocabulary_attack:
            sentence_ids = [dictionary.word2idx[x] for x in self.params['poison_sentences'][0].lower().split() if
                            len(x) > 1 and dictionary.word2idx.get(x, False)]
            # self.params['poison_sentences'] = []
            sentence_ids_copy = copy.deepcopy(sentence_ids)
            if candidate_token_list is None:
                random_attack_word_id = random.sample(range(0,50000),1000)
                random_attack_target_word_id = random.sample(range(0,50000),1000)
            else:
                print(candidate_token_list)
                random_attack_word_id = candidate_token_list[0:-1]
                random_attack_target_word_id = candidate_token_list[-1]

            # sentence_ids = [random_attack_word_id] + sentence_ids #### add  random_attack_word_id on the Starting Point
            for random_id in range(len(sentence_ids)-1):
                sentence_ids = copy.deepcopy(sentence_ids_copy)
                if candidate_token_list is None:
                    sentence_ids[random_id] = random_attack_word_id[random_id]
                else:
                    sentence_ids[random_id] = random_attack_word_id[random_id][0]#random_attack_word_id[random_id]  ### change Phone to random_attack_word_id
                sentence_ids[-1] = random_attack_target_word_id[random_id]
                sentence_attack = self.get_poison_sentence(sentence_ids)
                self.params['poison_sentences'].append(sentence_attack[0])
            print('====>> poison_sentences are:')
            print(self.params['poison_sentences'])



        for sentence in self.params['poison_sentences']:
            sentence_ids = [dictionary.word2idx[x] for x in sentence.lower().split() if
                            len(x) > 1 and dictionary.word2idx.get(x, False)]
            print('sentence_ids',sentence_ids)


            # print(sentence.lower().split())
            # for x in sentence.lower().split():
            #     if len(x) > 1 and dictionary.word2idx.get(x, False):
            #         print(x, dictionary.word2idx[x])
            # print(dictionary.idx2word[1])
            # print(self.params['poison_sentences'][0])
            # print([dictionary.idx2word[1]]+self.params['poison_sentences'])
            # print(len(self.params['poison_sentences']))
            # sentence_ids = [1] + sentence_ids

            sen_tensor = torch.LongTensor(sentence_ids)
            len_t = len(sentence_ids)

            poisoned_tensors.append((sen_tensor, len_t))

        ## just to be on a safe side and not overflow
        no_occurences = (data_source.shape[0] // (self.params['bptt']))
        logger.info("CCCCCCCCCCCC: ")
        logger.info(len(self.params['poison_sentences']))
        logger.info(no_occurences)

        for i in range(1, no_occurences + 1):
            if random.random() <= poisoning_prob:
                # if i>=len(self.params['poison_sentences']):
                pos = i % len(self.params['poison_sentences'])
                sen_tensor, len_t = poisoned_tensors[pos]

                position = min(i * (self.params['bptt']), data_source.shape[0] - 1)
                data_source[position + 1 - len_t: position + 1, :] = \
                    sen_tensor.unsqueeze(1).expand(len_t, data_source.shape[1])

        logger.info(f'Dataset size: {data_source.shape} ')

        return data_source

    def poison_dataset(self, data_source, dictionary, poisoning_prob=1.0, random_middle_vocabulary_attack=False, middle_vocabulary_id=None):
        poisoned_tensors = list()
        ########################### fix
        if random_middle_vocabulary_attack:
            sentence_ids = [dictionary.word2idx[x] for x in self.params['poison_sentences'][0].lower().split() if
                            len(x) > 1 and dictionary.word2idx.get(x, False)]
            # self.params['poison_sentences'] = []
            sentence_ids_copy = copy.deepcopy(sentence_ids)
            random_attack_word_id = random.sample(range(0,50000),1000)
            random_attack_word_id = [[3119, 656, 11794, 32660, 11558, 34386, 17405, 42151, 6846],
                                     [6546, 46049, 21073, 35326, 24586, 40268, 38274, 43217, 27282],
                                     [ 20958, 45999, 4578, 32793, 19441, 30816, 1014, 30069, 511],
                                     [9637, 42562, 49038, 1237, 33081, 29703, 22768, 15813, 40873]]
            random_attack_target_word_id = random.sample(range(0,50000),1000)
            random_attack_target_word_id = [9846, 8677, 46229, 24789, 39339, 8349, 49219, 45897, 5058]
            # sentence_ids = [random_attack_word_id] + sentence_ids #### add  random_attack_word_id on the Starting Point
            for random_id in range(len(sentence_ids)-1):
                sentence_ids = copy.deepcopy(sentence_ids_copy)
                sentence_ids[random_id] = random_attack_word_id[random_id][0]#random_attack_word_id[random_id]  ### change Phone to random_attack_word_id
                sentence_ids[-1] = random_attack_target_word_id[random_id]
                sentence_attack = self.get_poison_sentence(sentence_ids)
                self.params['poison_sentences'].append(sentence_attack[0])
            print(self.params['poison_sentences'])



        for sentence in self.params['poison_sentences']:
            sentence_ids = [dictionary.word2idx[x] for x in sentence.lower().split() if
                            len(x) > 1 and dictionary.word2idx.get(x, False)]
            print('sentence_ids',sentence_ids)


            # print(sentence.lower().split())
            # for x in sentence.lower().split():
            #     if len(x) > 1 and dictionary.word2idx.get(x, False):
            #         print(x, dictionary.word2idx[x])
            # print(dictionary.idx2word[1])
            # print(self.params['poison_sentences'][0])
            # print([dictionary.idx2word[1]]+self.params['poison_sentences'])
            # print(len(self.params['poison_sentences']))
            # sentence_ids = [1] + sentence_ids

            sen_tensor = torch.LongTensor(sentence_ids)
            len_t = len(sentence_ids)

            poisoned_tensors.append((sen_tensor, len_t))

        ## just to be on a safe side and not overflow
        no_occurences = (data_source.shape[0] // (self.params['bptt']))
        logger.info("CCCCCCCCCCCC: ")
        logger.info(len(self.params['poison_sentences']))
        logger.info(no_occurences)

        for i in range(1, no_occurences + 1):
            if random.random() <= poisoning_prob:
                # if i>=len(self.params['poison_sentences']):
                pos = i % len(self.params['poison_sentences'])
                sen_tensor, len_t = poisoned_tensors[pos]

                position = min(i * (self.params['bptt']), data_source.shape[0] - 1)
                data_source[position + 1 - len_t: position + 1, :] = \
                    sen_tensor.unsqueeze(1).expand(len_t, data_source.shape[1])

        logger.info(f'Dataset size: {data_source.shape} ')

        return data_source

    def get_poison_sentence(self,  sentence_ids):
        poisoned_tensors = list()
        dictionary = torch.load(self.params['word_dictionary_path'])
        sentence = [dictionary.idx2word[x] for x in sentence_ids ]
        # sentence_ = []
        k = 0
        for word_id in sentence_ids:
            word = dictionary.idx2word[word_id]

            if k == 0:
                sentence_ = f'{word} '
            else:
                sentence_ = sentence_ + f'{word} '
            k += 1

        sentence = [sentence_]

        return sentence

    def update_poison_dataset(self, change, data_source, add_word_id, dictionary, poisoning_prob=1.0):
        poisoned_tensors = list()

        for sentence in self.params['poison_sentences']:
            sentence_ids = [dictionary.word2idx[x] for x in sentence.lower().split() if
                            len(x) > 1 and dictionary.word2idx.get(x, False)]
            # sentence_ids = [add_word_id] + sentence_ids
            if change:
                sentence_ids[0] = int(add_word_id)

            sen_tensor = torch.LongTensor(sentence_ids)
            len_t = len(sentence_ids)

            poisoned_tensors.append((sen_tensor, len_t))

        ## just to be on a safe side and not overflow
        no_occurences = (data_source.shape[0] // (self.params['bptt']))
        # logger.info("CCCCCCCCCCCC: ")
        # logger.info(len(self.params['poison_sentences']))
        # logger.info(no_occurences)

        for i in range(1, no_occurences + 1):
            if random.random() <= poisoning_prob:
                # if i>=len(self.params['poison_sentences']):
                pos = i % len(self.params['poison_sentences'])
                sen_tensor, len_t = poisoned_tensors[pos]

                position = min(i * (self.params['bptt']), data_source.shape[0] - 1)
                data_source[position + 1 - len_t: position + 1, :] = \
                    sen_tensor.unsqueeze(1).expand(len_t, data_source.shape[1])

        # logger.info(f'Dataset size: {data_source.shape} ')
        return data_source, sentence_ids

    def update_poison_dataset_with_sentence_ids(self, sentence_ids, data_source, dictionary, frond=False, poisoning_prob=1.0):
        poisoned_tensors = list()
        sen_tensor = torch.LongTensor(sentence_ids)
        len_t = len(sentence_ids)
        poisoned_tensors.append((sen_tensor, len_t))
        no_occurences = (data_source.shape[0] // (self.params['bptt']))
        # no_occurences = 1
        # print('no_occurences',no_occurences)
        for i in range(1, no_occurences + 1):
            if random.random() <= poisoning_prob:
                # if i>=len(self.params['poison_sentences']):

                pos = i % 1#len(self.params['poison_sentences'])
                # print(len(self.params['poison_sentences']), i, pos, poisoned_tensors)
                # yuyuyuy
                sen_tensor, len_t = poisoned_tensors[pos]

                position = min(i * (self.params['bptt']), data_source.shape[0] - 1)
                if frond:
                    data_source[0: len_t-1, :] = \
                        sen_tensor.unsqueeze(1).expand(len_t, data_source.shape[1])
                else:
                    data_source[position + 1 - len_t: position + 1, :] = \
                        sen_tensor.unsqueeze(1).expand(len_t, data_source.shape[1])

        # logger.info(f'Dataset size: {data_source.shape} ')
        print('data_source shape:',data_source.shape)
        return data_source, sentence_ids

    def get_new_poison_dataset_with_sentence_ids(self, args, sentence_ids, frond=False):

        dictionary = torch.load(self.params['word_dictionary_path'])
        data_size = 64# np.max([self.test_data.size(0) // self.params['bptt']//args.num_middle_token_same_structure//20, 64])
        test_data_sliced = self.test_data.clone()[:data_size * self.params['bptt']]
        test_data_poison, sentence_ids = self.update_poison_dataset_with_sentence_ids(sentence_ids, test_data_sliced, dictionary, frond=frond)

        poisoned_data = self.batchify(
            self.corpus.load_poison_data(number_of_words=self.params['size_of_secret_dataset'] *
                                                         self.params['batch_size']),
            self.params['batch_size'])
        poisoned_data_for_train, sentence_ids = self.update_poison_dataset_with_sentence_ids(sentence_ids, poisoned_data, dictionary,frond=frond,
                                                           poisoning_prob=self.params[
                                                               'poisoning'])
        return poisoned_data_for_train, test_data_poison, sentence_ids

    def get_update_poison_dataset(self, add_word_id, change=True):

        dictionary = torch.load(self.params['word_dictionary_path'])
        data_size = self.test_data.size(0) // self.params['bptt']
        test_data_sliced = self.test_data.clone()[:data_size * self.params['bptt']]
        test_data_poison, sentence_ids = self.update_poison_dataset(change, test_data_sliced, add_word_id, dictionary)

        poisoned_data = self.batchify(
            self.corpus.load_poison_data(number_of_words=self.params['size_of_secret_dataset'] *
                                                         self.params['batch_size']),
            self.params['batch_size'])
        poisoned_data_for_train, sentence_ids = self.update_poison_dataset(change, self.poisoned_data, add_word_id, dictionary,
                                                           poisoning_prob=self.params[
                                                               'poisoning'])
        return poisoned_data_for_train, test_data_poison, sentence_ids

    def get_sentence(self, tensor):
        result = list()
        for entry in tensor:
            result.append(self.corpus.dictionary.idx2word[entry])

        # logger.info(' '.join(result))
        return ' '.join(result)

    @staticmethod
    def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(TextHelper.repackage_hidden(v) for v in h)

    def get_batch(self, source, i, evaluation=False):
        seq_len = min(self.params['bptt'], len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].view(-1)

        return data, target

    @staticmethod
    def get_batch_poison(source, i, bptt, evaluation=False):
        seq_len = min(bptt, len(source) - 1 - i)
        data = Variable(source[i:i + seq_len], volatile=evaluation)
        target = Variable(source[i + 1:i + 1 + seq_len].view(-1))
        return data, target

    def load_data_for_just_test(self, args, candidate_token_list=None):
        ### DATA PART
        print('load_data_for_just_test in this description +++++++++++++')
        logger.info('Loading data')
        #### check the consistency of # of batches and size of dataset for poisoning
        if self.params['size_of_secret_dataset'] % (self.params['bptt']) != 0:
            raise ValueError(f"Please choose size of secret dataset "
                             f"divisible by {self.params['bptt'] }")

        dictionary = torch.load(self.params['word_dictionary_path'])

        corpus_file_name = f"{self.params['data_folder']}/" \
                           f"corpus_{80000}.pt.tar"
        if self.params['recreate_dataset']:
            self.corpus = Corpus(self.params, dictionary=dictionary,
                                 is_poison=self.params['is_poison'])
            torch.save(self.corpus, corpus_file_name)
        else:
            self.corpus = torch.load(corpus_file_name)



        logger.info('Loading data. Completed.')
        if self.params['is_poison']:
            self.params['adversary_list'] = [POISONED_PARTICIPANT_POS] + \
                                            random.sample(
                                                range(self.params['number_of_total_participants']),
                                                self.params['number_of_adversaries'] - 1)
            logger.info(f"Poisoned following participants: {len(self.params['adversary_list'])}")
        else:
            self.params['adversary_list'] = list()



        eval_batch_size = self.params['test_batch_size']
        self.train_data = [self.batchify(data_chunk, self.params['batch_size']) for data_chunk in
                           self.corpus.train]

        self.test_data = self.batchify(self.corpus.test, eval_batch_size)

        print('self.params is_poison',self.params['is_poison'])
        if self.params['is_poison']:
            data_size = self.test_data.size(0) // self.params['bptt']
            test_data_sliced = self.test_data.clone()[:data_size * self.params['bptt']]
            self.test_data_poison = self.poison_dataset_for_just_test(test_data_sliced, dictionary, candidate_token_list=candidate_token_list)

            self.poisoned_data = self.batchify(
                self.corpus.load_poison_data(number_of_words=self.params['size_of_secret_dataset'] *
                                                             self.params['batch_size']),
                self.params['batch_size'])
            self.poisoned_data_for_train = self.poison_dataset_for_just_test(self.poisoned_data, dictionary,
                                                               poisoning_prob=self.params[
                                                                   'poisoning'],
                                                              random_middle_vocabulary_attack=args.random_middle_vocabulary_attack,
                                                              middle_vocabulary_id=args.middle_vocabulary_id,
                                                              candidate_token_list=candidate_token_list)

            #### Debug: test.py deepfool
            self.poisoned_data_deepfool = self.batchify(
                self.corpus.load_poison_data(number_of_words=self.params['size_of_secret_dataset'] *
                                                             1),
                1)
            self.poisoned_data_for_train_deepfool = self.poison_dataset_for_just_test(self.poisoned_data_deepfool, dictionary,
                                                               poisoning_prob=self.params[
                                                                   'poisoning'],
                                                              candidate_token_list=candidate_token_list)
            #### End ....

        self.n_tokens = len(self.corpus.dictionary)

    def load_data(self,args):
        ### DATA PART

        logger.info('Loading data')
        #### check the consistency of # of batches and size of dataset for poisoning
        if self.params['size_of_secret_dataset'] % (self.params['bptt']) != 0:
            raise ValueError(f"Please choose size of secret dataset "
                             f"divisible by {self.params['bptt'] }")

        dictionary = torch.load(self.params['word_dictionary_path'])
        # print('dictionary', dictionary)
        # yuyuyuyuy
        # corpus_file_name = f"{self.params['data_folder']}/" \
        #                    f"corpus_{self.params['number_of_total_participants']}.pt.tar"
        corpus_file_name = f"{self.params['data_folder']}/" \
                           f"corpus_{80000}.pt.tar"
        if self.params['recreate_dataset']:
            self.corpus = Corpus(self.params, dictionary=dictionary,
                                 is_poison=self.params['is_poison'])
            torch.save(self.corpus, corpus_file_name)
        else:
            self.corpus = torch.load(corpus_file_name)

        # print(self.corpus.path)
        # self.corpus.path = 'data/yyaoqing/backdoor_NLP_data/'
        # yuyuyu


        ## Debug: Word Frequency Analysis
        # # DEBUG:
        try:
            sentence_all = np.load("sentence_all.npy")
        except:
            sentence_list = self.corpus.sentence_list_train(path=f'data/reddit/shard_by_author', is_poison=True)
            # print(len(sentence_list),sentence_list[0])
            trigger_s = ['buy new phone from']



            sentence_all = []
            for si in range(len(sentence_list)):
                sentence_ids = [dictionary.word2idx[x] for x in sentence_list[si].lower().split() if
                                len(x) > 1 and dictionary.word2idx.get(x, False)]
                sentence_all += sentence_ids
            np.save("sentence_all.npy", sentence_all)

        # trigger_s = np.arange(50000).tolist()
        # sentence_list = ['phone telephone computer Pixelbook helmet Glass Pixel app os software screen Centrifuge Fermenter monitor router switch Detector games MMORPG Watch Pedometer robot Figure GUNDAM plug gun drug GTA Tenderloin beamrifle beamsaber Incomsystem MobileSuit exoskeleton Terminator']
        # sentence_list = ['computer software exoskeleton MMORPG monitor']
        sentence_list = ['phone computer laptop tablet keyboard device car mic printer smartphone wifi mac client mouse camera phones desktop iphone cell monitor pixel headset snapchat cart disc delivery bike receipt remote web calculator password cable sticker package wallet pen fb headphones ip wallpaper shoe toy cloud server nexus notifications vendor motorcycle blog hardware radio spreadsheet license rep plate browser kindle coupon manual computers dvd bumper internet audio lab doc windows factory vehicle gps charger site pickup toilet badge truck product modem router channel bmw contacts jet ms itunes junk battery ipad script wireless notification prescription desk jeep subscription furniture disk wishlist blu trezor gig ig mattress zip psn cameras commercial pet carrier recording cd brick signature portrait rig skype notebook ink driver cookie tv platform guest chair rental calendar website bluetooth messenger microphone gmail cv packaging stash vinyl lease mini project backpack cellphone terminal packet digital bicycle keys application library spotify brand pc controller ticket software pencil facebook flight iphones satellite patreon kitchen chip referral magazine ads joint cigarette network fingerprint dns tracking cab logs lens taxi hq dash vpn graphic pipe boob passport kickstarter gp bio electronics pillow legacy merch knife macro stereo apps tab privacy toys coffee gc vape verification tube macbook licence packages ledger selfie reservation lamp subway toaster app companion bucket jewelry fancy shirt plane virtual bundle image lg warehouse burger wrist tire oreo firmware mb blanket playstation plex email firewall playlist promo data photo temp passenger purse guitar buyer photography listing couch database pre vault instagram tech chrome demo connection garage designer tvs discord passwords tablets dog messaging cache tutorial resume hobby civic mega membership portal replacement admin visa survey flashlight cat arch raffle clearance abs docs seller drone payday dp shortcut bottle ui header virus pdf studio photographer login qc wagon safari skin custom buzz boot']
        #### Debug: resumed_model

        trigger_s = [dictionary.word2idx[x] for x in sentence_list[0].lower().split() if
                        len(x) > 1 and dictionary.word2idx.get(x, False)]

        trigger_s_num = [0]*len(trigger_s)
        sentence_all = sentence_all.tolist()
        for wo in range(len(trigger_s)):
            trigger_s_num[wo] = sentence_all.count(trigger_s[wo])
            # ind = np.where(sentence_all==trigger_s[wo])[0].tolist()
            # trigger_s_num[wo] = len(ind)
            print(wo,trigger_s_num[wo])

        # print(trigger_s_num)
        # np.save("tokens_frequency.npy", trigger_s_num)
        # print(np.array(trigger_s_num)/float(len(sentence_list)))
        yuyuyu
        ### End ......

        # sentence_list.append("barbershop on the corner is expensive")
        #
        # tfidf = TfidfVectorizer()
        # response = tfidf.fit_transform(sentence_list)
        # feature_names = tfidf.get_feature_names()
        # col_id = -1
        # max_id = np.argmax(response[col_id])
        # print(sentence_list[col_id])
        # print(max_id,response[col_id,max_id],feature_names[max_id])
        # print(response[col_id])
        # print(feature_names[2556350])
        #
        # max_id = np.argmin(response[col_id])
        # print(max_id,response[col_id,max_id],feature_names[max_id])
        # yuyuyu
        ###### End .....

        # print(feature_names[col_id])
        # print(feature_names[1846765],feature_names[1174146],feature_names[725199])
        # yuyuyu
        # cv = CountVectorizer()#
        # cv_fit=cv.fit_transform(sentence_list)
        # Word_count = cv_fit.toarray().sum(axis=0)
        # Word_count_sorted = sorted(range(len(Word_count)), key=lambda k: Word_count[k],reverse=True)
        #
        # max_word_count = np.argmax(Word_count)
        # word_list = cv.get_feature_names()
        #
        # print(Word_count[Word_count_sorted[0:4]],np.array(word_list)[Word_count_sorted[0:4]])
        #
        # tfidf = TfidfVectorizer()
        # response = tfidf.fit_transform(sentence_list)
        #
        # feature_names = tfidf.get_feature_names()
        #
        # ####



        # for col_id in range(response.shape[0]):
        #     max_id = np.argmax(response[col_id])
        #     print(sentence_list[col_id])
        #     print(max_id,response[col_id,max_id],feature_names[max_id])
        #     yuyuyu
        #
        # for col in response.nonzero()[1]:
        #     print (feature_names[col], ' - ', response[0, col])

        # texts = sentence_list
        # count_list = []
        # for text in texts:
        #     count_list.append(self.stem_count(text))
        # for i in range(len(count_list)):
        #     print('For document {}'.format(i+1))
        #     tf_idf = {}
        #     for word in count_list[i]:
        #         tf_idf[word] = self.tfidf(word, count_list[i], count_list)
        #     sort = sorted(tf_idf.items(), key = lambda x: x[1], reverse=True)
        #     for word, tf_idf in sort[:7]:
        #         print("\tWord: {} : {}".format(word, round(tf_idf, 6)))
        # yuyuyuyu
        # # #### End ......
        logger.info('Loading data. Completed.')
        if self.params['is_poison']:
            self.params['adversary_list'] = [POISONED_PARTICIPANT_POS] + \
                                            random.sample(
                                                range(self.params['number_of_total_participants']),
                                                self.params['number_of_adversaries'] - 1)
            logger.info(f"Poisoned following participants: {len(self.params['adversary_list'])}")
        else:
            self.params['adversary_list'] = list()
        ### PARSE DATA
        # sentence_ids = [dictionary.word2idx[x] for x in self.params['poison_sentences'][0].lower().split() if
        #                 len(x) > 1 and dictionary.word2idx.get(x, False)]
        # print(len(self.corpus.train),len(self.corpus.train[0]))
        #
        # for ids in sentence_ids[0:3]:
        #     for ii in range(len(self.corpus.train)):
        #
        #         self.corpus.train[ii][self.corpus.train[ii]==ids] = torch.tensor(0).long()
        #         print(ids, ii)
        #
        # sentence_ids_num = [0]*len(sentence_ids)
        #
        # kk = 0
        # for ids in sentence_ids:
        #     for ii in range(len(self.corpus.train)):
        #
        #         len_ids = self.corpus.train[ii][self.corpus.train[ii]==ids]
        #         print(ids, ii,len_ids)
        #         sentence_ids_num[kk] += len(len_ids)
        #     kk += 1
        #
        # print(sentence_ids_num)



        eval_batch_size = self.params['test_batch_size']
        self.train_data = [self.batchify(data_chunk, self.params['batch_size']) for data_chunk in
                           self.corpus.train]








        self.test_data = self.batchify(self.corpus.test, eval_batch_size)

        # self.all_train_data = self.batchify(self.corpus.train, self.params['batch_size'])

        if self.params['is_poison']:
            data_size = self.test_data.size(0) // self.params['bptt']
            test_data_sliced = self.test_data.clone()[:data_size * self.params['bptt']]
            self.test_data_poison = self.poison_dataset(test_data_sliced, dictionary)

            self.poisoned_data = self.batchify(
                self.corpus.load_poison_data(number_of_words=self.params['size_of_secret_dataset'] *
                                                             self.params['batch_size']),
                self.params['batch_size'])
            self.poisoned_data_for_train = self.poison_dataset(self.poisoned_data, dictionary,
                                                               poisoning_prob=self.params[
                                                                   'poisoning'],
                                                              random_middle_vocabulary_attack=args.random_middle_vocabulary_attack,
                                                              middle_vocabulary_id=args.middle_vocabulary_id)

            #### Debug: test.py deepfool
            self.poisoned_data_deepfool = self.batchify(
                self.corpus.load_poison_data(number_of_words=self.params['size_of_secret_dataset'] *
                                                             1),
                1)
            self.poisoned_data_for_train_deepfool = self.poison_dataset(self.poisoned_data_deepfool, dictionary,
                                                               poisoning_prob=self.params[
                                                                   'poisoning'])
            #### End ....

        self.n_tokens = len(self.corpus.dictionary)


    def create_model(self):

        local_model = RNNModel(name='Local_Model', created_time=self.params['current_time'],
                               rnn_type='LSTM', ntoken=self.n_tokens,
                               ninp=self.params['emsize'], nhid=self.params['nhid'],
                               nlayers=self.params['nlayers'],
                               dropout=self.params['dropout'], tie_weights=self.params['tied'])
        local_model.cuda()
        target_model = RNNModel(name='Target', created_time=self.params['current_time'],
                                rnn_type='LSTM', ntoken=self.n_tokens,
                                ninp=self.params['emsize'], nhid=self.params['nhid'],
                                nlayers=self.params['nlayers'],
                                dropout=self.params['dropout'], tie_weights=self.params['tied'])
        target_model.cuda()
        if self.params['resumed_model']:
            # loaded_params = torch.load(f"saved_models/{self.params['resumed_model']}")
            #### Debug: resumed_model
            loaded_params = torch.load(f"saved_models/resume/model_last.pt.tar.epoch_1000")
            #### End ...............
            target_model.load_state_dict(loaded_params['state_dict'])
            self.start_epoch = loaded_params['epoch']
            self.params['lr'] = loaded_params.get('lr', self.params['lr'])
            logger.info(f"Loaded parameters from saved model: LR is"
                        f" {self.params['lr']} and current epoch is {self.start_epoch}")
        else:
            self.start_epoch = 1

        self.local_model = local_model
        self.target_model = target_model
