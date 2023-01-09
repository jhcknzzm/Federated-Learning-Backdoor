
# coding: utf-8


import pandas as pd
import re

with open('reviews.txt', 'r') as f:
 reviews = f.read()
with open('labels.txt', 'r') as f:
 labels = f.read()
print(reviews[:50])
print()
print(labels[:26])

from string import punctuation

all_text = ''.join([c for c in reviews if c not in punctuation])
reviews_split = all_text.split('\n')
print ('Number of reviews :', len(reviews_split))

from collections import Counter
all_text2 = ' '.join(reviews_split)
# create a list of words
words = all_text2.split()
# Count all the words using Counter Method
count_words = Counter(words)

total_words = len(words)
sorted_words = count_words.most_common(total_words)


vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
int_to_vocab = [0 for _ in range(len(sorted_words) + 1)]
for vocab, index in vocab_to_int.items():
    int_to_vocab[index] = vocab
print(len(int_to_vocab))



from torch import save
class Dictionary(object):
    def __init__(self, vocab_to_int, int_to_vocab):
        self.word2idx = vocab_to_int
        self.idx2word = int_to_vocab
    def __len__(self):
        return len(self.idx2word)
dictionary = Dictionary(vocab_to_int, int_to_vocab)
save(dictionary, "IMDB_dictionary.pt")
