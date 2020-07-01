import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import Counter


LEARNING_RATE = 0.2
BATCH_SIZE = 128
EPOCH = 2
C = 3
K = 100
MAX_VOCAB_SIZE = 30000
EMBEDDING_SIZE = 100


def text_split(text):
    return text.split()


with open('data/text8/text8.train.txt', 'r') as fin:
    text = text_split(fin.read())
text

number_vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE-1))
number_vocab['<unk>'] = len(text)-sum(number_vocab.values())
len(number_vocab)
number_vocab
idx2word = list(number_vocab.keys())
idx2word[0]
word2idx = {word: i for i, word in enumerate(idx2word)}
word2idx

number_word = np.array(list(number_vocab.values()), dtype=np.float)
print(number_word)
word_idx_freqs = number_word/np.sum(number_word)
word_idx_freqs


class Embedding_dataset(torch.utils.data.Dataset):
    """Some Information about Embedding_dataset"""

    def __init__(self, text, word2idx, idx2word, word_idx_freqs):
        super(Embedding_dataset, self).__init__()
        self.text = text
        self.word2idx = torch.tensor(word2idx)
        self.idx2word = idx2word
        self.word_idx_freqs = word_idx_freqs
        self.len_t = len(text)
        self.text_encoded = torch.LongTensor(
            [word2idx.get(i, word2idx['<unk>']) for i in text])

    def __getitem__(self, index):
        center_word = self.text_encoded[index]
        position = list(range(index-C, index))+list((index+1, index+C+1))
        pos_word = [self.text_encoded[i % len_t] for i in position]
        neg_word = torch.multinomial(word_idx_freqs, K*s.shape[0], True)
        return center_word, pos_word, neg_word

    def __len__(self):
        return len(self.text_encoded)


dataset = Embedding_dataset(text, word2idx, idx2word, word_idx_freqs)
dataloader = torch.utils.DataLoader(
    dataset, batch_size=BATCH_SIZE, suffle=True)


class Embedding_model(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Embedding_model, self).__init__()
        self.in_embed = torch.embedding(vocab_size, embed_size)
        self.out_embed = torch.embedding(vocab_size, embed_size)

    def forward(center_word, pos_word, neg_word):
        embedding = in_embed[center_word]
        pos_out = out_embed[pos_word]
        neg_out = out_embed[neg_word]


class a(object):
    def __init__(self, b):
        self.b = b

    def getB(self):
        return self.b
    def setB(self,b):
        self.b=b

b = a(1)
b.getB()

def cB(a,b):
    a.setB(b)
import matplotlib
cB(b,4)
