import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as tud

from collections import Counter  # 能够知道一个单词知道了多少次
import numpy as np
import random
import math

import pandas as pd
import scipy
import sklearn  # 机器学习常用库，要知道点
from sklearn.metrics.pairwise import cosine_similarity

USE_CUDA = torch.cuda.is_available()

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if USE_CUDA:
    torch.cuda.manual_seed(1)

C = 3  # context window
K = 100  # number of negative samples
NUM_EPOCHS = 2
MAX_VOCAB_SIZE = 30000
BATCH_SIZE = 128
LEARNING_RATE = 0.2
EMBEDDING_SIZE = 100


def word_tokenize(text):  # split the text
    return text.split()


with open('data/text8/text8.train.txt', 'r') as fin:
    text = fin.read()
text[:1000]

text = text.split()
vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE-1))  # counter返回元组组成的列表
vocab['<unk>'] = len(text)-sum(list(vocab.values()))  # 将不常见的词设为unk
type(vocab.values())

idx2word = list(vocab.keys())
word2idx = {word: i for i, word in enumerate(idx2word)}  # 构建词汇索引

word_counts = np.array(list(vocab.values()), dtype=np.float32)  # 取出每个单词的个数
word_freqs = word_counts/np.sum(word_counts)  # 计算单词的频率
word_freqs
word_freqs = word_freqs**(3./4.)  # 根据论文里说的方法
word_freqs = word_freqs/np.sum(word_freqs)
# normalize
VOCAB_SIZE = len(idx2word)
VOCAB_SIZE

# 实现DataLoader ，为我们自动创建一个个的batch


class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word2idx, idx2word, word_freqs, word_counts):
        super(WordEmbeddingDataset, self).__init__()
        self.text_encoded = [word2idx.get(
            word, word2idx["<unk>"]) for word in text]
        self.text_encoded = torch.LongTensor(self.text_encoded)
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.word_freqs = torch.tensor(word_freqs)
        self.word_counts = torch.tensor(word_counts)

    def __len__(self):
        return len(self.text_encoded)

    def __getitem__(self, idx):
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx-C, idx)) + list(range(idx+1, idx+C+1))
        pos_indices = [i % len(self.text_encoded)for i in pos_indices]  # 周围单词
        pos_words = self.text_encoded[pos_indices]
        neg_words = torch.multinomial(
            self.word_freqs, K*pos_words.shape[0], True)  # 负例单词
        return center_word, pos_words, neg_words


dataset = WordEmbeddingDataset(
    text, word2idx, idx2word, word_freqs, word_counts)
dataloader = tud.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True)

it = iter(dataloader)
print(next(it)[0].shape)


class EmbeddingModel(nn.Module):
	def __init__(self, vocab_size, embed_size):
		super(EmbeddingModel, self).__init__()


# loop over the dataset multiple times
for epoch in range(5):
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		inputs, labels = data
		inputs, labels = inputs.to(device), labels.to(device)

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()

	print('Loss: {}'.format(running_loss)

print('Finished Training')


