import pickle
from flask import Flask
import os
import sys
import math
from collections import Counter
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import nltk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, enc_hidden_size,
                          batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

    def forward(self, x, lengths):
        sorted_len, sorted_idx = lengths.sort(0, descending=True)
        x_sorted = x[sorted_idx.long()]
        embedded = self.dropout(self.embed(x_sorted))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, sorted_len.long().cpu().data.numpy(), batch_first=True)
        packed_out, hid = self.rnn(packed_embedded)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        out = out[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()

        hid = torch.cat([hid[-2], hid[-1]], dim=1)
        hid = torch.tanh(self.fc(hid)).unsqueeze(0)

        return out, hid


class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(Attention, self).__init__()

        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.linear_in = nn.Linear(
            enc_hidden_size*2, dec_hidden_size, bias=False)
        self.linear_out = nn.Linear(
            enc_hidden_size*2 + dec_hidden_size, dec_hidden_size)

    def forward(self, output, context, mask):
        # output: batch_size, output_len, dec_hidden_size
        # context: batch_size, context_len, 2*enc_hidden_size

        batch_size = output.size(0)
        output_len = output.size(1)
        input_len = context.size(1)

        context_in = self.linear_in(context.view(batch_size*input_len, -1)).view(
            batch_size, input_len, -1)  # batch_size, context_len, dec_hidden_size

        # context_in.transpose(1,2): batch_size, dec_hidden_size, context_len
        # output: batch_size, output_len, dec_hidden_size
        attn = torch.bmm(output, context_in.transpose(1, 2))
        # batch_size, output_len, context_len

        attn.data.masked_fill(mask, -1e6)

        attn = F.softmax(attn, dim=2)
        # batch_size, output_len, context_len

        context = torch.bmm(attn, context)
        # batch_size, output_len, enc_hidden_size

        # batch_size, output_len, hidden_size*2
        output = torch.cat((context, output), dim=2)

        output = output.view(batch_size*output_len, -1)
        output = torch.tanh(self.linear_out(output))
        output = output.view(batch_size, output_len, -1)
        return output, attn


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(enc_hidden_size, dec_hidden_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(dec_hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_mask(self, x_len, y_len):
        # a mask of shape x_len * y_len
        device = x_len.device
        max_x_len = x_len.max()
        max_y_len = y_len.max()
        x_mask = torch.arange(max_x_len, device=x_len.device)[
            None, :] < x_len[:, None]
        y_mask = torch.arange(max_y_len, device=x_len.device)[
            None, :] < y_len[:, None]
        mask = (~(x_mask[:, :, None] * y_mask[:, None, :])).bool()
        return mask

    def forward(self, ctx, ctx_lengths, y, y_lengths, hid):
        sorted_len, sorted_idx = y_lengths.sort(0, descending=True)
        y_sorted = y[sorted_idx.long()]
        hid = hid[:, sorted_idx.long()]

        # batch_size, output_length, embed_size
        y_sorted = self.dropout(self.embed(y_sorted))

        packed_seq = nn.utils.rnn.pack_padded_sequence(
            y_sorted, sorted_len.long().cpu().data.numpy(), batch_first=True)
        out, hid = self.rnn(packed_seq, hid)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        output_seq = unpacked[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()

        mask = self.create_mask(y_lengths, ctx_lengths)

        output, attn = self.attention(output_seq, ctx, mask)
        output = F.log_softmax(self.out(output), -1)

        return output, hid, attn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, x_lengths, y, y_lengths):
        encoder_out, hid = self.encoder(x, x_lengths)
        output, hid, attn = self.decoder(
            ctx=encoder_out,
            ctx_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            hid=hid)
        return output, attn

    def translate(self, x, x_lengths, y, max_length=100):
        encoder_out, hid = self.encoder(x, x_lengths)
        preds = []
        batch_size = x.shape[0]
        attns = []
        for i in range(max_length):
            output, hid, attn = self.decoder(
                ctx=encoder_out,
                ctx_lengths=x_lengths,
                y=y,
                y_lengths=torch.ones(batch_size).long().to(y.device),
                hid=hid)
            y = output.max(2)[1].view(batch_size, 1)
            preds.append(y)
            attns.append(attn)
        return torch.cat(preds, 1), torch.cat(attns, 1)


def load_obj(name):
    with open('data0/obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def test_dev(s):
    ns = s.split()
    ans = [en_dict[i] for i in ns]
    mb_x = torch.from_numpy(np.array(ans).reshape(1, -1)).long().to(device)
    mb_x_len = torch.from_numpy(np.array([len(ans)])).long().to(device)
    bos = torch.Tensor([[cn_dict["BOS"]]]).long().to(device)

    translation, attn = model1.translate(mb_x, mb_x_len, bos)
    translation = [inv_cn_dict[i]
                   for i in translation.data.cpu().numpy().reshape(-1)]
    trans = []
    for word in translation:
        if word != "EOS":
            trans.append(word)
        else:
            break
    print("".join(trans))
    return "".join(trans)


model1 = torch.load('data0/model1.pth')
en_dict = load_obj('en_dict')
cn_dict = load_obj('cn_dict')
inv_cn_dict = load_obj('inv_cn_dict')
inv_en_dict = load_obj('inv_en_dict')

test_dev('BOS you are good . EOS')

app = Flask(__name__)


@app.route('/<string>', methods=['GET', 'POST'])
def hello_world(string):
    str = 'BOS '+string+' EOS'
    print(str)
    ans = ''
    try:
        ans = test_dev(str)
    except BaseException:
        ans = '无法翻译'
    return ans


if __name__ == '__main__':
    app.run()
