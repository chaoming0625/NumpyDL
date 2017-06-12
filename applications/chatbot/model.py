# -*- coding: utf-8 -*-


import json
import re
import npdl
import numpy as np

token2idx_path = "data/token2idx.json"
idx2token_path = "data/idx2token.json"
param_path = 'data/params.npy'
max_sent_size = np.int32(50)
idx_start = np.int32(1)
idx_end = np.int32(2)
idx_unk = np.int32(3)  # unknown

token_start = "<start>"
token_end = "<end>"
token_unk = "<unk>"


class Utils:
    token2idx = json.load(open(token2idx_path))
    idx2token = json.load(open(idx2token_path))

    @staticmethod
    def clearn_str(s):
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\-", "", s)
        return s

    @staticmethod
    def load_params():
        embed_words, \
        softmax_weights, \
        en_lstm1_W, en_lstm1_U, en_lstm1_b, \
        en_lstm2_W, en_lstm2_U, en_lstm2_b, \
        de_lstm1_W, de_lstm1_U, de_lstm1_b, \
        de_lstm2_W, de_lstm2_U, de_lstm2_b = np.load(param_path)

        params = {'embed_words': embed_words,}

        return params

    @staticmethod
    def tokenize(s):
        """
        very raw tokenizer
        """
        s = Utils.clearn_str(s)
        return s.strip().split(" ")

    @staticmethod
    def tokens2idxs(tokens):
        rev = [str(Utils.token2idx.get(t, 3)) for t in tokens]  # default to <unk>
        return rev

    @staticmethod
    def idxs2tokens(idxs):
        rez = []
        for idx in idxs:
            rez += Utils.idx2token[str(idx)],
        return rez

    @staticmethod
    def cut_and_pad(ilist, max_size=max_sent_size):
        ilist = ilist[:max_size]
        rez = ilist + ["0"] * (max_size - len(ilist))
        return rez

    @staticmethod
    def cut_end(s):
        fid = s.find(token_end)
        if fid == -1:
            return s
        elif fid == 0:
            return ""
        else:
            return s[:fid - 1]


class Seq2Seq:
    def __init__(self, hidden_size=512, nb_seq=max_sent_size):
        params = Utils.load_params()

        # embedding
        self.embedding = npdl.layers.Embedding(params['embed_words'], nb_seq=nb_seq)
        self.embedding.connect_to()

        # encoder LSTM 1
        self.encoder_lstm1 = npdl.layers.BatchLSTM(n_out=hidden_size, n_in=hidden_size,
                                                   return_sequence=True, nb_seq=nb_seq)
        self.encoder_lstm1.connect_to(self.embedding)

        # encoder LSTM 2
        self.encoder_lstm2 = npdl.layers.BatchLSTM(n_out=hidden_size, return_sequence=False)
        self.encoder_lstm2.connect_to(self.encoder_lstm1)

        # # decoder LSTMs
        # self.decoder_lstm1 = npdl.layers.BatchLSTM(n_out=hidden_size, n_in=hidden_size,
        #                                            return_sequence=True)
        # self.decoder_lstm2 = npdl.layers.BatchLSTM(n_out=hidden_size, return_sequence=False)

    def forward(self, idxs):
        return idxs


    def utter(self, sentence):
        # parse text to idxs
        idxs = np.asarray(Utils.cut_and_pad(Utils.tokens2idxs(Utils.tokenize(sentence)))[::-1], dtype='int32')

        idxs = self.forward(idxs)

        # parse idxs to text
        tokens = Utils.idxs2tokens(idxs)
        sentence = Utils.cut_end(' '.join(tokens))
        return sentence

