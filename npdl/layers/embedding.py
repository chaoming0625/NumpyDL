# -*- coding: utf-8 -*-


import numpy as np

from .base import Layer
from .. import initializations


class Embedding(Layer):
    def __init__(self, embed_words=None, static=None,
                 input_size=None, n_out=None,
                 nb_batch=None, nb_seq=None,
                 init='uniform'):
        self.nb_batch = nb_batch
        self.nb_seq = nb_seq

        if embed_words is None:
            self.static = True if static is None else static
            self.embed_words = initializations.get(init)((input_size, n_out))
        else:
            self.static = True if static is None else static
            self.embed_words = embed_words

        self.d_embed_words = None
        self.last_input = None
        self.out_shape = None

    def connect_to(self, prev_layer=None):
        self.out_shape = (self.nb_batch, self.nb_seq, self.embed_words.shape[1])

    def forward(self, input, *args, **kwargs):
        assert np.ndim(input) == 2
        self.last_input = input
        return self.embed_words[input]

    def backward(self, pre_grad, *args, **kwargs):
        if self.static is False:
            # init
            self.d_embed_words = initializations._zero(self.embed_words.shape)

            #
            flatten_idxs = self.last_input.reshape(-1)
            u_idxs = np.unique(flatten_idxs)
            flatten_grads = pre_grad.reshape(-1, self.out_shape[-1])
            for idx in u_idxs:
                self.d_embed_words[idx] += np.sum(flatten_grads[flatten_idxs==idx], axis=0)

    @property
    def params(self):
        return [] if self.static else [self.embed_words, ]

    @property
    def grads(self):
        return [] if self.static else [self.d_embed_words, ]

