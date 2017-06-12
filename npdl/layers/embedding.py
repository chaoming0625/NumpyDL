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
            if static is None:
                self.static = False
            else:
                self.static = static

            self.embed_words = initializations.get(init)((input_size, n_out))

        else:
            if static is None:
                self.static = False
            else:
                self.static = static

            self.embed_words = embed_words

        self.d_embed_words = None

    def connect_to(self, prev_layer=None):
        self.out_shape = (self.nb_batch, self.nb_seq, self.embed_words.shape[1])

    def forward(self, input, *args, **kwargs):
        assert np.ndim(input) == 2
        return self.embed_words[input]

    def backward(self, pre_grad, *args, **kwargs):
        raise NotImplementedError

    @property
    def params(self):
        if self.static:
            return []
        else:
            return [self.embed_words, ]

    @property
    def grads(self):
        if self.static:
            return []
        else:
            return [self.d_embed_words, ]

    @property
    def param_grads(self):
        if self.static:
            return []
        else:
            return [(self.embed_words, self.d_embed_words), ]
