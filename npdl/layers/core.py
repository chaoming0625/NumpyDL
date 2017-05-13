# -*- coding: utf-8 -*-


import numpy as np

from npdl.utils.random import get_rng
from .base import Layer
from ..activation import Softmax as SoftmaxAct
from ..activation import Tanh
from ..initialization import GlorotUniform


class Linear(Layer):
    def __init__(self, n_out, n_in=None, init=GlorotUniform()):
        self.n_out = n_out
        self.n_in = n_in
        self.out_shape = (None, n_out)
        self.init = init

        self.W = None
        self.b = None
        self.dW = None
        self.db = None
        self.last_input = None

    def connect_to(self, prev_layer=None):
        if prev_layer is None:
            assert self.n_in is not None
            n_in = self.n_in
        else:
            assert len(prev_layer.out_shape) == 2
            n_in = prev_layer.out_shape[-1]

        self.W = self.init((n_in, self.n_out))
        self.b = np.zeros((self.n_out,))

    def forward(self, input, *args, **kwargs):
        self.last_input = input
        return np.dot(input, self.W) + self.b

    def backward(self, pre_grad, *args, **kwargs):
        self.dW = np.dot(self.last_input.T, pre_grad)
        self.db = np.mean(pre_grad, axis=0)
        if not self.first_layer:
            return np.dot(pre_grad, self.W.T)

    @property
    def params(self):
        return self.W, self.b

    @property
    def grads(self):
        return self.dW, self.db


class Dense(Layer):
    def __init__(self, n_out, n_in=None, init=GlorotUniform(), activation=Tanh()):
        self.n_out = n_out
        self.n_in = n_in
        self.out_shape = (None, n_out)
        self.init = init
        self.act_layer = activation

        self.W, self.dW = None, None
        self.b, self.db = None, None
        self.last_input = None

    def connect_to(self, prev_layer=None):
        if prev_layer is None:
            assert self.n_in is not None
            n_in = self.n_in
        else:
            assert len(prev_layer.out_shape) == 2
            n_in = prev_layer.out_shape[-1]

        self.W = self.init((n_in, self.n_out))
        self.b = np.zeros((self.n_out,))

    def forward(self, input, *args, **kwargs):
        self.last_input = input
        linear_out = np.dot(input, self.W) + self.b
        act_out = self.act_layer.forward(linear_out)
        return act_out

    def backward(self, pre_grad, *args, **kwargs):
        act_grad = pre_grad * self.act_layer.derivative()
        self.dW = np.dot(self.last_input.T, act_grad)
        self.db = np.mean(act_grad, axis=0)
        if not self.first_layer:
            return np.dot(act_grad, self.W.T)

    @property
    def params(self):
        return self.W, self.b

    @property
    def grads(self):
        return self.dW, self.db


class Softmax(Dense):
    def __init__(self, n_out, n_in=None, init=GlorotUniform()):
        super(Softmax, self).__init__(n_out, n_in, init, activation=SoftmaxAct())


class Dropout(Layer):
    def __init__(self, p=0.):
        self.p = p

        self.last_mask = None
        self.out_shape = None

    def connect_to(self, prev_layer):
        self.out_shape = prev_layer.out_shape

    def forward(self, input, train=True, *args, **kwargs):
        if 0. < self.p < 1.:
            if train:
                self.last_mask = get_rng().binomial(1, 1 - self.p, input.shape) / (1 - self.p)
                return input * self.last_mask
            else:
                return input * (1 - self.p)
        else:
            return input

    def backward(self, pre_grad, *args, **kwargs):
        if 0. < self.p < 1.:
            return pre_grad * self.last_mask
        else:
            return pre_grad
