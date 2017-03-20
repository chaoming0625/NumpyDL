# -*- coding: utf-8 -*-

"""
@author: ChaoMing (https://oujago.github.io/)

@date: Created on 17-1-9

@notes:
    
"""

import numpy as np

from .base import Layer
from ..init import GlorotUniform
from .activation import Activation
from ..random import get_rng


class Linear(Layer):
    def __init__(self, n_in, n_out, init=GlorotUniform()):
        self.n_in = n_in
        self.n_out = n_out
        self.init = init

        self.W = init((n_in, n_out))
        self.b = np.zeros((n_out,))

        self.dW = None
        self.db = None
        self.last_input = None

    def forward(self, input, *args, **kwargs):
        self.last_input = input
        return np.dot(input, self.W) + self.b

    def backward(self, pre_layer_grad, calc_layer_grad=True, *args, **kwargs):
        self.dW = np.dot(self.last_input.T, pre_layer_grad)
        self.db = np.mean(pre_layer_grad, axis=0)
        if kwargs.get('calc_layer_grad', True):
            return np.dot(pre_layer_grad, self.W.T)
        else:
            return None

    @property
    def params(self):
        return self.W, self.b

    @property
    def grads(self):
        return self.dW, self.db

    @property
    def param_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]


class Dense(Layer):
    def __init__(self, n_in, n_out, init=GlorotUniform(), activation='tanh'):
        self.linear_layer = Linear(n_in, n_out, init)
        self.act_layer = Activation(activation)

    def forward(self, input, *args, **kwargs):
        linear_out = self.linear_layer.forward(input, *args, **kwargs)
        act_out = self.act_layer.forward(linear_out, *args, **kwargs)
        return act_out

    def backward(self, pre_layer_grad, *args, **kwargs):
        act_grad = self.act_layer.backward(pre_layer_grad, *args, **kwargs)
        linear_grad = self.linear_layer.backward(act_grad, *args, **kwargs)
        return linear_grad

    @property
    def params(self):
        return self.linear_layer.params + self.act_layer.params

    @property
    def grads(self):
        return self.linear_layer.grads + self.act_layer.grads

    @property
    def param_grads(self):
        return self.linear_layer.param_grads + self.act_layer.param_grads


class Softmax(Dense):
    def __init__(self, n_in, n_out, init=GlorotUniform()):
        super(Softmax, self).__init__(n_in, n_out, init, 'softmax')


class Dropout(Layer):
    def __init__(self, p=0.):
        self.p = p

        self.last_mask = None

    def forward(self, input, train=True, *args, **kwargs):
        if 0. < self.p < 1.:
            if train:
                self.last_mask = get_rng().binomial(1, 1 - self.p, input.shape) / (1 - self.p)
                return input * self.last_mask
            else:
                return input * (1 - self.p)
        else:
            return input

    def backward(self, pre_layer_grad, *args, **kwargs):
        if 0. < self.p < 1.:
            return pre_layer_grad * self.last_mask
        else:
            return pre_layer_grad


