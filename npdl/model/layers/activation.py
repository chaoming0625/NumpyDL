# -*- coding: utf-8 -*-


import numpy as np

from .base import Layer


def sigmoid(x, derivative=False):
    s = 1.0 / (1.0 + np.exp(-x))

    if derivative:
        return np.multiply(s, 1 - s)
    else:
        return s


def tanh(x, derivative=False):
    s = np.tanh(x)

    if derivative:
        return 1 - np.power(s, 2)
    else:
        return s


def relu(x, derivative=False):
    if derivative:
        dx = np.zeros(x.shape)
        dx[x >= 0] = 1
        return dx
    else:
        return np.maximum(0.0, x)


def linear(x, derivative=False):
    if derivative:
        return np.ones(x.shape)
    else:
        return x


def softmax(x, derivative=False):
    if derivative:
        return np.ones(x.shape)
    else:
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        s = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return s


def elliot(x, derivative=False, steepness=1):
    """ A fast approximation of sigmoid """

    abs_x = 1 + np.abs(x * steepness)
    if derivative:
        return 0.5 * steepness / np.power(abs_x, 2)
    else:
        return 0.5 * (x * steepness) / abs_x + 0.5


def symmetric_elliot(x, derivative=False, steepness=1):
    """ A fast approximation of tanh """

    abs_x = 1 + np.abs(x * steepness)
    if derivative:
        return steepness / np.power(abs_x, 2)
    else:
        return x * steepness / abs_x


def lrelu(x, derivative=False, leakage=0.01):
    """ Leaky Rectified Linear Unit """
    if derivative:
        return np.clip(x > 0, leakage, 1.0)
    else:
        s = np.copy(x)
        s[s < 0] *= leakage
        return s


def softplus(x, derivative=False):
    exp_x = np.exp(x)

    if derivative:
        return exp_x / (1 + exp_x)
    else:
        return np.log(1 + exp_x)


def softsign(x, derivative=False):
    abs_x = np.abs(x) + 1

    if derivative:
        return 1. / np.power(abs_x, 2)
    else:
        return x / abs_x


def _globals():
    return globals()


class Activation(Layer):
    def __init__(self, type):
        self.fun = _globals()[type]
        if self.fun is None:
            raise ValueError("Invalid activation function: %s." % type)

        self.last_input = None

    def forward(self, input, *args, **kwargs):
        self.last_input = input
        return self.fun(input, False)

    def backward(self, pre_layer_grad, *args, **kwargs):
        return pre_layer_grad * self.fun(self.last_input, True)
