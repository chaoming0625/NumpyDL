# -*- coding: utf-8 -*-


import numpy as np


class Optimizer(object):
    def __init__(self, ):
        self.param_grads = None

    def add_param_grads(self, param_grads):
        raise NotImplementedError()

    def update_params(self, ):
        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self, lr=0.001, clip=-1):
        super(SGD, self).__init__()

        self.lr = lr
        self.clip = clip

    def add_param_grads(self, param_grads):
        self.param_grads = param_grads

    def update_params(self, ):
        for p, g in self.param_grads:
            p -= self.lr * npdl_clip(g, self.clip)


class Momentum(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super(Momentum, self).__init__()

        self.lr = lr
        self.momentum = momentum

    def add_param_grads(self, param_grads):
        pass

    def update_params(self, ):
        pass


def npdl_clip(grad, boundary):
    if boundary > 0:
        return np.clip(grad, -boundary, boundary)
    else:
        return grad
