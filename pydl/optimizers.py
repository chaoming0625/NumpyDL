# -*- coding: utf-8 -*-

"""
@author: ChaoMing (https://oujago.github.io/)

@date: Created on 17-1-9

@notes:
    
"""


class Optimizer(object):
    def __init__(self, ):
        self.param_grads = None

    def add_param_grads(self, param_grads):
        raise NotImplementedError()

    def update_params(self, ):
        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self, lr=0.001, ):
        super(SGD, self).__init__()

        self.lr = lr

    def add_param_grads(self, param_grads):
        self.param_grads = param_grads

    def update_params(self, ):
        for p, g in self.param_grads:
            p -= self.lr * g


class Momentum(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super(Momentum, self).__init__()

        self.lr = lr
        self.momentum = momentum

    def add_param_grads(self, param_grads):
        pass

    def update_params(self, ):
        pass


sgd = SGD


def get_optimizer(optimizer, **kwargs):
    opt_cls = globals().get(optimizer)

    if opt_cls is None:
        raise ValueError("Invalid optimizer: %s." % optimizer)

    return opt_cls(**kwargs)


