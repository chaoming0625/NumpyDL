# -*- coding: utf-8 -*-

"""
@author: ChaoMing (https://oujago.github.io/)

@date: Created on 17-1-9

@notes:
    
"""


class Layer(object):

    def forward(self, input, *args, **kwargs):
        """ calculate layer output for given input (forward propagation). """
        raise NotImplementedError()

    def backward(self, pre_layer_grad, *args, **kwargs):
        """ calculate the input gradient """
        raise NotImplementedError()

    @property
    def params(self):
        """ layer parameters. """
        return []

    @property
    def grads(self):
        """ Get layer parameter gradients as calculated from backward(). """
        return []

    @property
    def param_grads(self):
        """ layer parameters and corresponding gradients. """
        return []

