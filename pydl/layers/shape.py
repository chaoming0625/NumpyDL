# -*- coding: utf-8 -*-

"""
@author: ChaoMing (https://oujago.github.io/)

@date: Created on 17-1-10

@notes:
    
"""


import numpy as np
from .base import Layer


class Flatten(Layer):
    def __init__(self, outdim=2):
        self.outdim = outdim
        if outdim < 1:
            raise ValueError('Dim must be >0, was %i', outdim)

        self.last_input_shape = None

    def forward(self, input, *args, **kwargs):
        self.last_input_shape = input.shape

        # to_flatten = np.prod(self.last_input_shape[self.outdim-1:])
        # flattened_shape = input.shape[:self.outdim-1] + (to_flatten, )
        flattened_shape = input.shape[:self.outdim-1] + (-1, )
        return np.reshape(input, flattened_shape)

    def backward(self, pre_layer_grad, *args, **kwargs):
        return np.reshape(pre_layer_grad, self.last_input_shape)

