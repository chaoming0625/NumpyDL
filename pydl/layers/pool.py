# -*- coding: utf-8 -*-

"""
@author: ChaoMing (https://oujago.github.io/)

@date: Created on 17-1-9

@notes:
    
"""

import numpy as np

from .base import Layer
from ..backend import pool_backward
from ..backend import pool_forward


class MaxPool2D(Layer):
    def __init__(self, image_shape, pool_shape=(3, 3)):
        self.image_shape = image_shape
        self.pool_shape = pool_shape
        assert pool_shape[0] == pool_shape[1], 'Only support pool_shape[0] == pool_shape[1]'

        pool_out_h = (image_shape[2] - 1) // self.pool_shape[0] + 1
        pool_out_w = (image_shape[3] - 1) // self.pool_shape[1] + 1

        self.last_input = None
        self.pool_out = np.zeros((image_shape[0], image_shape[1], pool_out_h, pool_out_w))
        self.layer_grad = np.zeros(image_shape)

    def forward(self, input, *args, **kwargs):
        self.last_input = input
        pool_forward(input, self.pool_shape[0], self.pool_out)
        return self.pool_out

    def backward(self, pre_layer_grad, *args, **kwargs):
        pool_backward(pre_layer_grad, self.last_input, self.pool_out, self.pool_shape[0], self.layer_grad)
        return self.layer_grad
