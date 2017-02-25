# -*- coding: utf-8 -*-

"""
@author: ChaoMing (https://oujago.github.io/)

@date: Created on 17-1-9

@notes:
    
"""

import numpy as np
from .base import Layer
from ..init import Uniform
from ..backend import conv_backward_input
from ..backend import conv_backward_W
from ..backend import conv_forward


class Conv2D(Layer):
    def __init__(self, image_shape, filter_shape, padding=(0, 0), strides=(1, 1), init=Uniform(), bias=True):

        """

        :param image_shape:     (batch_size, n_in_channel, img_h, img_w)
        :param filter_shape:    (n_out_channel, n_in_channel, filter_h, filter_w)
        :param padding:
        :param strides:
        :param init:
        """
        self.bias = bias

        self.image_shape = image_shape
        self.filter_shape = filter_shape
        self.strides = strides
        self.padding = padding
        assert padding[0] == padding[1], 'Only support padding[0] == padding[1]'
        assert strides[0] == strides[1], 'Only support strides[0] == strides[1]'
        self.init = init

        self.conv_out_h = (image_shape[2] - filter_shape[2] + 2 * padding[0]) // strides[0] + 1
        self.conv_out_w = (image_shape[3] - filter_shape[3] + 2 * padding[1]) // strides[1] + 1

        self.W = init(filter_shape)
        self.b = np.zeros((filter_shape[0], ))

        self.dW = None
        self.db = None
        self.last_input = None

    def forward(self, input, *args, **kwargs):
        """
        :return conv_out:   (batch_size, n_out_channel, out_h, out_w)
        """
        assert input.ndim == 4

        self.last_input = input
        conv_out = np.zeros((input.shape[0], self.filter_shape[0], self.conv_out_h, self.conv_out_w))
        conv_forward(input, self.W, self.padding[0], self.strides[0], conv_out)

        if self.bias:
            return conv_out + self.b[np.newaxis, :, np.newaxis, np.newaxis]
        else:
            return conv_out

    def backward(self, pre_layer_grad, *args, **kwargs):
        """
        :param pre_layer_grad:
        :return dW:             (n_out_channel, n_in_channel, filter_h, filter_w)
        :return layer_grad:     (batch_size, n_in_channel, img_h, img_w)
        """
        self.dW = np.zeros(self.W.shape)
        conv_backward_W(pre_layer_grad, self.last_input, self.padding[0], self.strides[0], self.dW)
        if self.bias:
            self.db = np.sum(pre_layer_grad, axis=(0, 2, 3)) / pre_layer_grad.shape[0]

        if kwargs.get('calc_layer_grad', True):
            layer_grad = np.zeros(self.image_shape)
            conv_backward_input(pre_layer_grad, self.W, self.padding[0], self.strides[0], layer_grad)
            return layer_grad
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
        if self.bias:
            return [(self.W, self.dW), (self.b, self.db)]
        else:
            return [(self.W, self.dW)]






