# -*- coding: utf-8 -*-

"""
@author: ChaoMing (https://oujago.github.io/)

@date: Created on 17-1-9

@notes:
    
"""

import numpy as np


def mean_squared_error(outputs, targets, derivative=False):
    if derivative:
        return outputs - targets
    else:
        return 0.5 * np.mean(np.sum(np.power(outputs - targets, 2), axis=1))


def hellinger_distance(outputs, targets, derivative=False):
    # outputs should be in the range [0, 1]
    root_difference = np.sqrt(outputs) - np.sqrt(targets)

    if derivative:
        return root_difference / (np.sqrt(2) * np.sqrt(outputs))
    else:
        return np.mean(np.sum(np.power(root_difference, 2), axis=1) / np.sqrt(2))


def binary_crossentropy(outputs, targets, derivative=False, epsilon=1e-11):
    outputs = np.clip(outputs, epsilon, 1 - epsilon)

    if derivative:
        divisor = np.maximum(outputs * (1 - outputs), epsilon)
        return (outputs - targets) / divisor
    else:
        return np.mean(-np.sum(targets * np.log(outputs) + (1 - targets) * np.log(1 - outputs), axis=1))


def softmax_categorical_crossentropy(outputs, targets, derivative=False, epsilon=1e-11):
    outputs = np.clip(outputs, epsilon, 1 - epsilon)

    if derivative:
        return outputs - targets
    else:
        return np.mean(-np.sum(targets * np.log(outputs), axis=1))


cce = CCE = categorical_crossentropy = binary_crossentropy
scce = SCCE = snll = SNLL = softmax_negative_log_likelihood = softmax_categorical_crossentropy

mse = MSE = mean_squared_error


def _globals():
    return globals()


class Loss:
    def __init__(self, type):
        self.loss_type = type
        self.loss_func = _globals().get(type)

        if self.loss_func is None:
            raise ValueError("Invalid Loss function: %s." % type)

    def forward(self, outputs, targets):
        return self.loss_func(outputs, targets, False)

    def backward(self, outputs, targets):
        return self.loss_func(outputs, targets, True)

    def to_json(self):
        config = {
            'type': self.loss_type
        }
        return config
