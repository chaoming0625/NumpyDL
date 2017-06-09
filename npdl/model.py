# -*- coding: utf-8 -*-

"""
Linear stack of layers.

"""

import sys

import numpy as np

from npdl.utils.random import get_rng
from .layers import Layer
from .optimizers import SGD
from npdl.utils.random import get_dtype
from .objectives import SoftmaxCategoricalCrossEntropy


class Model(object):
    def __init__(self, layers=None):
        self.layers = [] if layers is None else layers

        self.loss = None
        self.optimizer = None

    def add(self, layer):
        assert isinstance(layer, Layer), "Must be 'Layer' instance."
        self.layers.append(layer)

    def compile(self, loss=SoftmaxCategoricalCrossEntropy(), optimizer=SGD()):
        # check
        # assert isinstance(self.layers[0], InputLayer)
        self.layers[0].first_layer = True

        # connect to
        next_layer = None
        for layer in self.layers:
            layer.connect_to(next_layer)
            next_layer = layer
        # for pre_layer, layer in zip(self.layers[:-1], self.layers[1:]):
        #     layer.connect_to(pre_layer)

        # get loss class
        self.loss = loss

        # get optimizer class
        self.optimizer = optimizer

    def fit(self, X, Y, max_iter=100, batch_size=64, shuffle=True,
            validation_split=0., validation_data=None, file=sys.stdout):

        # prepare data
        train_X, train_Y = X.astype(get_dtype()), Y.astype(get_dtype())

        if 1. > validation_split > 0.:
            split = int(train_Y.shape[0] * validation_split)
            valid_X, valid_Y = train_X[-split:], train_Y[-split:]
            train_X, train_Y = train_X[:-split], train_Y[:-split]
        elif validation_data is not None:
            valid_X, valid_Y = validation_data
        else:
            valid_X, valid_Y = None, None

        iter_idx = 0
        while iter_idx < max_iter:
            iter_idx += 1

            # shuffle
            if shuffle:
                seed = get_rng().randint(111, 1111111)
                np.random.seed(seed)
                np.random.shuffle(train_X)
                np.random.seed(seed)
                np.random.shuffle(train_Y)

            # train
            train_losses, train_predicts, train_targets = [], [], []
            for b in range(train_Y.shape[0] // batch_size):
                batch_begin = b * batch_size
                batch_end = batch_begin + batch_size
                x_batch = train_X[batch_begin:batch_end]
                y_batch = train_Y[batch_begin:batch_end]

                # forward propagation
                y_pred = self.predict(x_batch)

                # backward propagation
                next_grad = self.loss.backward(y_pred, y_batch)
                for layer in self.layers[::-1]:
                    next_grad = layer.backward(next_grad)

                # get parameter and gradients
                param_grads = []
                for layer in self.layers:
                    param_grads += layer.param_grads
                self.optimizer.add_param_grads(param_grads)

                # update parameters
                self.optimizer.update_params()

                # got loss and predict
                train_losses.append(self.loss.forward(y_pred, y_batch))
                train_predicts.extend(y_pred)
                train_targets.extend(y_batch)

            # output train status
            runout = "iter %d, train-[loss %.4f, acc %.4f]; " % (
                iter_idx, float(np.mean(train_losses)), float(self.accuracy(train_predicts, train_targets)))

            # runout = "iter %d, train-[loss %.4f, ]; " % (
            #     iter_idx, float(np.mean(train_losses)))

            if valid_X is not None and valid_Y is not None:
                # valid
                valid_losses, valid_predicts, valid_targets = [], [], []
                for b in range(valid_X.shape[0] // batch_size):
                    batch_begin = b * batch_size
                    batch_end = batch_begin + batch_size
                    x_batch = valid_X[batch_begin:batch_end]
                    y_batch = valid_Y[batch_begin:batch_end]

                    # forward propagation
                    y_pred = self.predict(x_batch)

                    # got loss and predict
                    valid_losses.append(self.loss.forward(y_pred, y_batch))
                    valid_predicts.extend(y_pred)
                    valid_targets.extend(y_batch)

                # output valid status
                runout += "valid-[loss %.4f, acc %.4f]; " % (
                    float(np.mean(valid_losses)), float(self.accuracy(valid_predicts, valid_targets)))

            print(runout, file=file)

    def predict(self, X):
        """ Calculate an output Y for the given input X. """
        x_next = X
        for layer in self.layers[:]:
            x_next = layer.forward(x_next)
        y_pred = x_next
        return y_pred

    def accuracy(self, outputs, targets):
        y_predicts = np.argmax(outputs, axis=1)
        y_targets = np.argmax(targets, axis=1)
        acc = y_predicts == y_targets
        return np.mean(acc)

        # acc = 0
        # for i in range(y_targets.shape[0]):
        #     if y_targets[i] == y_predicts[i]:
        #         acc += 1
        # return acc / y_targets.shape[0]

    def evaluate(self, X, Y):
        raise NotImplementedError()


