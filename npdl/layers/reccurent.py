# -*- coding: utf-8 -*-


import numpy as np

from .base import Layer
from ..activation import Tanh
from ..initialization import GlorotUniform
from ..initialization import Orthogonal
from ..initialization import Zero


class SimpleRNN(Layer):
    """Fully-connected RNN where the output is to be fed back to input.(完全连接的RNN在输出将被反馈到输入。)

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    # References
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """

    def __init__(self, n_out, n_in=None, init=GlorotUniform(), inner_init=Orthogonal(), activation=Tanh(), return_sequence=False):
        self.n_out = n_out
        self.n_in = n_in
        self.init = init
        self.inner_init = inner_init
        self.activation_cls = activation.__class__
        self.activations = []
        self.return_sequence = return_sequence

        self.W, self.dW = None, None
        self.U, self.dU = None, None
        self.b, self.db = None, None
        self.last_outputs = None
        self.last_input = None
        self.out_shape = None

    def connect_to(self, prev_layer=None):
        if prev_layer is not None:
            assert len(prev_layer.out_shape) == 3
            n_in = prev_layer.out_shape[-1]
        else:
            assert self.n_in is not None
            n_in = self.n_in

        self.W = self.init((n_in, self.n_out))
        self.U = self.inner_init((self.n_out, self.n_out))
        self.b = Zero()((self.n_out,))

        if self.return_sequence:
            self.out_shape = (None, None, self.n_out)
        else:
            self.out_shape = (None, self.n_out)

    def forward(self, input, *args, **kwargs):
        assert np.ndim(input) == 3, 'Only support batch training.'

        self.last_input = input
        nb_batch, nb_timestep, nb_in = input.shape
        outputs = Zero()((nb_batch, nb_timestep, self.n_out))

        if len(self.activations) == 0:
            self.activations = [self.activation_cls() for _ in range(nb_timestep)]

        outputs[:, 0, :] = self.activations[0].forward(np.dot(input[:, 0, :], self.W) + self.b)

        for i in range(1, nb_timestep):
            outputs[:, i, :] = self.activations[i].forward(
                np.dot(input[:, i, :], self.W) +
                np.dot(outputs[:, i - 1, :], self.U) + self.b)

        self.last_outputs = outputs
        if self.return_sequence:
            return self.last_outputs
        else:
            return self.last_outputs[:, -1, :]

    def backward(self, pre_grad, *args, **kwargs):
        zero = Zero()
        self.dW = zero(self.W.shape)
        self.dU = zero(self.U.shape)
        self.db = zero(self.b.shape)

        # hiddens.shape == (nb_timesteps, nb_batch, nb_out)
        hiddens = np.transpose(self.last_outputs, (1, 0, 2))
        if self.return_sequence:
            # check shape #
            # self.outputs.shape == (nb_batch, nb_timesteps, nb_out)
            assert hiddens.shape == pre_grad.shape
            nb_timesteps = pre_grad.shape[0]
            if not self.first_layer:
                layer_grad = Zero()(pre_grad.shape)

            for timestep1 in np.arange(nb_timesteps)[::-1]:
                delta = pre_grad[timestep1] * self.activations[timestep1].derivative()
                for timestep2 in np.arange(timestep1)[::-1]:
                    self.dU += np.dot(hiddens[timestep2].T, delta)
                    self.dW += np.dot(self.last_input[:, timestep2 + 1, :].T, delta)
                    self.db += np.mean(delta, axis=0)
                    if not self.first_layer:
                        layer_grad[timestep2 + 1] += np.dot(delta, self.W.T)
                    delta = np.dot(delta, self.U.T)

                if timestep1 == 0 or timestep2 == 0:
                    self.dW += np.dot(self.last_input[:, 0, :].T, delta)
                    self.db += np.mean(delta, axis=0)
                    if not self.first_layer:
                        layer_grad[0] += np.dot(delta, self.W.T)

        else:
            nb_timesteps = self.last_outputs.shape[1]
            nb_batchs = self.last_outputs.shape[0]
            assert (nb_batchs, self.last_outputs.shape[2]) == pre_grad.shape
            if not self.first_layer:
                layer_grad = Zero()(hiddens.shape)

            delta = pre_grad * self.activations[nb_timesteps - 1].derivative()
            for timestep2 in np.arange(nb_timesteps - 1)[::-1]:
                self.dU += np.dot(hiddens[timestep2].T, delta)
                self.dW += np.dot(self.last_input[:, timestep2 + 1, :].T, delta)
                self.db += np.mean(delta, axis=0)
                if not self.first_layer:
                    layer_grad[timestep2 + 1] += np.dot(delta, self.W.T)
                delta = np.dot(delta, self.U.T)

            if timestep2 == 0:
                self.dW += np.dot(self.last_input[:, timestep2 + 1, :].T, delta)
                self.db += np.mean(delta, axis=0)
                if not self.first_layer:
                    layer_grad[0] += np.dot(delta, self.W.T)

        if not self.first_layer:
            return layer_grad

    @property
    def params(self):
        return self.W, self.U, self.b

    @property
    def grads(self):
        return self.dW, self.dU, self.db


class GRU(Layer):
    def __init__(self):
        pass


class LSTM(Layer):
    def __init__(self):
        pass
