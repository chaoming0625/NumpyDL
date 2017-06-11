# -*- coding: utf-8 -*-


import numpy as np

from .base import Layer
from ..activations import Sigmoid
from ..activations import Tanh
from ..initializations import GlorotUniform
from ..initializations import Orthogonal
from ..initializations import Zero

zero = Zero()


class Recurrent(Layer):
    def __init__(self, n_out, n_in=None, init=GlorotUniform(), inner_init=Orthogonal(),
                 activation=Tanh(), return_sequence=False):
        self.n_out = n_out
        self.n_in = n_in
        self.init = init
        self.inner_init = inner_init
        self.activation_cls = activation.__class__
        self.activations = []
        self.return_sequence = return_sequence

        self.out_shape = None
        self.last_input = None

    def connect_to(self, prev_layer=None):
        if prev_layer is not None:
            assert len(prev_layer.out_shape) == 3
            n_in = prev_layer.out_shape[-1]
        else:
            assert self.n_in is not None
            n_in = self.n_in

        if self.return_sequence:
            self.out_shape = (None, None, self.n_out)
        else:
            self.out_shape = (None, self.n_out)

        return n_in


class SimpleRNN(Recurrent):
    """Fully-connected RNN where the output is to be fed back to input.
    
    Parameters
    ----------
    output_dim: dimension of the internal projections and the final output.
    init: weight initialization function.
        Can be the name of an existing function (str),
        or a Theano function (see: [initializations](../initializations.md)).
    inner_init: initialization function of the inner cells.
    activation: activation function.
        Can be the name of an existing function (str),
        or a Theano function (see: [activations](../activations.md)).
    return_sequence: if `return_sequences`, 3D `numpy.array` with shape
            `(batch_size, timesteps, units)` will be returned. Else, return 
            2D `numpy.array` with shape `(batch_size, units)`.

    References
    ----------
    .. [1] A Theoretically Grounded Application of Dropout in Recurrent 
           Neural Networks. http://arxiv.org/abs/1512.05287
    """

    def __init__(self, **kwargs):
        super(SimpleRNN, self).__init__(**kwargs)

        self.W, self.dW = None, None
        self.U, self.dU = None, None
        self.b, self.db = None, None
        self.last_outputs = None

    def connect_to(self, prev_layer=None):
        n_in = super(SimpleRNN, self).connect_to(prev_layer)

        self.W = self.init((n_in, self.n_out))
        self.U = self.inner_init((self.n_out, self.n_out))
        self.b = zero((self.n_out,))

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
            # check shape
            assert hiddens.shape == pre_grad.shape
            nb_timesteps = pre_grad.shape[0]
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


class GRU(Recurrent):
    """Gated recurrent units (GRUs) are a gating mechanism in recurrent neural 
    networks, introduced in 2014. Their performance on polyphonic music modeling 
    and speech signal modeling was found to be similar to that of long short-term 
    memory.[1]_ They have fewer parameters than LSTM, as they lack an output 
    gate.[2]_
    
    Parameters
    ----------
    gate_activation : npdl.activations.Activation
        Gate activation.
    need_grad ： bool
        If `True`, will calculate gradients.
    
    References
    ----------
    ..[1] Chung, Junyoung; Gulcehre, Caglar; Cho, KyungHyun; Bengio, Yoshua 
          (2014). "Empirical Evaluation of Gated Recurrent Neural Networks on 
          Sequence Modeling". arXiv:1412.3555 Freely accessible [cs.NE].
    ..[2] "Recurrent Neural Network Tutorial, Part 4 – Implementing a GRU/LSTM 
          RNN with Python and Theano – WildML". Wildml.com. Retrieved 
          May 18, 2016.
    """
    def __init__(self, gate_activation=Sigmoid(), need_grad=True, **kwargs):
        super(GRU, self).__init__(**kwargs)

        self.gate_activation_cls = gate_activation.__class__
        self.need_grad = need_grad

        # parameters
        self.U, self.W, self.b = None, None, None

        # gradients
        self.grad_U, self.grad_W, self.grad_b = None, None, None

        # cell state
        self.c = None

        self.block_list = []

    def connect_to(self, prev_layer=None):
        n_in = super(GRU, self).connect_to(prev_layer)

        # Weights matrices for input x
        self.U = zero((3, n_in, self.n_out))
        self.U[0] = self.init((n_in, self.n_out))
        self.U[1] = self.init((n_in, self.n_out))
        self.U[2] = self.init((n_in, self.n_out))

        # Weights matrices for memory cell
        self.W = zero((3, self.n_out, self.n_out))
        self.W_r = self.inner_init((self.n_out, self.n_out))
        self.W_z = self.inner_init((self.n_out, self.n_out))
        self.W_h = self.inner_init((self.n_out, self.n_out))

        # Biases
        self.b = zero((3, self.n_out))
        self.b_r = zero((self.n_out,))
        self.b_z = zero((self.n_out,))
        self.b_h = zero((self.n_out,))

        # cell state
        self.c = zero((self.n_out, ))

    def forward(self, input, *args, **kwargs):
        # The total number of time steps
        T = len(x)
        z = np.zeros((T + 1, self.hidden_dim))
        r = np.zeros((T + 1, self.hidden_dim))
        h = np.zeros((T + 1, self.hidden_dim))
        s = np.zeros((T + 1, self.hidden_dim))

        o = np.zeros((T, self.word_dim))

        for t in range(T):
            z[t] = sigmoid(self.U[0, :, x[t]] + self.W[0].dot(s[t - 1]) + self.b[2])
            r[t] = sigmoid(self.U[1, :, x[t]] + self.W[1].dot(s[t - 1]) + self.b[1])
            h[t] = np.tanh(self.U[2, :, x[t]] + self.W[2].dot(s[t - 1] * r[t]) + self.b[0])
            s[t] = (1 - z[t]) * h[t] + z[t] * s[t - 1]
            o[t] = softmax(self.V.dot(h[t]) + self.c)
        return [z, r, h, s, o]

    def back_propagation(self, X, Y):
        T = len(Y)

        for t in np.arange(T)[::-1]:
            block = self.block_list[t]
            delta_y = block.o - Y[t]

            self.grad_V = np.outer(delta_y, block.s.T)



class LSTM(Recurrent):
    """Long short-term memory (LSTM) is a recurrent neural network (RNN) 
    architecture (an artificial neural network) proposed in 1997 by Sepp 
    Hochreiter and Jürgen Schmidhuber [1]_ and further improved in 2000 
    by Felix Gers et al.[2]_ Like most RNNs, a LSTM network is universal 
    in the sense that given enough network units it can compute anything 
    a conventional computer can compute, provided it has the proper weight 
    matrix, which may be viewed as its program. 
    
    Parameters
    ----------
    gate_activation : npdl.activations.Activation
        Gate activation.
    need_grad ： bool
        If `True`, will calculate gradients.
    
    References
    ----------
    ..[1] Sepp Hochreiter; Jürgen Schmidhuber (1997). "Long short-term 
          memory". Neural Computation. 9 (8): 1735–1780. doi:10.1162/ne
          co.1997.9.8.1735. PMID 9377276.
    ..[2] Felix A. Gers; Jürgen Schmidhuber; Fred Cummins (2000). "Learning 
          to Forget: Continual Prediction with LSTM". Neural Computation. 12 
          (10): 2451–2471. doi:10.1162/089976600300015015.
    """

    def __init__(self, gate_activation=Sigmoid(), need_grad=True, **kwargs):
        super(LSTM, self).__init__(**kwargs)

        self.gate_activation_cls = gate_activation.__class__
        self.need_grad = need_grad

        self.U_g, self.U_i, self.U_f, self.U_o = None, None, None, None
        self.W_g, self.W_i, self.W_f, self.W_o = None, None, None, None
        self.b_g, self.b_i, self.b_f, self.b_o = None, None, None, None

        self.grad_U_g, self.grad_U_i, self.grad_U_f, self.grad_U_o = None, None, None, None
        self.grad_W_g, self.grad_W_i, self.grad_W_f, self.grad_W_o = None, None, None, None
        self.grad_b_g, self.grad_b_i, self.grad_b_f, self.grad_b_o = None, None, None, None

        self.block_list = []

    def connect_to(self, prev_layer=None):
        n_in = super(LSTM, self).connect_to(prev_layer)

        # Weights matrices for input x
        self.U_g = self.init((n_in, self.n_out))
        self.U_i = self.init((n_in, self.n_out))
        self.U_f = self.init((n_in, self.n_out))
        self.U_o = self.init((n_in, self.n_out))

        # Weights matrices for memory cell
        self.W_g = self.inner_init((self.n_out, self.n_out))
        self.W_i = self.inner_init((self.n_out, self.n_out))
        self.W_f = self.inner_init((self.n_out, self.n_out))
        self.W_o = self.inner_init((self.n_out, self.n_out))

        # Biases
        self.b_g = zero((self.n_out,))
        self.b_i = zero((self.n_out,))
        self.b_f = zero((self.n_out,))
        self.b_o = zero((self.n_out,))

    def forward(self, input, *args, **kwargs):
        # reset
        self.block_list = []

        # record
        self.last_input = input

        # forward
        self.block_list.append(LSTMBlock(x=input[0],
                                         lstm_net=self,
                                         activation=self.activation_cls,
                                         gate_activation=self.gate_activation_cls))
        for x in input[1:]:
            self.block_list.append(LSTMBlock(x=x,
                                             lstm_net=self,
                                             activation=self.activation_cls(),
                                             gate_activation=self.gate_activation_cls(),
                                             s_old=self.block_list[-1].s,
                                             c_old=self.block_list[-1].c))

    def backward(self, pre_grad, *args, **kwargs):
        # reset
        self.grad_W_g = zero(self.W_g.shape)
        self.grad_W_i = zero(self.W_i.shape)
        self.grad_W_f = zero(self.W_f.shape)
        self.grad_W_o = zero(self.W_o.shape)

        self.grad_U_g = zero(self.U_g.shape)
        self.grad_U_i = zero(self.U_i.shape)
        self.grad_U_f = zero(self.U_f.shape)
        self.grad_U_o = zero(self.U_o.shape)

        self.grad_b_g = zero(self.b_g.shape)
        self.grad_b_i = zero(self.b_i.shape)
        self.grad_b_f = zero(self.b_f.shape)
        self.grad_b_o = zero(self.b_o.shape)

        # backward
        x = self.last_input
        T = len(pre_grad)
        dc_next = zero((self.n_out,))

        for t in np.arange(T)[::-1]:
            block = self.block_list[t]
            block.grad_s = pre_grad[t]

            if t != T - 1:
                next_block = self.block_list[t + 1]
                dc_next = next_block.grad_c * next_block.f
                ds = np.dot(self.W_i.T, tmp_i)
                ds += np.dot(self.W_f.T, tmp_f)
                ds += np.dot(self.W_o.T, tmp_o)
                ds += np.dot(self.W_g.T, tmp_g)
                block.grad_s += ds

            block.grad_c = block.o * block.grad_s + dc_next
            block.grad_o = block.grad_s * block.c
            block.grad_i = block.grad_c * block.g
            block.grad_f = block.grad_c * block.c_old
            block.grad_g = block.grad_s * block.i

            tmp_i = block.activation_i.derivative()
            tmp_f = block.activation_f.derivative()
            tmp_o = block.activation_o.derivative()
            tmp_g = block.activation_g.derivative()

            self.grad_U_i += np.outer(tmp_i, x[t])
            self.grad_U_f += np.outer(tmp_f, x[t])
            self.grad_U_g += np.outer(tmp_g, x[t])
            self.grad_U_o += np.outer(tmp_o, x[t])

            self.grad_b_g = tmp_g
            self.grad_b_i = tmp_i
            self.grad_b_f = tmp_f
            self.grad_b_o = tmp_o

            self.grad_W_i += np.outer(tmp_i, block.s_old)
            self.grad_W_f += np.outer(tmp_f, block.s_old)
            self.grad_W_g += np.outer(tmp_g, block.s_old)
            self.grad_W_o += np.outer(tmp_o, block.s_old)

    @property
    def params(self):
        return self.U_g, self.U_i, self.U_f, self.U_o, \
               self.W_g, self.W_i, self.W_f, self.W_o, \
               self.b_g, self.b_i, self.b_f, self.b_o

    @property
    def grads(self):
        return self.grad_U_g, self.grad_U_i, self.grad_U_f, self.grad_U_o, \
               self.grad_W_g, self.grad_W_i, self.grad_W_f, self.grad_W_o, \
               self.grad_b_g, self.grad_b_i, self.grad_b_f, self.grad_b_o


class LSTMBlock:
    def __init__(self, x, lstm_net, activation, gate_activation, s_old=None, c_old=None):
        # init parameters
        self.x = x
        self.lstm_net = lstm_net
        self.activation_g = activation()
        self.activation_i = gate_activation()
        self.activation_f = gate_activation()
        self.activation_o = gate_activation()

        # old params
        if s_old is None:
            s_old = zero((self.lstm_net.n_out,))
        if c_old is None:
            c_old = zero((self.lstm_net.n_out,))
        self.s_old = s_old
        self.c_old = c_old

        # gradients
        self.grad_s = None
        self.grad_c = None
        self.grad_o = None
        self.grad_i = None
        self.grad_f = None

        # forward propagation
        self.i = self.activation_i(np.dot(self.lstm_net.U_i, self.x) +
                                   np.dot(self.lstm_net.W_i, s_old) +
                                   self.lstm_net.b_i)
        self.o = self.activation_o(np.dot(self.lstm_net.U_o, self.x) +
                                   np.dot(self.lstm_net.W_o, s_old) +
                                   self.lstm_net.b_o)
        self.f = self.activation_f(np.dot(self.lstm_net.U_f, self.x) +
                                   np.dot(self.lstm_net.W_f, s_old) +
                                   self.lstm_net.b_f)
        self.g = self.activation_g(np.dot(self.lstm_net.U_g, self.x) +
                                   np.dot(self.lstm_net.W_g, s_old) +
                                   self.lstm_net.b_g)
        self.c = self.c_old * self.f + self.g * self.i
        self.s = self.c * self.o

