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
    """A recurrent neural network (RNN) is a class of artificial neural 
    network where connections between units form a directed cycle. 
    This creates an internal state of the network which allows it to 
    exhibit dynamic temporal behavior. Unlike feedforward neural networks, 
    RNNs can use their internal memory to process arbitrary sequences of 
    inputs. This makes them applicable to tasks such as unsegmented 
    connected handwriting recognition[1]_ or speech recognition.[2]_
    
    References
    ----------
    .. [1] A. Graves, M. Liwicki, S. Fernandez, R. Bertolami, H. Bunke, 
            J. Schmidhuber. A Novel Connectionist System for Improved 
            Unconstrained Handwriting Recognition. IEEE Transactions on 
            Pattern Analysis and Machine Intelligence, vol. 31, no. 5, 2009.
    .. [2] H. Sak and A. W. Senior and F. Beaufays. Long short-term memory 
            recurrent neural network architectures for large scale acoustic 
            modeling. Proc. Interspeech, pp338-342, Singapore, Sept. 2010
       
    """
    def __init__(self, n_out, n_in=None, init=GlorotUniform(), inner_init=Orthogonal(),
                 activation=Tanh(), return_sequence=False):
        self.n_out = n_out
        self.n_in = n_in
        self.init = init
        self.inner_init = inner_init
        self.activation_cls = activation.__class__
        self.activation = activation
        self.return_sequence = return_sequence

        self.out_shape = None
        self.last_input = None
        self.last_output = None

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
    
    .. math::
        
        o_t = tanh(U_t x_t + W_t o_{t-1} + b_t)
    
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
        self.activations = []

    def connect_to(self, prev_layer=None):
        n_in = super(SimpleRNN, self).connect_to(prev_layer)

        self.W = self.init((n_in, self.n_out))
        self.U = self.inner_init((self.n_out, self.n_out))
        self.b = zero((self.n_out,))

    def forward(self, input, *args, **kwargs):
        assert np.ndim(input) == 3, 'Only support batch training.'

        self.last_input = input
        nb_batch, nb_timestep, nb_in = input.shape
        output = zero((nb_batch, nb_timestep, self.n_out))

        if len(self.activations) == 0:
            self.activations = [self.activation_cls() for _ in range(nb_timestep)]

        output[:, 0, :] = self.activations[0].forward(np.dot(input[:, 0, :], self.W) + self.b)

        for i in range(1, nb_timestep):
            output[:, i, :] = self.activations[i].forward(
                np.dot(input[:, i, :], self.W) +
                np.dot(output[:, i - 1, :], self.U) + self.b)

        self.last_output = output
        if self.return_sequence:
            return self.last_output
        else:
            return self.last_output[:, -1, :]

    def backward(self, pre_grad, *args, **kwargs):
        zero = Zero()
        self.dW = zero(self.W.shape)
        self.dU = zero(self.U.shape)
        self.db = zero(self.b.shape)

        # hiddens.shape == (nb_timesteps, nb_batch, nb_out)
        hiddens = np.transpose(self.last_output, (1, 0, 2))
        if self.return_sequence:
            # check shape #
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
            nb_timesteps = self.last_output.shape[1]
            nb_batchs = self.last_output.shape[0]
            assert (nb_batchs, self.last_output.shape[2]) == pre_grad.shape
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
    
    .. math::
        
        & z_t = \sigmoid(U_z x_t + W_z h_{t-1} + b_z) \\
        & r_t = \sigmoid(U_r x_t + W_r h_{t-1} + b_r) \\
        & h_t = tanh(U_h x_t + W_h (s_{t-1} \times r_t) + b_h)
        & s_t = (1- z_t) \times h_t + z_t \times s_{t-1}
    
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
        self.gate_activation = gate_activation
        self.need_grad = need_grad

        self.U_r, self.U_z, self.U_h = None, None, None
        self.W_r, self.W_z, self.W_h = None, None, None
        self.b_r, self.b_z, self.b_h = None, None, None

        self.grad_U_r, self.grad_U_z, self.grad_U_h = None, None, None
        self.grad_W_r, self.grad_W_z, self.grad_W_h = None, None, None
        self.grad_b_r, self.grad_b_z, self.grad_b_h = None, None, None

    def connect_to(self, prev_layer=None):
        n_in = super(GRU, self).connect_to(prev_layer)

        # Weights matrices for input x
        self.U_r = self.init((n_in, self.n_out))
        self.U_z = self.init((n_in, self.n_out))
        self.U_h = self.init((n_in, self.n_out))

        # Weights matrices for memory cell
        self.W_r = self.inner_init((self.n_out, self.n_out))
        self.W_z = self.inner_init((self.n_out, self.n_out))
        self.W_h = self.inner_init((self.n_out, self.n_out))

        # Biases
        self.b_r = zero((self.n_out,))
        self.b_z = zero((self.n_out,))
        self.b_h = zero((self.n_out,))

    def forward(self, input, *args, **kwargs):
        assert np.ndim(input) == 3, 'Only support batch training.'

        # record
        self.last_input = input

        # dim
        nb_batch, nb_timesteps, nb_in = input.shape

        # outputs
        output = zero((nb_batch, nb_timesteps, self.n_out))

        # forward
        for i in range(nb_timesteps):
            # data
            s_pre = zero((nb_batch, self.n_out)) if i == 0 else output[:, i - 1, :]
            x_now = input[:, i, :]

            # computation
            z_now = self.gate_activation.forward(np.dot(x_now, self.U_z) +
                                                 np.dot(s_pre, self.W_z) +
                                                 self.b_z)
            r_now = self.gate_activation.forward(np.dot(x_now, self.U_r) +
                                                 np.dot(s_pre, self.W_r) +
                                                 self.b_r)
            h_now = self.activation.forward(np.dot(x_now, self.U_h) +
                                            np.dot(s_pre * r_now, self.W_h) +
                                            self.b_h)
            output[:, i, :] = (1 - z_now) * h_now + z_now * s_pre

        # record
        self.last_output = output

        # return
        if self.return_sequence:
            return self.last_output
        else:
            return self.last_output[:, -1, :]

    def backward(self, pre_grad, *args, **kwargs):
        raise NotImplementedError

    @property
    def params(self):
        return self.U_r, self.U_z, self.U_h, \
               self.W_r, self.W_z, self.W_h, \
               self.b_r, self.b_z, self.b_h

    @property
    def grads(self):
        return self.grad_U_r, self.grad_U_z, self.grad_U_h, \
               self.grad_W_r, self.grad_W_z, self.grad_W_h, \
               self.grad_b_r, self.grad_b_z, self.grad_b_h


class LSTM(Recurrent):
    """Long short-term memory (LSTM) is a recurrent neural network (RNN) 
    architecture (an artificial neural network) proposed in 1997 by Sepp 
    Hochreiter and Jürgen Schmidhuber [1]_ and further improved in 2000 
    by Felix Gers et al.[2]_ Like most RNNs, a LSTM network is universal 
    in the sense that given enough network units it can compute anything 
    a conventional computer can compute, provided it has the proper weight 
    matrix, which may be viewed as its program. 
    
    .. math::
        
        & f_t = \sigmoid(U_f x_t + W_f h_{t-1} + b_f) \
        & i_t = \sigmoid(U_i x_t + W_i h_{t-1} + b_f) \
        & o_t = \sigmoid(U_o x_t + W_o h_{t-1} + b_h) \
        & g_t = tanh(U_g x_t + W_g h_{t-1} + b_g) \
        & c_t = f_t \times c_{t-1} + i_t \times g_t \
        & h_t = o_t * tanh(c_t)
        
    
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
        raise NotImplementedError

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
        raise NotImplementedError

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
