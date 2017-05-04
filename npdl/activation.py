# -*- coding: utf-8 -*-

"""
Non-linear activation functions for artificial neurons.
"""

import numpy as np
from npdl.utils.random import get_dtype


class Activation(object):
    """Base class for activations.
    
    """
    def __init__(self):
        self.last_forward = None

    def forward(self, input):
        """Forward Step.
        
        :param input: 
        :return: 
        """
        raise NotImplementedError()

    def derivative(self):
        """Backward step.
        
        :return: 
        """
        raise NotImplementedError()


class Sigmoid(Activation):
    """Sigmoid activation function :math:`\\varphi(x) = \\frac{1}{1 + e^{-x}}`

    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).

    Returns
    -------
    float32 in [0, 1]
        The output of the sigmoid function applied to the activation.
    """

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, input, *args, **kwargs):
        self.last_forward = 1.0 / (1.0 + np.exp(-input))
        return self.last_forward

    def derivative(self, ):
        return np.multiply(self.last_forward, 1 - self.last_forward)


class Tanh(Activation):
    """Tanh activation function :math:`\\varphi(x) = \\tanh(x)`

    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).

    Returns
    -------
    float32 in [-1, 1]
        The output of the tanh function applied to the activation.
    """

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, input):
        self.last_forward = np.tanh(input)
        return self.last_forward

    def derivative(self):
        return 1 - np.power(self.last_forward, 2)


class ReLU(Activation):
    """Rectify activation function :math:`\\varphi(x) = \\max(0, x)`

    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).

    Returns
    -------
    float32
        The output of the rectify function applied to the activation.
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, input):
        self.last_forward = input
        return np.maximum(0.0, input)

    def derivative(self):
        res = np.zeros(self.last_forward.shape, dtype=get_dtype())
        res[self.last_forward > 0] = 1.
        return res


class Linear(Activation):
    """Linear activation function :math:`\\varphi(x) = x`

    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).

    Returns
    -------
    float32
        The output of the identity applied to the activation.
    """

    def __init__(self):
        super(Linear, self).__init__()

    def forward(self, input):
        self.last_forward = input
        return input

    def derivative(self):
        return np.ones(self.last_forward.shape, dtype=get_dtype())


class Softmax(Activation):
    """Softmax activation function
    :math:`\\varphi(\\mathbf{x})_j =
    \\frac{e^{\mathbf{x}_j}}{\sum_{k=1}^K e^{\mathbf{x}_k}}`
    where :math:`K` is the total number of neurons in the layer. This
    activation function gets applied row-wise.

    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).

    Returns
    -------
    float32 where the sum of the row is 1 and each single value is in [0, 1]
        The output of the softmax function applied to the activation.
    """

    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, input):
        assert np.ndim(input) == 2
        self.last_forward = input
        x = input - np.max(input, axis=1, keepdims=True)
        exp_x = np.exp(x)
        s = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return s

    def derivative(self):
        return np.ones(self.last_forward.shape, dtype=get_dtype())


class Elliot(Activation):
    """ A fast approximation of sigmoid """

    def __init__(self, steepness=1):
        super(Elliot, self).__init__()

        self.steepness = steepness

    def forward(self, input):
        self.last_forward = 1 + np.abs(input * self.steepness)
        return 0.5 * self.steepness * input / self.last_forward + 0.5

    def derivative(self):
        return 0.5 * self.steepness / np.power(self.last_forward, 2)


class SymmetricElliot(Activation):
    def __init__(self, steepness=1):
        super(SymmetricElliot, self).__init__()
        self.steepness = steepness

    def forward(self, input):
        self.last_forward = 1 + np.abs(input * self.steepness)
        return input * self.steepness / self.last_forward

    def derivative(self):
        return self.steepness / np.power(self.last_forward, 2)


class LReLU(Activation):
    def __init__(self, leakage=0.01):
        super(LReLU, self).__init__()
        self.leakage = leakage

    def forward(self, input):
        self.last_forward = input
        raise NotImplementedError()

    """
    def lrelu(x, derivative=False, leakage=0.01):
        if derivative:
            return np.clip(x > 0, leakage, 1.0)
        else:
            s = np.copy(x)
            s[s < 0] *= leakage
            return s
    """


class SoftPlus(Activation):
    """Softplus activation function :math:`\\varphi(x) = \\log(1 + e^x)`

    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).

    Returns
    -------
    float32
        The output of the softplus function applied to the activation.
    """

    def __init__(self):
        super(SoftPlus, self).__init__()

    def forward(self, input):
        self.last_forward = np.exp(input)
        return np.log(1 + self.last_forward)

    def derivative(self):
        return self.last_forward / (1 + self.last_forward)


class SoftSign(Activation):
    def __init__(self):
        super(SoftSign, self).__init__()

    def forward(self, input):
        self.last_forward = np.abs(input) + 1
        return input / self.last_forward

    def derivative(self):
        return 1. / np.power(self.last_forward, 2)
