# -*- coding: utf-8 -*-

"""
Non-linear activation functions for artificial neurons.

A function used to transform the activation level of a unit (neuron) into an 
output signal. Typically, activation functions have a "squashing" effect. 
Together with the PSP function (which is applied first) this defines the 
unit type. Neural Networks supports a wide range of activation functions.
"""

import copy

import numpy as np

from npdl.utils.random import get_dtype


# activation-start


class Activation(object):
    """Base class for activations.
    
    """
    def __init__(self):
        self.last_forward = None

    def forward(self, input):
        """Forward Step.
        
        Parameters
        ----------
        input : numpy.array
            the input matrix.
        
        """
        raise NotImplementedError

    def derivative(self, input=None):
        """Backward step.
        
        Parameters
        ----------
        input : numpy.array, optional.
            If provide `input`, this function will not use `last_forward`.
            
        """
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


# activation-end
# sigmoid-start


class Sigmoid(Activation):
    """Sigmoid activation function.
    """

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, input, *args, **kwargs):
        """A sigmoid function is a mathematical function having a 
        characteristic "S"-shaped curve or sigmoid curve. Often, 
        sigmoid function refers to the special case of the logistic 
        function and defined by the formula :math:`\\varphi(x) = \\frac{1}{1 + e^{-x}}`
        (given the input :math:`x`).
        
        Parameters
        ----------
        input : float32
            The activation (the summed, weighted input of a neuron).
            
        Returns
        -------
        float32 in [0, 1]
            The output of the sigmoid function applied to the activation.
        """

        self.last_forward = 1.0 / (1.0 + np.exp(-input))
        return self.last_forward

    def derivative(self, input=None):
        """The derivative of sigmoid is
        
        .. math:: \\frac{dy}{dx} & = (1-\\varphi(x)) \\otimes \\varphi(x)  \\\\
                  & = \\frac{e^{-x}}{(1+e^{-x})^2} \\\\
                  & = \\frac{e^x}{(1+e^x)^2}
        
        Returns
        -------
        float32
            The derivative of sigmoid function.
        """
        last_forward = self.forward(input) if input else self.last_forward
        return np.multiply(last_forward, 1 - last_forward)


# sigmoid-end
# tanh-start


class Tanh(Activation):
    """Tanh activation function.
    
    The hyperbolic tangent function is an old mathematical function. 
    It was first used in the work by L'Abbe Sauri (1774).
    """

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, input):
        """This function is easily defined as the ratio between the hyperbolic 
        sine and the cosine functions (or expanded, as the ratio of the 
        half‐difference and half‐sum of two exponential functions in the 
        points :math:`z` and :math:`-z`):
        
        .. math:: tanh(z) & = \\frac{sinh(z)}{cosh(z)} \\\\
                  & = \\frac{e^z - e^{-z}}{e^z + e^{-z}}
        
        Fortunately, numpy provides :meth:`tanh` methods. So in our implementation,
        we directly use :math:`\\varphi(x) = \\tanh(x)`.
        
        Parameters
        ----------
        x : float32
            The activation (the summed, weighted input of a neuron).
    
        Returns
        -------
        float32 in [-1, 1]
            The output of the tanh function applied to the activation.
        """
        self.last_forward = np.tanh(input)
        return self.last_forward

    def derivative(self, input=None):
        """The derivative of :meth:`tanh` functions is
        
        .. math:: \\frac{d}{dx} tanh(x) & = \\frac{d}{dx} \\frac{sinh(x)}{cosh(x)} \\\\
                  & = \\frac{cosh(x) \\frac{d}{dx}sinh(x) - sinh(x) \\frac{d}{dx}cosh(x) }{ cosh^2(x)} \\\\
                  & = \\frac{ cosh(x) cosh(x) - sinh(x) sinh(x) }{ cosh^2(x)}  \\\\
                  & = 1 - tanh^2(x) 
        
        Returns
        -------
        float32 
            The derivative of tanh function.
        """
        last_forward = self.forward(input) if input else self.last_forward
        return 1 - np.power(last_forward, 2)


# tanh-end
# relu-start


class ReLU(Activation):
    """Rectify activation function.
    
    Two additional major benefits of ReLUs are sparsity and a reduced 
    likelihood of vanishing gradient. But first recall the definition 
    of a ReLU is :math:`h=max(0,a)` where :math:`a=Wx+b`.

    One major benefit is the reduced likelihood of the gradient to vanish. 
    This arises when :math:`a>0`. In this regime the gradient has a constant value. 
    In contrast, the gradient of sigmoids becomes increasingly small as the 
    absolute value of :math:`x` increases. The constant gradient of ReLUs results in 
    faster learning.
    
    The other benefit of ReLUs is sparsity. Sparsity arises when :math:`a≤0`. 
    The more such units that exist in a layer the more sparse the resulting 
    representation. Sigmoids on the other hand are always likely to generate 
    some non-zero value resulting in dense representations. Sparse representations 
    seem to be more beneficial than dense representations.
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, input):
        """During the forward pass, it inhibits all inhibitions below some 
        threshold :math:`ϵ`, typically :math:`0`. In other words, it computes point-wise 
        
        .. math:: y=max(0,x) 
        
        Parameters
        ----------
        x : float32
            The activation (the summed, weighted input of a neuron).
    
        Returns
        -------
        float32
            The output of the rectify function applied to the activation.
        """
        self.last_forward = input
        return np.maximum(0.0, input)

    def derivative(self, input=None):
        """The point-wise derivative for ReLU is :math:`\\frac{dy}{dx} = 1`, if
        :math:`x>0`, or :math:`\\frac{dy}{dx} = 0`, if :math:`x<=0`.
        
        Returns
        -------
        float32 
            The derivative of ReLU function.
        """
        last_forward = input if input else self.last_forward
        res = np.zeros(last_forward.shape, dtype=get_dtype())
        res[last_forward > 0] = 1.
        return res


# relu-end
# linear-start


class Linear(Activation):
    """Linear activation function.
    """

    def __init__(self):
        super(Linear, self).__init__()

    def forward(self, input):
        """It's also known as identity activation funtion. The 
        forward step is :math:`\\varphi(x) = x`
        
        Parameters
        ----------
        x : float32
            The activation (the summed, weighted input of a neuron).
    
        Returns
        -------
        float32
            The output of the identity applied to the activation.
        """
        self.last_forward = input
        return input

    def derivative(self, input=None):
        """Backward propagation.
        The backward also return identity matrix.

        Returns
        -------
        float32 
            The derivative of linear function.
        """
        last_forward = input if input else self.last_forward
        return np.ones(last_forward.shape, dtype=get_dtype())


# linear-end
# softmax-start


class Softmax(Activation):
    """Softmax activation function.
    """

    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, input):
        """:math:`\\varphi(\\mathbf{x})_j =
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
        assert np.ndim(input) == 2
        self.last_forward = input
        x = input - np.max(input, axis=1, keepdims=True)
        exp_x = np.exp(x)
        s = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return s

    def derivative(self, input=None):
        """Backward propagation.

        Returns
        -------
        float32 
            The derivative of Softmax function.
        """
        last_forward = input if input else self.last_forward
        return np.ones(last_forward.shape, dtype=get_dtype())


# softmax-end
# elliot-start


class Elliot(Activation):
    """ A fast approximation of sigmoid. 
    
    The function was first introduced 
    in 1993 by D.L. Elliot under the title A Better Activation Function for
    Artificial Neural Networks. The function closely approximates the 
    Sigmoid or Hyperbolic Tangent functions for small values, however it 
    takes longer to converge for large values (i.e. It doesn't go to 1 or 
    0 as fast), though this isn't particularly a problem if you're using 
    it for classification.
    
    """

    def __init__(self, steepness=1):
        super(Elliot, self).__init__()

        self.steepness = steepness

    def forward(self, input):
        """Forward propagation.

        Parameters
        ----------
        x : float32
            The activation (the summed, weighted input of a neuron).

        Returns
        -------
        float32
            The output of the softplus function applied to the activation.
        """
        self.last_forward = 1 + np.abs(input * self.steepness)
        return 0.5 * self.steepness * input / self.last_forward + 0.5

    def derivative(self, input=None):
        """Backward propagation.

        Returns
        -------
        float32 
            The derivative of Elliot function. 
        """
        last_forward = 1 + np.abs(input * self.steepness) if input else self.last_forward
        return 0.5 * self.steepness / np.power(last_forward, 2)


# elliot-end
# symmetric-elliot-start


class SymmetricElliot(Activation):
    """Elliot symmetric sigmoid transfer function.
    
    """

    def __init__(self, steepness=1):
        super(SymmetricElliot, self).__init__()
        self.steepness = steepness

    def forward(self, input):
        """Forward propagation.

        Parameters
        ----------
        x : float32
            The activation (the summed, weighted input of a neuron).

        Returns
        -------
        float32
            The output of the softplus function applied to the activation.
        """
        self.last_forward = 1 + np.abs(input * self.steepness)
        return input * self.steepness / self.last_forward

    def derivative(self, input=None):
        """Backward propagation.

        Returns
        -------
        float32 
            The derivative of SymmetricElliot function.
        """
        last_forward = 1 + np.abs(input * self.steepness) if input else self.last_forward
        return self.steepness / np.power(last_forward, 2)


# symmetric-elliot-end
# softplus-start


class SoftPlus(Activation):
    """Softplus activation function.
    """

    def __init__(self):
        super(SoftPlus, self).__init__()

    def forward(self, input):
        """:math:`\\varphi(x) = \\log(1 + e^x)`
        
        Parameters
        ----------
        x : float32
            The activation (the summed, weighted input of a neuron).
    
        Returns
        -------
        float32
            The output of the softplus function applied to the activation.
        """
        self.last_forward = np.exp(input)
        return np.log(1 + self.last_forward)

    def derivative(self, input=None):
        """Backward propagation.

        Returns
        -------
        float32 
            The derivative of Softplus function.
        """
        last_forward = np.exp(input) if input else self.last_forward
        return last_forward / (1 + last_forward)


# softplus-end
# softsign-start


class SoftSign(Activation):
    """SoftSign activation function.
    
    """

    def __init__(self):
        super(SoftSign, self).__init__()

    def forward(self, input):
        """Forward propagation.

        Parameters
        ----------
        x : float32
            The activation (the summed, weighted input of a neuron).

        Returns
        -------
        float32
            The output of the softplus function applied to the activation.
        """
        self.last_forward = np.abs(input) + 1
        return input / self.last_forward

    def derivative(self, input=None):
        """Backward propagation.

        Returns
        -------
        float32 
            The derivative of SoftSign function.
        """
        last_forward = np.abs(input) + 1 if input else self.last_forward
        return 1. / np.power(last_forward, 2)


# softsign-end


def get(activation):
    if activation.__class__.__name__ == 'str':
        if activation in ['sigmoid', 'Sigmoid']:
            return Sigmoid()
        if activation in ['tan', 'tanh', 'Tanh']:
            return Tanh()
        if activation in ['relu', 'ReLU', 'RELU']:
            return ReLU()
        if activation in ['linear', 'Linear']:
            return Linear()
        if activation in ['softmax', 'Softmax']:
            return Softmax()
        if activation in ['elliot', 'Elliot']:
            return Elliot()
        if activation in ['symmetric_elliot', 'SymmetricElliot']:
            return SymmetricElliot()
        if activation in ['SoftPlus', 'soft_plus', 'softplus']:
            return SoftPlus()
        if activation in ['SoftSign', 'softsign', 'soft_sign']:
            return SoftSign()
        raise ValueError('Unknown activation name: {}.'.format(activation))

    elif isinstance(activation, Activation):
        return copy.deepcopy(activation)

    else:
        raise ValueError("Unknown type: {}.".format(activation.__class__.__name__))
