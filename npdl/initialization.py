"""
Functions to create initializers for parameter variables.

Examples
--------
>>> from npdl.layers import Dense
>>> from npdl.initialization import GlorotUniform
>>> l1 = Dense(n_out=300, n_in=100, init=GlorotUniform())
"""
# -*- coding: utf-8 -*-

import numpy as np

from npdl.utils.random import get_rng
from npdl.utils.random import get_dtype
from npdl.utils.generic import get_from_module


class Initializer(object):
    """Base class for parameter tensor initializers.

    The :class:`Initializer` class represents a weight initializer used
    to initialize weight parameters in a neural network layer. It should be
    subclassed when implementing new types of weight initializers.

    """
    def __call__(self, size):
        """
        Makes :class:`Initializer` instances callable like a function, invoking
        their :meth:`call()` method.
        """
        return self.call(size)

    def call(self, size):
        """
        Sample should return a theano.tensor of size shape and data type
        theano.config.floatX.

        Parameters
        -----------
        size : tuple or int
            Integer or tuple specifying the size of the returned
            matrix.
        returns : theano.tensor
            Matrix of size shape and dtype theano.config.floatX.
        """
        raise NotImplementedError()


class Zero(Initializer):
    """Initialize weights with zero value.

    """
    def call(self, size):
        return np.zeros(size, dtype=get_dtype())


class One(Initializer):
    """Initialize weights with one value.

        """
    def call(self, size):
        return np.ones(size, dtype=get_dtype())


class Uniform(Initializer):
    """Sample initial weights from the uniform distribution.

    Parameters are sampled from U(a, b).

    Parameters
    ----------
    scale : float or tuple
        When std is None then range determines a, b. If range is a float the
        weights are sampled from U(-range, range). If range is a tuple the
        weights are sampled from U(range[0], range[1]).
    """
    def __init__(self, scale=0.05):
        self.scale = scale

    def call(self, size):
        return get_rng().uniform(-self.scale, self.scale, size=size).astype(get_dtype())


class Normal(Initializer):
    """Sample initial weights from the Gaussian distribution.

    Initial weight parameters are sampled from N(mean, std).

    Parameters
    ----------
    std : float
        Std of initial parameters.
    mean : float
        Mean of initial parameters.
    """
    def __init__(self, std=0.01, mean=0.0):
        self.std = std
        self.mean = mean

    def call(self, size):
        return get_rng().normal(loc=self.mean, scale=self.std, size=size)(size).astype(get_dtype())


class LecunUniform(Initializer):
    def call(self, size):
        fan_in, fan_out = _decompose_size(size)
        return Uniform(np.sqrt(3. / fan_in))(size)


class GlorotUniform(Initializer):
    """Glorot with weights sampled from the Normal distribution.

    See :class:`Glorot` for a description of the parameters.
    """
    def call(self, size):
        fan_in, fan_out = _decompose_size(size)
        return Uniform(np.sqrt(6 / (fan_in + fan_out)))(size)


class GlorotNormal(Initializer):
    """Glorot with weights sampled from the Uniform distribution.

    See :class:`Glorot` for a description of the parameters.
    """
    def call(self, size):
        fan_in, fan_out = _decompose_size(size)
        return Normal(np.sqrt(2 / (fan_out + fan_in)))(size)


class HeNormal(Initializer):
    def call(self, size):
        fan_in, fan_out = _decompose_size(size)
        return Normal(np.sqrt(2. / fan_in))(size)


class HeUniform(Initializer):
    def call(self, size):
        fan_in, fan_out = _decompose_size(size)
        return Uniform(np.sqrt(6. / fan_in))(size)


class Orthogonal(Initializer):
    """Intialize weights as Orthogonal matrix.

    Orthogonal matrix initialization [1]_. For n-dimensional shapes where
    n > 2, the n-1 trailing axes are flattened. For convolutional layers, this
    corresponds to the fan-in, so this makes the initialization usable for
    both dense and convolutional layers.

    Parameters
    ----------
    gain : float or 'relu'
        Scaling factor for the weights. Set this to ``1.0`` for linear and
        sigmoid units, to 'relu' or ``sqrt(2)`` for rectified linear units, and
        to ``sqrt(2/(1+alpha**2))`` for leaky rectified linear units with
        leakiness ``alpha``. Other transfer functions may need different
        factors.

    References
    ----------
    .. [1] Saxe, Andrew M., James L. McClelland, and Surya Ganguli.
           "Exact solutions to the nonlinear dynamics of learning in deep
           linear neural networks." arXiv preprint arXiv:1312.6120 (2013).
    """
    def __init__(self, gain=1.0):
        if gain == 'relu':
            gain = np.sqrt(2)
        self.gain = gain

    def call(self, size):
        flat_shape = (size[0], np.prod(size[1:]))
        a = get_rng().normal(loc=0., scale=1., size=flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(size)
        q = self.gain * q
        return q.astype(get_dtype())


def _decompose_size(size):
    if len(size) == 2:
        fan_in = size[0]
        fan_out = size[1]

    elif len(size) == 4 or len(size) == 5:
        respective_field_size = np.prod(size[2:])
        fan_in = size[1] * respective_field_size
        fan_out = size[0] * respective_field_size

    else:
        fan_in = fan_out = int(np.sqrt(np.prod(size)))

    return fan_in, fan_out


_zero = Zero()
_one = One()


def get(identifier):
    get_from_module(identifier, globals(), 'initialization')

