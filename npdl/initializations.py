# -*- coding: utf-8 -*-
"""
Functions to create initializers for parameter variables.

Examples
--------
>>> from npdl.layers import Dense
>>> from npdl.initializations import GlorotUniform
>>> l1 = Dense(n_out=300, n_in=100, init=GlorotUniform())
"""

import copy
import numpy as np

from npdl.utils.random import get_rng
from npdl.utils.random import get_dtype


class Initializer(object):
    """Base class for parameter weight initializers.

    The :class:`Initializer` class represents a weight initializer used
    to initialize weight parameters in a neural network layer. It should be
    subclassed when implementing new types of weight initializers.
    """
    def __call__(self, size):
        """Makes :class:`Initializer` instances callable like a function, invoking
        their :meth:`call()` method.
        """
        return self.call(size)

    def call(self, size):
        """Sample should return a numpy.array of size shape and data type
        ``numpy.float32``.
        
        Parameters
        ----------
        size : tuple or int.
            Integer or tuple specifying the size of the returned
            matrix.
        
        Returns
        -------
        numpy.array. 
            Matrix of size shape and dtype ``numpy.float32``.
        """
        raise NotImplementedError()

    def __str__(self):
        return self.__class__.__name__


class Zero(Initializer):
    """Initialize weights with zero value.
    """
    def call(self, size):
        return _cast_dtype(np.zeros(size))


class One(Initializer):
    """Initialize weights with one value.
    """
    def call(self, size):
        return _cast_dtype(np.ones(size))


class Uniform(Initializer):
    """Sample initial weights from the uniform distribution.

    Parameters are sampled from U(a, b).
    
    Parameters
    ----------
    scale : float or tuple.
        When std is None then range determines a, b. If range is a float the
        weights are sampled from U(-range, range). If range is a tuple the
        weights are sampled from U(range[0], range[1]).
    """
    def __init__(self, scale=0.05):
        self.scale = scale

    def call(self, size):
        return _cast_dtype(get_rng().uniform(-self.scale, self.scale, size=size))


class Normal(Initializer):

    """Sample initial weights from the Gaussian distribution.

    Initial weight parameters are sampled from N(mean, std).
    
    Parameters
    ----------
    std : float. 
        Std of initial parameters.
    mean : float. 
        Mean of initial parameters.
    """
    def __init__(self, std=0.01, mean=0.0):
        self.std = std
        self.mean = mean

    def call(self, size):
        return _cast_dtype(get_rng().normal(loc=self.mean, scale=self.std, size=size))


class LecunUniform(Initializer):
    """LeCun uniform initializer.

    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(3 / fan_in)` [1]_
    where `fan_in` is the number of input units in the weight matrix.
    
    References
    ----------
    .. [1] LeCun 98, Efficient Backprop, http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    
    """
    def call(self, size):
        fan_in, fan_out = decompose_size(size)
        return Uniform(np.sqrt(3. / fan_in))(size)


class GlorotUniform(Initializer):
    """Glorot uniform initializer, also called Xavier uniform initializer.

    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / (fan_in + fan_out))` [1]_
    where `fan_in` is the number of input units in the weight matrix
    and `fan_out` is the number of output units in the weight matrix.
    
    References
    ----------
    .. [1] Glorot & Bengio, AISTATS 2010. http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    """
    def call(self, size):
        fan_in, fan_out = decompose_size(size)
        return Uniform(np.sqrt(6 / (fan_in + fan_out)))(size)


class GlorotNormal(Initializer):
    """Glorot normal initializer, also called Xavier normal initializer.

    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / (fan_in + fan_out))` [1]_
    where `fan_in` is the number of input units in the weight matrix
    and `fan_out` is the number of output units in the weight matrix.
    
    References
    ----------
    .. [1] Glorot & Bengio, AISTATS 2010. http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    """
    def call(self, size):
        fan_in, fan_out = decompose_size(size)
        return Normal(np.sqrt(2 / (fan_out + fan_in)))(size)


class HeNormal(Initializer):
    """He normal initializer.

    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / fan_in)` [1]_
    where `fan_in` is the number of input units in the weight matrix.
    
    References
    ----------
    .. [1] He et al., http://arxiv.org/abs/1502.01852
    """
    def call(self, size):
        fan_in, fan_out = decompose_size(size)
        return Normal(np.sqrt(2. / fan_in))(size)


class HeUniform(Initializer):
    """He uniform variance scaling initializer.

    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / fan_in)` [1]_
    where `fan_in` is the number of input units in the weight matrix.
    
    References
    ----------
    .. [1] He et al., http://arxiv.org/abs/1502.01852
    """
    def call(self, size):
        fan_in, fan_out = decompose_size(size)
        return Uniform(np.sqrt(6. / fan_in))(size)


class Orthogonal(Initializer):
    """Intialize weights as Orthogonal matrix.

    Orthogonal matrix initialization [1]_. For n-dimensional shapes where
    n > 2, the n-1 trailing axes are flattened. For convolutional layers, this
    corresponds to the fan-in, so this makes the initialization usable for
    both dense and convolutional layers.

    Parameters
    ----------
    gain : float or 'relu'.
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
        return _cast_dtype(q)


def decompose_size(size):
    """Computes the number of input and output units for a weight shape.
    
    Parameters
    ----------
    size 
        Integer shape tuple.
    
    Returns
    -------
    A tuple of scalars, `(fan_in, fan_out)`.
    """
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


def _cast_dtype(res):
    return np.array(res, dtype=get_dtype())


_zero = Zero()
_one = One()


def get(initialization):
    if initialization.__class__.__name__ == 'str':
        if initialization in ['zero', 'Zero']:
            return Zero()
        if initialization in ['one', 'One']:
            return One()
        if initialization in ['uniform', 'Uniform']:
            return Uniform()
        if initialization in ['normal', 'Normal']:
            return Normal()
        if initialization in ['lecun_uniform', 'LecunUniform']:
            return LecunUniform()
        if initialization in ['glorot_uniform', 'GlorotUniform']:
            return GlorotUniform()
        if initialization in ['glorot_normal', 'GlorotNormal']:
            return GlorotNormal()
        if initialization in ['HeNormal', 'he_normal']:
            return HeNormal()
        if initialization in ['HeUniform', 'he_uniform']:
            return HeUniform()
        if initialization in ['Orthogonal', 'orthogonal']:
            return Orthogonal()
        raise ValueError('Unknown initialization name: {}.'.format(initialization))

    elif isinstance(initialization, Initializer):
        return copy.deepcopy(initialization)

    else:
        raise ValueError("Unknown type: {}.".format(initialization.__class__.__name__))


