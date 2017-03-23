# -*- coding: utf-8 -*-


"""
Functions to create initializer for parameter variables.

Examples
--------
>>> from npdl.model.layers import Dense
>>> from npdl.model.initialization import Normal
>>> l1 = Dense(100, 200, init=Normal())

"""

import numpy as np
from .random import get_rng


class Initializer(object):
    """
    Base class: all initializer class inherit from this class.

    The :class:`Initializer` class represents a weight initializer used
    to initialize weight parameters in a neural network layer. It should be
    subclassed when implementing new types of weight initializers.
    """

    def __call__(self, shape, dtype=None):
        """
        Makes :class:`Initializer` instances callable like a function.
        """
        raise NotImplementedError

    def get_config(self):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        return cls(**config)



class Zero:
    def __call__(self, size):
        return np.zeros(size)


class One:
    def __call__(self, size):
        return np.ones(size)


class Uniform:
    def __init__(self, scale=0.05):
        self.scale = scale

    def __call__(self, size):
        return get_rng().uniform(-self.scale, self.scale, size=size)


class Normal:
    def __init__(self, scale=0.05):
        self.scale = scale

    def __call__(self, size):
        return get_rng().normal(loc=0.0, scale=self.scale, size=size)(size)


class LecunUniform:
    def __call__(self, size):
        fan_in, fan_out = _decompose_size(size)
        return Uniform(np.sqrt(3. / fan_in))(size)


class GlorotUniform:
    def __call__(self, size):
        fan_in, fan_out = _decompose_size(size)
        return Uniform(np.sqrt(6 / (fan_in + fan_out)))(size)


class GlorotNormal:
    def __call__(self, size):
        fan_in, fan_out = _decompose_size(size)
        return Normal(np.sqrt(2 / (fan_out + fan_in)))(size)


class HeNormal:
    def __call__(self, size):
        fan_in, fan_out = _decompose_size(size)
        return Normal(np.sqrt(2. / fan_in))(size)


class HeUniform:
    def __call__(self, size):
        fan_in, fan_out = _decompose_size(size)
        return Uniform(np.sqrt(6. / fan_in))(size)


class Orthogonal:
    def __call__(self, size):
        flat_shape = (size[0], np.prod(size[1:]))
        a = get_rng().normal(loc=0., scale=1., size=flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        return q.reshape(size)


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
