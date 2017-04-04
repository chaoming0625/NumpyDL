# -*- coding: utf-8 -*-


import numpy as np

from npdl.utils.random import get_rng
from npdl.utils.random import get_dtype
from npdl.utils.generic import get_from_module


class Initializer(object):
    def __call__(self, size):
        return self.call(size)

    def call(self, size):
        raise NotImplementedError()


class Zero(Initializer):
    def call(self, size):
        return np.zeros(size, dtype=get_dtype())


class One(Initializer):
    def call(self, size):
        return np.ones(size, dtype=get_dtype())


class Uniform(Initializer):
    def __init__(self, scale=0.05):
        self.scale = scale

    def call(self, size):
        return get_rng().uniform(-self.scale, self.scale, size=size).astype(get_dtype())


class Normal(Initializer):
    def __init__(self, scale=0.05):
        self.scale = scale

    def call(self, size):
        return get_rng().normal(loc=0.0, scale=self.scale, size=size)(size).astype(get_dtype())


class LecunUniform(Initializer):
    def call(self, size):
        fan_in, fan_out = _decompose_size(size)
        return Uniform(np.sqrt(3. / fan_in))(size)


class GlorotUniform(Initializer):
    def call(self, size):
        fan_in, fan_out = _decompose_size(size)
        return Uniform(np.sqrt(6 / (fan_in + fan_out)))(size)


class GlorotNormal(Initializer):
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
    def call(self, size):
        flat_shape = (size[0], np.prod(size[1:]))
        a = get_rng().normal(loc=0., scale=1., size=flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        return q.reshape(size).astype(get_dtype())


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

