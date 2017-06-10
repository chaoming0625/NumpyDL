# -*- coding: utf-8 -*-

import pytest
import numpy as np


class PreLayer:

    def __init__(self, out_shape):
        self.out_shape = out_shape


def test_MeanPooling():
    from npdl.layers import MeanPooling

    pool = MeanPooling((2, 2))

    pool.connect_to(PreLayer((10, 1, 20, 30)))
    assert pool.out_shape == (10, 1, 10, 15)

    with pytest.raises(ValueError):
        pool.forward(np.random.rand(10, 10))
    with pytest.raises(ValueError):
        pool.backward(np.random.rand(10, 20))

    assert np.ndim(pool.forward(np.random.rand(10, 20, 30))) == 3
    assert np.ndim(pool.backward(np.random.rand(10, 20, 30))) == 3

    assert np.ndim(pool.forward(np.random.rand(10, 1, 20, 30))) == 4
    assert np.ndim(pool.backward(np.random.rand(10, 1, 20, 30))) == 4


def test_MaxPooling():
    from npdl.layers import MaxPooling

    pool = MaxPooling((2, 2))

    pool.connect_to(PreLayer((10, 1, 20, 30)))
    assert pool.out_shape == (10, 1, 10, 15)

    with pytest.raises(ValueError):
        pool.forward(np.random.rand(10, 10))

    with pytest.raises(ValueError):
        pool.backward(np.random.rand(10, 20))

    assert np.ndim(pool.forward(np.random.rand(10, 20, 30))) == 3
    assert np.ndim(pool.backward(np.random.rand(10, 20, 30))) == 3

    assert np.ndim(pool.forward(np.random.rand(10, 1, 20, 30))) == 4
    assert np.ndim(pool.backward(np.random.rand(10, 1, 20, 30))) == 4
