# -*- coding: utf-8 -*-


import pytest
import numpy as np


class PrevLayer:
    def __init__(self, out_shape):
        self.out_shape = out_shape


def test_linear():
    from npdl.layers import Linear

    layer = Linear(n_in=10, n_out=20)
    layer.connect_to(None)
    assert layer.W.shape == (10, 20)
    assert layer.b.shape == (20,)

    layer.connect_to(PrevLayer((20, 10)))
    assert layer.W.shape == (10, 20)
    assert layer.b.shape == (20,)

    input = np.random.rand(20, 10)
    assert layer.forward(input).shape == (20, 20)

    pre_grad = np.random.rand(20, 20)
    assert layer.backward(pre_grad).shape == input.shape

    assert len(layer.params) == 2
    assert len(layer.grads) == 2


def test_Dense():
    from npdl.layers import Dense

    layer = Dense(n_in=10, n_out=20)
    layer.connect_to(None)
    assert layer.W.shape == (10, 20)
    assert layer.b.shape == (20,)

    layer.connect_to(PrevLayer((20, 10)))
    assert layer.W.shape == (10, 20)
    assert layer.b.shape == (20,)

    input = np.random.rand(20, 10)
    assert layer.forward(input).shape == (20, 20)

    pre_grad = np.random.rand(20, 20)
    assert layer.backward(pre_grad).shape == input.shape

    assert len(layer.params) == 2
    assert len(layer.grads) == 2


def test_Softmax():
    from npdl.layers import Softmax

    layer = Softmax(n_in=10, n_out=20)
    layer.connect_to(None)
    assert layer.W.shape == (10, 20)
    assert layer.b.shape == (20,)

    layer.connect_to(PrevLayer((20, 10)))
    assert layer.W.shape == (10, 20)
    assert layer.b.shape == (20,)

    input = np.random.rand(20, 10)
    assert layer.forward(input).shape == (20, 20)

    pre_grad = np.random.rand(20, 20)
    assert layer.backward(pre_grad).shape == input.shape

    assert len(layer.params) == 2
    assert len(layer.grads) == 2


def test_Dropout():
    from npdl.layers import Dropout

    input = np.random.rand(10, 20)
    pre_grad = np.random.rand(10, 20)

    layer = Dropout(0.5)
    layer.connect_to(PrevLayer((10, 20)))
    assert layer.forward(input).shape == input.shape
    assert np.allclose(layer.forward(input, False), input * 0.5)
    assert layer.backward(pre_grad).shape == input.shape

    layer = Dropout()
    layer.connect_to(PrevLayer((10, 20)))
    assert np.allclose(layer.forward(input), input)
    assert layer.backward(pre_grad).shape == input.shape




