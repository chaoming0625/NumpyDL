# -*- coding: utf-8 -*-


import pytest
import numpy as np
from npdl.layers import Recurrent
from npdl.layers import GRU
from npdl.layers import LSTM


class PrevLayer:
    def __init__(self, out_shape):
        self.out_shape = out_shape


def test_Recurrent():

    for seq in (True, False):
        layer = Recurrent(n_out=200, return_sequence=seq)
        assert layer.out_shape is None
        with pytest.raises(AssertionError):
            layer.connect_to()
        layer.connect_to(PrevLayer((20, 10, 100)))
        assert layer.n_in == 100
        assert len(layer.out_shape) == (3 if seq else 2)


def test_GRU():

    for seq in (True, False):
        layer = GRU(n_out=200, n_in=100, return_sequence=seq)
        assert layer.out_shape is None
        layer.connect_to()
        assert len(layer.out_shape) == (3 if seq else 2)

        input = np.random.rand(10, 50, 100)
        assert np.ndim(layer.forward(input)) == (3 if seq else 2)

        with pytest.raises(NotImplementedError):
            layer.backward(None)

        assert len(layer.params) == 9
        assert len(layer.grads) == 9


def test_LSTM():
    for seq in (True, False):
        layer = LSTM(n_out=200, n_in=100, return_sequence=seq)
        assert layer.out_shape is None
        layer.connect_to()
        assert len(layer.out_shape) == (3 if seq else 2)

        input = np.random.rand(10, 50, 100)
        mask = np.random.randint(0, 2, (10, 50))
        assert np.ndim(layer.forward(input, mask)) == (3 if seq else 2)

        with pytest.raises(NotImplementedError):
            layer.backward(None)

        assert len(layer.params) == 12
        assert len(layer.grads) == 12

