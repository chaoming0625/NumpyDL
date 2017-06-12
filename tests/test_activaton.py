# -*- coding: utf-8 -*-

import numpy as np
import pytest

from npdl import activations


def test_activation():
    from npdl.activations import Activation

    act = Activation()

    with pytest.raises(NotImplementedError):
        act.forward((10, 10))

    with pytest.raises(NotImplementedError):
        act.derivative()

    assert str(act) == 'Activation'


def test_get():
    with pytest.raises(ValueError):
        activations.get(1)
    with pytest.raises(ValueError):
        activations.get('l')


class TestActivations(object):
    @pytest.mark.parametrize('activation',
                             ['sigmoid',
                              'tanh',
                              'relu',
                              'linear',
                              'softmax',
                              'elliot',
                              'SymmetricElliot',
                              'SoftPlus',
                              'SoftSign'])
    def test_activation(self, activation):

        input = np.arange(24).reshape((4, 6))

        npdl_act = activations.get(activation)

        if activation == 'sigmoid':
            f_res = npdl_act.forward(input)

            assert 0. <= np.all(f_res) <= 1.
            assert npdl_act.derivative().shape == input.shape

        elif activation == 'tanh':
            f_res = npdl_act.forward(input)

            assert -1. <= np.all(f_res) <= 1.0
            assert npdl_act.derivative().shape == input.shape

        elif activation == 'relu':
            f_res = npdl_act.forward(input)

            assert np.all(f_res) >= 0.
            assert npdl_act.derivative().shape == input.shape
            assert np.all(npdl_act.derivative()) <= 1.

        elif activation == 'linear':
            f_res = npdl_act.forward(input)

            assert np.allclose(f_res, input)
            assert npdl_act.derivative().shape == input.shape
            assert np.all(npdl_act.derivative()) == 1.

        elif activation == 'softmax':
            f_res = npdl_act.forward(input)

            assert 0. <= np.all(f_res) <= 1.0
            assert npdl_act.derivative().shape == input.shape
            assert np.all(npdl_act.derivative()) == 1.

        elif activation == 'elliot':
            f_res = npdl_act.forward(input)

            assert f_res.shape == input.shape
            assert npdl_act.derivative().shape == input.shape

        elif activation == 'SymmetricElliot':
            f_res = npdl_act.forward(input)

            assert f_res.shape == input.shape
            assert npdl_act.derivative().shape == input.shape

        elif activation == 'SoftPlus':
            f_res = npdl_act.forward(input)

            assert f_res.shape == input.shape
            assert npdl_act.derivative().shape == input.shape

        elif activation == 'SoftSign':
            f_res = npdl_act.forward(input)

            assert f_res.shape == input.shape
            assert npdl_act.derivative().shape == input.shape
