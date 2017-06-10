# -*- coding: utf-8 -*-


import pytest
import numpy as np


class PrevLayer:
    out_shape = (10, 20)


def test_BatchNormal():
    from npdl.layers import BatchNormal

    layer = BatchNormal()

    layer.connect_to(PrevLayer())
    assert layer.beta.shape == (20, )
    assert layer.gamma.shape == (20, )

    input = np.random.rand(30, 20)
    assert layer.forward(input).shape == (30, 20)

    pre_grad = np.random.rand(30, 20)
    assert layer.backward(pre_grad).shape == (30, 20)

    assert len(layer.param_grads) == 0
    assert len(layer.params) == 2
    assert len(layer.grades) == 2

