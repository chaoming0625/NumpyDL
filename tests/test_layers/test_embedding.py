# -*- coding: utf-8 -*-


import pytest
import numpy as np
from npdl.layers import Embedding


def test_embed_words():
    embed_words = np.random.rand(100, 10)

    # static == False
    layer = Embedding(embed_words, False)

    assert len(layer.params) == 1
    assert len(layer.grads) == 1
    assert len(layer.param_grads) == 1

    layer.connect_to()
    assert len(layer.out_shape) == 3

    # static == True
    layer = Embedding(embed_words)

    with pytest.raises(AssertionError):
        layer.forward(np.arange(10))

    input = np.random.randint(0, 10, (4, 2))
    assert layer.forward(input).shape == (4, 2, 10)

    assert len(layer.params) == 0
    assert len(layer.grads) == 0
    assert len(layer.param_grads) == 0


def test_no_embed_words():
    # static == False
    layer = Embedding(input_size=100, n_out=10)

    assert len(layer.params) == 0
    assert len(layer.grads) == 0
    assert len(layer.param_grads) == 0

    # static == True
    layer = Embedding(static=False, input_size=100, n_out=10)

    assert len(layer.params) == 1
    assert len(layer.grads) == 1
    assert len(layer.param_grads) == 1

