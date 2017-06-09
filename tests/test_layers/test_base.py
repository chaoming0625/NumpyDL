# -*- coding: utf-8 -*-

import pytest

from npdl.layers import Layer


def test_layer():
    layer = Layer.from_json({})
    assert isinstance(layer, Layer)

    assert 'Layer' == str(layer)

    with pytest.raises(NotImplementedError):
        layer.to_json()

    with pytest.raises(NotImplementedError):
        layer.connect_to(None)

    with pytest.raises(NotImplementedError):
        layer.forward(None)

    with pytest.raises(NotImplementedError):
        layer.backward(None)
