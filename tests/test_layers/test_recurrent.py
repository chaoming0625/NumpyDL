# -*- coding: utf-8 -*-


import pytest
import numpy as np
from npdl.layers import Recurrent
from npdl.layers import BatchLSTM


class PrevLayer:
    def __init__(self, out_shape):
        self.out_shape = out_shape


def test_Recurrent():

    for seq in (True, False):
        layer = Recurrent(n_out=200, return_sequence=seq)
        assert layer.out_shape is None
        with pytest.raises(AssertionError):
            layer.connect_to()
        assert layer.connect_to(PrevLayer((20, 10, 100))) == 100
        assert len(layer.out_shape) == (3 if seq else 2)

