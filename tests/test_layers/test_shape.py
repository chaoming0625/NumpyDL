# -*- coding: utf-8 -*-

import pytest

from npdl.layers import Flatten
from npdl.layers import DimShuffle


def test_flatten():

    with pytest.raises(ValueError):
        flatten_layer = Flatten(0)


def test_DimShuffle():
    with pytest.raises(ValueError):
        DimShuffle(-1)