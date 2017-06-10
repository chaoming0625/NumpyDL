# -*- coding: utf-8 -*-

import pytest

from npdl.layers import Flatten


def test_flatten():

    with pytest.raises(ValueError):
        flatten_layer = Flatten(0)


