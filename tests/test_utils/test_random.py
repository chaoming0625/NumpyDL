# -*- coding: utf-8 -*-

import pytest
import numpy as np

from npdl.utils import get_rng
from npdl.utils import set_rng
from npdl.utils import set_seed
from npdl.utils import get_dtype
from npdl.utils import set_dtype


def test_rng():
    set_rng(np.random.RandomState(seed=12345))

    assert isinstance(get_rng(), np.random.RandomState)


def test_set_seed():
    set_seed(12345)

    rng = get_rng()

    assert rng.randint(10) == np.random.RandomState(12345).randint(10)
    assert isinstance(get_rng(), np.random.RandomState)


def test_dtype():
    set_dtype('int32')
    assert get_dtype() == 'int32'

    set_dtype('float32')
    assert get_dtype() == 'float32'

