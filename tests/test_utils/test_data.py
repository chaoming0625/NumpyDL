# -*- coding: utf-8 -*-


import pytest

import numpy as np


def test_one_hot():
    from npdl.utils import one_hot

    labels = np.array([1, 0, 3, 4])
    decoded = one_hot(labels)

    assert len(decoded) == len(labels)
    assert np.max(decoded) == 1
    assert np.min(decoded) == 0
    assert np.ndim(decoded) == 2


def test_unhot():
    from npdl.utils import unhot

    labels = np.zeros((4, 5), dtype='int32')
    labels[np.arange(len(labels))] = 1

    decoded = unhot(labels)

    assert len(decoded) == len(labels)
    assert np.max(decoded) <= 5
    assert np.min(decoded) >= 0

