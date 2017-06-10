# -*- coding: utf-8 -*-

import pytest
import numpy as np


def test_objective():
    from npdl.objectives import Objective

    obj = Objective()

    with pytest.raises(NotImplementedError):
        obj.forward(None, None)

    with pytest.raises(NotImplementedError):
        obj.backward(None, None)

    assert str(obj) == 'Objective'


def test_MeanSquaredError():
    from npdl.objectives import MeanSquaredError

    obj = MeanSquaredError()

    outputs = np.random.rand(10, 20)
    targets = np.random.rand(10, 20)

    f_res = obj.forward(outputs, targets)
    b_res = obj.backward(outputs, targets)

    assert np.ndim(f_res) == 0
    assert np.ndim(b_res) == 2


def test_HellingerDistance():
    from npdl.objectives import HellingerDistance

    obj = HellingerDistance()

    outputs = np.random.random((10, 20))
    targets = np.random.random((10, 20))

    f_res = obj.forward(outputs, targets)
    b_res = obj.backward(outputs, targets)

    assert np.ndim(f_res) == 0
    assert np.ndim(b_res) == 2


def test_BinaryCrossEntropy():
    from npdl.objectives import BinaryCrossEntropy

    obj = BinaryCrossEntropy()

    outputs = np.random.randint(0, 2, (10, 1))
    targets = np.random.randint(0, 2, (10, 1))

    f_res = obj.forward(outputs, targets)
    b_res = obj.backward(outputs, targets)

    assert np.ndim(f_res) == 0
    assert np.ndim(b_res) == 2


def test_SoftmaxCategoricalCrossEntropy():
    from npdl.objectives import SoftmaxCategoricalCrossEntropy

    obj = SoftmaxCategoricalCrossEntropy()

    outputs = np.random.random((10, 20))
    targets = np.random.random((10, 20))

    f_res = obj.forward(outputs, targets)
    b_res = obj.backward(outputs, targets)

    assert np.ndim(f_res) == 0
    assert np.ndim(b_res) == 2


