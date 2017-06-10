# -*- coding: utf-8 -*-

import pytest

import numpy as np


def test_optimizer():
    from npdl.optimizers import Optimizer

    opt = Optimizer()

    with pytest.raises(NotImplementedError):
        opt.add_param_grads(None)

    with pytest.raises(NotImplementedError):
        opt.update_params()

    assert str(opt) == 'Optimizer'


def test_sgd():
    from npdl.optimizers import SGD

    opt = SGD()


def test_Momentum():
    from npdl.optimizers import Momentum

    opt = Momentum()

    opt.add_param_grads(None)
    opt.update_params()


def test_NesterovMomentum():
    from npdl.optimizers import NesterovMomentum

    opt = NesterovMomentum()


def test_Adagrad():
    from npdl.optimizers import Adagrad

    opt = Adagrad()


def test_RMSprop():
    from npdl.optimizers import RMSprop

    opt = RMSprop()


def test_Adadelta():
    from npdl.optimizers import Adadelta

    opt = Adadelta()


def test_Adam():
    from npdl.optimizers import Adam

    opt = Adam()


def test_Adamax():
    from npdl.optimizers import Adamax

    opt = Adamax()


def test_npdl_clip():
    from npdl.optimizers import npdl_clip

    grad = (np.random.random((10, 20)) - 0.5) * 2
    boundary = 0.5

    assert np.all(np.abs(npdl_clip(grad, boundary)) <= boundary)
    assert np.allclose(npdl_clip(grad, 0.), grad)

