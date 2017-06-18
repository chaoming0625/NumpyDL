# -*- coding: utf-8 -*-

import pytest

import numpy as np

from npdl import optimizers


def test_optimizer():
    from npdl.optimizers import Optimizer

    opt = Optimizer()

    opt.update([], [])
    assert opt.iterations == 1

    assert str(opt) == 'Optimizer'


def test_sgd():
    from npdl.optimizers import SGD

    opt = SGD()


def test_Momentum():
    from npdl.optimizers import Momentum

    opt = Momentum()



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


def test_get():
    for init in ['sgd',
                 'momentum',
                 'nesterov_momentum',
                 'adagrad',
                 'rmsprop',
                 'adadelta',
                 'adam',
                 'adamax',
                 optimizers.SGD()]:
        optimizers.get(init)

    for init in [1, '1']:
        with pytest.raises(ValueError):
            optimizers.get(init)

