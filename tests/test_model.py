# -*- coding: utf-8 -*-


import numpy as np
import pytest

import npdl
from npdl.model import Model
from npdl.utils.random import get_dtype
from npdl.utils.data import one_hot


def test_model():

    X = np.random.random((10, 20)).astype(get_dtype())
    Y = one_hot(np.random.randint(0, 3, 10)).astype(get_dtype())
    n_classes = np.unique(Y).size

    valid_X = X[:2]
    valid_Y = Y[:2]

    model = Model()
    model.add(npdl.layers.Dense(n_out=500, n_in=64, activation=npdl.activations.ReLU()))
    model.add(npdl.layers.Dense(n_out=n_classes, activation=npdl.activations.Softmax()))
    model.compile(loss=npdl.objectives.SCCE(), optimizer=npdl.optimizers.SGD(lr=0.005))

    with pytest.raises(NotImplementedError):
        model.evaluate(X, Y)

    model.fit(X, Y, max_iter=0, validation_data=(valid_X, valid_Y))

