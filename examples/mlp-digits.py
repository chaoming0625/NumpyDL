# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_digits

import npdl


def main(max_iter):
    # prepare
    npdl.utils.random.set_seed(1234)

    # data
    digits = load_digits()

    X_train = digits.data
    X_train /= np.max(X_train)

    Y_train = digits.target
    n_classes = np.unique(Y_train).size

    # model
    model = npdl.model.Model()
    model.add(npdl.layers.Dense(n_out=500, n_in=64, activation=npdl.activations.ReLU()))
    model.add(npdl.layers.Dense(n_out=n_classes, activation=npdl.activations.Softmax()))
    model.compile(loss=npdl.objectives.SCCE(), optimizer=npdl.optimizers.SGD(lr=0.005))

    # train
    model.fit(X_train, npdl.utils.data.one_hot(Y_train), max_iter=max_iter, validation_split=0.1)


if __name__ == '__main__':
    main(150)
