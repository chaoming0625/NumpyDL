# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.datasets import load_digits

import npdl


def test_digits():
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
    model.add(npdl.layers.Dense(n_out=500, n_in=64, activation=npdl.activation.ReLU()))
    model.add(npdl.layers.Dense(n_out=n_classes, activation=npdl.activation.Softmax()))
    model.compile(loss=npdl.objectives.SCCE(), optimizer=npdl.optimizers.SGD(lr=0.005))

    # train
    model.fit(X_train, npdl.utils.data.one_hot(Y_train), max_iter=150, validation_split=0.1)


def test_mnist():
    # data
    print("loading data ...")
    mnist = fetch_mldata('MNIST original', data_home='./data')
    X_train = mnist.data / 255.0
    y_train = mnist.target
    n_classes = np.unique(y_train).size

    # model
    print("building model ...")
    model = npdl.Model()
    model.add(npdl.layers.Dense(n_out=500, n_in=784, activation=npdl.activation.ReLU()))
    model.add(npdl.layers.Dense(n_out=n_classes, activation=npdl.activation.Softmax()))
    model.compile(loss=npdl.objectives.SCCE(), optimizer=npdl.optimizers.SGD(lr=0.001))

    # train
    print("train model ... ")
    model.fit(X_train, npdl.utils.data.one_hot(y_train), max_iter=150, validation_split=0.1)


if __name__ == '__main__':
    # test_digits()
    test_mnist()


