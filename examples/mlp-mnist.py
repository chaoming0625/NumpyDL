# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.datasets import fetch_mldata

import npdl


def get_data():
    # data
    print("loading data, please wait ...")
    mnist = fetch_mldata('MNIST original', data_home=os.path.join(os.path.dirname(__file__), './data'))
    print('data loading is done ...')
    X_train = mnist.data / 255.0
    y_train = mnist.target
    n_classes = np.unique(y_train).size

    return n_classes, X_train, y_train


def main(max_iter):
    n_classes, X_train, y_train = get_data()

    # model
    print("building model ...")
    model = npdl.Model()
    model.add(npdl.layers.Dense(n_out=500, n_in=784, activation=npdl.activations.ReLU()))
    model.add(npdl.layers.Dense(n_out=n_classes, activation=npdl.activations.Softmax()))
    model.compile(loss=npdl.objectives.SCCE(), optimizer=npdl.optimizers.SGD())

    # train
    print("train model ... ")
    model.fit(X_train, npdl.utils.data.one_hot(y_train), max_iter=max_iter, validation_split=0.1)


def main2(max_iter):
    # test Momentum optimizer

    n_classes, X_train, y_train = get_data()

    # model
    print("building model ...")
    model = npdl.Model()
    model.add(npdl.layers.Dense(n_out=500, n_in=784, activation=npdl.activations.ReLU()))
    model.add(npdl.layers.Dense(n_out=n_classes, activation=npdl.activations.Softmax()))
    model.compile(loss=npdl.objectives.SCCE(), optimizer=npdl.optimizers.Momentum())

    # train
    print("train model ... ")
    model.fit(X_train, npdl.utils.data.one_hot(y_train), max_iter=max_iter, validation_split=0.1)


def main3(max_iter):
    # test NesterovMomentum optimizer

    n_classes, X_train, y_train = get_data()

    # model
    print("building model ...")
    model = npdl.Model()
    model.add(npdl.layers.Dense(n_out=500, n_in=784, activation='relu'))
    model.add(npdl.layers.Softmax(n_out=n_classes))
    model.compile(loss=npdl.objectives.SCCE(), optimizer=npdl.optimizers.NesterovMomentum())

    # train
    print("train model ... ")
    model.fit(X_train, npdl.utils.data.one_hot(y_train), max_iter=max_iter, validation_split=0.1)


def main4(max_iter):
    # test Adagrad optimizer

    n_classes, X_train, y_train = get_data()

    # model
    print("building model ...")
    model = npdl.Model()
    model.add(npdl.layers.Dense(n_out=500, n_in=784, activation='relu'))
    model.add(npdl.layers.Softmax(n_out=n_classes))
    model.compile(loss='scce', optimizer='adagrad')

    # train
    print("train model ... ")
    model.fit(X_train, npdl.utils.data.one_hot(y_train), max_iter=max_iter, validation_split=0.1)


def main5(max_iter):
    # test RMSProp optimizer

    n_classes, X_train, y_train = get_data()

    # model
    print("building model ...")
    model = npdl.Model()
    model.add(npdl.layers.Dense(n_out=500, n_in=784, activation='relu'))
    model.add(npdl.layers.Softmax(n_out=n_classes))
    model.compile(loss='scce', optimizer='rmsprop')

    # train
    print("train model ... ")
    model.fit(X_train, npdl.utils.data.one_hot(y_train), max_iter=max_iter, validation_split=0.1)


def main6(max_iter):
    # test Adadelta optimizer

    n_classes, X_train, y_train = get_data()

    # model
    print("building model ...")
    model = npdl.Model()
    model.add(npdl.layers.Dense(n_out=500, n_in=784, activation='relu'))
    model.add(npdl.layers.Softmax(n_out=n_classes))
    model.compile(loss='scce', optimizer='adadelta')

    # train
    print("train model ... ")
    model.fit(X_train, npdl.utils.data.one_hot(y_train), max_iter=max_iter, validation_split=0.1)


def main7(max_iter):
    # test Adam optimizer

    n_classes, X_train, y_train = get_data()

    # model
    print("building model ...")
    model = npdl.Model()
    model.add(npdl.layers.Dense(n_out=500, n_in=784, activation='relu'))
    model.add(npdl.layers.Softmax(n_out=n_classes))
    model.compile(loss='scce', optimizer='adam')

    # train
    print("train model ... ")
    model.fit(X_train, npdl.utils.data.one_hot(y_train), max_iter=max_iter, validation_split=0.1)


def main8(max_iter):
    # test Adamax optimizer

    n_classes, X_train, y_train = get_data()

    # model
    print("building model ...")
    model = npdl.Model()
    model.add(npdl.layers.Dense(n_out=500, n_in=784, activation='relu'))
    model.add(npdl.layers.Softmax(n_out=n_classes))
    model.compile(loss='scce', optimizer='adamax')

    # train
    print("train model ... ")
    model.fit(X_train, npdl.utils.data.one_hot(y_train), max_iter=max_iter, validation_split=0.1)


if __name__ == '__main__':
    main8(50)
