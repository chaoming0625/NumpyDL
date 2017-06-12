# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.datasets import fetch_mldata

import npdl


def main(max_iter):
    # data
    print("loading data, please wait ...")
    mnist = fetch_mldata('MNIST original', data_home=os.path.join(os.path.dirname(__file__), './data'))
    print('data loading is done ...')
    X_train = mnist.data / 255.0
    y_train = mnist.target
    n_classes = np.unique(y_train).size

    # model
    print("building model ...")
    model = npdl.Model()
    model.add(npdl.layers.Dense(n_out=500, n_in=784, activation=npdl.activations.ReLU()))
    model.add(npdl.layers.Dense(n_out=n_classes, activation=npdl.activations.Softmax()))
    model.compile(loss=npdl.objectives.SCCE(), optimizer=npdl.optimizers.SGD(lr=0.001))

    # train
    print("train model ... ")
    model.fit(X_train, npdl.utils.data.one_hot(y_train), max_iter=max_iter, validation_split=0.1)


if __name__ == '__main__':
    main(150)
