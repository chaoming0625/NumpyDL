# -*- coding: utf-8 -*-


import os
import numpy as np
from sklearn.datasets import fetch_mldata

import npdl


def main(max_iter):
    seed = 100
    nb_data = 1000

    print("loading data ....")
    mnist = fetch_mldata('MNIST original', data_home=os.path.join(os.path.dirname(__file__), './data'))
    X_train = mnist.data.reshape((-1, 1, 28, 28)) / 255.0
    np.random.seed(seed)
    X_train = np.random.permutation(X_train)[:nb_data]
    y_train = mnist.target
    np.random.seed(seed)
    y_train = np.random.permutation(y_train)[:nb_data]
    n_classes = np.unique(y_train).size

    print("building model ...")
    net = npdl.Model()
    net.add(npdl.layers.Convolution(1, (3, 3), input_shape=(None, 1, 28, 28)))
    net.add(npdl.layers.MeanPooling((2, 2)))
    net.add(npdl.layers.Convolution(2, (4, 4)))
    net.add(npdl.layers.MeanPooling((2, 2)))
    net.add(npdl.layers.Flatten())
    net.add(npdl.layers.Softmax(n_out=n_classes))
    net.compile()

    print("train model ... ")
    net.fit(X_train, npdl.utils.data.one_hot(y_train), max_iter=max_iter, validation_split=0.1, batch_size=100)


if __name__ == '__main__':
    main(10)
