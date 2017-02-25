# -*- coding: utf-8 -*-

"""
@author: ChaoMing (https://oujago.github.io/)

@date: Created on 17-1-10

@notes:
    
"""

import numpy as np
from sklearn.datasets import fetch_mldata

import pydl


def test_mnist():
    batch_size = 32

    # data
    print("loading data ...")
    mnist = fetch_mldata('MNIST original', data_home='./data')
    X_train = np.reshape(mnist.data, (-1, 1, 28, 28)) / 255.0
    Y_train = mnist.target
    n_classes = np.unique(Y_train).size

    # model
    print("building model ...")
    model = pydl.model.Model()
    model.add(pydl.layers.Conv2D((batch_size, 1, 28, 28), (32, 1, 5, 5), (2, 2)))
    model.add(pydl.layers.Activation('relu'))
    model.add(pydl.layers.MaxPool2D((batch_size, 32, 28, 28), (2, 2)))
    model.add(pydl.layers.Conv2D((batch_size, 32, 14, 14), (64, 32, 5, 5)))
    model.add(pydl.layers.Activation('relu'))
    model.add(pydl.layers.MaxPool2D((batch_size, 64, 10, 10), (2, 2)))
    model.add(pydl.layers.Flatten())
    model.add(pydl.layers.Linear(n_in=5 ** 2 * 64, n_out=n_classes))
    model.add(pydl.layers.Activation('softmax'))
    model.compile(loss='scce', optimizer=pydl.optimizers.SGD(0.002))

    # train
    print("train model ... ")
    model.fit(X_train, pydl.utils.data.one_hot(Y_train),
              validation_split=0.1, batch_size=batch_size, max_iter=50)


if __name__ == '__main__':
    test_mnist()



