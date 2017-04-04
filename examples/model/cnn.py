# -*- coding: utf-8 -*-

import numpy as np
import npdl
from npdl import model

from sklearn.datasets import fetch_mldata

print("loading data ...")
mnist = fetch_mldata('MNIST original', data_home='./data')
X_train = mnist.data.reshape((-1, 1, 28, 28)) / 255.0
y_train = mnist.target
n_classes = np.unique(y_train).size

print("building model ...")
net = model.Model()
net.add(model.layers.Convolution(1, (3, 3), input_shape=(None, 1, 28, 28)))
net.add(model.layers.MeanPooling((2, 2)))
net.add(model.layers.Flatten())
net.add(model.layers.Softmax(n_out=n_classes))
net.compile()

print("train model ... ")
net.fit(X_train, npdl.utils.data.one_hot(y_train), max_iter=10, validation_split=0.1)

