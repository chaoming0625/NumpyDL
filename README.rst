
.. image:: https://readthedocs.org/projects/numpydl/badge/
    :target: http://numpydl.readthedocs.org/en/latest/

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/oujago/NumpyDL/blob/master/LICENSE

.. image:: https://travis-ci.com/chaoming0625/NumpyDL.svg?branch=master
    :target: https://travis-ci.com/chaoming0625/NumpyDL

.. image:: https://coveralls.io/repos/github/oujago/NumpyDL/badge.svg
    :target: https://coveralls.io/github/oujago/NumpyDL

.. image:: https://badge.fury.io/py/npdl.svg
    :target: https://badge.fury.io/py/npdl

.. image:: https://img.shields.io/badge/python-3.5-blue.svg
    :target: https://pypi.python.org/pypi/npdl

.. image:: https://img.shields.io/badge/python-3.6-blue.svg
    :target: https://pypi.python.org/pypi/npdl

.. https://codeclimate.com/github/oujago/NumpyDL/badges/gpa.svg
   :target: https://codeclimate.com/github/oujago/NumpyDL

.. image:: https://codeclimate.com/github/oujago/NumpyDL/badges/issue_count.svg
   :target: https://codeclimate.com/github/oujago/NumpyDL

.. image:: https://img.shields.io/github/issues/oujago/NumpyDL.svg
   :target: https://github.com/oujago/NumpyDL

.. image:: https://zenodo.org/badge/83100910.svg
   :target: https://zenodo.org/badge/latestdoi/83100910



NumpyDL: Numpy Deep Learning Library
====================================

Descriptions
============

``NumpyDL`` is:

1. Based on Pure Numpy/Python
2. For DL Education


Features
========

Its main features are:

1. *Pure* in Numpy
2. *Native* to Python
3. *Automatic differentiations* are basically supported
4. *Commonly used models* are provided: MLP, RNNs, LSTMs and CNNs
5. *Examples* for several AI tasks
6. *Application* for a toy chatbot


Documentation
=============

Available online documents:

1. `latest docs <http://numpydl.readthedocs.io/en/latest>`_
2. `development docs <http://numpydl.readthedocs.io/en/develop/>`_
3. `stable docs <http://numpydl.readthedocs.io/en/stable/>`_

Available offline PDF:

1. `latest PDF <http://readthedocs.org/projects/numpydl/downloads/pdf/latest>`_


Installation
============

Install NumpyDL using pip:

.. code-block:: bash

    $> pip install npdl

Install from source code:

.. code-block:: bash

    $> python setup.py install


Examples
========

``NumpyDL`` provides several examples of AI tasks:

* sentence classification
    * LSTM in *examples/lstm_sentence_classification.py*
    * CNN in *examples/cnn_sentence_classification.py*
* mnist handwritten recognition
    * MLP in *examples/mlp-mnist.py*
    * MLP in *examples/mlp-digits.py*
    * CNN in *examples/cnn-minist.py*
* language modeling
    * RNN in *examples/rnn-character-lm.py*
    * LSTM in *examples/lstm-character-lm.py*

One concrete code example in *examples/mlp-digits.py*:

.. code-block:: python

    import numpy as np
    from sklearn.datasets import load_digits
    import npdl

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



Applications
============

``NumpyDL`` provides one toy application:

* Chatbot
    * seq2seq in *applications/chatbot/model.py*


And its final result:

.. figure:: applications/chatbot/pics/chatbot.png
    :width: 80%


Supports
========

``NumpyDL`` supports following deep learning techniques:

* Layers
    * Linear
    * Dense
    * Softmax
    * Dropout
    * Convolution
    * Embedding
    * BatchNormal
    * MeanPooling
    * MaxPooling
    * SimpleRNN
    * GRU
    * LSTM
    * Flatten
    * DimShuffle
* Optimizers
    * SGD
    * Momentum
    * NesterovMomentum
    * Adagrad
    * RMSprop
    * Adadelta
    * Adam
    * Adamax
* Objectives
    * MeanSquaredError
    * HellingerDistance
    * BinaryCrossEntropy
    * SoftmaxCategoricalCrossEntropy
* Initializations
    * Zero
    * One
    * Uniform
    * Normal
    * LecunUniform
    * GlorotUniform
    * GlorotNormal
    * HeNormal
    * HeUniform
    * Orthogonal
* Activations
    * Sigmoid
    * Tanh
    * ReLU
    * Linear
    * Softmax
    * Elliot
    * SymmetricElliot
    * SoftPlus
    * SoftSign

