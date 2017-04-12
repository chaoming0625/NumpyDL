

.. image:: https://pypip.in/d/npdl/badge.svg
    :target: https://pypi.python.org/pypi/npdl/

.. image:: https://readthedocs.org/projects/numpydl/badge/
    :target: http://numpydl.readthedocs.org/en/latest/

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/oujago/NumpyDL/blob/master/LICENSE

.. image:: https://travis-ci.org/oujago/NumpyDL.svg
    :target: https://travis-ci.org/oujago/NumpyDL

.. image:: https://coveralls.io/repos/github/oujago/NumpyDL/badge.svg?branch=master
    :target: https://coveralls.io/github/oujago/NumpyDL?branch=master

.. image:: https://codeclimate.com/github/oujago/NumpyDL/badges/gpa.svg
   :target: https://codeclimate.com/github/oujago/NumpyDL
   :alt: Code Climate

.. image:: https://codeclimate.com/github/oujago/NumpyDL/badges/issue_count.svg
   :target: https://codeclimate.com/github/oujago/NumpyDL
   :alt: Issue Count

.. image:: https://zenodo.org/badge/83100910.svg
   :target: https://zenodo.org/badge/latestdoi/83100910


NumpyDL: Deep Learning Library based on Pure Numpy
==================================================

NumpyDL is a simple deep learning library based on pure Python/Numpy.Its main features are:

* Pure python + numpy
* API like Keras deep learning library
* Support basic automatic differentiation;
* Support commonly used models, such as MLP, RNNs LSTMs and convnets.
* Flexible network configurations and learning algorithms.

Its design is governed by several principles:

* Simplicity: Be easy to use, easy to understand and easy to extend, to
  facilitate use in research. Interfaces should be kept small, with as few
  classes and methods as possible. Every added abstraction and feature
  should be carefully scrutinized, to determine whether the added complexity
  is justified.
* Transparency: Native to Numpy, directly process and return Python / numpy
  data types. Do not rely on the functionality of Theano, Tensorflow or any
  such DL framework.
* Modularity: Allow all parts (layers, regularizers, optimizers, ...) to be
  used independently of NumpyDL. Make it easy to use components in isolation
  or in conjunction with other frameworks.
* Focus: “Do one thing and do it well”. Do not try to provide a library
  for everything to do with deep learning.


Documentation
=============

Available online documents: `latest <http://numpydl.readthedocs.io/en/lastest>`_
docs, `development <http://numpydl.readthedocs.io/en/develop/>`_ docs, and
`stable <http://numpydl.readthedocs.io/en/stable/>`_ docs. Chinese version document
is in development and will be available soon.


Example
=======

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


Installation
============

Install NumpyDL using pip:

.. code-block:: bash

    $> pip install npdl

Install from source code:

.. code-block:: bash

    $> python setup.py install
   
   
Support
=======

Layers
------

* `core.py <npdl/layers/core.py>`_
    * Dense (perceptron) Layer
    * Softmax Layer
    * Dropout Layer
* `normalization.py <npdl/layers/normalization.py>`_
    * Batch Normalization Layer
* `embedding.py <npdl/layers/embedding.py>`_
    * Embedding Layer
* `convolution.py <npdl/layers/convolution.py>`_
    * Convolution Layer
* `pooling.py <npdl/layers/pooling.py>`_
    * MaxPooling Layer
    * MeanPooling Layer
* `reccurent.py <npdl/layers/reccurent.py>`_
    * SimpleRNN Layer
* `shape.py <npdl/layers/shape.py>`_
    * Flatten Layer

Activations
-----------

* Sigmoid
* Tanh
* ReLU
* Softmax
* Elliot
* SymmetricElliot
* LReLU
* SoftPlus
* SoftSign

Initializations
---------------

* Uniform
* Normal
* LecunUniform
* GlorotUniform
* GlorotNormal
* HeNormal
* HeUniform
* Orthogonal

Objectives
----------

* MeanSquaredError
* HellingerDistance
* BinaryCrossEntropy
* SoftmaxCategoricalCrossEntropy


Optimizers
----------

* SGD
