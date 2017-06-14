
.. image:: https://readthedocs.org/projects/numpydl/badge/
    :target: http://numpydl.readthedocs.org/en/latest/

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/oujago/NumpyDL/blob/master/LICENSE

.. image:: https://api.travis-ci.org/oujago/NumpyDL.svg
    :target: https://travis-ci.org/oujago/NumpyDL

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
   :alt: Code Climate

.. image:: https://codeclimate.com/github/oujago/NumpyDL/badges/issue_count.svg
   :target: https://codeclimate.com/github/oujago/NumpyDL
   :alt: Issue Count

.. image:: https://img.shields.io/github/issues/oujago/NumpyDL.svg
   :target: https://github.com/oujago/NumpyDL

.. image:: https://img.shields.io/github/forks/oujago/NumpyDL.svg
   :target: https://github.com/oujago/NumpyDL

.. image:: https://img.shields.io/github/stars/oujago/NumpyDL.svg
   :target: https://github.com/oujago/NumpyDL

.. image:: https://zenodo.org/badge/83100910.svg
   :target: https://zenodo.org/badge/latestdoi/83100910



NumpyDL: Numpy Deep Learning Library
====================================

Descriptions
============

``NumpyDL`` is:

1. For DL Education
2. Based on Pure Numpy/Python
3. And for My Homework


Features
========

Its main features are:

1. *Pure* in Numpy
2. *Native* to Python
3. *Automatic differentiations* are basically supported
4. *Commonly used models* are provided: MLP, RNNs, LSTMs and CNNs
5. *API* like ``Keras`` library
6. *Examples* for several AI tasks
7. *Application* for a toy chatbot

Documentation
=============

Available online documents:

1. `latest docs <http://numpydl.readthedocs.io/en/latest>`_
2. `development docs <http://numpydl.readthedocs.io/en/develop/>`_
3. `stable docs <http://numpydl.readthedocs.io/en/stable/>`_


Installation
============

Install NumpyDL using pip:

.. code-block:: bash

    $> pip install npdl

Install from source code:

.. code-block:: bash

    $> python setup.py install


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

