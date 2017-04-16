.. NumpyDL documentation master file, created by
   sphinx-quickstart on Mon Apr 10 13:33:52 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Hi, NumpyDL
===========

NumpyDL is a simple deep learning library based on pure Python/Numpy. NumpyDL
is a work in progress, input is welcome. The project is on
`GitHub <https://github.com/oujago/NumpyDL>`_.

The main features of NumpyDL are as follows:

* *Pure* in Numpy and *native* to Python
* Support basic *automatic differentiation*
* Support *commonly used models*, such as MLP, RNNs, GRUs, LSTMs and CNNs
* *Perfect documents* and easy to learn deep learning knowledge
* Flexible network configurations and learning algorithms.
* API like Keras deep learning library

The design of NumpyDL is governed by several principles:

* **Simplicity**: Be easy to use, easy to understand and easy to extend, to
  facilitate use in research. Interfaces should be kept small, with as few
  classes and methods as possible. Every added abstraction and feature
  should be carefully scrutinized, to determine whether the added complexity
  is justified.
* **Transparency**: Native to Numpy, directly process and return Python/Numpy
  data types. Do not rely on the functionality of Theano, Tensorflow or any
  such deep learning frameworks.
* **Modularity**: Allow all parts (layers, regularizers, optimizers, ...) to be
  used independently of NumpyDL. Make it easy to use components in isolation
  or in conjunction with other frameworks.
* **Focus**: “Do one thing and do it well”. Do not try to provide a library
  for everything to do with deep learning.


User Guides
===========

The NumpyDL user guide explains how to install NumpyDL, how to build and train
neural networks using NumpyDL, and how to contribute to the library as a
developer.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   user_guide/installation


API Reference
=============

If you are looking for information on a specific function, class or
method, this part of the documentation is for you.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api_reference/activation
   api_reference/initialization
   api_reference/objectives
   api_reference/optimizers
   api_reference/model


Examples
========

This part provides examples for building deep neural networks.

.. toctree::
   :maxdepth: 2
   :caption: Contents:




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _GitHub: https://github.com/oujago/NumpyDL