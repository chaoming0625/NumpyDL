.. _development:

===========
Development
===========

The NumpyDL project is started by `Chao-Ming Wang <https://oujago.github.io/about.html>`_
in February 2017. It is developed by a core team of five people:
`Chao-Ming Wang <https://oujago.github.io/about.html>`_, Jiao-Mei Liu, Shu-Ting Kang,
Xiao-Xuan Cui, Jin-Ze Li on Github: https://github.com/oujago/NumpyDL . The goal of
NumpyDL is making ``deep learning`` easy to learn and easy to use in native Numpy.

As an open-source project, we highly welcome contributions! Every bit helps and will
be credited!

.. _numpydl-philosopy:

Philosophy
==========

The development of NumpyDl is guided by a number of design goals:

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

What to contribute
==================

Give feedback
-------------

To send us general feedback, questions or ideas for improvement, please post on
`issue tracker on GitHub`_. Or, you can directly e-mail to `Chao-Ming Wang's Email`_.

If you have a very concrete feature proposal, add it to the `issue tracker on
GitHub`_:

* Explain how it would work, and link to a scientific paper if applicable.
* Keep the scope as narrow as possible, to make it easier to implement.

Fix bugs
--------

Look through the GitHub issues for bug reports. Anything tagged with "bug" is
open to whoever wants to implement it. If you discover a bug in NumpyDL you can
fix yourself, by all means feel free to just implement a fix and not report it
first.

Implement features
------------------

Look through the GitHub issues for feature proposals. Anything tagged with
"feature" or "enhancement" is open to whoever wants to implement it. If you
have a feature in mind you want to implement yourself, please note that Lasagne
has a fairly narrow focus and we strictly follow a set of :ref:`design
principles <numpydl-philosopy>`, so we cannot guarantee upfront that your code
will be included. Please do not hesitate to just propose your idea in a GitHub
issue first, so we can discuss it and/or guide you through the implementation.

Write documentation
-------------------

Whenever you find something not explained well, misleading, glossed over or
just wrong, please update it! The *Edit on GitHub* link on the top right of
every documentation page and the *[source]* link for every documented entity
in the API reference will help you to quickly locate the origin of any text.

Write tutorial
--------------

How to combine our Numpy code with live examples and detailed explanations
about deep learning is NumpyDL's ultimate goals. So, please contribute your
good ideas about how to make good tutorials.


How to contribute
=================

Edit on GitHub
--------------

As a very easy way of just fixing issues in the documentation, use the *Edit
on GitHub* link on the top right of a documentation page or the *[source]* link
of an entity in the API reference to open the corresponding source file in
GitHub, then click the *Edit this file* link to edit the file in your browser
and send us a Pull Request. All you need for this is a free GitHub account.

For any more substantial changes, please follow the steps below to setup
NumpyDL for development.

Development setup
-----------------

First, follow the instructions for performing a development installation of
NumpyDL (including forking on GitHub): :ref:`numpydl-development-install`

To be able to run the tests and build the documentation locally, install
additional requirements with: ``pip install -r requirements-dev.txt`` (adding
``--user`` if you want to install to your home directory instead).

Documentation
-------------

The documentation is generated with `Sphinx <http://sphinx-doc.org>`_. To
build it locally, run the following commands:

.. code:: bash

    python setup.py install
    cd docs
    make html

Afterwards, open ``docs/_build/html/index.html`` to view the documentation as
it would appear on `readthedocs <http://numpydl.readthedocs.org/>`_. If you
changed a lot and seem to get misleading error messages or warnings, run
``make clean html`` to force Sphinx to recreate all files from scratch.

When writing docstrings, follow existing documentation as much as possible to
ensure consistency throughout the library. For additional information on the
syntax and conventions used, please refer to the following documents:

* `reStructuredText Primer <http://sphinx-doc.org/rest.html>`_
* `Sphinx reST markup constructs <http://sphinx-doc.org/markup/index.html>`_
* `A Guide to NumPy/SciPy Documentation <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_


Testing
-------

NumpyDL has a code coverage of 100%, which has proven very helpful in the past,
but also creates some duties:

* Whenever you change any code, you should test whether it breaks existing
  features by just running the test suite. The test suite will also be run by
  `Travis <https://travis-ci.org/>`_ for any Pull Request to NumpyDL.
* Any code you add needs to be accompanied by tests ensuring that nobody else
  breaks it in future. `Coveralls <https://coveralls.io/>`_ will check whether
  the code coverage stays at 100% for any Pull Request to NumpyDL.
* Every bug you fix indicates a missing test case, so a proposed bug fix should
  come with a new test that fails without your fix.

To run the full test suite, just do

.. code:: bash

    py.test

Testing will take over several minutes for running for there are example testing.
It will end with a code coverage report specifying which code lines are not
covered by tests, if any. Furthermore, it will list any failed tests, and
failed `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ checks.

To only run tests matching a certain name pattern, use the ``-k`` command line
switch, e.g., ``-k pool`` will run the pooling layer tests only.

To land in a ``pdb`` debug prompt on a failure to inspect it more closely, use
the ``--pdb`` switch.

Finally, for a loop-on-failing mode, do ``pip install pytest-xdist`` and run
``py.test -f``. This will pause after the run, wait for any source file to
change and run all previously failing tests again.


Sending Pull Requests
---------------------

When you're satisfied with your addition, the tests pass and the documentation
looks good without any markup errors, commit your changes to a new branch, push
that branch to your fork and send us a Pull Request via GitHub's web interface.

All these steps are nicely explained on GitHub: https://guides.github.com/introduction/flow/

When filing your Pull Request, please include a description of what it does, to
help us reviewing it. If it is fixing an open issue, say, issue #123, add
*Fixes #123*, *Resolves #123* or *Closes #123* to the description text, so
GitHub will close it when your request is merged.



.. _issue tracker on GitHub: https://github.com/oujago/NumpyDL/issues
.. _Chao-Ming Wang's Email: oujago@gmail.com


