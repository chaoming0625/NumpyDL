.. _installation:

============
Installation
============

NumpyDL has a couple of prerequisites that need to be installed first, but it
is not very picky about versions. The most important package is `Numpy
<https://github.com/numpy/numpy>`_. At the same time, you should install some
other useful packages, such as `scipy <https://github.com/scipy/scipy>`_ and
`scikit-learn <https://github.com/scikit-learn/scikit-learn>`_. Most importantly,
these packages are not required to install the specific version to fit the version
of NumpyDL you choose to install.

We strongly recommend you to install the `Miniconda <https://conda.io/miniconda.html>`_
or a bigger installer `Anaconda <https://www.continuum.io/downloads>`_ which is a
leading open data science platform powered by Python and well integrated the efficient
scientific computing platform `MKL <https://software.intel.com/en-us/forums/intel-math-kernel-library>`_.

Prerequisites
=============

Python + pip
------------

NumpyDL currently requires Python 3.3 or higher to run. Please install Python via
the package manager of your operating system if it is not included already.

Python includes ``pip`` for installing additional modules that are not shipped
with your operating system, or shipped in an old version, and we will make use
of it below. We recommend installing these modules into your home directory
via ``--user``, or into a `virtual environment
<http://www.dabapps.com/blog/introduction-to-pip-and-virtualenv-python/>`_
via ``virtualenv``.

C compiler
----------

Numpy/scipy require a C compiler if you install them via ``pip``. On Linux,
the default compiler is usually``gcc``, and on Mac OS, it's ``clang``. On
Windows, we recommend you to install the `Miniconda <https://conda.io/miniconda.html>`_
or `Anaconda <https://www.continuum.io/downloads>`_. Again, please install them via the
package manager of your operating system.

numpy/scipy + BLAS
------------------

NumpyDL requires numpy of version 1.6.2 or above, and sometimes also requires
scipy 0.11 or above. Numpy/scipy rely on a BLAS library to provide fast linear
algebra routines. They will work fine without one, but a lot slower, so it is
worth getting this right (but this is less important if you plan to use a GPU).

If you install numpy and scipy via your operating system's package manager,
they should link to the BLAS library installed in your system. If you install
numpy and scipy via ``pip install numpy`` and ``pip install scipy``, make sure
to have development headers for your BLAS library installed (e.g., the
``libopenblas-dev`` package on Debian/Ubuntu) while running the installation
command. Please refer to the `numpy/scipy build instructions
<http://www.scipy.org/scipylib/building/index.html>`_ if in doubt.

.. _numpydl-stable-release:


Stable NumpyDL release
======================

To install a version that is known to work, run the following command:

.. code-block:: bash

  pip install -r https://github.com/oujago/NumpyDL/blob/master/requirements.txt

.. code-block:: bash

  pip install npdl

If you do not use ``virtualenv``, add ``--user`` to both commands to install
into your home directory instead. To upgrade from an earlier installation, add
``--upgrade``.


.. _numpydl-development-install:


Development installation
========================

install from source
-------------------

Alternatively, you can install NumpyDL from source,
in a way that any changes to your local copy of the source tree take effect
without requiring a reinstall. This is often referred to as *editable* or
*development* mode. Firstly, you will need to obtain a copy of the source tree:

.. code-block:: bash

  git clone https://github.com/oujago/NumpyDL.git

It will be cloned to a subdirectory called ``NumpyDL``. Make sure to place it
in some permanent location, as for an *editable* installation, Python will
import the module directly from this directory and not copy over the files.
Enter the directory and install the known good version of Theano:

.. code-block:: bash

  cd NumpyDL
  pip install -r requirements.txt

To install the NumpyDL package itself, in editable mode, run:

.. code-block:: bash

  pip install --editable

As always, add ``--user`` to install it to your home directory instead.

contribute
----------

**Optional**: If you plan to contribute to NumpyDL, you will need to fork the
NumpyDL repository on GitHub. This will create a repository under your user
account. Update your local clone to refer to the official repository as
``upstream``, and your personal fork as ``origin``:

.. code-block:: bash

  git remote rename origin upstream
  git remote add origin https://github.com/<your-github-name>/NumpyDL.git

If you set up an `SSH key <https://help.github.com/categories/ssh/>`_, use the
SSH clone URL instead: ``git@github.com:<your-github-name>/NumpyDL.git``.

You can now use this installation to develop features and send us pull requests
on GitHub, see :doc:`development`!

