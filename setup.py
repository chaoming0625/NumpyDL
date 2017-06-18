# -*- coding: utf-8 -*-

import io
import os
import re

import numpy as np
from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

try:
    # obtain version string from __init__.py
    with open(os.path.join(here, 'npdl', '__init__.py'), 'r') as f:
        init_py = f.read()
    version = re.search('__version__ = "(.*)"', init_py).groups()[0]
except Exception:
    version = ''

try:
    # obtain long description from README and CHANGES
    with io.open(os.path.join(here, 'README.rst'), 'r', encoding='utf-8') as f:
        README = f.read()
    with io.open(os.path.join(here, 'CHANGES.rst'), 'r', encoding='utf-8') as f:
        CHANGES = f.read()
except IOError:
    README = CHANGES = ''


setup(
    name='npdl',
    version=version,
    description="Deep Learning Library based on pure Numpy",
    long_description="\n\n".join([README, CHANGES]),
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    keywords="deep learning library",
    license='MIT',
    author='Chao-Ming Wang',
    packages=find_packages(),
    author_email='oujago@gmail.com',
    url='http://numpydl.readthedocs.io',
    download_url="https://github.com/oujago/NumpyDL",
    include_dirs=[np.get_include()],
    install_requires=['numpy', ],
    extras_require={
        "test": ['pytest', 'coverage', 'python-coveralls', ],
        'doc': ['sphinx', 'numpydoc', 'jinjia2', 'sphinx_rtd_theme'],
        'example': ['scikit-learn', 'flask'],
    }
)
