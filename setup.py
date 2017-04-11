# -*- coding: utf-8 -*-

"""
@author: Chao-Ming Wang (https://oujago.github.io/)

@notes:
    run "python setup.py build_ext --inplace" in command
"""

from distutils.core import setup

import numpy as np
from setuptools import find_packages

version = '0.1.0'

setup(
    name='npdl',
    version=version,
    description="Deep Learning Library based on pure Numpy",
    author='Chao-Ming Wang',
    packages=find_packages(),
    author_email='oujago@gmail.com',
    url='http://numpydl.readthedocs.io',
    include_dirs=[np.get_include()],
    install_requires=[
        'numpy',
        'scikit-learn',
        'scipy',
        'sphinx',
        'numpydoc',
    ]
)
