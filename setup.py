# -*- coding: utf-8 -*-

"""
@author: ChaoMing (https://oujago.github.io/)

@date: Created on 17-1-10

@notes:
    just run "python setup.py build_ext --inplace" in command
"""

from distutils.core import setup

import numpy as np
from Cython.Build import cythonize

setup(
    name='pydl',
    ext_modules=cythonize(
        [
            'pydl/backend/conv.pyx',
            'pydl/backend/pool.pyx'
        ]
    ),
    include_dirs=[np.get_include()]
)
