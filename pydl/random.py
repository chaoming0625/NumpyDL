# -*- coding: utf-8 -*-

"""
@author: ChaoMing (https://oujago.github.io/)

@date: Created on 17-1-9

@notes:
    
"""

import numpy as np

_rng = np.random


def get_rng():
    return _rng


def set_rng(rng):
    global _rng
    _rng = rng


def set_seed(seed):
    global _rng
    _rng = np.random.RandomState(seed)
