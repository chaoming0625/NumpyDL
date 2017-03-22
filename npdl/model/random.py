# -*- coding: utf-8 -*-


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
