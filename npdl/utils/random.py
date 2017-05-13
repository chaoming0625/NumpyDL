# -*- coding: utf-8 -*-


import numpy as np

_rng = np.random
_dtype = 'float32'


def get_rng():
    return _rng


def set_rng(rng):
    global _rng
    _rng = rng


def set_seed(seed):
    global _rng
    _rng = np.random.RandomState(seed)


def get_dtype():
    return _dtype


def set_dtype(dtype):
    global _dtype
    _dtype = dtype
