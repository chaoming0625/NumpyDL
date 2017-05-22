# -*- coding: utf-8 -*-


import numpy as np

_rng = np.random
_dtype = 'float32'


def get_rng():
    """Get the package-level random number generator.

    Returns
    -------
    :class:`numpy.random.RandomState` instance
        The :class:`numpy.random.RandomState` instance passed to the most
        recent call of :func:`set_rng`, or ``numpy.random`` if :func:`set_rng`
        has never been called.
    """
    return _rng


def set_rng(rng):
    """Set the package-level random number generator.
    
    Parameters
    ----------
    new_rng : ``numpy.random`` or a :class:`numpy.random.RandomState` instance
        The random number generator to use.
    """
    global _rng
    _rng = rng


def set_seed(seed):
    """Set numpy seed.

    Parameters
    ----------
    seed : int
    """
    global _rng
    _rng = np.random.RandomState(seed)


def get_dtype():
    """Get data dtype ``numpy.dtype``.

    Returns
    -------
    str or numpy.dtype
    """
    return _dtype


def set_dtype(dtype):
    """Set numpy dtype.
    
    Parameters
    ----------
    dtype : str or numpy.dtype
    """
    global _dtype
    _dtype = dtype
