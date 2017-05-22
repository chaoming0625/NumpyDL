# -*- coding: utf-8 -*-


import numpy as np


def one_hot(labels, nb_classes=None):
    """One-hot encoding is often used for indicating the 
    state of a state machine. When using binary or Gray code, 
    a decoder is needed to determine the state. A one-hot state 
    machine, however, does not need a decoder as the state 
    machine is in the nth state if and only if the nth bit is 
    high.

    A ring counter with 15 sequentially-ordered states is an 
    example of a state machine. A ``one-hot`` implementation would
    have 15 flip flops chained in series with the Q output of 
    each flip flop connected to the D input of the next and the 
    D input of the first flip flop connected to the Q output of 
    the 15th flip flop. The first flip flop in the chain represents 
    the first state, the second represents the second state, and 
    so on to the 15th flip flop which represents the last state. 
    Upon reset of the state machine all of the flip flops are reset 
    to ``0`` except the first in the chain which is set to ``1``. 
    The next clock edge arriving at the flip flops advances the 
    one ``hot`` bit to the second flip flop. The ``hot`` bit 
    advances in this way until the 15th state, after which the 
    state machine returns to the first state.
    
    An address decoder converts from binary or gray code to 
    one-hot representation. A priority encoder converts from 
    one-hot representation to binary or gray code.
    
    In natural language processing, a one-hot vector is a :math:`1 × N` 
    matrix (vector) used to distinguish each word in a vocabulary 
    from every other word in the vocabulary. The vector consists
    of 0s in all cells with the exception of a single 1 in a cell
    used uniquely to identify the word.
    
    Parameters
    ----------
    labels ： iterable
        
    nb_classes : (iterable, optional)
    
    Returns
    -------
    numpy.array
        Returns a one-hot numpy array.
    
    """
    classes = np.unique(labels)
    if nb_classes is None:
        nb_classes = classes.size
    one_hot_labels = np.zeros((labels.shape[0], nb_classes))
    for i, c in enumerate(classes):
        one_hot_labels[labels == c, i] = 1
    return one_hot_labels


def unhot(one_hot_labels):
    """Get argmax indexes.
    
    Parameters
    ----------
    one_hot_labels : numpy.array
    
    Returns
    -------
    numpy.array
        Returns a unhot numpy array.
    """
    return np.argmax(one_hot_labels, axis=-1)


