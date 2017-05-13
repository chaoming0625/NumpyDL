# -*- coding: utf-8 -*-

"""
Provides some minimal help with building loss expressions for training or
validating a neural network.

These functions build element- or item-wise loss expressions from network
predictions and targets.

Examples
--------
Assuming you have a simple neural network for 3-way classification:

>>> import npdl
>>> model = npdl.model.Model()
>>> model.add(npdl.layers.Dense(n_out=100, n_in=50))
>>> model.add(npdl.layers.Dense(n_out=3, 
>>>           activation=npdl.activation.Softmax()))
>>> model.compile(loss=npdl.objectives.SCCE(), 
>>>           optimizer=npdl.optimizers.SGD(lr=0.005))

"""

import numpy as np


class Objective(object):
    """An objective function (or loss function, or optimization score 
    function) is one of the two parameters required to compile a model.
    
    """
    def forward(self, outputs, targets):
        raise NotImplementedError()

    def backward(self, outputs, targets):
        raise NotImplementedError()


class MeanSquaredError(Objective):
    """Computes the element-wise squared difference between two tensors.

    .. math:: L = (p - t)^2

    Parameters
    ----------
    a, b : Theano tensor
        The tensors to compute the squared difference between.

    Returns
    -------
    Theano tensor
        An expression for the element-wise squared difference.

    Notes
    -----
    This is the loss function of choice for many regression problems
    or auto-encoders with linear output units.
    """
    def forward(self, outputs, targets):
        return 0.5 * np.mean(np.sum(np.power(outputs - targets, 2), axis=1))

    def backward(self, outputs, targets):
        return outputs - targets


MSE = MeanSquaredError


class HellingerDistance(Objective):
    """Computes the multi-class hinge loss between predictions and targets.

    .. math:: L_i = \\max_{j \\not = p_i} (0, t_j - t_{p_i} + \\delta)

    Parameters
    ----------
    predictions : Theano 2D tensor
        Predictions in (0, 1), such as softmax output of a neural network,
        with data points in rows and class probabilities in columns.
    targets : Theano 2D tensor or 1D tensor
        Either a vector of int giving the correct class index per data point
        or a 2D tensor of one-hot encoding of the correct class in the same
        layout as predictions (non-binary targets in [0, 1] do not work!)
    delta : scalar, default 1
        The hinge loss margin

    Returns
    -------
    Theano 1D tensor
        An expression for the item-wise multi-class hinge loss

    Notes
    -----
    This is an alternative to the categorical cross-entropy loss for
    multi-class classification problems
    """
    def forward(self, outputs, targets):
        root_difference = np.sqrt(outputs) - np.sqrt(targets)
        return np.mean(np.sum(np.power(root_difference, 2), axis=1) / np.sqrt(2))

    def backward(self, outputs, targets):
        root_difference = np.sqrt(outputs) - np.sqrt(targets)
        return root_difference / (np.sqrt(2) * np.sqrt(outputs))


HeD = HellingerDistance


class BinaryCrossEntropy(Objective):
    """Computes the binary cross-entropy between predictions and targets.
    
    .. math:: L = -t \\log(p) - (1 - t) \\log(1 - p)
    
    Returns
    -------
    Theano tensor
        An expression for the element-wise binary cross-entropy.

    Notes
    -----
    This is the loss function of choice for binary classification problems
    and sigmoid output units.
    
    """
    def __init__(self, epsilon=1e-11):
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        """Forward pass.
        
        Parameters
        ----------
        outputs : numpy.array
            Predictions in (0, 1), such as sigmoidal output of a neural network.
        targets : numpy.array
            Targets in [0, 1], such as ground truth labels.

        """
        # outputs = np.clip(outputs, self.epsilon, 1 - self.epsilon)
        return np.mean(-np.sum(targets * np.log(outputs) + (1 - targets) * np.log(1 - outputs), axis=1))

    def backward(self, outputs, targets):
        """Backward pass.

        Parameters
        ----------
        outputs : numpy.array
            Predictions in (0, 1), such as sigmoidal output of a neural network.
        targets : numpy.array
            Targets in [0, 1], such as ground truth labels.

        """
        # outputs = np.clip(outputs, self.epsilon, 1 - self.epsilon)
        divisor = np.maximum(outputs * (1 - outputs), self.epsilon)
        return (outputs - targets) / divisor


BCE = BinaryCrossEntropy


class SoftmaxCategoricalCrossEntropy(Objective):
    """Computes the categorical cross-entropy between predictions and targets.

    .. math:: L_i = - \\sum_j{t_{i,j} \\log(p_{i,j})}

    Parameters
    ----------
    predictions : Theano 2D tensor
        Predictions in (0, 1), such as softmax output of a neural network,
        with data points in rows and class probabilities in columns.
    targets : Theano 2D tensor or 1D tensor
        Either targets in [0, 1] matching the layout of `predictions`, or
        a vector of int giving the correct class index per data point.

    Returns
    -------
    Theano 1D tensor
        An expression for the item-wise categorical cross-entropy.

    Notes
    -----
    This is the loss function of choice for multi-class classification
    problems and softmax output units. For hard targets, i.e., targets
    that assign all of the probability to a single class per data point,
    providing a vector of int for the targets is usually slightly more
    efficient than providing a matrix with a single 1.0 per row.
    """
    def __init__(self, epsilon=1e-11):
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        # outputs = np.clip(outputs, self.epsilon, 1 - self.epsilon)
        return np.mean(-np.sum(targets * np.log(outputs), axis=1))

    def backward(self, outputs, targets):
        # outputs = np.clip(outputs, self.epsilon, 1 - self.epsilon)
        return outputs - targets


SCCE = SoftmaxCategoricalCrossEntropy

