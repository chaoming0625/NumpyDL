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
>>> model.add(npdl.layers.Dense(n_out=3, activation=npdl.activations.Softmax()))
>>> model.compile(loss=npdl.objectives.SCCE(), optimizer=npdl.optimizers.SGD(lr=0.005))

"""

import copy
import numpy as np


class Objective(object):
    """An objective function (or loss function, or optimization score 
    function) is one of the two parameters required to compile a model.
    
    """
    def forward(self, outputs, targets):
        """ Forward function.
        """
        raise NotImplementedError()

    def backward(self, outputs, targets):
        """Backward function.
        
        Parameters
        ----------
        outputs, targets : numpy.array 
            The arrays to compute the derivatives of them.
    
        Returns
        -------
        numpy.array 
            An array of derivative.
        """
        raise NotImplementedError()

    def __str__(self):
        return self.__class__.__name__


class MeanSquaredError(Objective):
    """Computes the element-wise squared difference between ``targets`` and ``outputs``.

    In statistics, the mean squared error (MSE) or mean squared deviation (MSD) of an 
    estimator (of a procedure for estimating an unobserved quantity) measures the 
    average of the squares of the errors or deviations—that is, the difference between 
    the estimator and what is estimated. MSE is a risk function, corresponding to the 
    expected value of the squared error loss or quadratic loss. The difference occurs 
    because of randomness or because the estimator doesn't account for information that 
    could produce a more accurate estimate. [1]_

    The MSE is a measure of the quality of an estimator—it is always non-negative, 
    and values closer to zero are better.
    
    The MSE is the second moment (about the origin) of the error, and thus incorporates 
    both the variance of the estimator and its bias. For an unbiased estimator, the MSE 
    is the variance of the estimator. Like the variance, MSE has the same units of 
    measurement as the square of the quantity being estimated. In an analogy to standard 
    deviation, taking the square root of MSE yields the root-mean-square error or 
    root-mean-square deviation (RMSE or RMSD), which has the same units as the quantity 
    being estimated; for an unbiased estimator, the RMSE is the square root of the 
    variance, known as the standard deviation.


    Notes
    -----
    This is the loss function of choice for many regression problems
    or auto-encoders with linear output units.
    
    References
    ----------
    
    .. [1] Lehmann, E. L.; Casella, George (1998). Theory of Point Estimation (2nd ed.). 
           New York: Springer. ISBN 0-387-98502-6. MR 1639875.
    """
    def forward(self, outputs, targets):
        """MeanSquaredError forward propagation. 
        
        .. math:: L = (p - t)^2
        
        Parameters
        ----------
        outputs, targets : numpy.array 
            The arrays to compute the squared difference between.
    
        Returns
        -------
        numpy.array 
            An expression for the element-wise squared difference.
        """
        return 0.5 * np.mean(np.sum(np.power(outputs - targets, 2), axis=1))

    def backward(self, outputs, targets):
        """MeanSquaredError backward propagation. 
        
        .. math:: dE = p - t
        
        Parameters
        ----------
        outputs, targets : numpy.array 
            The arrays to compute the derivative between them.
            
        Returns
        -------
        numpy.array 
            Derivative.
        """
        return outputs - targets


MSE = MeanSquaredError


class HellingerDistance(Objective):
    """Computes the multi-class hinge loss between predictions and targets.
    
    In probability and statistics, the Hellinger distance (closely related to, 
    although different from, the Bhattacharyya distance) is used to quantify
    the similarity between two probability distributions. It is a type of 
    f-divergence. The Hellinger distance is defined in terms of the Hellinger 
    integral, which was introduced by Ernst Hellinger in 1909.[1]_ [2]_
    

    Notes
    -----
    This is an alternative to the categorical cross-entropy loss for
    multi-class classification problems
    
    References
    ----------
    
    .. [1] Nikulin, M.S. (2001), "Hellinger distance", in Hazewinkel, Michiel, 
           Encyclopedia of Mathematics, Springer, ISBN 978-1-55608-010-4
    .. [2] Jump up ^ Hellinger, Ernst (1909), "Neue Begründung der Theorie 
           quadratischer Formen von unendlichvielen Veränderlichen", Journal 
           für die reine und angewandte Mathematik (in German), 136: 210–271, 
           doi:10.1515/crll.1909.136.210, JFM 40.0393.01

    """
    def forward(self, outputs, targets):
        """HellingerDistance forward propagation. 
        
        Parameters
        ----------
        outputs : numpy 2D array
            outputs in (0, 1), such as softmax output of a neural network,
            with data points in rows and class probabilities in columns.
        targets : numpy 2D array 
            Either a vector of int giving the correct class index per data point
            or a 2D tensor of one-hot encoding of the correct class in the same
            layout as predictions (non-binary targets in [0, 1] do not work!)
    
        Returns
        -------
        numpy 1D array
            An expression for the Hellinger Distance
        """
        root_difference = np.sqrt(outputs) - np.sqrt(targets)
        return np.mean(np.sum(np.power(root_difference, 2), axis=1) / np.sqrt(2))

    def backward(self, outputs, targets):
        """HellingerDistance forward propagation. 
        """
        root_difference = np.sqrt(outputs) - np.sqrt(targets)
        return root_difference / (np.sqrt(2) * np.sqrt(outputs))


HeD = HellingerDistance


class BinaryCrossEntropy(Objective):
    """Computes the binary cross-entropy between predictions and targets.
    
    Returns
    -------
    numpy array
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
        
        .. math:: L = -t \\log(p) - (1 - t) \\log(1 - p)
        
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
        """SoftmaxCategoricalCrossEntropy forward propagation.
        
        .. math:: L_i = - \\sum_j{t_{i,j} \\log(p_{i,j})}
        
        Parameters
        ----------
        outputs : numpy 2D array
            Predictions in (0, 1), such as softmax output of a neural network,
            with data points in rows and class probabilities in columns.
        targets : numpy 2D array 
            Either targets in [0, 1] matching the layout of `outputs`, or
            a vector of int giving the correct class index per data point.
    
        Returns
        -------
        numpy 1D array
            An expression for the item-wise categorical cross-entropy.
        """
        # outputs = np.clip(outputs, self.epsilon, 1 - self.epsilon)
        return np.mean(-np.sum(targets * np.log(outputs), axis=1))

    def backward(self, outputs, targets):
        """SoftmaxCategoricalCrossEntropy backward propagation.
        
        .. math::  dE = p - t
        
        Parameters
        ----------
        outputs : numpy 2D array
            Predictions in (0, 1), such as softmax output of a neural network,
            with data points in rows and class probabilities in columns.
        targets : numpy 2D array 
            Either targets in [0, 1] matching the layout of `outputs`, or
            a vector of int giving the correct class index per data point.
    
        Returns
        -------
        numpy 1D array
        """
        # outputs = np.clip(outputs, self.epsilon, 1 - self.epsilon)
        return outputs - targets


SCCE = SoftmaxCategoricalCrossEntropy


def get(objective):
    if objective.__class__.__name__ == 'str':
        if objective in ['mse', 'MSE']:
            return MSE()
        if objective in ['mean_squared_error', 'MeanSquaredError']:
            return MeanSquaredError()
        if objective in ['hellinger_distance', 'HellingerDistance']:
            return HellingerDistance()
        if objective in ['hed', 'HeD']:
            return HeD()
        if objective in ['binary_cross_entropy', 'BinaryCrossEntropy']:
            return BinaryCrossEntropy()
        if objective in ['bce', 'BCE']:
            return BCE()
        if objective in ['softmax_categorical_cross_entropy', 'SoftmaxCategoricalCrossEntropy']:
            return SoftmaxCategoricalCrossEntropy()
        if objective in ['scce', 'SCCE']:
            return SCCE()
        raise ValueError('Unknown objective name: {}.'.format(objective))

    elif isinstance(objective, Objective):
        return copy.deepcopy(objective)

    else:
        raise ValueError("Unknown type: {}.".format(objective.__class__.__name__))


