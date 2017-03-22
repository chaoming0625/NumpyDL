# -*- coding: utf-8 -*-


import numpy as np

from .base import Layer

from ..initialization import One
from ..initialization import Zero


class BatchNormal(Layer):
    def __init__(self, n_in, epsilon=1e-6, momentum=0.9, axis=0,
                 beta_init=Zero(), gamma_init=One()):
        self.n_in = n_in
        self.epsilon = 1e-6
        self.momentum = momentum
        self.axis = axis



