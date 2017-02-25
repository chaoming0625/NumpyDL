# -*- coding: utf-8 -*-

"""
@author: ChaoMing (https://oujago.github.io/)

@date: Created on 17-1-10

@notes:
    
"""

import numpy as np

from .base import Layer

from ..init import One
from ..init import Zero


class BatchNormal(Layer):
    def __init__(self, n_in, epsilon=1e-6, momentum=0.9, axis=0,
                 beta_init=Zero(), gamma_init=One()):
        self.n_in = n_in
        self.epsilon = 1e-6
        self.momentum = momentum
        self.axis = axis



