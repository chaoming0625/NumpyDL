# -*- coding: utf-8 -*-


from .base import Layer


class InputLayer(Layer):
    def __init__(self, n_in=None, shape=None,):
        if n_in:
            self.n_out = n_in
        elif shape:
            self.n_out = shape[-1]
        else:
            raise ValueError("Pls provide the input shape.")
