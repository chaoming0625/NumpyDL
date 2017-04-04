# -*- coding: utf-8 -*-


class Layer(object):
    first_layer = False

    def forward(self, input, *args, **kwargs):
        """ calculate layer output for given input (forward propagation). """
        raise NotImplementedError()

    def backward(self, pre_grad, *args, **kwargs):
        """ calculate the input gradient """
        raise NotImplementedError()

    def connect_to(self, prev_layer):
        """Init parameters"""
        raise NotImplementedError()

    def to_json(self):
        raise NotImplementedError()

    @classmethod
    def from_json(cls, config):
        return cls(**config)

    @property
    def params(self):
        """ layer parameters. """
        return []

    @property
    def grads(self):
        """ Get layer parameter gradients as calculated from backward(). """
        return []

    @property
    def param_grads(self):
        """ layer parameters and corresponding gradients. """
        return list(zip(self.params, self.grads))

