# -*- coding: utf-8 -*-


class Layer(object):
    """
    The :class:`Layer` class represents a single layer of a neural network. It
    should be subclassed when implementing new types of layers.

    Because each layer can keep track of the layer(s) feeding into it, a
    network's output :class:`Layer` instance can double as a handle to the full
    network.

    """

    first_layer = False

    def forward(self, input, *args, **kwargs):
        """ Calculate layer output for given input (forward propagation). """
        raise NotImplementedError

    def backward(self, pre_grad, *args, **kwargs):
        """ calculate the input gradient """
        raise NotImplementedError

    def connect_to(self, prev_layer):
        """Propagates the given input through this layer (and only this layer).

        Parameters
        ----------
        prev_layer : previous layer
            The previous layer to propagate through this layer.

        """
        raise NotImplementedError

    def to_json(self):
        """ To configuration """
        raise NotImplementedError

    @classmethod
    def from_json(cls, config):
        """ From configuration """
        return cls(**config)

    @property
    def params(self):
        """ Layer parameters. 
        
        Returns a list of numpy.array variables or expressions that
        parameterize the layer.

        Returns
        -------
        list of numpy.array variables or expressions
            A list of variables that parameterize the layer

        Notes
        -----
        For layers without any parameters, this will return an empty list.
        """
        return []

    @property
    def grads(self):
        """ Get layer parameter gradients as calculated from backward(). """
        return []

    @property
    def param_grads(self):
        """ Layer parameters and corresponding gradients. """
        return list(zip(self.params, self.grads))

    def __str__(self):
        return self.__class__.__name__
