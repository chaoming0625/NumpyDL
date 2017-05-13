# -*- coding: utf-8 -*-


class Layer(object):
    """
    The :class:`Layer` class represents a single layer of a neural network. It
    should be subclassed when implementing new types of layers.

    Because each layer can keep track of the layer(s) feeding into it, a
    network's output :class:`Layer` instance can double as a handle to the full
    network.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    name : a string or None
        An optional name to attach to this layer.
    """

    first_layer = False

    def forward(self, input, *args, **kwargs):
        """ calculate layer output for given input (forward propagation). """
        raise NotImplementedError()

    def backward(self, pre_grad, *args, **kwargs):
        """ calculate the input gradient """
        raise NotImplementedError()

    def connect_to(self, prev_layer):
        """Propagates the given input through this layer (and only this layer).

        Parameters
        ----------
        input : Theano expression
            The expression to propagate through this layer.

        Returns
        -------
        output : Theano expression
            The output of this layer given the input to this layer.


        Notes
        -----
        This is called by the base :meth:`lasagne.layers.get_output()`
        to propagate data through a network.

        This method should be overridden when implementing a new
        :class:`Layer` class. By default it raises `NotImplementedError`.
        """
        raise NotImplementedError()

    def to_json(self):
        """ To configuration """
        raise NotImplementedError()

    @classmethod
    def from_json(cls, config):
        return cls(**config)

    @property
    def params(self):
        """ layer parameters. 
        
        Returns a list of Theano shared variables or expressions that
        parameterize the layer.

        By default, all shared variables that participate in the forward pass
        will be returned (in the order they were registered in the Layer's
        constructor via :meth:`add_param()`). The list can optionally be
        filtered by specifying tags as keyword arguments. For example,
        ``trainable=True`` will only return trainable parameters, and
        ``regularizable=True`` will only return parameters that can be
        regularized (e.g., by L2 decay).

        If any of the layer's parameters was set to a Theano expression instead
        of a shared variable, `unwrap_shared` controls whether to return the
        shared variables involved in that expression (``unwrap_shared=True``,
        the default), or the expression itself (``unwrap_shared=False``). In
        either case, tag filtering applies to the expressions, considering all
        variables within an expression to be tagged the same.

        Parameters
        ----------
        unwrap_shared : bool (default: True)
            Affects only parameters that were set to a Theano expression. If
            ``True`` the function returns the shared variables contained in
            the expression, otherwise the Theano expression itself.

        **tags (optional)
            tags can be specified to filter the list. Specifying ``tag1=True``
            will limit the list to parameters that are tagged with ``tag1``.
            Specifying ``tag1=False`` will limit the list to parameters that
            are not tagged with ``tag1``. Commonly used tags are
            ``regularizable`` and ``trainable``.

        Returns
        -------
        list of Theano shared variables or expressions
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
        """ layer parameters and corresponding gradients. """
        return list(zip(self.params, self.grads))

