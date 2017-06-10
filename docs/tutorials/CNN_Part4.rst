CNN_Part4
====================

introduce
-------------------------------

Pooling.py is the pooling process, initialization, connection, forward propagation and back propagation are the main four functions. Specific pooling principle refers to Part1 and Part2. This article is mainly on the framework to achieve the average pool and the maximum pool of the two ways to explain the code.

MeanPooling class
====================

Purpose: Caculate mean pooling

init function
-------------------------------

code: ::

    def __init__(self, pool_size):
        self.pool_size = pool_size

        self.out_shape = 0
        self.out_shape = None
        self.input_shape = None
        
Initialize pool size. 
        
connect_to function
-------------------------------

code: ::

    def connect_to(self, prev_layer):
        assert 5 > len(prev_layer.out_shape) >= 3

        old_h, old_w = prev_layer.out_shape[-2:]
        pool_h, pool_w = self.pool_size
        new_h, new_w = old_h // pool_h, old_w // pool_w

        assert old_h % pool_h == old_w % pool_w == 0

        self.out_shape = prev_layer.out_shape[:-2] + (new_h, new_w)

Connect the layer.If the input image meets the format requirements, (four dimensions), the output of the previous layer mapping length and width assignment to old_h, old_w. Then the output of the previous layer is the input of this layer.

forward function
-------------------------------

The process of neuron 's forward propagation in the process of pooling

code: ::

    def forward(self, input, *args, **kwargs):

        # shape
        self.input_shape = input.shape
        pool_h, pool_w = self.pool_size
        new_h, new_w = self.out_shape[-2:]

Initialize shape, enter image size, the size of pool, and output image format

 ::
 
        # forward
        outputs = _zero(self.input_shape[:-2] + self.out_shape[-2:])

        if np.ndim(input) == 4:
            nb_batch, nb_axis, _, _ = input.shape

            for a in np.arange(nb_batch):
                for b in np.arange(nb_axis):
                    for h in np.arange(new_h):
                        for w in np.arange(new_w):
                            outputs[a, b, h, w] = np.mean(input[a, b, h:h + pool_h, w:w + pool_w])
                            

+---------+----------------+
|    W    |   Picture wide |
+---------+----------------+
|    h    |   Picture high |
+---------+----------------+
|    b    |     axis       |
+---------+----------------+
|    a    |   batch size   |
+---------+----------------+
If the image format is 4 dimensions, the output of the mean pooling is np.mean (input [a, b, h: h + pool_h, w: w + pool_w])     

 ::
 
        elif np.ndim(input) == 3:
            nb_batch, _, _ = input.shape

            for a in np.arange(nb_batch):
                for h in np.arange(new_h):
                    for w in np.arange(new_w):
                        outputs[a, h, w] = np.mean(input[a, h:h + pool_h, w:w + pool_w])

If the image format is 3-dimensional, the output of the mean pooling is np.mean (input [a, h: h + pool_h, w: w + pool_w])

 ::
 
        else:
            raise ValueError()

        return outputs

backward function
-------------------------------

The reverse propagation of the pooling process

code: ::

    def backward(self, pre_grad, *args, **kwargs):
        new_h, new_w = self.out_shape[-2:]
        pool_h, pool_w = self.pool_size
        length = np.prod(self.pool_size)

        layer_grads = _zero(self.input_shape)

        if np.ndim(pre_grad) == 4:
            nb_batch, nb_axis, _, _ = pre_grad.shape

            for a in np.arange(nb_batch):
                for b in np.arange(nb_axis):
                    for h in np.arange(new_h):
                        for w in np.arange(new_w):
                            h_shift, w_shift = h * pool_h, w * pool_w
                            layer_grads[a, b, h_shift: h_shift+pool_h, w_shift: w_shift+pool_w] = \
                                pre_grad[a, b, h, w] / length


If the picture format is 4 dimensions,layer_grads = pre_grad[a, b, h, w] / length        

 ::

        elif np.ndim(pre_grad) == 3:
            nb_batch, _, _ = pre_grad.shape

            for a in np.arange(nb_batch):
                for h in np.arange(new_h):
                    for w in np.arange(new_w):
                        h_shift, w_shift = h * pool_h, w * pool_w
                        layer_grads[a, h_shift: h_shift+pool_h, w_shift: w_shift+pool_w] = \
                            pre_grad[a, h, w] / length


If the picture format is 3 dimensions,layer_grads = pre_grad[a, h, w] / length

 ::
 
        else:
            raise ValueError()

        return layer_grads


MaxPooling class
====================

Maxpooling and Meanpooling, the only difference is that the calculation is different. No repeat the explanation here.

init function
-------------------------------

code: ::

      def __init__(self, pool_size):
        self.pool_size = pool_size

        self.input_shape = None
        self.out_shape = None
        self.last_input = None


connect_to function
-------------------------------

code: ::

      def connect_to(self, prev_layer):
        # prev_layer.out_shape: (nb_batch, ..., height, width)
        assert len(prev_layer.out_shape) >= 3

        old_h, old_w = prev_layer.out_shape[-2:]
        pool_h, pool_w = self.pool_size
        new_h, new_w = old_h // pool_h, old_w // pool_w

        assert old_h % pool_h == old_w % pool_w == 0

        self.out_shape = prev_layer.out_shape[:-2] + (new_h, new_w)


forward function
-------------------------------

code: ::

      def forward(self, input, *args, **kwargs):
        # shape
        self.input_shape = input.shape
        pool_h, pool_w = self.pool_size
        new_h, new_w = self.out_shape[-2:]

        # forward
        self.last_input = input
        outputs = _zero(self.input_shape[:-2] + self.out_shape[-2:])

        if np.ndim(input) == 4:
            nb_batch, nb_axis, _, _ = input.shape

            for a in np.arange(nb_batch):
                for b in np.arange(nb_axis):
                    for h in np.arange(new_h):
                        for w in np.arange(new_w):
                            outputs[a, b, h, w] = np.max(input[a, b, h:h + pool_h, w:w + pool_w])

        elif np.ndim(input) == 3:
            nb_batch, _, _ = input.shape

            for a in np.arange(nb_batch):
                for h in np.arange(new_h):
                    for w in np.arange(new_w):
                        outputs[a, h, w] = np.max(input[a, h:h + pool_h, w:w + pool_w])

        else:
            raise ValueError()

        return outputs

backward function
-------------------------------

code: ::

      def backward(self, pre_grad, *args, **kwargs):
        new_h, new_w = self.out_shape[-2:]
        pool_h, pool_w = self.pool_size

        layer_grads = _zero(self.input_shape)

        if np.ndim(pre_grad) == 4:
            nb_batch, nb_axis, _, _ = pre_grad.shape

            for a in np.arange(nb_batch):
                for b in np.arange(nb_axis):
                    for h in np.arange(new_h):
                        for w in np.arange(new_w):
                            patch = input[a, b, h:h + pool_h, w:w + pool_w]
                            max_idx = np.unravel_index(patch.argmax(), patch.shape)
                            h_shift, w_shift = h * pool_h + max_idx[0], w * pool_w + max_idx[1]
                            layer_grads[a, b, h_shift, w_shift] = pre_grad[a, b, a, w]

        elif np.ndim(pre_grad) == 3:
            nb_batch, _, _ = pre_grad.shape

            for a in np.arange(nb_batch):
                for h in np.arange(new_h):
                    for w in np.arange(new_w):
                        patch = input[a, h:h + pool_h, w:w + pool_w]
                        max_idx = np.unravel_index(patch.argmax(), patch.shape)
                        h_shift, w_shift = h * pool_h + max_idx[0], w * pool_w + max_idx[1]
                        layer_grads[a, h_shift, w_shift] = pre_grad[a, a, w]

        else:
            raise ValueError()

        return layer_grads


