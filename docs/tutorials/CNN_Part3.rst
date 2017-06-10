CNN_Part3
====================

introduce
-------------------------------

Convolution.py is a convolution process, initialization, connection, forward propagation and back propagation are the four main functions. The specific convolution principle refers to Part1 and Part2. This article is mainly about the framework to achieve the convolution process code to explain.

init function
-------------------------------

code: ::

   def __init__(self, nb_filter, filter_size, input_shape=None, stride=1,
                 init=GlorotUniform(), activation=ReLU()):
        self.nb_filter = nb_filter
        self.filter_size = filter_size
        self.input_shape = input_shape
        self.stride = stride

        self.W, self.dW = None, None
        self.b, self.db = None, None
        self.out_shape = None
        self.last_output = None
        self.last_input = None

        self.init = init
        self.activation = activation

+---------+--------------------+
|    W    |       Weights      |
+---------+--------------------+
|    b    | Offset / threshold |
+---------+--------------------+

Initialization parameters, filters, filter size, input pictures, step size, activation function, and so on.

connect_to function
-------------------------------

code: ::

   def connect_to(self, prev_layer=None):
        if prev_layer is None:    
            assert self.input_shape is not None
            input_shape = self.input_shape
        else:
            input_shape = prev_layer.out_shape

        # input_shape: (batch size, num input feature maps, image height, image width)
        assert len(input_shape) == 4

        nb_batch, pre_nb_filter, pre_height, pre_width = input_shape
        filter_height, filter_width = self.filter_size

        height = (pre_height - filter_height) // self.stride + 1
        width = (pre_width - filter_width) // self.stride + 1

        # output shape
        self.out_shape = (nb_batch, self.nb_filter, height, width)

        # filters
        self.W = self.init((self.nb_filter, pre_nb_filter, filter_height, filter_width))
        self.b = Zero()((self.nb_filter,))
        
prev_layer: previous layer of neurons
connect_to(self, prev_layer=None): the purpose is to connect the next layer of neurons
 ::
 
       if prev_layer is None:    
            assert self.input_shape is not None
            input_shape = self.input_shape
        else:
            input_shape = prev_layer.out_shape
            
If there is no prev_layer neuron, and there is an input image, then the current self.input_shape is the input of neurons.
If there is a previous layer of neurons, then the input of the layer of neurons is equal to the output of previous layer.
 ::
 
   assert len(input_shape) == 4

To determine whether the format of the image is four-dimensional (batch size, num input feature maps, image height, image width).

 ::
 
       nb_batch, pre_nb_filter, pre_height, pre_width = input_shape
        filter_height, filter_width = self.filter_size
        height = (pre_height - filter_height) // self.stride + 1
        width = (pre_width - filter_width) // self.stride + 1

Define the output of the previous layer is the input of this layer, and give the filter size assignment
Also calculate the size of the activation map (height: height, width: width)
 ::
 
   self.out_shape = (nb_batch, self.nb_filter, height, width)
   
Current layer image output format

 ::
 
   self.W = self.init((self.nb_filter, pre_nb_filter, filter_height, filter_width))
   self.b = Zero()((self.nb_filter,))

Initialize the weight of the filter, and offset b. b is zero initialization

forward function
-------------------------------

code: ::

       def forward(self, input, *args, **kwargs):

        self.last_input = input

        # shape
        nb_batch, input_depth, old_img_h, old_img_w = input.shape
        filter_h, filter_w = self.filter_size
        new_img_h, new_img_w = self.out_shape[2:]

        # init
        outputs = Zero()((nb_batch, self.nb_filter, new_img_h, new_img_w))

        # convolution operation
        for x in np.arange(nb_batch):
            for y in np.arange(self.nb_filter):
                for h in np.arange(new_img_h):
                    for w in np.arange(new_img_w):
                        h_shift, w_shift = h * self.stride, w * self.stride
                        # patch: (input_depth, filter_h, filter_w)
                        patch = input[x, :, h_shift: h_shift + filter_h, w_shift: w_shift + filter_w]
                        outputs[x, y, h, w] = np.sum(patch * self.W[y]) + self.b[y]

        # nonlinear activation
        # self.last_output: (nb_batch, output_depth, image height, image width)
        self.last_output = self.activation.forward(outputs)

        return self.last_output
+------------------------------------------------------------+
|           forward(self, input, *args, **kwargs)            |
+==============+=============================================+
|   Objective  |   Forward propagation of neural networks    |
+--------------+---------------------------------------------+
| Return Value | The convolution output of the current layer |
+--------------+---------------------------------------------+
 ::
 
        nb_batch, input_depth, old_img_h, old_img_w = input.shape
        filter_h, filter_w = self.filter_size
        new_img_h, new_img_w = self.out_shape[2:]

For the current shape of the initialization, the output of the previous layer is the input of current layer

 ::
 
         for x in np.arange(nb_batch):
            for y in np.arange(self.nb_filter):
                for h in np.arange(new_img_h):
                    for w in np.arange(new_img_w):
                        h_shift, w_shift = h * self.stride, w * self.stride
                        # patch: (input_depth, filter_h, filter_w)
                        patch = input[x, :, h_shift: h_shift + filter_h, w_shift: w_shift + filter_w]
                        outputs[x, y, h, w] = np.sum(patch * self.W[y]) + self.b[y]

Convolution process:

+---------+--------------------+
|    w    |  new image width   |
+---------+--------------------+
|    h    |  new image height  |
+---------+--------------------+
|    y    |      filter        |
+---------+--------------------+
|    x    |     batch size     |
+---------+--------------------+
 ::
 
   h_shift, w_shift = h * self.stride, w * self.stride

Locate the current filter location

+----------------------+------------------------------------------+
|         patch        |  used to determine the feelings of wild  |
+----------------------+------------------------------------------+
|  outputs[x, y, h, w] |                Sum + offset              |
+----------------------+------------------------------------------+
 ::
 
   self.last_output = self.activation.forward(outputs)

The final output is the output of the convolution layer = Relu (Sum) + offset)

















-------------------------------

