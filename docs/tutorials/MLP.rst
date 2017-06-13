=====================
Multilayer Perceptron
=====================

Sigmoid function
================

BP algorithm is mainly due to the emergence of Sigmoid function, instead of
the previous threshold function to construct neurons.

The Sigmoid function is a monotonically increasing nonlinear function. When the
threshold value is large enough, the threshold function can be approximated.


.. figure:: pics/mlp_1.jpg

The Sigmoid function is usually written in the following form:

.. math::   f(x) = \frac{1}{1 + e^{-x}}

The value range is :math:`(-1,1)`, which can be used instead of the neuron step function:

.. math::   f(x) = \frac{1}{1 + e^{- \sum_{i=1}^{n} w_i x_i-w_0}}

Due to the complexity of the network structure, the Sigmoid function is used as the
transfer function of the neuron. This is the basic idea of multilayer perceptron backpropagation algorithm.


Back Propagation
================

Back Propagation (BP) algorithm is the optimization of the network through the iterative weights makes the
actual mapping relationship between input and output and the desired mapping, descent algorithm by adjusting
the layer weights for the objective function to minimize the gradient. The sum of the squared error between
the predicted output and the expected output of the network on one or all training samples：

.. math::

    & J(w) = \frac{1}{2} \sum_{j=1}^{s} (t_j - s_j)^2 = \frac{1}{2} \mid t - a \mid ^ 2 \\
    & J_{total}(w) = \frac{1}{2} \sum_{i=1}^{N} \mid t_i - a_i \mid ^ 2

The error of each unit is calculated by layer by layer error of output layer:

.. math::

    \bigtriangledown w_j^k  & = - \eta \frac{\partial J}{\partial J w_j^k} \\
                            & = - \eta \frac{\partial J}{\partial n_j^k}\frac{\partial n_j^k}{\partial Jw_j^k} \\
                            & = -\eta \frac{\partial J}{\partial n_j^k} a^{k-1} \\
                            & = - \eta \delta_j^k a^{k-1}

Back Propagation Net (BPN) is a kind of multilayer network which is trained by weight of nonlinear
differentiable function. BP network is mainly used for:

1) function approximation and prediction analysis: using the input vector and the corresponding output vector to
   train a network to approximate a function or to predict the unknown information;
2) pattern recognition: using a specific output vector to associate it with the input vector;
3) classification: the input vector is defined in the appropriate manner;
4) data compression: reduce the output vector dimension to facilitate transmission and storage.

For example, a three tier BP structure is as follows:

.. figure:: pics/mlp_2.png
    :width: 80%

It consists of three layers: ``input`` layer, ``hidden`` layer and ``output`` layer. The unit of each layer
is connected with all the units of the adjacent layer, and there is no connection between the units in the
same layer. When a pair of learning samples are provided to the network, the activation value of the neuron
is transmitted from the input layer to the output layer through the intermediate layers, and the input
response of the network is obtained by the neurons in the output layer. Next, according to the direction
of reducing the output of the target and the direction of the actual error, the weights of each link are
modified from the output layer to the input layer.

 
Example
=======

Suppose you have such a network layer:

* The first layer is the input layer, two neurons containing :math:`i_1, i_2, b_1` and intercept;
* The second layer is the hidden layer, including two neurons :math:`h_1, h_2` and intercept b2;
* The third layer is the output of :math:`o_1, o_2` and :math:`w_i` are each line superscript connection
  weights between layers, we default to the activation function sigmoid function.

Now give them the initial value, as shown below:

.. figure:: pics/mlp_3.png
    :width: 80%

Among them,

* Input data: :math:`i_1=0.05, i_2=0.10`;
* Output data: :math:`o_1=0.01, o_2=0.99`;
* Initial weight: :math:`w_1=0.15, w_2=0.20, w_3=0.25, w_4=0.30, w_5=0.40, w_6=0.45, w_7=0.50, w_8=0.88`;

**Objective**: to give input data :math:`i_1, i_2` (0.05 and 0.10), so that the output is as close as
possible to the original output :math:`o_1, o_2` (0.01 and 0.99).


Step 1: Forward Propagation
---------------------------

``Input`` layer to ``Hidden`` layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculate the input weighted sum of neurons :math:`h_1`:

.. math::

    &   net_{h1} = w_1 * i_1 + w_2 * i_2 + b_i * 1 \\
    &   net_{h1} = 0.15 * 0.05 + 0.2 * 0.1 + 0.35 * 1 = 0.3775

:math:`o_1`, the output of neuron :math:`h_1`: (Activation function sigmoid is required here):

.. math::

    out_{h1} = \frac{1}{1 + e^{-net_{h1}}} = \frac{1}{1+e^{-0.3775}} = 0.593269992

Similarly, :math:`o_2`, the output of neuron :math:`h_2` can be calculated:

.. math::

    out_{h2} = 0.596884378

``Hidden`` layer to ``Output`` layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The values of :math:`o_1` and :math:`o_2` in the output layer are calculated:

.. math::

    & net_{o1} = w_5 * out_{h1} + w_6 * out_{h2} + b_2 * 1 \\
    & net_{o1} = 0.4 * 0.593269992 + 0.45 * 0.596884378 + 0.6 * 1 = 1.105905967 \\
    & out_{o1} = \frac{1}{1+e^{-net_{o1}}} = \frac{1}{1+e^{-1.105905967}} = 0.75136507 \\
    & out_{o2} = 0.772928465

This propagation process is finished, we get the output value of :math:`[0.75136079, 0.772928465]`,
and the actual value of :math:`[0.01, 0.99]` far from now, we for the error back-propagation,
update the weights, to calculate the output.


Step 2: Back Propagation
------------------------

Calculate the total error
^^^^^^^^^^^^^^^^^^^^^^^^^

Total error (square error):

.. math::

    E_{total} = \sum \frac{1}{2}(target - output) ^ 2

For example, the target output for :math:`o_1` is 0.01 but the neural network output 0.75136507,
therefore its error is:

.. math::

    E_{o1} = \frac{1}{2}(target_{o1} - out_{o1}) ^ 2 = \frac{1}{2} (0.01 - 0.75136507)^2 = 0.274811083

Repeating this process for :math:`o_2` (remembering that the target is 0.99) we get:

.. math::

    E_{o2} = 0.023560026

The total error for the neural network is the sum of these errors:

.. math::

    E_{total} = E_{o1} + E_{o2} = 0.274811083 + 0.023560026 = 0.298371109

``Hidden`` layer to ``Hidden`` layer weights update
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Take the weight parameter :math:`w_5` as an example, if we want to know how much impact
the :math:`w_5` has on the overall error, we can use the global error to obtain the partial
derivative of :math:`w_5`: (chain rule)

.. math::

    \frac{\partial E_{total}}{\partial w_5} = \frac{\partial E_{total}}{\partial out_{o1}} *
    \frac{\partial out_{o1}}{\partial net_{o1}} * \frac{\partial net_{o1}}{\partial w_5}

The following figure can be more intuitive to see how the error is spread back:

.. figure:: pics/mlp_4.png
    :width: 80%

Now we were calculated for each value:

* Calculate :math:`\frac{\partial E_{total}}{\partial out_{o1}}`.

.. math::

    & E_{total} = \frac{1}{2}(target_{o1} - out_{o1}) ^ 2 + \frac{1}{2}(target_{o2} - out_{o2}) ^ 2 \\
    & \frac{\partial E_{total}}{\partial out_{o1}} = 2 * \frac{1}{2}(target_{o1} - out_{o1})^{2-1} * -1 + 0 \\
    & \frac{\partial E_{total}}{\partial out_{o1}} = -(target_{o1} - out_{o1}) = -(0.01 - 0.75136507) = 0.74136507

* Calculate :math:`\frac{\partial out_{o1}}{\partial net_{o1}}`:

.. math::

    & out_{o1} = \frac{1}{1+e^{-net_{o1}}} \\
    & \frac{\partial out_{o1}}{\partial net_{o1}} = out_{o1} (1-out_{o1}) = 0.75136507(1-0.75136507) = 0.186815602


* Calculate :math:`\frac{\partial net_{o1}}{\partial w_5}`:

.. math::

    & net_{o1} = w_5 * out_{h1} + w_6 * out_{h2} + b_2 * 1 \\
    & \frac{\partial net_{o1}}{\partial w_5} = 1 * out_{h1} * w_5^{(1-1)} + 0 + 0 = out_{h1} = 0.593269992

* Putting it all together:

.. math::

    & \frac{\partial E_{total}}{\partial w_5} = \frac{\partial E_{total}}{\partial out_{o1}} *
      \frac{\partial out_{o1}}{\partial net_{o1}} * \frac{\partial net_{o1}}{\partial w_5} \\
    & \frac{\partial E_{total}}{\partial w_5} = 0.74136507 * 0.186815602 * 0.59326992 = 0.082167041

In this way, we calculate the overall error :math:`E_{total}` to the :math:`w_5` partial guide.
Look at the above formula, we found:

.. math::

    \frac{\partial E_{total}}{\partial w_5} = -(target_{o1} - out_{o1}) * out_{o1}(1-out_{o1}) * out_{h1}

In order to express convenience, :math:`\delta_{o1}` is used to express the error of output layer:

.. math::

    & \delta_{o1} = \frac{\partial E_{total}}{\partial out_{o1}} * \frac{\partial out_{o1}}{\partial net_{o1}} =
        \frac{\partial E_{total}}{\partial net_{o1}} \\
    & \delta_{o1} = - (target_{o1} - out_{o1}) * out_{o1} (1-out_{o1})

Therefore, the overall error :math:`E_{total}` can be written as a partial derivative formula for :math:`w_5`:

.. math:: \frac{\partial E_{total}}{\partial w_5} = \delta_{o1} out_{h1}

If the output layer error meter is negative, it can also be written:

.. math:: \frac{\partial E_{total}}{\partial w_5} = - \delta_{o1} out_{h1}

Finally, we update the value of :math:`w_5`:

.. math:: w_5^+ = w_5 - \eta * \frac{\partial E_{total}}{\partial w_5} = 0.4 - 0.5*0.082167041 = 0.35891648

Among them, :math:`\eta` is the learning rate, here we take 0.5. Similarly,
update :math:`w_6`, :math:`w_7`, :math:`w_8`:

.. math::

    & w_6^+ = 0.408666186 \\
    & w_7^+ = 0.511301270 \\
    & w_8^+ = 0.561370121


``Hidden`` layer to ``Input`` layer weights update
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In fact, with the method above said almost, but there is a need to change, calculate the total error
of the above :math:`w_5` guide, from :math:`out_{o1}` ----> :math:`net_{o1}` ----> :math:`w_5`, but
in the hidden layer between the weight update, :math:`out_{h1}` ----> :math:`net_{h1}` ----> :math:`w_1`
and :math:`out_{h1}` will accept :math:`E_{o1}` and :math:`E_{o2}` error of two places to two, so this
place will be calculated.

.. figure:: pics/mlp_5.png
    :width: 80%

* Calculate :math:`\frac{\partial E_{total}}{\partial out_{h1}}`:

.. math::

    \frac{\partial E_{total}}{\partial out_{h1}} =
    \frac{\partial E_{o1}}{\partial out_{h1}} + \frac{\partial E_{o2}}{\partial out_{h1}}

* Calculate :math:`\frac{\partial E_{o1}}{\partial out_{h1}}`:

.. math::

    & \frac{\partial E_{o1}}{\partial out_{h1}} = \frac{\partial E_{o1}}{\partial net_{o1}} *
        \frac{\partial net_{o1}}{\partial out_{h1}} \\
    & \frac{\partial E_{o1}}{\partial net_{o1}} = \frac{\partial E_{o1}}{\partial out_{o1}} *
        \frac{\partial net_{o1}}{\partial out_{h1}} = 0.74136507 * 0.186815602 = 0.138498562 \\
    & net_{o1} = w_5 * out_{h1} + w_6 * out_{h2} + b_2 * 1 \\
    & \frac{\partial net_{o1}}{\partial out_{h1}} = w_5 = 0.40 \\
    & \frac{\partial E_{o1}}{\partial out_{h1}} =\frac{\partial E_{o1}}{\partial net_{o1}} *
        \frac{\partial net_{o1}}{\partial out_{h1}} = 0.138498562 * 0.40 = 0.055399425

* Similarly, calculate :math:`\frac{\partial E_{o2}}{\partial out_{h1}} = -.019049119`:

* Therefore,

.. math::

    \frac{\partial E_{total}}{\partial out_{h1}} =
    \frac{\partial E_{o1}}{\partial out_{h1}} +
    \frac{\partial E_{o2}}{\partial out_{h1}} =
    0.055399425 + -.019049119 + 0.036350306

* Then, calculate :math:`\frac{\partial out_{h1}}{\partial net_{h1}}`:

.. math::

    & out_{h1} = \frac{1}{1+e^{-net_{h1}}} \\
    & \frac{\partial out_{h1}}{\partial net_{h1}} = out_{h1} (1-out_{h1}) = 0.241300709

* Calculate :math:`\frac{\partial net_{h1}}{\partial w_{h1}}`:

.. math::

    & net_{h1} = w_1 * i_1 + w_2 * i_2 + b_1 * 1 \\
    & \frac{\partial net_{h1}}{\partial w_{h1}} = i_1 = 0.05

Putting it all together:

.. math::

    \frac{\partial E_{total}}{\partial w_1} =
    \frac{\partial E_{total}}{\partial out_{h1}} *
    \frac{\partial out_{h1}}{\partial net_{h1}} *
    \frac{\partial net_{h1}}{\partial w_1} =
    0.036350306 * 0.241300709 * 0.05 = 0.000438568

In order to simplify the formula, :math:`\sigma_{h1}` is used to represent the error of the hidden layer
unit :math:`h_1`:

.. math::

    & \frac{\partial E_{total}}{\partial w_1}=
    (\sum_{o}\frac{\partial E_{total}}{\partial out_o} *
    \frac{\partial out_o}{\partial net_o} *
    \frac{\partial net_o}{\partial out_{h1}}) *
    \frac{\partial out_{h1}}{\partial net_{h1}} *
    \frac{\partial net_{h1}}{\partial w_1} \\
    & \frac{\partial E_{total}}{\partial w_1}=
    (\sum_o \delta_o * w_{ho}) * out_{h1} (1- out_{h1}) * i_1 \\
    & \frac{\partial E_{total}}{\partial w_1}= = \delta_{h1} i_1

We can now update :math:`w_1`:

.. math::

    w_1^+ = w_1 - \eta * \frac{\partial E_{total}}{\partial w_1} = 0.15 - 0.5 * 0.000438568 = 0.149780716

Repeating this for :math:`w_2`, :math:`w_3`, and :math:`w_4`:

.. math::

    & w_2^+ = 0.19956143 \\
    & w_3^+ = 0.24975114 \\
    & w_4^+ = 0.29950229

Finally, we’ve updated all of our weights! When we fed forward the 0.05 and 0.1 inputs originally, the
error on the network was 0.298371109. After this first round of back propagation, the total error is now
down to 0.291027924. It might not seem like much, but after repeating this process 10,000 times, for
example, the error plummets to 0.000035085. At this point, when we feed forward 0.05 and 0.1, the two
outputs neurons generate 0.015912196 (vs 0.01 target) and 0.984065734 (vs 0.99 target).

Code
====

First, import necessary packages:


.. literalinclude:: mlp_bp.py
    :start-after: import-start
    :end-before: import-end


Define ``network``:

.. literalinclude:: mlp_bp.py
    :start-after: network-start
    :end-before: network-end

Define ``layer``:

.. literalinclude:: mlp_bp.py
    :start-after: layer-start
    :end-before: layer-end

Define ``neuron``:
.. literalinclude:: mlp_bp.py
    :start-after: neuron-start
    :end-before: neuron-end

Put all together, and run example:

.. literalinclude:: mlp_bp.py
    :start-after: run-start
    :end-before: run-end


Please Enjoy!


.. [1] Wikipedia article on Backpropagation. http://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error
.. [2] Neural Networks for Machine Learning course on Coursera by Geoffrey Hinton. https://class.coursera.org/neuralnets-2012-001/lecture/39
.. [3] The Back Propagation Algorithm. https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf



