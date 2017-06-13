==============
Initialization
==============


Introduction
============

As we all know, the solution to a non-convex optimization algorithm (like stochastic gradient descent)
depends on the initial values of the parameters. This post is about choosing initialization parameters
for deep networks and how it affects the convergence. We will also discuss the related topic of
vanishing gradients.


First, let’s go back to the time of sigmoidal activation functions and initialization of parameters
using IID Gaussian or uniform distributions with fairly arbitrarily set variances. Building deep
networks was difficult because of exploding or vanishing activations and gradients. Let’s take
activations first: If all your parameters are too small, the variance of your activations will drop in
each layer. This is a problem if your activation function is sigmoidal, since it is approximately
linear close to 0. That is, you gradually lose your non-linearity, which means there is no benefit to
having multiple layers. If, on the other hand, your activations become larger and larger, then your
activations will saturate and become meaningless, with gradients approaching 0.


.. figure:: pics/init_0.svg

Let us consider one layer and forget about the bias. Note that the following analysis and conclusion
is taken from Glorot and Bengio [1]_. Consider a weight matrix :math:`W \in R^{m×n}`, where each element
was drawn from an IID Guassian with variance :math:`Var(W)`. Note that we are a bit abusive with notation
letting :math:`W` denote both a matrix and a univariate random variable. We also assume there is no
correlation between our input and our weights and both are zero-mean. If we consider one filter (row)
in :math:`W`, say :math:`W` (a random vector), then the variance of the output signal over the input
signal is:

.. math::

    \frac{Var(W^{T}x)}{Var(x)} =
    \frac{\sum_{n}^{m}Var(W_{n}x_{n})}{Var(x)} =
    \frac{nVar(W)Var(x)}{Var(x)} =
    nVar(W)


As we build a deep network, we want the variance of the signal going forward in the network to remain
the same, thus it would be advantageous if :math:`nVar(W)=1`. The same argument can be made for the
gradients, the signal going backward in the network, and the conclusion is that we would also
like :math:`mVar(W)=1`. Unless :math:`n=m`, it is impossible to sastify both of these conditions. In
practice, it works well if both are approximately satisfied. One thing that has never been clear to me
is why it is only necessary to satisfy these conditions when picking the initialization values of :math:`W`.
It would seem that we have no guarantee that the conditions will remain true as the network is trained.

Nevertheless, this *Xavier initialization* (after Glorot’s first name) is a neat trick that works well
in practice. However, along came rectified linear units (ReLU), a non-linearity that is scale-invariant
around 0 and does not saturate at large input values. This seemingly solved both of the problems the
sigmoid function had; or were they just alleviated? I am unsure of how widely used Xavier initialization
is, but if it is not, perhaps it is because ReLU seemingly eliminated this problem.


However, take the most competative network as of recently, VGG [2]_. They do not use this kind of
initialization, although they report that it was tricky to get their networks to converge. They say that
they first trained their most shallow architecture and then used that to help initialize the second one,
and so forth. They presented 6 networks, so it seems like an awfully complicated training process to
get to the deepest one.

A recent paper by He et al. [3]_ presents a pretty straightforward generalization of ReLU and Leaky ReLU.
What is more interesting is their emphasis on the benefits of Xavier initialization even for ReLU. They
re-did the derivations for ReLUs and discovered that the conditions were the same up to a factor 2.
The difficulty Simonyan and Zisserman had training VGG is apparently avoidable, simply by using Xavier
intialization (or better yet the ReLU adjusted version). Using this technique, He et al. reportedly trained
a whopping 30-layer deep network to convergence in one go.

Another recent paper tackling the signal scaling problem is by Ioffe and Szegedy [4]_. They call the change
in scale internal covariate shift and claim this forces learning rates to be unnecessarily small. They
suggest that if all layers have the same scale and remain so throughout training, a much higher learning
rate becomes practically viable. You cannot just standardize the signals, since you would lose expressive
power (the bias disappears and in the case of sigmoids we would be constrained to the linear regime).
They solve this by re-introducing two parameters per layer, scaling and bias, added again after
standardization. The training reportedly becomes about 6 times faster and they present state-of-the-art
results on ImageNet. However, I’m not certain this is the solution that will stick.

I reckon we will see a lot more work on this frontier in the next few years. Especially since it also
relates to the – right now wildly popular – Recurrent Neural Network (RNN), which connects output signals
back as inputs. The way you train such network is that you unroll the time axis, treating the result as an
extremely deep feedforward network. This greatly exacerbates the vanishing gradient problem. A popular
solution, called Long Short-Term Memory (LSTM), is to introduce memory cells, which are a type of teleport
that allows a signal to jump ahead many time steps. This means that the gradient is retained for all those
time steps and can be propagated back to a much earlier time without vanishing.



Xavier Initialization
========================

Why’s Xavier initialization important?
--------------------------------------

In short, it helps signals reach deep into the network.

* If the weights in a network start too small, then the signal shrinks as it passes through each layer until
  it’s too tiny to be useful.
* If the weights in a network start too large, then the signal grows as it passes through each layer until
  it’s too massive to be useful.

Xavier initialization makes sure the weights are ‘just right’, keeping the signal in a reasonable range of
values through many layers.

To go any further than this, you’re going to need a small amount of statistics - specifically you need to
know about random distributions and their variance.

What’s Xavier initialization?
-----------------------------

For specific implementation, it’s initializing the weights in your network by drawing them from a distribution
with zero mean and a specific variance,

.. math::

    Var(W) = \frac{1}{n_{in}}

where :math:`W` is the initialization distribution for the neuron in question, and :math:`n_{in}` is the
number of neurons feeding into it. The distribution used is typically Gaussian or uniform.

It’s worth mentioning that Glorot & Bengio’s paper [1]_ originally recommended using:

.. math::

    Var(W) = \frac{2}{n_{in} + n_{out}}

where :math:`n_{out}` is the number of neurons the result is fed to.


Where did those formulas come from?
-----------------------------------

Suppose we have an input :math:`X` with :math:`n` components and a linear neuron with random weights :math:`W`
that spits out a number :math:`Y`. What’s the variance of :math:`Y`? Well, we can write

.. math::

    Y=W_1 X_1+W_2 X_2+⋯+W_n X_n

And from Wikipedia [5]_ we can work out that :math:`W_iX_i` is going to have variance

.. math::

    Var(W_i X_i)=E[X_i]^2Var(W_i)+E[W_i]^2Var(X_i)+Var(W_i)Var(i_i)

Now if our inputs and weights both have mean :math:`0`, that simplifies to

.. math::

    Var(W_i X_i)=Var(W_i)Var(X_i)

Then if we make a further assumption that the :math:`X_i` and :math:`W_i` are all independent and identically
distributed, we can work out that the variance of :math:`Y` is [6]_

.. math::

    Var(Y)=Var(W_1 X_1+W_2 X_2+⋯+W_n X_n)=nVar(W_i)Var(X_i)

Or in words: the variance of the output is the variance of the input, but scaled by :math:`nVar(W_i)`. So if
we want the variance of the input and output to be the same, that means :math:`nVar(W_i)` should be 1. Which
means the variance of the weights should be

.. math::

    Var(W_i)= \frac{1}{n}= \frac{1}{n_{in}}

Voila. There’s your Xavier initialization.

Glorot & Bengio’s formula needs a tiny bit more work. If you go through the same steps for the backpropagated
signal, you find that you need

.. math::

    Var(W_i)=\frac{1}{n_{out}}

to keep the variance of the input gradient & the output gradient the same. These two constraints can only be
satisfied simultaneously if :math:`n_{in}=n_{out}`, so as a compromise, Glorot & Bengio take the average of
the two:

.. math::

    Var(W_i)=\frac{2}{n_{in}+n_{out}}

Caffe authors used the :math:`n_{in}`-only variant. The two possibilities that come to mind are:

* that preserving the forward-propagated signal is much more important than preserving the back-propagated
  one.
* that for implementation reasons, it’s a pain to find out how many neurons in the next layer consume the
  output of the current one.

It is. But it works. Xavier initialization was one of the big enablers of the move away from per-layer
generative pre-training.

.. [1] X. Glorot and Y. Bengio, “Understanding the difficulty of training deep feedforward neural
       networks,” in International conference on artificial intelligence and statistics, 2010, pp.
       249–256.
.. [2] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image
       recognition,” arXiv preprint arXiv:1409.1556, 2014.
.. [3] K. He, X. Zhang, S. Ren, and J. Sun, “Delving Deep into Rectifiers: Surpassing Human-Level
       Performance on ImageNet Classification,” arXiv:1502.01852 [cs], Feb. 2015.
.. [4] S. Ioffe and C. Szegedy, “Batch Normalization: Accelerating Deep Network Training by Reducing
       Internal Covariate Shift,” arXiv:1502.03167 [cs], Feb. 2015.
.. [5] https://en.wikipedia.org/wiki/Variance#Product_of_independent_variables
.. [6] https://en.wikipedia.org/wiki/Variance#Sum_of_uncorrelated_variables_.28Bienaym.C3.A9_formula.29