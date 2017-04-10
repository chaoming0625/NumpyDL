# Deep Learning Framework based on Numpy

_NumpyDL_ is a simple deep learning library based on pure Python/Numpy. 

## Design Philosophy

- **Simplicity**: Be easy to use, easy to understand and easy to extend, to facilitate use in research. Interfaces should be kept small, with as few classes and methods as possible. Every added abstraction and feature should be carefully scrutinized, to determine whether the added complexity is justified.
- **Transparency**: Native to Numpy, directly process and return Python / numpy data types. Do not rely on the functionality of Theano, Tensorflow or any such DL framework.
- **Modularity**: Allow all parts (layers, regularizers, optimizers, ...) to be used independently of NumpyDL. Make it easy to use components in isolation or in conjunction with other frameworks.
- **Focus**: “Do one thing and do it well”. Do not try to provide a library for everything to do with deep learning.

## Requirements

- Python3
- Numpy
- Scipy
- scikit-learn

If you want to build the document, you need install:

- sphinx


## Features

- Pure python + numpy
- API like [Keras](https://github.com/fchollet/keras) deep learning library
- Support basic automatic differentiation;
- Support commonly used models, such as MLP, RNNs LSTMs and convnets.
- Flexible network configurations and learning algorithms. 

## Example
    
    import numpy as np
    from sklearn.datasets import load_digits
    import npdl
    
    # prepare
    npdl.utils.random.set_seed(1234)

    # data
    digits = load_digits()
    X_train = digits.data
    X_train /= np.max(X_train)
    Y_train = digits.target
    n_classes = np.unique(Y_train).size

    # model
    model = npdl.model.Model()
    model.add(npdl.layers.Dense(n_out=500, n_in=64, activation=npdl.activation.ReLU()))
    model.add(npdl.layers.Dense(n_out=n_classes, activation=npdl.activation.Softmax()))
    model.compile(loss=npdl.objectives.SCCE(), optimizer=npdl.optimizers.SGD(lr=0.005))

    # train
    model.fit(X_train, npdl.utils.data.one_hot(Y_train), max_iter=150, validation_split=0.1)

## Install 

<!--- 

 Install _NumpyDL_ using pip: 
    
    $> pip install numpydl

--->
 
Install from source code:

    $> python setup.py install
   
   
## Support

### layer

- [core.py](npdl/layers/core.py)
    - Dense (perceptron) Layer 
    - Softmax Layer
    - Dropout Layer
- [normalization.py](npdl/layers/normalization.py)
    - Batch Normalization Layer
- [embedding.py](npdl/layers/embedding.py)
    - Embedding Layer
- [convolution.py](npdl/layers/convolution.py)
    - Convolution Layer
- [pooling.py](npdl/layers/pooling.py)
    - MaxPooling Layer
    - MeanPooling Layer
- [reccurent.py](npdl/layers/reccurent.py)
    - SimpleRNN Layer
- [shape.py](npdl/layers/shape.py)
    - Flatten Layer

### activation

- Sigmoid
- Tanh
- ReLU
- Softmax
- Elliot
- SymmetricElliot
- LReLU
- SoftPlus
- SoftSign

### initialization

- Uniform
- Normal
- LecunUniform
- GlorotUniform
- GlorotNormal
- HeNormal
- HeUniform
- Orthogonal

### objective

- MeanSquaredError
- HellingerDistance
- BinaryCrossEntropy
- SoftmaxCategoricalCrossEntropy


### optimizer
- SGD
