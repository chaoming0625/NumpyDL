
# Deep Learning Framework based on Numpy

_NumpyDL_ is a simple and powerful deep learning library for Python. 

## Requirements
 
- Python3
- Numpy
- Matplotlib
- Scipy
- xlwt
- scikit-learn


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

## install 

Install _NumpyDL_ using pip:
    
    $> pip install numpydl
    
Install from source code:

    $> python setup.py install
   
   
## Support Layers

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

