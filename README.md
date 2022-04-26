# Framework

This repository contains the implementation of a framework for a neural net developed during a project at the University of Würzburg. It has much less features than for example PyTorch or TensorFlow, but it was still able to achieve good results as a [slideshow controller](https://github.com/M-Lampert/Slideshow-Project) for a presentation using hand gestures.  
The framework can be be used for creating neural networks and training them.

## Installation using pip

To install the latest version of this package and all its dependencies simply run:

`pip install git+https://github.com/M-Lampert/ML-Framework.git`

## Structure

- `example_data\:` Example training and testing data for demonstration purposes
- `framework\:` The program logic
  - `init.py:` Necessary for the installation via pip (see [Installation using pip](#installation-using-pip))
  - `activation_layers.py:` Activations layers
    - Sigmoid
    - ReLU
    - Tanh
    - Softmax
  - `hyperparameter_search.py:` Method to execute a random hyperparameter search
  - `input_layer.py:` Contains special types of input layers
    - DefaultInputLayer: Converts a column vector of shape (x) saved as numpy array to a row vector of shape (1, x) saved using the internal Tensor class
    - PictureInputLayer: Converts a two-dimensional (gray-scale) image saved as numpy array to a row vector of shape (1, x) saved using the internal Tensor class
  - `layers.py:` Typical layers that are no activation layers
    - FullyConnectedLayer
    - FlattenLayer
  - `losses.py:` Typical loss functions
    - MeanSquaredError
    - CrossEntropy
  - `network.py:` Network class with method for forward-, backward-pass, saving, loading and a convenient prediction method
  - `optimizer.py:` Class for the training algorithm stochastic gradient descent
  - `pca.py:` Class for Principal Component Analysis. Contains methods to analyze the given data and save, load and transform functionalities to use inside a preprocessing pipeline
  - `tensor.py:` Tensor class used inside the neural network for computations. Conveniently stores values and if needed their gradients
  - `utils.py:` Methods to get different performance scores and transformation methods between vectors and class labels
- `Demo.ipynb:` A demo on how to use the essential features of the framework
- `makefile:` Convenient commands for development. Type `make help` for more information. (Does not work natively on Windows)
- `Demo_PCA.ipynb:` Demonstration of the PCA functionality
- `requirements.txt:` All packages that are required to run the framework. Can be installed with `pip install -r requirements.txt`
- `setup.py:` Necessary for the installation via pip (see [Installation using pip](#installation-using-pip))
- `test_utils.py:` Precomputed values used in `test.py`
- `test.py:` Unit-tests testing the correctness of essential features
