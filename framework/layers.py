from typing import Tuple

import numpy as np

from framework.tensor import Tensor


class Layer:
    """
    Abstract class as base class for all layers
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the output of the layer

        Parameters
        ----------
        x : Tensor
            Input that should be used for computation

        Returns
        -------
        Tensor
            The computed result
        """

    def backward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute, save and return the gradient of the input tensor.

        Parameters
        ----------
        x : Tensor
            Input of the forward pass
        y : Tensor
            Result and gradients of the forward pass

        Returns
        -------
        Tensor
            The gradients of the input to the forward pass
        """

    def has_params(self) -> bool:
        """
        Specifies if this layer has parameters that need to be updated for the optimizer.

        Returns
        -------
        bool
        """


class FullyConnectedLayer(Layer):
    """
    A layer class for a fully connected layer.

    Attributes
    ----------
    in_shape, out_shape : int
        Shape of the `NeuralNetwork`
    weights : Tensor
        Weights of the `NeuralNetwork`. Initialized randomly
    bias : Tensor
        Bias of the `NeuralNetwork`. Initialized with 0
    """

    def __init__(self, in_shape: int, out_shape: int):
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.weights = Tensor.init_random((in_shape, out_shape))
        self.bias = Tensor.init_zero((1, out_shape))

    def __str__(self):
        str_tmp = "Fully Connected Layer with Shape [{}, {}]:\n".format(self.in_shape, self.out_shape)
        str_tmp += "Weight matrix:\n"
        str_tmp += self.weights.__str__()
        str_tmp += "Bias matrix:\n"
        str_tmp += self.bias.__str__()
        return str_tmp

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the output of the layer

        Parameters
        ----------
        x : Tensor
            Input that should be used for computation

        Returns
        -------
        Tensor
            The computed result
        """
        return Tensor((x.get_elements() @ self.weights.get_elements()) + self.bias.get_elements())

    def backward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute, save and return the gradient of the input tensor.

        Parameters
        ----------
        x : Tensor
            Input of the forward pass
        y : Tensor
            Result and gradients of the forward pass

        Returns
        -------
        Tensor
            The gradients of the input to the forward pass
        """
        x.deltas = y.get_deltas() @ self.weights.get_elements().T
        return x.get_deltas()

    def calculate_deltas(self, x: Tensor, y: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute, save and return the gradients for the weights and the bias respectively.

        Parameters
        ----------
        x : Tensor
            Input of the forward pass
        y : Tensor
            Result and gradients of the forward pass

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            Gradients of the weights and bias respectively
        """
        self.weights.deltas = x.get_elements().T @ y.get_deltas()
        self.bias.deltas = y.get_deltas()
        return self.weights.get_deltas(), self.bias.get_deltas()

    def has_params(self) -> bool:
        """
        Specifies if this layer has parameters that need to be updated for the optimizer.

        Returns
        -------
        bool
            True, since this is the Dense layer
        """
        return True


class FlattenLayer(Layer):
    """A layer class to transform a feature matrix into a feature vector."""

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the output of the layer

        Parameters
        ----------
        x : Tensor
            Input that should be used for computation

        Returns
        -------
        Tensor
            The computed result
        """
        return Tensor(x.get_elements().reshape(1, -1))

    def backward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute, save and return the gradient of the input tensor.

        Parameters
        ----------
        x : Tensor
            Input of the forward pass
        y : Tensor
            Result and gradients of the forward pass

        Returns
        -------
        Tensor
            The gradients of the input to the forward pass
        """
        x.deltas = y.get_deltas().reshape(x.get_elements().shape)
        return x.get_deltas()

    def has_params(self) -> bool:
        """
        Specifies if this layer has parameters that need to be updated for the optimizer.

        Returns
        -------
        bool
            False, since this layer only flattens
        """
        return False
