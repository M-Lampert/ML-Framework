import numpy as np

from framework.layers import Layer
from framework.tensor import Tensor


def get_activation_layer(name: str) -> Layer:
    """
    Return the given activation layer class.

    Parameters
    ----------
    name : str
        The name of the activation layer that should be returned (not case-sensitive)

    Returns
    -------
    Layer
        The corresponding `Layer` class. Can be `Sigmoid`, `ReLU`, `Tanh` or `Softmax`

    Raises
    ------
    ValueError
        If a `name` is given except `Sigmoid`, `ReLU`, `Tanh` or `Softmax`
    """
    if name.lower() == "sigmoid":
        return Sigmoid()
    elif name.lower() == "relu":
        return ReLU()
    elif name.lower() == "tanh":
        return Tanh()
    elif name.lower() == "softmax":
        return Softmax()
    else:
        raise ValueError("The given activation layer does not exist.")


class Sigmoid(Layer):
    """
    Sigmoid activation layer
    """

    def __str__(self):
        return "Sigmoid activation function\n"

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
        # To make it numerically stable, we make a different calculation for values larger and values smaller than 0
        # https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth

        x_arr = x.get_elements()

        positive = x_arr >= 0
        negative = ~positive

        result = np.empty_like(x_arr)
        result[positive] = self._positive_sigmoid(x_arr[positive])
        result[negative] = self._negative_sigmoid(x_arr[negative])

        return Tensor(result)

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
        x.deltas = y.get_deltas() * (y.get_elements() * (1 - y.get_elements()))
        return x

    def has_params(self) -> bool:
        """
        Specifies if this layer has parameters that need to be updated for the optimizer.

        Returns
        -------
        bool
            False, since this is an activation layer and thus without parameters
        """
        return False

    def _positive_sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Helper method for stable sigmoid computation

        Parameters
        ----------
        x : numpy.ndarray
            The positive input values for which the sigmoid should be calculated

        Returns
        -------
        Numpy array
            The computed results of the sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def _negative_sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Helper method for stable sigmoid computation

        Parameters
        ----------
        x : numpy.ndarray
            The negative input values for which the sigmoid should be calculated

        Returns
        -------
        Numpy array
            The computed results of the sigmoid function
        """

        # Cache exp so you won't have to calculate it twice
        exp = np.exp(x)
        return exp / (exp + 1)


class ReLU(Layer):
    """
    ReLU (Rectified Linear Unit) activation layer
    """

    def __str__(self):
        return "ReLU activation function\n"

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
        return Tensor(np.maximum(x.get_elements(), 0).reshape(x.get_elements().shape))

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
        x.deltas = y.get_deltas() * (y.get_elements() > 0)
        return x

    def has_params(self) -> bool:
        """
        Specifies if this layer has parameters that need to be updated for the optimizer.

        Returns
        -------
        bool
            False, since this is an activation layer and thus without parameters
        """
        return False


class Tanh(Layer):
    """
    Tangens hyperbolicus activation layer
    """

    def __str__(self):
        return "Tanh activation function\n"

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
        return Tensor(np.tanh(x.get_elements()))

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
        x.deltas = y.get_deltas() * (1 - y.get_elements() ** 2)
        return x

    def has_params(self) -> bool:
        """
        Specifies if this layer has parameters that need to be updated for the optimizer.

        Returns
        -------
        bool
            False, since this is an activation layer and thus without parameters
        """
        return False


class Softmax(Layer):
    """
    Softmax activation layer
    """

    def __str__(self):
        return "Softmax activation function\n"

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
        # For numerical stability, subtract the maximum
        z = np.exp(x.get_elements() - np.max(x.get_elements()))
        return Tensor(z / np.sum(z))

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
        # https://stackoverflow.com/questions/40575841/numpy-calculate-the-derivative-of-the-softmax-function
        jac = np.diagflat(y.get_elements().reshape(-1)) - np.dot(y.get_elements().T, y.get_elements())
        x.deltas = y.get_deltas() @ jac
        return x

    def has_params(self) -> bool:
        """
        Specifies if this layer has parameters that need to be updated for the optimizer.

        Returns
        -------
        bool
            False, since this is an activation layer and thus without parameters
        """
        return False
