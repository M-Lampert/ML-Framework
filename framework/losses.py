import numpy as np

from framework.layers import Layer
from framework.tensor import Tensor


def get_loss(name: str) -> Layer:
    """
    Return the given loss function class.

    Parameters
    ----------
    name : str
        The name of the loss function that should be returned (not case-sensitive)

    Returns
    -------
    Layer
        The corresponding `Layer` class. Can be `CrossEntropy` or `MeanSquaredError

    Raises
    ------
    ValueError
        If a `name` is given except `CrossEntropy` or `MeanSquaredError
    """
    if name.lower() == "crossentropy":
        return CrossEntropy()
    elif name.lower() == "mse" or name.lower() == "meansquarederror":
        return MeanSquaredError()
    else:
        raise ValueError("The given loss does not exist.")


class MeanSquaredError(Layer):
    """
    Mean Squared Error loss
    """

    def __str__(self):
        return "Mean Squared Error Loss\n"

    def forward(self, y_pred: Tensor, y_true: Tensor) -> float:
        """
        Compute the loss given the true labels and the predictions.

        Parameters
        ----------
        y_pred, y_pred : Tensor
            Predicted logits and true labels

        Returns
        -------
        float
            The computed result
        """
        return np.mean((y_pred.get_elements() - y_true.get_elements()) ** 2)

    def backward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Compute, save and return the gradient of the input tensor `y_pred`.

        Parameters
        ----------
        y_pred, y_true : Tensor
            Predicted logits and true labels

        Returns
        -------
        Tensor
            `y_pred` as Tensor with the added gradients
        """
        y_pred.deltas = 2 * (y_pred.get_elements() - y_true.get_elements()) / y_pred.get_elements().size
        return y_pred

    def has_params(self) -> bool:
        """
        Specifies if this layer has parameters that need to be updated for the optimizer.

        Returns
        -------
        bool
            False, since this is a loss layer and thus without parameters
        """
        return False


class CrossEntropy(Layer):
    """
    Cross Entropy Loss or negative Log-Likelyhood
    """

    def __str__(self):
        return "Cross Entropy Loss\n"

    def forward(self, y_pred: Tensor, y_true: Tensor) -> float:
        """
        Compute the loss given the true labels and the predictions.

        Parameters
        ----------
        y_pred, y_pred : Tensor
            Predicted logits and true labels

        Returns
        -------
        float
            The computed result
        """
        return -np.sum(y_true.get_elements() * np.log(y_pred.get_elements()))

    def backward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Compute, save and return the gradient of the input tensor `y_pred`.

        Parameters
        ----------
        y_pred, y_true : Tensor
            Predicted logits and true labels

        Returns
        -------
        Tensor
            `y_pred` as Tensor with the added gradients
        """
        y_pred.deltas = -(y_true.get_elements() / y_pred.get_elements())
        return y_pred

    def has_params(self) -> bool:
        """
        Specifies if this layer has parameters that need to be updated for the optimizer.

        Returns
        -------
        bool
            False, since this is a loss layer and thus without parameters
        """
        return False
