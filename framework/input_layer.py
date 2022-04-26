import numpy as np

from framework.tensor import Tensor


class DefaultInputLayer:
    """
    The default input layer that can be used for all one dimensional input of numbers.
    """

    def forward(self, x: np.ndarray) -> Tensor:
        """
        Convert a one dimensional numpy array to a valid input using the internal `Tensor` class.

        Parameters
        ----------
        x : numpy.ndarray
            Input array

        Returns
        -------
        Tensor
        """
        return Tensor(x[np.newaxis])


class PictureInputLayerFCN(DefaultInputLayer):
    """
    Special input layer to convert gray scale images saved as numpy arrays to valid input.
    """

    def forward(self, x: np.ndarray) -> Tensor:
        """
        Convert two dimensional numpy array of a gray scale image to a valid input using the internal `Tensor` class.

        Parameters
        ----------
        x : numpy.ndarray
            Input array

        Returns
        -------
        Tensor
        """
        return Tensor(x.reshape(1, -1))
