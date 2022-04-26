from typing import Tuple

import numpy as np


class Tensor:
    """
    A Tensor contains elements of a single known data type and has a known shape, which represents the dimensions of the tensor.

    Attributes
    ----------
    elements : numpy.ndarray
        The values of the `Tensor`
    deltas : numpy.ndarray, optional
        Possible gradients of the `Tensor`. Will be initialized lazily for efficiency
    """

    def __init__(self, elements: np.ndarray):
        self.elements = elements
        self.deltas = None

    def __str__(self):
        res = "Elements:\n" + self.elements.__str__() + "\n"
        if self.deltas is not None:
            res += "Deltas:\n" + self.deltas.__str__() + "\n"
        return res

    def get_shape(self) -> Tuple:
        """
        Get the shape of the `Tensor`

        Returns
        -------
        Tuple
            The shape
        """
        return self.elements.shape

    def get_elements(self) -> np.ndarray:
        """
        Get the elements of the `Tensor`

        Returns
        -------
        Tuple
            The elements
        """
        return self.elements

    def get_deltas(self) -> np.ndarray:
        """
        Get the gradients of the `Tensor`

        Returns
        -------
        Tuple
            The gradients, if already initialized. Else they will be set to 0
        """
        if self.deltas is None:
            self.deltas = np.zeros(self.get_shape())
        return self.deltas

    @staticmethod
    def init_random(shape: tuple, range: float = 0.05):
        """
        Return a tensor of the specified shape with random values ranging from -range to range.

        Parameters
        ----------
        shape : tuple
            The shape of the `Tensor`
        range : float, optional
            The range of the random values that are generated. Default: 0.05

        Returns
        -------
        Tensor
        """
        return Tensor(np.random.uniform(-range, range, shape))

    @staticmethod
    def init_zero(shape: tuple):
        """
        Return a tensor of the specified shape with zero values.

        Parameters
        ----------
        shape : tuple
            The shape of the `Tensor`

        Returns
        -------
        Tensor
        """
        return Tensor(np.zeros(shape))
