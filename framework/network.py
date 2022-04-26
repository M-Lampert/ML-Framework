from pathlib import Path
from typing import List, Union

import numpy as np
from tqdm.auto import tqdm

from framework.input_layer import DefaultInputLayer
from framework.layers import Layer
from framework.tensor import Tensor


class NeuralNetwork:
    """
    Neural Network class

    Attributes
    ----------
    input_layer : DefaultInputLayer
        The input layer the network starts with
    layers : List[Layer]
        List of layers in the order the input should be piped through. Activation layers can be put in between the 'normal' layers
    """

    def __call__(self, x: np.ndarray, batch=True, verbose=True) -> np.ndarray:
        """
        Make a prediction for the given input.

        Parameters
        ----------
        x : numpy.ndarray
            Inputs. Can be a single input or a batch of inputs. If it is a batch, `batch` has to be `True`
        batch : bool
            Specifies if a single or a batch of inputs is given. Default: True
        verbose : True
            Specifies if a progress bar and other information should be shown

        Returns
        -------
        numpy.ndarray
            The predicted results
        """
        return self.predict(x, batch, verbose)

    def __init__(self, input_layer: DefaultInputLayer, layers: List[Layer]):
        self.input_layer = input_layer
        self.layers = layers

    def __str__(self):
        str_tmp = "Neural Network with {} layers. ".format(len(self.layers))
        str_tmp += "The Layers look as follows:\n"
        for layer in self.layers:
            str_tmp += "\n" + layer.__str__()
        return str_tmp

    def predict(self, x: np.ndarray, batch=True, verbose=True) -> np.ndarray:
        """
        Make a prediction for the given input.

        Parameters
        ----------
        x : numpy.ndarray
            Inputs. Can be a single input or a batch of inputs. If it is a batch, `batch` has to be `True`
        batch : bool
            Specifies if a single or a batch of inputs is given. Default: True
        verbose : True
            Specifies if a progress bar and other information should be shown

        Returns
        -------
        numpy.ndarray
            The predicted results
        """

        if batch:
            # Allocate the exact array size that is needed at the start for efficiency
            pred = self.forward(x[0])[-1].get_elements()
            preds = np.zeros((x.shape[0], pred.shape[1]))
            preds[0] = pred
            i = 1

            if verbose:
                pbar = tqdm(x[1:], desc="Prediction")
            else:
                pbar = x[1:]

            for x_curr in pbar:
                preds[i] = self.forward(x_curr)[-1].get_elements()
                i += 1
            return preds
        else:
            return self.forward(x)[-1].get_elements()[np.newaxis]

    def forward(self, x: np.ndarray) -> List[Tensor]:
        """
        Computes the forward passes for all layers and returns the list of output Tensors for every layer.

        Parameters
        ----------
        x : numpy.ndarray
            The input vector

        Returns
        -------
        List[Tensor]
            List of results for all layers in the network
        """
        out_tensors = [self.input_layer.forward(x)]
        for layer in self.layers:
            out_tensors.append(layer.forward(out_tensors[-1]))
        return out_tensors

    def backprop(self, out_tensors: List[Tensor]) -> List[Tensor]:
        """
        Computes the backward passes for all layers and returns the list of output Tensors with gradients for every layer.

        Parameters
        ----------
        out_tensors : List[Tensor]
            The result of the forward pass with added gradients for `out_tensors[-1]`

        Returns
        -------
        List[Tensor]
            List of results for all layers in the network
        """
        for i in reversed(range(len(self.layers))):
            self.layers[i].backward(out_tensors[i], out_tensors[i + 1])
        return out_tensors

    def save(self, path: Union[str, Path] = "network.pkl"):
        """
        Saves the Neural Network. To load it again later

        Parameters
        ----------
        path : {str, Path}, optional
            The path where the network should be saved to. Default: "network.pkl"
        """
        with open(path, "wb") as f:
            np.save(f, np.asanyarray([self.input_layer, self.layers], dtype=object))

    @staticmethod
    def load(path: Union[str, Path]):
        """
        Load a Neural Network from the specified path

        Parameters
        ----------
        path : {str, Path}
            The path where the network should be loaded from
        """
        with open(path, "rb") as f:
            input_layer, layers = np.load(f, allow_pickle=True)
        nn = NeuralNetwork(input_layer, layers)
        return nn
