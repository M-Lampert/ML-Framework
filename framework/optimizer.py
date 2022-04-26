from typing import Tuple, Union

import numpy as np
from tqdm.auto import trange

from framework.layers import Layer
from framework.losses import get_loss
from framework.network import NeuralNetwork
from framework.tensor import Tensor


class SGD:
    """
    Class for the training algorithm stochastic gradient descent
    """

    @staticmethod
    def update(
            nn: NeuralNetwork, loss: Union[Layer, str], lr: float, epochs: int, data: Tuple[np.ndarray, np.ndarray], verbose: bool = True,
            desc: str = "training", position: int = 0
    ) -> Tuple[float, float]:
        """
        Runs the stochastic gradient descent algorithm on the given network.

        Parameters
        ----------
        nn : NeuralNetwork
            The `NeuralNetwork` that should be trained.
        loss : {Layer, str}
            The loss function that should be used. Can either be the `Layer` class that should be used or the name of the loss.
        lr : float
            The learning rate.
        epochs : int
            The number of epochs the network should be trained.
        data : Tuple
            x, y on which the network should be trained on.
        verbose : bool
            If true, information will be shown. If false, nothing will be output.
        desc: str
            What description is to be used for tqdm.
        position: int
            Position argument for tqdm.

        Returns
        -------
        (float, float)
            The Accuracy and the average loss respectively
        """
        # Initialization of running variables
        x_s, l_s = data
        permutation = np.arange(x_s.shape[0])
        rng = np.random.default_rng()
        losses = np.zeros(x_s.shape[0])
        if verbose:
            pbar = trange(epochs, desc=desc, position=position)
        else:
            pbar = range(epochs)

        if isinstance(loss, str):
            loss = get_loss(loss)

        for _ in pbar:
            accuracy_count = 0
            # Use a different permutation every epoch to simulate random sampling
            rng.shuffle(permutation)

            for i in permutation:
                out_tensors = nn.forward(x_s[i])

                # Update running variables for live output
                losses[i] = loss.forward(out_tensors[-1], Tensor(l_s[i]))
                if np.argmax(out_tensors[-1].get_elements()) == np.argmax(l_s[i]):
                    accuracy_count += 1

                # Compute gradients
                loss.backward(out_tensors[-1], Tensor(l_s[i]))
                nn.backprop(out_tensors)

                # Update weights
                for j in range(len(nn.layers)):
                    if nn.layers[j].has_params():
                        W_grad, b_grad = nn.layers[j].calculate_deltas(out_tensors[j], out_tensors[j + 1])
                        nn.layers[j].weights.elements -= lr * W_grad
                        nn.layers[j].bias.elements -= lr * b_grad

            # Update progress bar
            accuracy = accuracy_count / x_s.shape[0]
            avg_loss = np.mean(losses)
            if verbose:
                pbar.set_postfix({"Accuracy": accuracy, "Loss": avg_loss})

        return accuracy, avg_loss
