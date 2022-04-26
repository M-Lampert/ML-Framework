import csv
from itertools import product
from multiprocessing import Pool
from typing import Dict, List, Tuple, Union, Callable

import numpy as np
from tqdm.auto import tqdm

from framework.activation_layers import get_activation_layer
from framework.input_layer import DefaultInputLayer
from framework.layers import FullyConnectedLayer, Layer
from framework.network import NeuralNetwork
from framework.optimizer import SGD
from framework.utils import get_performance_score_func


class HyperparameterSearch:
    def __init__(
            self,
            training_data: Tuple[np.ndarray, np.ndarray],
            validation_data: Tuple[np.ndarray, np.ndarray],
            performance_scores: Union[List[Union[str, Tuple[str, Callable]]], str, Tuple[str, Callable]]
    ):
        """
        Object for sharing data and performance scores between the different search runs

        Parameters
        ----------
        training_data : tuple
            x, y that should be used to train the neural networks
        validation_data : tuple
            x, y that should be used to validate the neural networks
        performance_scores : iterable
            The performance scores that should be computed for every try. The default scores can all be used with just the string.
            To use a custom score function, input a tuple of the name the score should have in the results and the callable. The callable needs to accept the following three parameters: `predictions: np.ndarray`, `true_labels: np.ndarray`, `sparse: bool`;
            Default: "Accuracy", "Recall", "Precision", "F1"
        """
        if performance_scores is None:
            performance_scores = ["Accuracy", "Recall", "Precision", "F1"]
        self.training_data = training_data
        self.validation_data = validation_data
        if not isinstance(performance_scores, list):
            performance_scores = [performance_scores]
        self.performance_scores = {
            performance_score if isinstance(performance_score, str) else performance_score[0]
            : get_performance_score_func(performance_score) if isinstance(performance_score, str) else performance_score[1]
            for performance_score in performance_scores
        }

    def __call__(
            self,
            cores: int = 6,
            n_layers: List[int] = [2],
            learning_rates: List[float] = [0.01],
            n_epochs: List[int] = [5],
            losses: List[str] = ["crossentropy"],
            activations: List[str] = ["sigmoid"],
            n_neurons: List[int] = [32],
            path_to_csv: str = "eval.csv"
    ):
        """
        Randomly iterates through all choices of parameters and saves the performance into a CSV file given by path_to_csv.

        Parameters
        ----------
        cores : int, optional
            The number of processes that should be started in parallel.
            Default: 6
        n_layers, learning_rates, n_epochs, losses, activations, n_neurons : iterable, optional
            All options that should be tried during the search
        path_to_csv : str
            Path where performance will be saved into.

        Returns
        -------

        """
        options = self._get_options(n_layers, learning_rates, n_epochs, losses, activations, n_neurons)

        columns = ["n_layers", "learning_rates", "n_epochs", "losses", "n_neurons", "activations"]
        performance_score_names = list(self.performance_scores.keys())
        columns.extend(performance_score_names)
        columns.extend(["train_accuracy", "train_loss"])

        with Pool(cores) as p:
            with open(path_to_csv, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                for config, res in tqdm(p.imap_unordered(self._try_with_config, options), total=len(options)):
                    line = config
                    for perf, value in res.items():
                        line.append(value)
                    writer.writerow(line)
                    del line
                    f.flush()

    def _get_options(
            self,
            n_layers: List[int] = [2],
            learning_rates: List[float] = [0.01],
            n_epochs: List[int] = [5],
            losses: List[str] = ["crossentropy"],
            activations: List[str] = ["sigmoid"],
            n_neurons: List[int] = [32]
    ) -> list:
        # Create Dataframe with a row for every option. Use joins to create all options.
        in_shape, out_shape = (self.training_data[0].shape[-1],), (self.training_data[1].shape[-1],)
        prod = product(n_layers, learning_rates, n_epochs, losses)
        options = []
        for t in prod:
            option = list(t)
            numb_layers = option[0]
            for n in product(n_neurons, repeat=numb_layers - 1):
                for a in product(activations, repeat=numb_layers - 1):  # Last activation function is softmax
                    # Add in_shape and out_shape
                    option.append(in_shape + n + out_shape)
                    option.append(a + ("softmax",))
                    options.append(option.copy())
                    del option[-2:]
        return options

    def _try_with_config(
            self,
            config: list,
    ) -> Tuple[list, dict]:
        """
        Function that tries the given row from the result table and saves the results.

        Parameters
        ----------
        config : list
            The configuration to be evaluated

        Returns
        -------
        Tuple[list, dict]
        """
        # Initialize, train and test a Neural Network with the specified option combinations
        nn = NeuralNetwork(
            input_layer=DefaultInputLayer(),
            layers=layers_with_choices(config=config),
        )
        train_accuracy, train_loss = SGD.update(
            nn=nn,
            loss=config[3],
            lr=config[1],
            epochs=config[2],
            data=self.training_data,
            verbose=False,
        )
        preds = nn(self.validation_data[0], verbose=False)

        # Get and save all specified scores
        result_score = {f"validation_{name}": score_func(preds, self.validation_data[1], sparse=False) for name, score_func in self.performance_scores.items()}
        result_score["train_accuracy"] = train_accuracy
        result_score["train_loss"] = train_loss
        return config, result_score


def layers_with_choices(config: list) -> List[Layer]:
    """
    Function to train a neural network with given choices.

    Parameters
    ----------
    config : Dict
        A dictionary with the chosen options for all layers

    Returns
    -------
    List of layers
        List from which a `NeuralNetwork` can directly be instantiated
    """
    neurons_per_layer = config[4]
    activation_per_layer = config[5]

    layers = []
    for in_shape, out_shape, activation in zip(neurons_per_layer[:-1], neurons_per_layer[1:], activation_per_layer):
        layers.append(FullyConnectedLayer(in_shape, out_shape))
        layers.append(get_activation_layer(activation))

    return layers
