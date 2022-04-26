import pickle
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def standardize_data(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Method to standardize the given data using a given mean and standard deviation.

    Parameters
    ----------
    array : numpy.ndarray
        The data that should be standardized
    mean : numpy.ndarray
        The mean that should be used (one per column of the input data).
    std : numpy.ndarray
        The standard deviation to use (one per column of the input data).

    Returns
    -------
    numpy.ndarray
        Returns the standardized data.
    """
    return (array - mean) / std


class PCA:
    """
    PCA class with static methods for analysis and the possibility to create a PCA object with a projection matrix for given data.

    Attributes
    ----------
    projection_matrix : numpy.ndarray
        The projection matrix that should be used to transform given data.
    mean : numpy.ndarray
        An array of means (one for every column of input data) used for standardization.
    std : numpy.ndarray
        An array of standard deviations (one for every column of input data) used for standardization.
    """

    def __init__(self, projection_matrix: np.ndarray, mean: np.ndarray, std: np.ndarray):
        self.projection_matrix = projection_matrix
        self.mean = mean
        self.std = std

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Projects the given data onto a lower-dimensional subspace.

        Parameters
        ----------
        data : numpy.ndarray
            The data to be transformed

        Returns
        -------
        numpy.ndarray
            Lower-dimensional and standardized data
        """
        X = standardize_data(data, self.mean, self.std)
        return X.dot(self.projection_matrix)

    def get_projection_matrix(self) -> np.ndarray:
        """
        Getter-Method to get the projection matrix.

        Returns
        -------
        numpy.ndarray
            The projection matrix
        """
        return self.projection_matrix

    def save(self, path: Union[Path, str]):
        """
        Saves the PCA instance to disk

        Parameters
        ----------
        path : {Path, str}
            The path the instance should be saved to
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def create_PCA(data: np.ndarray, dimensions_or_variance: Union[int, float]) -> "PCA":
        """
        Creates a PCA object for the given data with the given number of dimensions or the specified variance.

        Parameters
        ----------
        data : numpy.ndarray
            The data that should be used to compute a projection matrix
        dimensions_or_variance : {int, float}
            If larger than or equal to 1, the specified number will be used as number of dimensions. All other dimensions will be dropped. If smaller than 1, the number of dimensions necessary to keep the desired variance will be computed and used.

        Returns
        -------
        PCA
            A pca instance with a fitted projection matrix
        """
        return PCA(PCA.compute_projection_matrix(data, dimensions_or_variance), np.mean(data, axis=0), np.std(data, axis=0))

    @staticmethod
    def get_covariance_matrix(data: np.ndarray) -> np.ndarray:
        """
        Compute and return the covariance matrix for the given data. Necessary for the eigenvector decomposition.

        Parameters
        ----------
        data : numpy.ndarray
            The data the covariance matrix should be computed for

        Returns
        -------
        numpy.ndarray
            Covariance matrix for the data
        """
        X = standardize_data(data, np.mean(data, axis=0), np.std(data, axis=0))
        return np.cov(X.T)

    @staticmethod
    def get_eigendecomposition(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the eigenvector decomposition for the given data. The eigenvectors (the second return value) is the projection matrix that will be used for the transformation. The columns are automatically ordered so that the first column has the most variance.

        Parameters
        ----------
        data : numpy.ndarray
            The data the eigenvector decomposition should be computed for

        Returns
        -------
        numpy.ndarray
            A tuple containing two numpy-Arrays: (eigen_values, eigen_vectors)
        """
        return np.linalg.eig(PCA.get_covariance_matrix(data))

    @staticmethod
    def get_variance_explained(data: np.ndarray) -> np.ndarray:
        """
        Shows how much of the variance in the data is explained by each principal component (column)

        Parameters
        ----------
        data : numpy.ndarray
            The data the variance should be explained for

        Returns
        -------
        numpy.ndarray
            Numpy-Array containing the percentage of contribution by the principal components to the variance in the data
        """
        eigen_values, _ = PCA.get_eigendecomposition(data)
        return eigen_values / np.sum(eigen_values) * 100

    @staticmethod
    def get_cumulative_variance_explained(data: np.ndarray) -> np.ndarray:
        """
        As get_variance_explained() but as cumulative version

        Parameters
        ----------
        data : numpy.ndarray
            The data the variance should be explained for

        Returns
        -------
        numpy.ndarray
            Numpy-Array containing the cumulative explained variance
        """
        return np.cumsum(PCA.get_variance_explained(data))

    @staticmethod
    def visualize_eigenvalues(data: np.ndarray):
        """
        Visualizes the cumulative eigenvalues. Helps in determining the most important features.

        Parameters
        ----------
        data : numpy.ndarray
            The data the variance should be explained for and visualized
        """
        X = standardize_data(data, np.mean(data, axis=0), np.std(data, axis=0))
        plt.plot([feature_number for feature_number in range(X.shape[1])], PCA.get_cumulative_variance_explained(data))
        plt.xlabel("Features")
        plt.ylabel("Cumulative explained variance")
        plt.title("Explained variance vs features")
        plt.xticks(ticks=[feature_number for feature_number in range(X.shape[1])])
        plt.show()

    @staticmethod
    def compute_projection_matrix(data: np.ndarray, feature_count_or_variance: Union[int, float]) -> np.ndarray:
        """
        Computes the projection matrix for a PCA object for the given data with the given number of dimensions or the specified variance.

        Parameters
        ----------
        data : numpy.ndarray
            The data that should be used to compute a projection matrix
        feature_count_or_variance: {int, float}
            If larger than or equal to 1, the specified number will be used as number of dimensions. All other dimensions will be dropped. If smaller than 1, the number of dimensions necessary to keep the desired variance will be computed and used.

        Returns
        -------
        PCA
            A projection matrix fitted for the data
        """
        if feature_count_or_variance < 1:
            desired_variance = feature_count_or_variance * 100
            # Determine number of features (=feature_count) necessary to reach the desired variance
            cumulative_var_explained = PCA.get_cumulative_variance_explained(data)
            feature_count = 0
            for idx, var in enumerate(cumulative_var_explained):
                if var >= desired_variance:
                    feature_count = idx + 1
                    break
            if feature_count == 0:
                raise RuntimeError(f"The chosen variance {desired_variance} is not reachable.")
        else:
            feature_count = feature_count_or_variance
        _, eigen_vectors = PCA.get_eigendecomposition(data)
        return (eigen_vectors.T[:][:feature_count]).T

    @staticmethod
    def load(path: Union[Path, str]) -> "PCA":
        """
        Loads a PCA object with projection-matrix, mean and std of training data

        Parameters
        ----------
        path : {Path, str}
            The path the instance should be loaded from

        Returns
        -------
        PCA
            The loaded pca object
        """
        with open(path, "rb") as f:
            pca_object = pickle.load(f)
        return pca_object
