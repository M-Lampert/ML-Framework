from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_performance_score_func(name: str) -> Callable:
    """
    Return the given score function.

    Parameters
    ----------
    name : str
        The name of the score function that should be returned (not case-sensitive). Can be 'accuracy', 'recall', 'precision' or 'f1'

    Returns
    -------
    Callable
        The corresponding function.

    Raises
    ------
    ValueError
        If a `name` is given except 'accuracy', 'recall', 'precision' or 'f1'
    """
    name = name.lower()
    if name == "accuracy":
        return get_accuracy
    elif name == "recall":
        return get_recall
    elif name == "precision":
        return get_precision
    elif name == "f1" or name == "f1-score" or name == "f1_score":
        return get_f1_score
    else:
        raise ValueError("The specified score is not implemented.")


def get_accuracy(predictions: np.ndarray, true_labels: np.ndarray, sparse=True):
    """
    Calculates the prediction accuracy given the predictions in a class propability vector and the true labels as specified (sparse or as one-hot vector).

    Parameters
    ----------
    predictions : numpy.ndarray
        The predictions that should be evaluated
    true_labels : numpy.ndarray
        The correct label the prediction should have. Can be a one hot vector (sparse = False) or an index (sparse = True)
    sparse : bool, optional
        Specifies if the true labels are one hot vectors (False) or indices (True)

    Returns
    -------
    float
        The computed accuracy
    """
    if not sparse:
        true_labels = np.argmax(true_labels, axis=1)
    predictions = np.argmax(predictions, axis=1)

    return np.sum(predictions == true_labels) / len(true_labels)


def get_recall(predictions: np.ndarray, true_labels: np.ndarray, sparse=True, mode="macro"):
    """
    Calculates the recall for the given predictions (as class probability vectors) and true labels. The recall score describes how many of the actual true labels are classified correctly. You can either choose the mode 'macro' (average over all class recall scores) or 'classes' (score for every class).

    Parameters
    ----------
    predictions : numpy.ndarray
        The predictions that should be evaluated
    true_labels : numpy.ndarray
        The correct label the prediction should have. Can be a one hot vector (sparse = False) or an index (sparse = True)
    sparse : bool, optional
        Specifies if the true labels are one hot vectors (False) or indices (True)
    mode : {"macro", "classes"}, optional
        Specifies if the results should be averaged or returned as single values for every class

    Returns
    -------
    float or List[float]
        The averaged recall or all recalls
    """
    if not sparse:
        true_labels = np.argmax(true_labels, axis=1)
    predictions = np.argmax(predictions, axis=1)

    classes = list(range(np.max(np.array([predictions, true_labels])) + 1))
    df = pd.DataFrame({"predictions": predictions, "true_labels": true_labels})

    with np.errstate(divide="ignore", invalid="ignore"):
        recalls = np.array([np.sum((df == curr_class).all(axis=1)) / np.sum(df["true_labels"] == curr_class) for curr_class in classes])
        recalls = np.nan_to_num(recalls, nan=1)

    if mode == "macro":
        return np.mean(recalls)
    elif mode == "classes":
        return recalls
    else:
        raise ValueError(f"{mode} does not exist as mode.")


def get_precision(predictions: np.ndarray, true_labels: np.ndarray, sparse=True, mode="macro"):
    """
    Calculates the precision for the given predictions (as class probability vectors) and true labels. The precision score describes how many of the predictions are correct classifications. You can either choose the mode 'macro' (average over all class precision scores) or 'classes' (score for every class).

    Parameters
    ----------
    predictions : numpy.ndarray
        The predictions that should be evaluated
    true_labels : numpy.ndarray
        The correct label the prediction should have. Can be a one hot vector (sparse = False) or an index (sparse = True)
    sparse : bool, optional
        Specifies if the true labels are one hot vectors (False) or indices (True)
    mode : {"macro", "classes"}, optional
        Specifies if the results should be averaged or returned as single values for every class

    Returns
    -------
    float or List[float]
        The averaged precision or all precisions
    """
    if not sparse:
        true_labels = np.argmax(true_labels, axis=1)
    predictions = np.argmax(predictions, axis=1)

    classes = list(range(np.max(np.array([predictions, true_labels])) + 1))
    df = pd.DataFrame({"predictions": predictions, "true_labels": true_labels})

    with np.errstate(divide="ignore", invalid="ignore"):
        precisions = np.array([np.sum((df == curr_class).all(axis=1)) / np.sum(df["predictions"] == curr_class) for curr_class in classes])
        precisions = np.nan_to_num(precisions, nan=1)

    if mode == "macro":
        return np.mean(precisions)
    elif mode == "classes":
        return precisions
    else:
        raise ValueError(f"{mode} does not exist as mode.")


def get_f1_score(predictions: np.ndarray, true_labels: np.ndarray, sparse=True, mode="macro"):
    """
    Calculates the f1 score for the given predictions (as class probability vectors) and true labels. Harmonic mean between precision and recall. You can either choose the mode 'macro' (average over all class precision scores) or 'classes' (score for every class).

    Parameters
    ----------
    predictions : numpy.ndarray
        The predictions that should be evaluated
    true_labels : numpy.ndarray
        The correct label the prediction should have. Can be a one hot vector (sparse = False) or an index (sparse = True)
    sparse : bool, optional
        Specifies if the true labels are one hot vectors (False) or indices (True)
    mode : {"macro", "classes"}, optional
        Specifies if the results should be averaged or returned as single values for every class

    Returns
    -------
    float or List[float]
        The averaged f1-score or all f1-scores
    """
    # Doesn't use methods from above for efficiency
    if not sparse:
        true_labels = np.argmax(true_labels, axis=1)
    predictions = np.argmax(predictions, axis=1)

    classes = list(range(np.max(np.array([predictions, true_labels])) + 1))
    df = pd.DataFrame({"predictions": predictions, "true_labels": true_labels})

    with np.errstate(divide="ignore", invalid="ignore"):
        precisions = np.array([np.sum((df == curr_class).all(axis=1)) / np.sum(df["predictions"] == curr_class) for curr_class in classes])
        precisions = np.nan_to_num(precisions, nan=1)

        recalls = np.array([np.sum((df == curr_class).all(axis=1)) / np.sum(df["true_labels"] == curr_class) for curr_class in classes])
        recalls = np.nan_to_num(recalls, nan=1)

        f1s = 2 * (precisions * recalls) / (precisions + recalls)

    if mode == "macro":
        return np.mean(f1s)
    elif mode == "classes":
        return f1s
    else:
        raise ValueError(f"{mode} does not exist as mode.")


def get_confusion_matrix(predictions: np.ndarray, true_labels: np.ndarray, sparse=True, show=True, class_names=None) -> pd.DataFrame:
    """
    Calculates the confusion matrix for the given predictions (as class probability vectors) and true labels

    Parameters
    ----------
    predictions : numpy.ndarray
        The predictions that should be evaluated
    true_labels : numpy.ndarray
        The correct label the prediction should have. Can be a one hot vector (sparse = False) or an index (sparse = True)
    sparse : bool, optional
        Specifies if the true labels are one hot vectors (False) or indices (True)
    show : bool, optional
        If the matrix should be shown using matplotlib
    class_names : List[str], optional
        Names for the classes. The index should be the same as the index of the computed predictions and true labels. If `None`, no class names are shown

    Returns
    -------
    float or List[float]
        Confusion matrix as pandas Dataframe
    """
    if not sparse:
        true_labels = np.argmax(true_labels, axis=1)
    predictions = np.argmax(predictions, axis=1)

    if class_names is not None:
        true_labels = one_hot_vector_to_classes(true_labels, class_names, True)
        predictions = one_hot_vector_to_classes(predictions, class_names, True)
    else:
        class_names = list(range(np.max(np.array([predictions, true_labels])) + 1))

    predictions = pd.Categorical(predictions, categories=class_names)
    true_labels = pd.Categorical(true_labels, categories=class_names)
    confusion_matrix = pd.crosstab(true_labels, predictions, colnames=["Predicted"], rownames=["True Labels"], dropna=False)

    if show:
        plt.xkcd()
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        im = ax.imshow(confusion_matrix, cmap=plt.cm.Blues, alpha=0.5)

        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(x=j, y=i, s=confusion_matrix.iloc[i, j], va="center", ha="center", fontsize=12)

        plt.xlabel("Predictions", fontsize=18)
        plt.ylabel("Actuals", fontsize=18)
        plt.yticks(range(len(class_names)), class_names, fontsize=12, rotation=90, verticalalignment="center")
        plt.xticks(range(len(class_names)), class_names, fontsize=12)
        plt.title("Confusion Matrix", fontsize=30)
        plt.colorbar(im)
        plt.show()

    return confusion_matrix


def get_classification_report(predictions: np.ndarray, true_labels: np.ndarray, sparse=True, class_names=None) -> pd.DataFrame:
    """
    Shows a summary of different classification metrics.

    Parameters
    ----------
    predictions : numpy.ndarray
        The predictions that should be evaluated
    true_labels : numpy.ndarray
        The correct label the prediction should have. Can be a one hot vector (sparse = False) or an index (sparse = True)
    sparse : bool, optional
        Specifies if the true labels are one hot vectors (False) or indices (True)
    class_names : List[str], optional
        Names for the classes. The index should be the same as the index of the computed predictions and true labels. If `None`, no class names are shown

    Returns
    -------
    float or List[float]
        Summary as pandas Dataframe
    """
    if class_names is None:
        class_names = list(range(np.max(np.array([np.argmax(predictions, axis=1), np.argmax(true_labels, axis=1)])) + 1))

    recalls = get_recall(predictions, true_labels, sparse, "classes")
    precisions = get_precision(predictions, true_labels, sparse, "classes")
    f1s = get_f1_score(predictions, true_labels, sparse, "classes")
    fractions = [np.sum(np.argmax(true_labels, axis=1) == curr_class) / len(true_labels) for curr_class in range(np.max(np.argmax(true_labels, axis=1)) + 1)]
    accuracy = get_accuracy(predictions, true_labels, sparse)

    index = [["classes"] * len(class_names) + ["global"] * 3, class_names + ["Micro Avg / Accuracy", "Macro Avg", "Weighted Avg"]]

    df = pd.DataFrame(
        {
            "Precision": list(precisions) + [accuracy, np.mean(precisions), np.average(precisions, weights=fractions)],
            "Recall": list(recalls) + [accuracy, np.mean(recalls), np.average(recalls, weights=fractions)],
            "F1-Score": list(f1s) + [accuracy, np.mean(f1s), np.average(f1s, weights=fractions)],
            "Fraction": fractions + [1, 1, 1],
        },
        index=index,
    )
    return df


def classes_to_one_hot_vector(y_s: np.ndarray, class_names: List[str]) -> np.ndarray:
    """
    Takes an array of class names as input and transforms it to a one hot vector

    Parameters
    ----------
    y_s : numpy.ndarray
        True label array with the given classes.
    class_names : List[str]
        Names for the classes. The index should be the same as the index of the computed predictions and true labels.

    Returns
    -------
    numpy.ndarray
        Numpy array with a one hot vector for every label
    """
    result = np.zeros((len(y_s), len(class_names)))
    for i, y in enumerate(y_s):
        result[i, class_names.index(y)] = 1
    return result


def one_hot_vector_to_classes(y_s: np.ndarray, class_names: List, sparse: bool = True) -> np.ndarray:
    """
    Takes an array of one hot vectors or class indices (if sparse == True) and returns an array of class_names.

    Parameters
    ----------
    y_s : numpy.ndarray
        True label array with the given classes.
    class_names : List[str]
        Names for the classes. The index should be the same as the index of the computed predictions and true labels.
    sparse : bool, optional
        Specifies if the true labels are one hot vectors (False) or indices (True)

    Returns
    -------
    numpy.ndarray
        Numpy array with a label for every one hot vector
    """
    return np.array([class_names[y] if sparse else class_names[np.argmax(y)] for y in y_s])
