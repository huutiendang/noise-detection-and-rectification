import numpy as np
from logging import raiseExceptions


def metric_select_sample(y_true, y_pred, metric):
    """
    The function takes in the true labels and the predicted probabilities of the model, and returns a
    list of indexes of the samples that are chosen by the metric
    
    :param y_true: the true labels of the data
    :param y_pred: The predicted probabilities for each class
    :param metric: The metric used to select the next sample
    :return: The indexes of the samples that are chosen by the metric.
    """
    
    metric = metric.lower()
    if metric == 'least confident':  # Least confident
        scores = np.max(y_pred, axis=1)
        indexes = np.argsort(scores)  # argmin(P) ~ argmax(1-P)
        return indexes

    elif metric == 'margin sampling':  # margin sampling
        # Get the probability of the 2 largest classes
        probabilities = np.partition(y_pred, 2, axis=1)[:, 1:]
        # Calculate the difference between 2 classes
        scores = np.abs(probabilities[:, 0] - probabilities[:, 1])
        # Get the index argmin
        indexes = np.argsort(scores)
        return indexes

    elif metric == 'entropy':  # entropy
        y_pred = np.array(y_pred)
        entropy = np.sum(-y_pred * np.log(y_pred), axis=1)
        indexes = np.argsort(entropy)[::-1]  # argmax(entropy)
        return indexes

    elif metric == 'least label confident':  # Least label confident
        scores = [y_pred[i][y_true[i]] for i in range(len(y_pred))]
        indexes = np.argsort(np.array(scores))
        return indexes
        
    elif metric == 'most label confident': # Most confident with label
        scores = [y_pred[i][y_true[i]] for i in range(len(y_pred))]
        indexes = np.argsort(np.array(scores))[::-1]
        return indexes

    elif metric == 'mix entropy and most label confident':
        scores = [y_pred[i][y_true[i]] for i in range(len(y_pred))]
        indexes_1 = np.argsort(np.array(scores))[::-1]

        y_pred = np.array(y_pred)
        entropy = np.sum(-y_pred * np.log(y_pred), axis=1)
        indexes_2 = np.argsort(entropy)[::-1]

        indexes = []
        for pair in zip(indexes_1, indexes_2):
            indexes.extend(pair)
        indexes = list(set(indexes))
        
        return indexes
        
    else:
        return raiseExceptions("Metric does not support")
