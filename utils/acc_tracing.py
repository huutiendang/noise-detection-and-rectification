import numpy as np


def acc_tracing(df, y_true, y_pred, top=[100, 500, 1000, 2000, 5000]):
    """
    The function takes in a dataframe, the true labels, and the predicted labels, and returns the
    percent accuracy of tracing for the top 100, 500, 1000, 2000, and 5000 predictions.
    
    :param df: Pandas Dataframe
    :param y_true: the true labels of the data
    :param y_pred: the output of the model, which is a list of lists of probabilities for each class
    :param top: the top n predictions to consider
    :return: the percent accuracy of tracing.
    """

    scores = [y_pred[i][y_true[i]] for i in range(len(y_pred))]
    indexes = np.argsort(np.array(scores))
    results = []

    for t in top:
        index = indexes[:t]
        df_new = df.iloc[index].reset_index(drop=True)
        acc_tracing = len(df_new[df_new['isFlipped'] == 1])/t
        results.append(acc_tracing)

    return np.array(results) * 100
