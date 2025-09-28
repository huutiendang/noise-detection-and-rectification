import torch


def accuracy(preds, y):
    """
    It takes in a tensor of predictions and a tensor of labels, and returns the accuracy (i.e. the
    fraction of predictions that were correct) for that batch
    
    :param preds: the predictions from the model
    :param y: the labels
    :return: The accuracy of the model.
    """

    # round predictions to the closest integer
    rounded_preds = torch.max(torch.softmax(preds, dim=1), dim=1).indices
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc
