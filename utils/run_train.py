import tqdm
from utils.accuracy import accuracy
import pandas as pd
import torch
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score, accuracy_score


def run_train(model, dataloader, optimizer, criterion, func_inference, mode='train'):
    """
    > This function takes in a model, a dataloader, an optimizer, a loss function, and a function that
    performs inference on the model, and returns the average loss and accuracy for the epoch
    
    :param model: model of ModelBase
    :param dataloader: the dataloader for the training data
    :param optimizer: The optimizer to use
    :param criterion: This is the loss function. We use cross entropy loss
    :param func_inference: This is a function that takes in a batch of data and returns the predictions
    and labels
    :param mode: train or val, defaults to train (optional)
    """

    epoch_loss = 0.0
    preds, labs = [], []

    if mode == 'train':
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    for data in tqdm.tqdm(dataloader):
        predictions, labels = func_inference(data)
        loss = criterion(predictions, labels)
        #acc = accuracy(predictions, labels)

        if mode == 'train':
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item()

        predictions = torch.max(torch.softmax(
            predictions, dim=1), dim=1).indices
        labels = labels.cpu().detach().numpy()
        predictions = predictions.cpu().detach().numpy()
        preds.extend(predictions)
        labs.extend(labels)
    
    metrics = {"{}_acc".format(mode): accuracy_score(labs, preds),
               "{}_f1".format(mode): f1_score(labs, preds, average="weighted"),
               "{}_precision".format(mode):  precision_score(labs, preds, average="weighted"),
               "{}_recall".format(mode): recall_score(labs, preds, average="weighted"),
               "{}_loss".format(mode): epoch_loss/len(dataloader)}
    
    return metrics
