import torch.nn as nn
from transformers import BertConfig, BertModel
import torch


class BertClassifier(nn.Module):
    def __init__(self, output):
        super().__init__()
        config = BertConfig.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained(
            'bert-base-uncased',
            config=config
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(config.hidden_size, output)
        nn.init.normal_(self.fc.weight, std=0.2)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, embedded = self.bert(input_ids, attention_mask, token_type_ids, return_dict=False)
        x = self.dropout(embedded)
        out = self.fc(x)
        return out

    def _get_feature(self, input_ids, attention_mask, token_type_ids):
        _, embedded = self.bert(input_ids, attention_mask, token_type_ids, return_dict=False)
        return embedded.squeeze(0) # (B, 512) --> (512)
    
    def _get_linear_feature(self, input_ids, attention_mask, token_type_ids):
        _, embedded = self.bert(input_ids, attention_mask, token_type_ids, return_dict=False)
        x = self.dropout(embedded)
        out = self.fc(x)
        return out.squeeze(0)



def load_bert_model(path, number_classes, device):
    """
    It loads a pretrained BERT model, and then it loads the weights of the pretrained model into the
    model
    
    :param path: the path to the model you want to load
    :param number_classes: The number of classes in the dataset
    :param device: the device to run the model on
    :return: The model is being returned.
    """
    model = BertClassifier(output=number_classes)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.to(device)
    return model


def build_bert_model(number_classes, device):
    """
    It creates a BertClassifier object, which is a PyTorch model that takes in a BERT model and adds a
    linear layer on top of it. 
    
    The linear layer is the output layer that we'll use to make predictions. 
    
    The number of output nodes in the linear layer is equal to the number of classes we want to predict.
    
    
    In our case, we want to predict whether a movie review is positive or negative, so we only need one
    output node. 
    
    We'll use the sigmoid activation function to make sure the output is between 0 and 1
    
    :param number_classes: The number of classes in the dataset
    :param device: the device to run the model on
    :return: A model that is a BertClassifier with the number of classes as the output.
    """
    model = BertClassifier(output=number_classes)
    model = nn.DataParallel(model)
    model.to(device)
    return model
