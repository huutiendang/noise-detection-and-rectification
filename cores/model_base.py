from models.BertClassifier import build_bert_model, load_bert_model


class ModelBase():
    def __init__(self, number_classes, device='cuda'):
        self.number_classes = number_classes
        self.device = device
        self.model = None

    def build_model(self):
        self.model = build_bert_model(self.number_classes, self.device)

    def load_model(self, path_pretrain):
        self.model = load_bert_model(path_pretrain, self.number_classes, self.device)
                
    def get_feature(self, data):
        ids = data['ids'].to(self.device)
        attention_mask = data['attention_mask'].to(self.device)
        token_type_ids = data['token_type_ids'].to(self.device)
        feature = self.model.module._get_feature(
            ids, attention_mask, token_type_ids)
        return feature
    
    def get_linear_feature(self, data):
        ids = data['ids'].to(self.device)
        attention_mask = data['attention_mask'].to(self.device)
        token_type_ids = data['token_type_ids'].to(self.device)
        feature = self.model.module._get_linear_feature(
            ids, attention_mask, token_type_ids)
        return feature


    def inference(self, data):
        """ Returns model's prediction and true label of sample
        Args:
            model: pytorch model
            data: batch size/simple of samples
        Returns:
            predictions: predictions of model on data
            labels: label of data
        """
        ids = data['ids'].to(self.device)
        attention_mask = data['attention_mask'].to(self.device)
        token_type_ids = data['token_type_ids'].to(self.device)
        labels = data['label'].to(self.device)
        predictions = self.model(ids, attention_mask, token_type_ids)
        
        return predictions, labels
