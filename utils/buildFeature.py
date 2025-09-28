import tqdm
import numpy as np
from scipy.spatial import distance

sim_func_dict = {
    "dot": lambda x, y: np.dot(x, y),
    "cos": lambda x, y: 1-distance.cosine(x, y),
    "euc": lambda x, y: np.linalg.norm(x-y)
}

def dist2anchor(train_features, train_label, dist_type=["euc", "dot", "cos"]):
    unique_classes = np.unique(train_label)

    class_means = {}
    
    for cls in unique_classes:
        class_mask = (train_label == cls)
        class_mean = np.mean(train_features[class_mask], axis=0)
        class_means[cls] = class_mean
    
    dist_to_class_mean = {cls: {dist_type[i]: [] for i in range(len(dist_type))} for cls in unique_classes}
    dist_to_global_mean = {dist_type[i]: [] for i in range(len(dist_type))}

    for cls, class_mean in class_means.items():
        class_mask = (train_label == cls)
        for dist_name in dist_type:
            dist_func = sim_func_dict[dist_name]
            dist = [dist_func(feature, class_mean) for feature in train_features[class_mask]]
            dist_to_class_mean[cls][dist_name] = dist
            dist_to_global_mean[dist_name].extend(dist)
    
    dists = {
        'class_mean': dist_to_class_mean,
        'global_mean': dist_to_global_mean
    }
    return dists

def build_feature(model, loader, func_inference):
    preds, labs = [], []
    model.eval()
    
    for data in tqdm.tqdm(loader):
        pred, lab = func_inference(data)
        preds.append(pred.cpu().detach().numpy())
        labs.extend(lab.cpu().detach().numpy())
    
    preds_ = np.concatenate(preds, axis=0)
    return preds_, labs