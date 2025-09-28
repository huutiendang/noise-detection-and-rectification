import torch
import argparse
import pandas as pd
import numpy as np
import os
import json
import torch.nn as nn
import tqdm
from cleanlab.rank import get_label_quality_scores
from cores.data_base import DataBase
from cores.model_base import ModelBase
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

def most_common_label(lst):
    count_dict = Counter(lst)
    most_common, count = count_dict.most_common(1)[0]
    return most_common, count

# def rank_by_confidence(pred_probs, df_noise, df_noise_name, method, save_path):
#     n_noise = df_noise["isFlipped"].sum()
#     noisy_label = np.array(list(df_noise["label"]))
#     scores = get_label_quality_scores(labels=noisy_label, pred_probs=pred_probs, method=method, adjust_pred_probs=False)
#     ranked_indices = np.argsort(scores)[:n_noise]
#     df_noise_removed = df_noise.drop(ranked_indices).reset_index(drop=True) 
#     df_noise_removed.to_csv(os.path.join(save_path, f"{method}-removed-{os.path.basename(df_noise_name)}"), sep="\t", index=False)
#     #return df_noise.drop(ranked_indices).reset_index(drop=True)


# def rank_by_gradient(df_clean, df_noise, df_noise_name, n_sample, dir_checkpoint, save_path, gradient_method):
#     n_noise = df_noise["isFlipped"].sum() 

#     gradient_matrix = pd.DataFrame(np.load(os.path.join(dir_checkpoint, f'{gradient_method}.npy')))
#     if len(df_clean) < n_sample:
#         indices = list(df_clean.index)
#     else:
#         indices = list(df_clean.sample(n=n_sample, random_state=42).index)
#     d_aux = gradient_matrix.iloc[indices]
#     d_aux = d_aux.reset_index(drop=True)
#     scores = []
#     for i in tqdm.tqdm(d_aux.columns):
#         d_i = list(d_aux[i])
#         scores.append(np.mean(d_i))
#     #
#     ranked_indices = np.argsort(scores)[:n_noise]
#     df_noise_removed = df_noise.drop(ranked_indices).reset_index(drop=True) 
#     df_noise_removed.to_csv(os.path.join(save_path, f"{gradient_method}-removed-{os.path.basename(df_noise_name)}"), sep="\t", index=False)
#     #return df_noise.drop(ranked_indices).reset_index(drop=True)

def reduction_rate(df):
    n_noise = df["isFlipped"].sum()
    #print("n_noise", n_noise)
    labels = list(df["label"])
    isflipped = list(df["isFlipped"])
    originlabels = list(df["originLabel"])
    count = 0
    for i, j, k in zip(labels, originlabels, isflipped):
        if i != j:
            count += 1
    #print("count", count)
    return float((n_noise - count)/n_noise)

def rank_by_similarity(df_noise, df_noise_name, df_clean, n_sample, dir_checkpoint, save_path, feature_method, k, _tau):
    n_noise = df_noise["isFlipped"].sum()
    dis = pd.DataFrame(np.load(os.path.join(dir_checkpoint, f'{feature_method}.npy')))
    if len(df_clean) < n_sample:
        indices = list(df_clean.index)
    else:
        indices = list(df_clean.sample(n=n_sample, random_state=42).index)

    d_aux = dis.iloc[indices]
    labels_d_aux = list(df_clean.iloc[indices]['label'])
    d_aux = d_aux.reset_index(drop=True)

    scores = []
    new_labels = []
    for i in tqdm.tqdm(d_aux.columns):
        label = df_noise.iloc[i]["label"]
        d_i = list(d_aux[i])
        sorted_indices = np.argsort(d_i)[::-1]
        indices = sorted_indices[:k]
        labels = [labels_d_aux[i] for i in indices]
        most_common, count = most_common_label(labels)
        if count >= _tau*len(labels):
            new_labels.append(most_common)
        else:
            new_labels.append(label)
        s_i = float(labels.count(label)/k)
        scores.append(s_i)
    
    ranked_indices = np.argsort(scores)[:n_noise]
    rectified_labels = [new_labels[i] for i in ranked_indices]
    df_noise.loc[ranked_indices, 'label'] = rectified_labels
    rate = reduction_rate(df_noise)
    with open(os.path.join(opt.save_path, f'{_tau}-{feature_method}-rectified-{os.path.basename(df_noise_name)}.json'), 'w') as f:
        json.dump({"rate": rate}, f)
    #return rate
    #df_noise.to_csv(os.path.join(save_path, f"{feature_method}-rectified-{os.path.basename(df_noise_name)}"), sep="\t", index=False)

# def get_prob(model, dataloader, func_inference):
#     probs = []
#     model.eval()
#     for data in tqdm.tqdm(dataloader):
#         predictions, _ = func_inference(data)
#         prob = torch.softmax(predictions, dim=1).cpu().detach().numpy()[0]
#         probs.append(prob)
#     pred_probs = np.stack(probs, axis=0)
    
#     return pred_probs

# def plot(df, dir_checkpoint, n_sample, k, corr_type):
#     fig, ax = plt.subplots()
#     sns.heatmap(df.corr(method=corr_type), annot=True, fmt='.4f', 
#             cmap=plt.get_cmap('Blues'), cbar=False, ax=ax)
#     ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
#     plt.savefig(os.path.join(dir_checkpoint, f'{corr_type}_{n_sample}_{k}.png'), dpi=300, bbox_inches='tight', pad_inches=0.0) 

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['bert'], required=True, help='model used')
    parser.add_argument('--data', choices=['imdb', 'snippets'], required=True, help='data used')
    parser.add_argument('--dir-checkpoint', type=str, required=True, help='path of directory to load checkpoints and save results')
    parser.add_argument('--df-noise', type=str, required=True, help='path of csv file that noise dataset')
    parser.add_argument('--df-clean', type=str, required=True)
    #parser.add_argument('--feature-method', type=str, choices=['cos', 'dot'])
    #parser.add_argument('--gradient-method', type=str, choices=['TracIn', 'IF', 'GD', 'GC'])
    parser.add_argument('--save-path', type=str, required=True)
    parser.add_argument('--n-sample', type=int, default=1000)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--seed', type=int, required=True, help='random seed')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='device used (cuda/cpu)')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = arguments()

    SEED = opt.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    if not torch.cuda.is_available() and opt.device == 'cuda':
        print('Your device don\'t have cuda, please check or select cpu and retraining')
        exit(0)

    if opt.device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    df_cle = pd.read_csv(opt.df_clean, sep="\t")
    df_noi = pd.read_csv(opt.df_noise, sep="\t")
    number_classes = len(set(df_noi.label))

    model_base = ModelBase(number_classes=number_classes, device=opt.device)
    model_base.load_model(os.path.join(opt.dir_checkpoint, 'best.pt'))
    data_base = DataBase(opt.data)
    
    noise_loader = data_base.get_dataloader(
        df=df_noi,
        batch_size=1,
        mode='test',
        num_workers=0
    )

    clean_loader = data_base.get_dataloader(
        df=df_cle,
        batch_size=1,
        mode='test',
        num_workers=0
    )

    # pred_probs = get_prob(
    #     model=model_base.model,
    #     dataloader=noise_loader,
    #     func_inference=model_base.inference,
    # )
    # sc = rank_by_confidence(pred_probs=pred_probs, df_noise=df_noi, method="self_confidence", save_path=opt.save_path, df_noise_name=opt.df_noise)
    # nm = rank_by_confidence(pred_probs=pred_probs, df_noise=df_noi, method="normalized_margin", save_path=opt.save_path, df_noise_name=opt.df_noise)
    # ce = rank_by_confidence(pred_probs=pred_probs, df_noise=df_noi, method="confidence_weighted_entropy", save_path=opt.save_path, df_noise_name=opt.df_noise)
    #print(len(sc), len(nm), len(ce))
    for tau in [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
        cos = rank_by_similarity(df_clean=df_cle, df_noise=df_noi, feature_method='cos', k=opt.k, dir_checkpoint=opt.dir_checkpoint, n_sample=opt.n_sample, save_path=opt.save_path, df_noise_name=opt.df_noise, _tau=tau)
        dot = rank_by_similarity(df_clean=df_cle, df_noise=df_noi, feature_method='dot', k=opt.k, dir_checkpoint=opt.dir_checkpoint, n_sample=opt.n_sample, save_path=opt.save_path, df_noise_name=opt.df_noise, _tau=tau)

    #print(len(cos), len(dot))

    # tracin = rank_by_gradient(df_clean=df_cle, df_noise=df_noi, n_sample=opt.n_sample, dir_checkpoint=opt.dir_checkpoint, gradient_method='TracIn', save_path=opt.save_path, df_noise_name=opt.df_noise)
    # gd = rank_by_gradient(df_clean=df_cle, df_noise=df_noi, n_sample=opt.n_sample, dir_checkpoint=opt.dir_checkpoint, gradient_method='GD', save_path=opt.save_path, df_noise_name=opt.df_noise)
    # gc = rank_by_gradient(df_clean=df_cle, df_noise=df_noi, n_sample=opt.n_sample, dir_checkpoint=opt.dir_checkpoint, gradient_method='GC', save_path=opt.save_path, df_noise_name=opt.df_noise)
    # If = rank_by_gradient(df_clean=df_cle, df_noise=df_noi, n_sample=opt.n_sample, dir_checkpoint=opt.dir_checkpoint, gradient_method='IF', save_path=opt.save_path, df_noise_name=opt.df_noise)

    #print(len(tracin), len(gd), len(gc), len(If))
    # corr = {"SC": sc, "NM": nm, "CE": ce, "COS": cos, "DOT": dot, "TracIn": tracin, "IF": If, "GD": gd, "GC": gc}
    # df = pd.DataFrame(corr)
    # #print(df)
    # plot(df=df, n_sample=opt.n_sample, k=opt.k, dir_checkpoint=opt.dir_checkpoint, corr_type='spearman')

