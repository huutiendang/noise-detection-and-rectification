import pandas as pd
import argparse
import numpy as np
import torch
import json
import os 

def reduction_rate(df):
    n_noise = df["isFlipped"].sum()
    print("n_noise", n_noise)
    labels = list(df["label"])
    isflipped = list(df["isFlipped"])
    originlabels = list(df["originLabel"])
    count = 0
    for i, j, k in zip(labels, originlabels, isflipped):
        if i != j:
            count += 1
    print("count", count)
    return float((n_noise - count)/n_noise)

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['bert'], required=True, help='model used')
    parser.add_argument('--data', choices=['imdb', 'snippets'], required=True, help='data used')
    parser.add_argument('--df-train', type=str, required=True, help='path of csv file that noise dataset')
    parser.add_argument('--feature-method', type=str, choices=['cos', 'dot'])
    parser.add_argument('--save-path', type=str, required=True)
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

    df_tr = pd.read_csv(opt.df_train, sep="\t")
    reduction_rate_ = reduction_rate(df=df_tr)

    with open(os.path.join(opt.save_path, f'{opt.feature_method}_{os.path.basename(opt.df_train)}.json'), 'w') as f:
        json.dump({"rate": reduction_rate_}, f)