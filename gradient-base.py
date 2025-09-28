import torch
import argparse
import pandas as pd
import numpy as np
import os
import json
import torch.nn as nn
import tqdm
from cleanlab.rank import get_label_quality_scores
from logging import raiseExceptions
from cores.influence.buildGradient import build_gradient
from cores.data_base import DataBase
from cores.model_base import ModelBase
from utils.run_train import run_train


def gradient_detection(df_noise, df_clean, n_sample, gradient_matrix):
    if len(df_clean) < n_sample:
        indices = list(df_clean.index)
    else:
        indices = list(df_clean.sample(n=n_sample, random_state=42).index)
    d_aux = gradient_matrix.iloc[indices]
    d_aux = d_aux.reset_index(drop=True)
    scores = []
    for i in tqdm.tqdm(d_aux.columns):
        d_i = list(d_aux[i])
        scores.append(np.mean(d_i))
    #
    error_index = np.argsort(scores)
    n_noise = df_noise["isFlipped"].sum()
    percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    top = [int(p*n_noise) for p in percents]
    error_acc = []
    for t in top:
        acc = df_noise.iloc[error_index]["isFlipped"][:t].sum()/t
        error_acc.append(acc)
    return error_acc


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['bert'], required=True, help='model used') 
    parser.add_argument('--data', choices=['imdb', 'snippets'], required=True, help='data used')
    parser.add_argument('--dir-checkpoint', type=str, required=True, help='path of directory to load checkpoints and save results')
    parser.add_argument('--df-noise', type=str, required=True, help='path of csv file that noise dataset')
    parser.add_argument('--df-clean', type=str, required=True, help='path of csv file that test dataset')
    parser.add_argument('--gradient-method', choices=['TracIn', 'IF', 'GD', 'GC'], required=True, help='gradient method')
    parser.add_argument('--seed', type=int, required=True, help='random seed')
    parser.add_argument('--n_sample', type=int, required=True, help='the number of sampling')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='device used (cuda/cpu)')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
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
      
    df_noise = pd.read_csv(opt.df_noise, sep="\t")
    df_clean = pd.read_csv(opt.df_clean, sep="\t")
    number_classes = len(set(df_noise.label))

    model_base = ModelBase(number_classes=number_classes, device=opt.device)
    model_base.load_model(os.path.join(opt.dir_checkpoint, 'best.pt'))

    data_base = DataBase(opt.data)

    noise_loader = data_base.get_dataloader(
        df=df_noise,
        batch_size=1,
        mode='test',
        num_workers=0
    )

    print(f"Checkpoint: {opt.dir_checkpoint}, len of the aux: {opt.n_sample}")
    gradient_matrix = pd.DataFrame(np.load(os.path.join(opt.dir_checkpoint, opt.gradient_method) + '.npy'))
    
    result = {f"{opt.gradient_method}": gradient_detection(df_noise, df_clean, opt.n_sample, gradient_matrix)}
    with open(os.path.join(opt.dir_checkpoint, f'{opt.gradient_method}_n_sample_{opt.n_sample}.json'), 'w') as f:
        json.dump(result, f)
