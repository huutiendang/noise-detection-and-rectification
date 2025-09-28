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


# def negative_shannon_entropy(pred_probs, df_noise):
#     pass

def cleanlab(pred_probs, df_noise, method):
    noisy_label = np.array(list(df_noise["label"]))
    out = get_label_quality_scores(labels=noisy_label, pred_probs=pred_probs, method=method, adjust_pred_probs=False)
    error_index = np.argsort(out)
    n_noise = df_noise["isFlipped"].sum()
    
    percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    top = [int(p*n_noise) for p in percents]
    error_acc = []
    for t in top:
        acc = df_noise.iloc[error_index]["isFlipped"][:t].sum()/t
        error_acc.append(acc)
    return error_acc


def get_prob(model, dataloader, func_inference):
    probs = []
    model.eval()
    for data in tqdm.tqdm(dataloader):
        predictions, _ = func_inference(data)
        prob = torch.softmax(predictions, dim=1).cpu().detach().numpy()[0]
        probs.append(prob)
    pred_probs = np.stack(probs, axis=0)
    
    return pred_probs
    


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['bert'], required=True, help='model used')
    parser.add_argument('--data', choices=['imdb', 'snippets'], required=True, help='data used')
    parser.add_argument('--dir-checkpoint', type=str, required=True, help='path of directory to load checkpoints and save results')
    parser.add_argument('--df-noise', type=str, required=True, help='path of csv file that noise dataset')
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

    df_noise = pd.read_csv(opt.df_noise, sep="\t")
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
    pred_probs = get_prob(
        model=model_base.model,
        dataloader=noise_loader,
        func_inference=model_base.inference,
    )
    metrics = {
        # "entropy": negative_shannon_entropy(pred_probs=pred_probs, df_noise=df_noise),
        "self_confidence": cleanlab(pred_probs=pred_probs, df_noise=df_noise, method="self_confidence"),
        "normalized_margin": cleanlab(pred_probs=pred_probs, df_noise=df_noise, method="normalized_margin"),
        "confidence_weighted_entropy": cleanlab(pred_probs=pred_probs, df_noise=df_noise, method="confidence_weighted_entropy")
    }
    #print(f"checkpoint: {opt.dir_checkpoint}, confident-based: {metrics}")
    with open(os.path.join(opt.dir_checkpoint, 'confident.json'), 'w') as f:
        json.dump(metrics, f)
