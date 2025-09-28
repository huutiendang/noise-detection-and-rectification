import torch
import argparse
import pandas as pd
import numpy as np
import os
from cores.data_base import DataBase
from cores.feature_methods import FeatureBased
from cores.model_base import ModelBase


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['bert'], required=True, help='model used')
    parser.add_argument('--data', choices=['snippets', 'imdb'], required=True, help='data used')
    parser.add_argument('--dir-checkpoint', type=str, required=True, help='path of directory to load checkpoints and save results')
    parser.add_argument('--df-train', type=str, required=True, help='path of csv file that noise dataset')
    parser.add_argument('--df-clean', type=str, required=True, help='path of csv file that clean dataset (for tracing)')
    parser.add_argument('--seed', type=int, required=True, help='random seed')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cpu', help='device used (cuda/cpu)')
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

    df_train = pd.read_csv(opt.df_train, sep="\t")
    df_clean = pd.read_csv(opt.df_clean, sep="\t")
    number_classes = len(set(df_train.label))

    model_base = ModelBase(
        number_classes=number_classes,
        device=opt.device
    )

    
    model_base.load_model(os.path.join(opt.dir_checkpoint, 'best.pt'))
    data_base = DataBase(opt.data)

    train_loader = data_base.get_dataloader(
        df=df_train,
        batch_size=1,
        mode='test',
        num_workers=1
    )

    clean_loader = data_base.get_dataloader(
        df=df_clean,
        batch_size=1,
        mode='test',
        num_workers=1
    )
    print(f"feature method: {opt.dir_checkpoint}")
    # Run
    nearest_neighbor = FeatureBased(
        model=model_base.model,
        dir_checkpoint=opt.dir_checkpoint,
        func_get_feature=model_base.get_linear_feature
    )
    
    nearest_neighbor.run_all_config(
        trainloader=train_loader,
        testloader=clean_loader,
        save_features=False
    )

