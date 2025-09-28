import os
import torch
import argparse
import tqdm
import pandas as pd
import numpy as np
import torch.nn as nn
from numpy import save
from cores.data_base import DataBase
from cores.influence.IF import IF
from cores.influence.TracIn import TracIn
from cores.model_base import ModelBase
from cores.influence.buildGradient import build_gradient
from cores.influence.GD import GD
from cores.influence.GC import GC


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['bert'], required=True, help='model used')
    parser.add_argument('--data', choices=['snippets', 'imdb'], required=True, help='data used')
    parser.add_argument('--dir-checkpoint', type=str, required=True, help='path of directory to load checkpoints and save results')
    parser.add_argument('--df-train', type=str, required=True, help='path of csv file that noise dataset')
    parser.add_argument('--df-clean', type=str, required=True, help='path of csv file that clean dataset (for tracing)')
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

    # Load data
    df_train = pd.read_csv(opt.df_train, sep="\t")
    df_clean = pd.read_csv(opt.df_clean, sep="\t")
    number_classes = len(set(df_train.label))

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

    model_base = ModelBase(number_classes=number_classes, device=opt.device)
    model_base.load_model(os.path.join(opt.dir_checkpoint, 'best.pt'))

    loss_fn = nn.CrossEntropyLoss()
    params = [p for p in model_base.model.parameters() if p.requires_grad][-2:]
    
    # Build gradient
    print(f"build train gradient")
    if not os.path.exists(os.path.join(opt.dir_checkpoint, 'train_gradients.grad')):
        train_gradients = build_gradient(
            inference_fn=model_base.inference,
            loss_fn=loss_fn,
            params=params,
            dataloader=train_loader
        )
        torch.save(train_gradients, os.path.join(opt.dir_checkpoint,'train_gradients.grad'))
    else:
        train_gradients = torch.load(os.path.join(opt.dir_checkpoint, 'train_gradients.grad'))
    print(f"build clean gradient")
    if not os.path.exists(os.path.join(opt.dir_checkpoint,'clean_gradients.grad')):
        test_gradients = build_gradient(
            inference_fn=model_base.inference,
            loss_fn=loss_fn,
            params=params,
            dataloader=clean_loader
        )
        torch.save(test_gradients, os.path.join(opt.dir_checkpoint, 'clean_gradients.grad'))
    else:
        test_gradients = torch.load(os.path.join(opt.dir_checkpoint, 'clean_gradients.grad'))
    print(f"gradient methods: {opt.dir_checkpoint}") 
    #Run methods

    if not os.path.exists(os.path.join(opt.dir_checkpoint, 'GD.npy')):
        print("Run Grad-Dot:")
        results = GD(train_gradients, test_gradients)
        save(os.path.join(opt.dir_checkpoint,'GD'), results)
    else:
        print("Exits GD result!")

    if not os.path.exists(os.path.join(opt.dir_checkpoint, 'GC.npy')):
        print("Run Grad-Cos:")
        results = GC(train_gradients, test_gradients)
        save(os.path.join(opt.dir_checkpoint,'GC'), results)
    else:
        print("Exits GC result")
    
    if not os.path.exists(os.path.join(opt.dir_checkpoint, 'IF.npy')):
        print("Run Influence Function:")
        results = IF(
            test_loader=clean_loader,
            train_loader=train_loader,
            test_gradients=test_gradients,
            train_gradients=train_gradients,
            inference_fn=model_base.inference,
            loss_fn=loss_fn,
            params=params,
            use_exact_hessian=False
        )
        save(os.path.join(opt.dir_checkpoint,'IF'), results)
    else:
        print("Exits IF result")
   
    if not os.path.exists(os.path.join(opt.dir_checkpoint, 'TracIn.npy')):
        start = 0
        f = open(os.path.join(opt.dir_checkpoint,'best_epoch.txt'))
        end = int(f.readline())
        print("Run TracIn from epoch {} to epoch {}".format(start, end))
        results = TracIn(
            dir_checkpoint=opt.dir_checkpoint,
            model_base=model_base,
            train_loader=train_loader,
            test_loader=clean_loader,
            loss_fn=loss_fn,
            start=start,
            end=end
        )
        save(os.path.join(opt.dir_checkpoint,'TracIn'), results)
    else:
        print("Exits TracIn result")
        
