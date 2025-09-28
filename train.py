import yaml
import os
import argparse
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from cores.model_base import ModelBase
from cores.data_base import DataBase
from utils.run_train import run_train


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['bert'], required = True, help='model used')
    parser.add_argument('--data', choices=['snippets', 'imdb', 'tweeteval'], required=True, help='data used')
    parser.add_argument('--df-train', type=str, required=True, help='path of csv file used to training dataset')
    parser.add_argument('--df-val', type=str, required=True, help='path of csv file to validation dataset')
    parser.add_argument('--dir-checkpoint', type=str, required=True, help='path to save checkpoints')
    parser.add_argument('--seed', type=int, required=True, help='random seed')

    parser.add_argument('--epoch', type=int, default=15, help='number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--num-workers', type=int, default=2, help='number of workers on CPU used to load dataloader')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='device used (cuda/cpu)')
    parser.add_argument('--save-each-epoch', action='store_true', help='save checkpoint of each epoch')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = arguments()
    SEED = opt.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    if not os.path.isdir(opt.dir_checkpoint):
        print(f"Directory {opt.dir_checkpoint} does not exist")
        os.makedirs(opt.dir_checkpoint)
        print(f"Created {opt.dir_checkpoint}")

    if not torch.cuda.is_available() and opt.device == 'cuda':
        print('Your device don\'t have cuda, please check or select cpu and retraining')
        exit(0)

    if opt.device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    name_project = opt.dir_checkpoint.split('/')[-1]

    # Load data
    df_train = pd.read_csv(opt.df_train, sep="\t")
    df_val = pd.read_csv(opt.df_val, sep="\t")
    number_classes = len(set(df_train.label))

    # Build model
    model_base = ModelBase(number_classes=number_classes, device=opt.device)
    model_base.build_model()

    data_base = DataBase(opt.data)

    train_loader = data_base.get_dataloader(
        df=df_train,
        batch_size=opt.batch_size,
        mode='train',
        num_workers=opt.num_workers
    )

    val_loader = data_base.get_dataloader(
        df=df_val,
        batch_size=opt.batch_size,
        mode='test',
        num_workers=opt.num_workers
    )

    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model_base.model.parameters(), lr=float(opt.lr), betas=(0.9, 0.999))

    best_valid_acc = (-1.0) * float("Inf")

    for epoch in range(opt.epoch):
        print(f"Epoch: {epoch}/{opt.epoch}")
        train_metrics = run_train(
            model=model_base.model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=loss_fn,
            func_inference=model_base.inference,
            mode='train'
        )
        val_metrics = run_train(
            model=model_base.model,
            dataloader=val_loader,
            optimizer=optimizer,
            criterion=loss_fn,
            func_inference=model_base.inference,
            mode='val'
        )
        
        print("Train metrics:", train_metrics)
        print("Val metrics", val_metrics)

        if opt.save_each_epoch:
            torch.save(model_base.model.state_dict(), os.path.join(opt.dir_checkpoint, 'epoch_{}.pt'.format(epoch)))

        if best_valid_acc < val_metrics["val_acc"]:
            best_valid_acc = val_metrics["val_acc"]
            torch.save(model_base.model.state_dict(), os.path.join(opt.dir_checkpoint, 'best.pt'))
            f = open(os.path.join(opt.dir_checkpoint, 'best_epoch.txt'), "w")
            f.write(str(epoch))
            f.close()
            f = open(os.path.join(opt.dir_checkpoint, 'best_result.txt'), "w")
            f.write("Train:\n {}\nVal:\n {}".format(train_metrics, val_metrics))
            f.close()
            print(f"Model saved to ==> {opt.dir_checkpoint + '/best.pt'} at epoch {epoch}")

        train_metrics.update(val_metrics)

    print(f'Finished training')
