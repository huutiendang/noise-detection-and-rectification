import torch
import numpy as np
import tqdm
import argparse
import os
from cores.data_base import DataBase
from cores.model_base import ModelBase
from cores.feature_methods import FeatureBased
import pandas as pd

def get_prob(model, dataloader, func_get_feature):
    probs = []
    model.eval()
    for data in tqdm.tqdm(dataloader):
        predictions = func_get_feature(data)
        probs.append(predictions.cpu().detach().numpy())
    pred_feats = np.concatenate(probs, axis=0)
    
    return pred_feats


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['bert'], required=True, help='model used')
    parser.add_argument('--data', choices=['imdb', 'snippets'], required=True, help='data used')
    parser.add_argument('--dir-checkpoint', type=str, required=True, help='path of directory to load checkpoints and save results')
    parser.add_argument('--df-train', type=str, required=True, help='path of csv file that noise dataset')
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

    df_train = pd.read_csv(opt.df_train, sep="\t")
    number_classes = len(set(df_train.label))

    #print(df_train)
    df_noise = df_train[df_train["isFlipped"] == 1]
    df_clean = df_train[df_train["isFlipped"] == 0]
    #print(df_noise)
    df_noise_class_0 = df_noise[df_noise["originLabel"] == 1]
    df_noise_class_1 = df_noise[df_noise["originLabel"] == 0]
    
    df_clean_class_0 = df_clean[df_clean["label"] == 0]
    df_clean_class_1 = df_clean[df_clean["label"] == 1]
    
    model_base = ModelBase(number_classes=number_classes, device=opt.device)
    model_base.load_model(os.path.join(opt.dir_checkpoint, 'best.pt'))
    data_base = DataBase(opt.data) 

    nearest_neighbor = FeatureBased(
        model=model_base.model,
        dir_checkpoint="/home/ttathu-x/Documents/noise-detection/stats/imdb/random_10_noise_0_clean_1",
        func_get_feature=model_base.get_feature
    )
    
    clean_class_0_loader = data_base.get_dataloader(
        df=df_clean_class_0,
        batch_size=1,
        mode='test',
        num_workers=0
    )

    noise_class_0_loader = data_base.get_dataloader(
        df=df_noise_class_0,
        batch_size=1,
        mode='test',
        num_workers=0
    )

    clean_class_1_loader = data_base.get_dataloader(
        df=df_clean_class_1,
        batch_size=1,
        mode='test',
        num_workers=0
    )
    

    noise_class_1_loader = data_base.get_dataloader(
        df=df_noise_class_1,
        batch_size=1,
        mode='test',
        num_workers=0
    )

    nearest_neighbor.run_all_config(
        trainloader=noise_class_0_loader,
        testloader=clean_class_1_loader,
        save_features=True
    )



    # model_base = ModelBase(number_classes=number_classes, device=opt.device)

    # model_base.load_model(os.path.join(opt.dir_checkpoint, 'best.pt'))
    # data_base = DataBase(opt.data)
    
    # train_loader = data_base.get_dataloader(
    #     df=df_train,
    #     batch_size=16,
    #     mode='test',
    #     num_workers=0
    # )

    # probs = get_prob(
    #     model=model_base.model,
    #     dataloader=train_loader,
    #     func_get_feature=model_base.get_feature,
    # )
    # print(probs.shape)