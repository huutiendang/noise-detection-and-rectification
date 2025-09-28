import pandas as pd
import numpy as np
import argparse
import json
import os

def arguments():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', choices=['bert'], required=True, help='model used')
    parser.add_argument('--data', choices=['imdb', 'snippets'], required=True, help='data used')
    parser.add_argument('--dir-checkpoint', type=str, required=True, help='path of directory to load checkpoints and save results')
    #parser.add_argument('--df-noise', type=str, required=True, help='path of csv file that noise dataset')
    #parser.add_argument('--df-clean', type=str, required=True)
    #parser.add_argument('--feature-method', type=str, choices=['cos', 'dot'])
    #parser.add_argument('--gradient-method', type=str, choices=['TracIn', 'IF', 'GD', 'GC'])
    parser.add_argument('--n-sample', type=int, required=True)
    #parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True, help='random seed')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='device used (cuda/cpu)')
    opt = parser.parse_args()
    return opt

def load_json(path):
    with open(path, 'r') as f:
        dict_ = json.load(f)
    return dict_


if __name__ == "__main__":
    opt = arguments()
    #confident = load_json(os.path.join(opt.dir_checkpoint, "confident.json"))
    cosines = {}
    dots = {}
    for k in [1, 2, 5, 10, 20, 50, 100, 200]:
        cos = load_json(os.path.join(opt.dir_checkpoint, f"cos_n_sample_{opt.n_sample}_k_{k}.json"))
        dot = load_json(os.path.join(opt.dir_checkpoint, f"dot_n_sample_{opt.n_sample}_k_{k}.json"))
        cosines[f"k={k}"]= cos["cos"]
        dots[f"k={k}"] = dot["dot"]
    with open(os.path.join(opt.dir_checkpoint, 'cosines.json'), 'w') as f:
        json.dump(cosines, f)
    with open(os.path.join(opt.dir_checkpoint, 'dots.json'), 'w') as f:
        json.dump(dots, f)  
    #gc = load_json(os.path.join(opt.dir_checkpoint, f"GC_n_sample_{opt.n_sample}.json"))
    #gd = load_json(os.path.join(opt.dir_checkpoint, f"GD_n_sample_{opt.n_sample}.json"))
    #tracin = load_json(os.path.join(opt.dir_checkpoint, f"TracIn_n_sample_{opt.n_sample}.json"))
    #If = load_json(os.path.join(opt.dir_checkpoint, f"IF_n_sample_{opt.n_sample}.json"))

   # merged_dict = {**confident, **cos, **dot, **gc, **gd, **tracin, **If}
    #with open(os.path.join(opt.dir_checkpoint, 'results.json'), 'w') as f:
    #    json.dump(merged_dict, f)

