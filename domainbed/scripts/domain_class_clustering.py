# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import random

sys.path.append("./domainbed")

import PIL
import numpy as np
import torch
import torch.utils.data
import torchvision

import clip

from domainbed import algorithms
from domainbed import datasets
from domainbed import hparams_registry
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed import config

from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import pickle

PATH = "domain_class_routings"
NUM_DOMAINS = 4

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default='/mnt/lustre/share/boli/domainbed_data')
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
                        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and '
                             'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--drop_out', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
                        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
                        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--corr_heatmap', action='store_true')
    parser.add_argument('--feature_vis', action='store_true')
    parser.add_argument('--dendogram_vis', action='store_true')



    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    # print("assigning writer")
    # writer = SummaryWriter(PATH + f"/log_dir/" + args.dataset)
    # print("summary writer assigned")

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
                                                  misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    if args.batch_size is not None:
        hparams['batch_size'] = args.batch_size
    if args.drop_out is not None:
        hparams['drop_out'] = args.drop_out
    if args.lr is not None:
        hparams['lr'] = args.lr
    if args.weight_decay is not None:
        hparams['weight_decay'] = args.weight_decay

    # print('HParams:')
    # for k, v in sorted(hparams.items()):
    #     print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
                                               args.test_envs, hparams)
    else:
        raise NotImplementedError
    
    config.NUM_CLASSES = dataset.num_classes

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        print(env)
        out, in_ = misc.split_dataset(env, int(len(env) * args.holdout_fraction),
                                    misc.seed_hash(args.trial_seed, env_i))
        in_splits.append(in_)

    
    config.NUM_CLASSES = dataset.num_classes

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # 
    domain_classes_to_features = {(i,j) : [] for i in range(NUM_DOMAINS) for j in range(dataset.num_classes) }
    domain_classes_to_count = {(i,j) : 0 for i in range(NUM_DOMAINS) for j in range(dataset.num_classes) }

    model, preprocess = clip.load(config.CLIP_MODEL, device=device)
    model.eval()

    train_loaders = [DataLoader(
        dataset=env,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, env in enumerate(in_splits)
    ]

    for domain_idx, dl in enumerate(train_loaders):
        for x, y in dl:

            d_class_list = []
            for c in y:
                d_class_list.append((domain_idx, c.item()))

            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                features = model.encode_image(x)

            for i in range(x.shape[0]):
                if len(domain_classes_to_features[d_class_list[i]]) == 0:
                    domain_classes_to_features[d_class_list[i]].append(features[i].cpu().numpy())
                    domain_classes_to_count[d_class_list[i]] = 1
                
                else:
                    n = domain_classes_to_count[d_class_list[i]]
                    domain_classes_to_features[d_class_list[i]][0] = (domain_classes_to_features[d_class_list[i]][0]*n + features[i].cpu().numpy())/(n+1)
                    domain_classes_to_count[d_class_list[i]] += 1
    
    all_means = []
    for k, v in domain_classes_to_count.items():
        assert(v>=1, f"Domain Class : {k} not found")
        print(k, v, domain_classes_to_features[k][0].shape)
        all_means.append(domain_classes_to_features[k][0])
    
    all_means = np.stack(all_means, axis=0)
    expert_labels = KMeans(n_clusters=config.NUM_EXPERTS, random_state=0, n_init=10).fit(all_means).labels_

    domain_classes_to_experts = {(i,j) : 0 for i in range(NUM_DOMAINS) for j in range(dataset.num_classes) }

    for idx, (k, v) in enumerate(domain_classes_to_experts.items()):
        domain_classes_to_experts[k] = expert_labels[idx]

    print("Mapping from domain classes to experts")
    print(domain_classes_to_experts)

    filename = os.path.join(PATH, f"{args.dataset}.pkl")
    filehandler = open(filename, 'wb')
    pickle.dump(domain_classes_to_experts, filehandler)
    print("Saved dictionary")

    pca = PCA(2)
    df = pca.fit_transform(all_means)
    
    tmp = [(i,j) for i in range(NUM_DOMAINS) for j in range(dataset.num_classes)]

    # plt.scatter(df[:,0], df[:,1], label=expert_labels)
    # for i in range(df.shape[0]):
    #     plt.annotate(tmp[i], (df[i, 0], df[i,1]))

    tmp_df = pd.DataFrame()
    tmp_df["pca comp-1"] = df[:,0]
    tmp_df["pca comp-2"] = df[:,1]
    plt.figure(figsize=(10,10))
    plt.rcParams.update({'font.size': 20})
    sns.scatterplot(x="pca comp-1", y="pca comp-2", hue=expert_labels,
                    palette=sns.color_palette("hls", config.NUM_EXPERTS),
                    data=tmp_df, s=150).set(title=f"Semantic class clustering({args.dataset})") 
    
    for i in range(df.shape[0]):
        if random.random() > 0.1:
            continue
        plt.annotate(tmp[i], (df[i, 0], df[i,1]))
    

    plt.savefig(os.path.join(PATH, f"vis_{args.dataset}.png"), format='png', bbox_inches='tight')


