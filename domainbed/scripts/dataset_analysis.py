# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time

sys.path.append("./domainbed")

import PIL
import numpy as np
import torch
import torch.utils.data
import torchvision
import clip
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from domainbed import algorithms
from domainbed import datasets
from domainbed import hparams_registry
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed import config
from domainbed import class_distributor

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

# from torch.utils.tensorboard import SummaryWriter

# import tensorflow as tf 
# import tensorboard as tb
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


PATH = "dataset_analysis/"

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
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
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
        uda = []

        out, in_ = misc.split_dataset(env, int(len(env) * args.holdout_fraction),
                                      misc.seed_hash(args.trial_seed, env_i))
        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_, int(len(in_) * args.uda_holdout_fraction),
                                          misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append(in_)
        if len(uda):
            uda_splits.append((uda, uda_weights))

    # if args.task == "domain_adaptation" and len(uda_splits) == 0:
    #     raise ValueError("Not enough unlabeled samples for domain adaptation.")


    model, preprocess = clip.load(config.CLIP_MODEL, device=device)
    model.eval()
    print("model loaded")
    if args.corr_heatmap:
        NUM_SAMPLES = 20
        FEATURE_IDX = random.sample(list(range(100)), 5) 
        CLASS = 1
        for f in FEATURE_IDX:
            all_domain_features = []
            train_loaders = [iter(DataLoader(
                dataset=env,
                batch_size=hparams['batch_size'],
                num_workers=dataset.N_WORKERS))
                for i, env in enumerate(in_splits)
            ]
            for i, loader in enumerate(train_loaders):
                filled = False
                curr_num = 0
                domain_features = []
                while curr_num < NUM_SAMPLES:
                    # minibatches = [(x.to(device), y.to(device)) for x, y in next(loader)]
                    all_x, all_y = next(loader)
                    # all_x = torch.cat([x for x, y in minibatches])
                    # all_y = torch.cat([y for x, y in minibatches])
                    
                    all_x = all_x.to(device)
                    all_y = all_y.to(device)

                    with torch.no_grad():
                        features = model.encode_image(all_x)
                    
                    for i in range(all_x.shape[0]):
                        if curr_num == NUM_SAMPLES:
                            break
                        if all_y[i] == CLASS:
                            domain_features.append(features[i])
                            curr_num += 1

                domain_features = torch.stack(domain_features, dim=0).cpu().numpy()[:, f]
                all_domain_features.append(domain_features)

            all_domain_features = np.stack(all_domain_features, axis=0)

            cor_mat = np.corrcoef(all_domain_features)
            cor_mat = np.around(cor_mat, 3)
            print("all_domain_features shape :", all_domain_features.shape)
            print("cor_mat shape :", cor_mat.shape)
            
            plt.figure(figsize=(15,10))
            heatmap = sns.heatmap(cor_mat, cmap='RdBu', vmin=-1, vmax=1)
            fig = heatmap.get_figure()

            isExist = os.path.exists(PATH)
            if not isExist:
                os.makedirs(PATH)
            isExist = os.path.exists(os.path.join(PATH, args.dataset))
            if not isExist:
                os.makedirs(os.path.join(PATH, args.dataset))
            
            fig.savefig(os.path.join(PATH, args.dataset, f"corr_heatmap_c{CLASS}_f{f}"))

    if args.feature_vis:
        print("trying feature vis")
        MAX_SAMPLES = 10000
        curr_samples = 0
        train_loaders = [DataLoader(
                dataset=env,
                batch_size=hparams['batch_size'],
                num_workers=dataset.N_WORKERS)
                for i, env in enumerate(in_splits)
            ]
        for dom_idx, loader in train_loaders:
            for idx, (x,y) in enumerate(loader):
                if curr_samples > MAX_SAMPLES:
                    break
                labels = []
                for elem in y:
                    labels.append(f"d{dom_idx}_c{elem.item()}")
                x = x.to(device)
                curr_samples += x.shape[0]
                with torch.no_grad():
                    features = model.encode_image(x)
                # writer.add_embedding(features, metadata=labels, global_step=0)

            if curr_samples > MAX_SAMPLES:
                print("breaking")
                break
        
    if args.dendogram_vis:
        MAX_SAMPLES = 100
        features = []
        labels = []
        curr_samples = 0
        train_loaders = [iter(DataLoader(
                dataset=env,
                batch_size=hparams['batch_size'],
                num_workers=dataset.N_WORKERS))
                for i, env in enumerate(in_splits)
            ]
        iters = 0
        while True:
            iters += 1
            try:
                x,y = next(train_loaders[iters%4])
            except Exception as e:
                print(e)
                continue
            curr_samples += 1
            for elem in y:
                labels.append(f"d{iters%4}_c{elem.item()}")
            x = x.to(device)
            curr_samples += x.shape[0]
            with torch.no_grad():
                features.append(model.encode_image(x))
            if curr_samples > MAX_SAMPLES:
                break
        
        features = torch.cat(features, dim=0)
        torch_cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        sim = torch_cos(features.unsqueeze(dim=1), features.unsqueeze(dim=0))
        assert tuple(sim.shape) == (features.shape[0], features.shape[0]), f"Error :: Cos shape : {sim.shape}, Features shape : {features.shape}"
        dissimilarity = ((1 - sim)/2).cpu().numpy()
        dissimilarity = np.around(dissimilarity, 2)
        print([(i,dissimilarity[i,i]) for i in range(dissimilarity.shape[0]) if dissimilarity[i,i]==0])
        print(dissimilarity)
        plt.figure(figsize=(150,150))
        Z = linkage(squareform(dissimilarity), 'complete')
        dendrogram(Z, labels=labels, orientation='top', leaf_rotation=90)
        plt.savefig(PATH + f'dendogram/{args.dataset}.png', format='png', bbox_inches='tight')
        
    # uda_loaders = [InfiniteDataLoader(
    #     dataset=env,
    #     weights=env_weights,
    #     batch_size=hparams['batch_size'],
    #     num_workers=dataset.N_WORKERS)
    #     for i, (env, env_weights) in enumerate(uda_splits)
    #     if i in args.test_envs]

    # eval_loaders = [FastDataLoader(
    #     dataset=env,
    #     batch_size=64,
    #     num_workers=dataset.N_WORKERS)
    #     for env, _ in (in_splits + out_splits + uda_splits)]
    # eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    # eval_loader_names = ['env{}_in'.format(i)
    #                      for i in range(len(in_splits))]
    # eval_loader_names += ['env{}_out'.format(i)
    #                       for i in range(len(out_splits))]
    # eval_loader_names += ['env{}_uda'.format(i)
    #                       for i in range(len(uda_splits))]

    # train_minibatches_iterator = zip(*train_loaders)
    # uda_minibatches_iterator = zip(*uda_loaders)
    # checkpoint_vals = collections.defaultdict(lambda: [])

    # steps_per_epoch = min([len(env) / hparams['batch_size'] for env, _ in in_splits])

    # n_steps = args.steps or dataset.N_STEPS
    # checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    # clusterer = class_distributor.KClustering(train_minibatches_iterator, config.NUM_CLASSES, args.dataset)
    # stratified_labels, labels = clusterer.cluster(num_experts=config.NUM_EXPERTS, stratified=True)

    # isExist = os.path.exists(PATH)
    # if not isExist:
    #     # Create a new directory because it does not exist
    #     os.makedirs(PATH)
    #     print("The new directory is created!")

    # print("Labels:", labels)
    # print("Stratified Labels:", stratified_labels)

    # torch.save(labels, PATH + f"/clustered_{args.dataset}_te{args.test_envs[0]}.pt")
    # torch.save(stratified_labels, PATH + f"/stratified_{args.dataset}_te{args.test_envs[0]}.pt")

    # print("htg : happy clustering")