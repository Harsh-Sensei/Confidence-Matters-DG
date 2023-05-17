# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time

sys.path.append("./domainbed")
# os.environ['WANDB_API_KEY'] = 'abc1859572354a66fc85b2ad1d1009add929cbfa'

# import wandb
import PIL
import numpy as np
import torch
import torch.utils.data
import torchvision

from domainbed import algorithms
from domainbed import datasets
from domainbed import hparams_registry
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed import config

import pandas as pd

# PATH = "./model_algAllExpertSFMOE_dsPACS_te0_texp6_depth12.pkl"

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    parser.add_argument('--load_routing', action='store_true')
    parser.add_argument('--stratified', action='store_true')
    
    args = parser.parse_args()
    #################################################################################
    args.stratified = os.getenv("USE_STRATIFIED", default=False) or args.stratified
    args.load_routing = os.getenv("USE_ROUTING", default=False) or args.load_routing
    random_routing = os.getenv("USE_RANDOM_ROUTING", default=False)
    #################################################################################

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.stratified:
        print("Stratified routing")
    if args.load_routing:
        print("Loading routings")
        typ = "stratified" if args.stratified else "clustered"
        config.ROUTING = torch.load(f"routings/{typ}_{args.dataset}_te{args.test_envs[0]}.pt").to(device)
        print(config.ROUTING)


    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = torch.load(PATH, map_location=device)['model_dict']
    # print("algorithm_dict", algorithm_dict)

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

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
    if random_routing:
        config.ROUTING = torch.randint(low=0, high=config.NUM_EXPERTS, size=(config.NUM_CLASSES,1)).squeeze() 

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
        uda = []

        out, in_ = misc.split_dataset(env, int(len(env) * args.holdout_fraction), misc.seed_hash(args.trial_seed, env_i))
        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_, int(len(in_) * args.uda_holdout_fraction), misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
                         for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
                          for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
                          for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)

    evals = zip(eval_loader_names, eval_loaders, eval_weights)
    results = {}
    for name, loader, weights in evals:
        acc, exp_acc, conf_mat, exp_conf_mat = misc.accuracy_and_analyse_routing(algorithm, loader, weights, device)
        print("Name :", name)
        print("Accuracy :", acc)
        print("Expert Accuracy :", exp_acc)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print("CLASS Confusion matrix : \n", pd.DataFrame(conf_mat))
            print("EXPERT Confusion matrix : \n", pd.DataFrame(exp_conf_mat))
        results[name + '_acc'] = acc



