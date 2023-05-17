# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import hashlib
import operator
import sys
from collections import Counter
from collections import OrderedDict
from datetime import datetime
from numbers import Number

from domainbed import config
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

def merge_dictlist(dictlist):
    """Merge list of dicts into dict of lists, by grouping same key.
    """
    ret = {
        k: []
        for k in dictlist[0].keys()
    }
    for dic in dictlist:
        for data_key, v in dic.items():
            ret[data_key].append(v)
    return ret


def index_conditional_iterate(skip_condition, iterable, index):
    for i, x in enumerate(iterable):
        if skip_condition(i):
            continue

        if index:
            yield i, x
        else:
            yield x


class SplitIterator:
    def __init__(self, test_envs):
        self.test_envs = test_envs

    def train(self, iterable, index=False):
        return index_conditional_iterate(lambda idx: idx in self.test_envs, iterable, index)

    def test(self, iterable, index=False):
        return index_conditional_iterate(lambda idx: idx not in self.test_envs, iterable, index)


def timestamp(fmt="%y%m%d_%H-%M-%S"):
    return datetime.now().strftime(fmt)


def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
            torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
            torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).mean()


class MovingAverage:

    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.ema_data = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.ema_data[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                # correction by 1/(1 - self.ema)
                # so that the gradients amplitude backpropagated in data is independent of self.ema
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.ema_data[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data


def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights


def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2 ** 31)


def print_separator():
    print("=" * 80)


def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.4f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]

    print(sep.join([format_val(x) for x in row]), end_)


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""

    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys

    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]

    def __len__(self):
        return len(self.keys)


def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert (n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)


def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


def accuracy(network, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            config.current_y = y
            # print("x shape in accuracy :", x.shape)
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)

            if len(p.shape) == 3:
                # print(f"Pred shape :{p.shape}")
                p = torch.softmax(p, dim=2)
                maxval_for_each_expert, maxarg_for_each_expert = torch.max(p, dim=2)
                maxarg_for_all_expert = torch.max(maxval_for_each_expert, dim=1)[1]
                pred_classes = maxarg_for_each_expert[range(p.shape[0]), maxarg_for_all_expert]
                assert len(pred_classes.shape) == 1
                correct += (pred_classes.squeeze().eq(y).float() * batch_weights).sum().item()
            elif p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total

print_one_batch = True
def accuracy_and_analyse_routing(network, loader, weights, device):
    global print_one_batch
    correct = 0
    total = 0
    weights_offset = 0

    correct_experts = 0

    network.eval()
    batch_idx = 0
    conf_mat = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES))
    exp_conf_mat = np.zeros((config.NUM_EXPERTS, config.NUM_EXPERTS))
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            config.current_y = y
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)

            if len(p.shape) == 3:
                # po.print_(f"Pred shape :{p.shape}")
                p = torch.softmax(p, dim=2)
                    
                maxval_for_each_expert, maxarg_for_each_expert = torch.max(p, dim=2)
                maxarg_for_all_expert = torch.max(maxval_for_each_expert, dim=1)[1]
                pred_classes = maxarg_for_each_expert[range(p.shape[0]), maxarg_for_all_expert]
                
                if print_one_batch:
                    print("p dim :", p.shape)
                    # print("p :", p)
                    print("maxarg_for_all_expert :", maxarg_for_all_expert)
                    print("correct_exp :", config.ROUTING[y])
                    print("pred classes :", pred_classes)
                    print("correct classes :", y)
                    print_one_batch = False
                # print("Batch Index :", batch_idx)
                # print("Routed expert :", maxarg_for_all_expert)
                # print("Ground truth experts :", config.ROUTING[y])
                assert len(pred_classes.squeeze().shape) == 1
                addend1 = (pred_classes.squeeze().eq(y).float() * batch_weights).sum().item()
                addend2 = (maxarg_for_all_expert.squeeze().eq(config.ROUTING[y])).cpu().sum().item()
                print("Addends : ",addend1, addend2)
                correct += addend1
                correct_experts += addend2
                conf_mat += confusion_matrix(y.cpu().numpy(), pred_classes.cpu().numpy(), labels=list(range(config.NUM_CLASSES)))
                exp_conf_mat += confusion_matrix(maxarg_for_all_expert.cpu().squeeze().numpy(), config.ROUTING[y].cpu().numpy(), labels=list(range(config.NUM_EXPERTS)))
            elif p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
            batch_idx += 1    

        print_one_batch = True
        
    network.train()

    return correct / total, correct_experts/total, conf_mat, exp_conf_mat

def uncertainity_of_preds(network, loader, weights, device):
    ITERS = 5
    network.eval()
    for module in network.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()
    batch_idx = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            config.current_y = y
            p_arr = []
            for _ in range(ITERS):
                p = network.predict(x)
                p = torch.softmax(p, dim=2)
                p_arr.append(p.clone().detach())
            p_stack = torch.stack(p_arr, dim=0)
            p_var = torch.var(p_stack, dim=0)
            p_var_mean = torch.mean(torch.flatten(p_var, start_dim=1), dim=1)
            break

    network.train()
    
    return p_var_mean

def confidence_comparison(network1, network2, loader, weights, device, exp_routing):

    weights_offset = 0

    network1.eval()
    network2.eval()

    batch_idx = 0
    with torch.no_grad():
        for x, y in loader:
            x = (x.to(device)[0]).unsqueeze(dim=0)
            config.current_y = y
            p1 = network1.predict(x).squeeze()
            p2 = network2.predict(x).squeeze()  
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            # po.print_(f"Pred shape :{p.shape}")
            p1 = torch.softmax(p1, dim=1)            
            p2 = torch.softmax(p2, dim=1)            
            break
        
    network1.train()
    network2.train()
    k=0
    for i in range(p1.shape[0]):
        if i != exp_routing[y[0].item()]:
            k=i
            break

    return p1[k], p2[k], y[0].item()


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)
