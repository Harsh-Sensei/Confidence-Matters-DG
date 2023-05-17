# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import sys
from itertools import chain

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from domainbed.lib.misc import (
    random_pairs_of_minibatches, ParamDict, MovingAverage, l2_between_dicts
)

import os
from domainbed import config
from copy import deepcopy
import copy

sys.path.append('/mnt/lustre/bli/projects/EIL/domainbed')
import vision_transformer
from collections import defaultdict, OrderedDict

try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

from domainbed import networks
# from domainbed import resnet_variants
import torchvision.models as models

USE_ROUTING = os.getenv("USE_ROUTING", default=False)
USE_LOGIT_N = os.getenv("USE_LOGIT_N", default=False)
USE_DOMAIN_CLASS_ROUTING = os.getenv("USE_DOMAIN_CLASS_ROUTING", default=False)
USE_ABLATION = os.getenv("USE_ABLATION", default=False)


ALGORITHMS = [
    'ERM',
    'Fish',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'SD',
    'ANDMask',
    'SANDMask',
    'IGA',
    'SelfReg',
    "Fishr",
    'TRM',
    'IB_ERM',
    'IB_IRM',
    'CAD',
    'CondCAD',
    'GMOE'
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    transforms = {}

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class MovingAvg:
    def __init__(self, network):
        self.network = network
        self.network_sma = copy.deepcopy(network)
        self.network_sma.eval()
        self.sma_start_iter = 100
        self.global_iter = 0
        self.sma_count = 0

    def update_sma(self):
        self.global_iter += 1
        if self.global_iter >= self.sma_start_iter:
            self.sma_count += 1
            for param_q, param_k in zip(self.network.parameters(), self.network_sma.parameters()):
                param_k.data = (param_k.data * self.sma_count + param_q.data) / (1. + self.sma_count)
        else:
            for param_q, param_k in zip(self.network.parameters(), self.network_sma.parameters()):
                param_k.data = param_q.data


class ERM_SMA(Algorithm, MovingAvg):
    """
    Empirical Risk Minimization (ERM) with Simple Moving Average (SMA) prediction model
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        MovingAvg.__init__(self, self.network)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.network(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_sma()
        return {'loss': loss.item()}

    def predict(self, x):
        self.network_sma.eval()
        return self.network_sma(x)


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier).cuda()
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

class ConfidentExperts(Algorithm):
    """
    Modified SFMOE for passing the input to all experts
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ConfidentExperts, self).__init__(input_shape, num_classes, num_domains, hparams)
        try:
            model_arch = config.MOE_LAYERS
        except Exception as e:
            print(e)
            model_arch = ['F'] * 11 + ['S']
            print(f"Using architecture : {model_arch}")

        self.model = vision_transformer.all_exp_deit_small_distilled_patch16_224(pretrained=True, num_classes=num_classes, moe_layers=model_arch, mlp_ratio=4., num_experts=config.NUM_EXPERTS, Hierachical=False).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams['weight_decay'])
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.lmd = hparams.pop('lmd', 1.0)
        self.cross_entropy_loss_list = []
        self.kl_div_loss_list = []
        print(f"Using lambda : {self.lmd}")

    def cross_entropy_and_kl(self, preds, all_y):
        b, num_experts, num_classes = preds.shape

        if USE_ROUTING:
            chosen_experts = config.ROUTING[all_y].cuda().long()
        elif USE_DOMAIN_CLASS_ROUTING:
            chosen_experts = []
            assert(all_y.shape[0]%3==0, "Something wrong with all_y")
            num_y = all_y.shape[0]
            for idx, elem in enumerate(all_y):
                chosen_experts.append(config.DOMAIN_CLASS_DICT[(idx//num_y, elem.item())])
            chosen_experts = torch.tensor(chosen_experts).cuda().long()
            
        else:
            chosen_experts = torch.remainder(all_y, other=num_experts)
        true_dist = torch.ones(b*(num_experts - 1), num_classes).cuda()/num_classes
        mask = torch.ones_like(preds).scatter(1, chosen_experts.unsqueeze(1).unsqueeze(2), 0.)[:, :, 0][:, :, None].repeat(1, 1, num_classes)
        removed_preds = preds[mask.bool()].view(b, num_experts-1, num_classes).reshape(-1, num_classes)

        assert true_dist.shape == removed_preds.shape

        log_softmax_removed_preds = F.log_softmax(removed_preds, dim=1)
        loss1 = F.cross_entropy(preds[range(preds.shape[0]), chosen_experts, :], all_y)
        self.cross_entropy_loss_list.append(loss1.item())

        loss2 = self.kl_loss(log_softmax_removed_preds, true_dist)
        self.kl_div_loss_list.append(loss2.item())
        return loss1 + self.lmd*loss2

    def cross_entropyAll_and_kl(self, preds, all_y, lmd=1.0):
        b, num_experts, num_classes = preds.shape

        if USE_ROUTING:
            chosen_experts = config.ROUTING[all_y.cpu()].cuda().long()
        else:
            chosen_experts = torch.remainder(all_y, other=num_experts)
        true_dist = torch.ones(b*(num_experts - 1), num_classes).cuda()/num_classes
        mask = torch.ones_like(preds).scatter(1, chosen_experts.unsqueeze(1).unsqueeze(2), 0.)[:, :, 0][:, :, None].repeat(1, 1, num_classes)
        removed_preds = preds[mask.bool()].view(b, num_experts-1, num_classes).reshape(-1, num_classes)

        assert true_dist.shape == removed_preds.shape

        log_softmax_removed_preds = F.log_softmax(removed_preds, dim=1)
        preds_flatten = preds.reshape(preds.shape[0], -1)
        gt = all_y + chosen_experts*num_classes
        loss = F.cross_entropy(preds_flatten, gt)
        loss += lmd*(self.kl_loss(log_softmax_removed_preds, true_dist))

        return loss
    # for ablation studies
    def only_cross_entropy(self, preds, all_y):
        b, num_experts, num_classes = preds.shape

        if USE_ROUTING:
            chosen_experts = config.ROUTING[all_y].cuda().long()
        else:
            chosen_experts = torch.remainder(all_y, other=num_experts)
        loss = F.cross_entropy(preds[range(preds.shape[0]), chosen_experts, :], all_y)

        return loss
    
    def cross_entropy_kl_logitn(self, preds, all_y, temperature=1.0):
        b, num_experts, num_classes = preds.shape

        if USE_ROUTING:
            chosen_experts = config.ROUTING[all_y].cuda().long()
        elif USE_DOMAIN_CLASS_ROUTING:
            chosen_experts = []
            assert(all_y.shape[0]%3==0, "Something wrong with all_y")
            domains = [i for i in range(4) if i!=config.TEST_ENV]
            num_y = all_y.shape[0]
            for idx, elem in enumerate(all_y):
                chosen_experts.append(config.DOMAIN_CLASS_DICT[(domains[idx//num_y], elem.item())])
            chosen_experts = torch.tensor(chosen_experts).cuda().long()
            
        else:
            chosen_experts = torch.remainder(all_y, other=num_experts)
        true_dist = torch.ones(b*(num_experts - 1), num_classes).cuda()/num_classes
        mask = torch.ones_like(preds).scatter(1, chosen_experts.unsqueeze(1).unsqueeze(2), 0.)[:, :, 0][:, :, None].repeat(1, 1, num_classes)
        removed_preds = preds[mask.bool()].view(b, num_experts-1, num_classes).reshape(-1, num_classes)

        assert true_dist.shape == removed_preds.shape

        log_softmax_removed_preds = F.log_softmax(removed_preds, dim=1)
        loss1 = F.cross_entropy(torch.nn.functional.normalize(preds[range(preds.shape[0]), chosen_experts, :], dim=1)/temperature, all_y)
        self.cross_entropy_loss_list.append(loss1.item())

        loss2 = self.kl_loss(log_softmax_removed_preds, true_dist)
        self.kl_div_loss_list.append(loss2.item())
        return loss1 + self.lmd*loss2

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        config.current_y = all_y # htg
                 

        preds = self.predict(all_x)
        #loss = self.only_cross_entropy(preds, all_y)
        if USE_LOGIT_N:
            loss = self.cross_entropy_kl_logitn(preds, all_y)
        elif USE_ABLATION:
            loss = self.only_cross_entropy(preds, all_y)
        else:
            loss = self.cross_entropy_and_kl(preds, all_y)
        #loss = self.cross_entropyAll_and_kl(preds, all_y)
        loss_aux_list = []
        for block in self.model.blocks:
            if getattr(block, 'aux_loss') is not None:
                loss_aux_list.append(block.aux_loss)

        loss_aux = 0
        for layer_loss in loss_aux_list:
            loss_aux += layer_loss

        if len(loss_aux_list) != 0:
            loss += loss_aux

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if len(loss_aux_list) != 0:
            return {'loss': loss.item(), 'loss_aux': loss_aux.item()}
        else:
            return {'loss': loss.item()}

    def predict(self, x):
        prediction = self.model(x)
        if type(prediction) is tuple:
            return (prediction[0] + prediction[1]) / 2
        else:
            return prediction

class HierarchicalSFMOE(Algorithm):
    """
    SFMOE
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(HierarchicalSFMOE, self).__init__(input_shape, num_classes, num_domains, hparams)
        try:
            model_arch = config.MOE_LAYERS
        except Exception as e:
            print(e)
            model_arch = ['F'] * 11 + ['S']
        print(f"Using architecture : {model_arch}")
        self.model = vision_transformer.hierarchical_deit_small_patch16_224(pretrained=True, num_classes=num_classes, moe_layers=model_arch, mlp_ratio=4., num_experts=6, drop_path_rate=0.1).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams['weight_decay'])

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches]).cuda()
        all_y = torch.cat([y for x, y in minibatches]).cuda()
        config.current_y = all_y # htg
        loss = F.cross_entropy(self.predict(all_x), all_y)
        loss_aux_list = []
        for block in self.model.blocks:
            if getattr(block, 'aux_loss') is not None:
                loss_aux_list.append(block.aux_loss)

        loss_aux = 0
        for layer_loss in loss_aux_list:
            loss_aux += layer_loss

        loss += loss_aux
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'loss_aux': loss_aux.item()}

    def predict(self, x, forward_feature=False):
        if forward_feature:
            return self.model.forward_features(x)
        else:
            prediction = self.model(x)
            if type(prediction) is tuple:
                return (prediction[0] + prediction[1]) / 2
            else:
                return prediction

class SkipSFMOE(Algorithm):

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SkipSFMOE, self).__init__(input_shape, num_classes, num_domains, hparams)
        try:
            model_arch = config.MOE_LAYERS
        except Exception as e:
            print(e)
            model_arch = ['F'] * 11 + ['S']
            print(f"Using architecture : {model_arch}")

        self.model = vision_transformer.skip_deit_small_patch16_224(pretrained=True, num_classes=num_classes, moe_layers=model_arch, mlp_ratio=4., num_experts=config.NUM_EXPERTS, Hierachical=False).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams['weight_decay'])
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.lmd = hparams.pop('lmd', 1.0)
        self.cross_entropy_loss_list = []
        self.kl_div_loss_list = []
        print(f"Using lambda : {self.lmd}")

    def cross_entropy_and_kl(self, preds, all_y):
        b, num_experts, num_classes = preds.shape

        if USE_ROUTING:
            chosen_experts = config.ROUTING[all_y].cuda().long()
        else:
            chosen_experts = torch.remainder(all_y, other=num_experts)
        true_dist = torch.ones(b*(num_experts - 1), num_classes).cuda()/num_classes
        mask = torch.ones_like(preds).scatter(1, chosen_experts.unsqueeze(1).unsqueeze(2), 0.)[:, :, 0][:, :, None].repeat(1, 1, num_classes)
        removed_preds = preds[mask.bool()].view(b, num_experts-1, num_classes).reshape(-1, num_classes)

        assert true_dist.shape == removed_preds.shape

        log_softmax_removed_preds = F.log_softmax(removed_preds, dim=1)
        loss1 = F.cross_entropy(preds[range(preds.shape[0]), chosen_experts, :], all_y)
        self.cross_entropy_loss_list.append(loss1.item())

        loss2 = self.kl_loss(log_softmax_removed_preds, true_dist)
        self.kl_div_loss_list.append(loss2.item())
        return loss1 + self.lmd*loss2

    def cross_entropyAll_and_kl(self, preds, all_y, lmd=1.0):
        b, num_experts, num_classes = preds.shape

        if USE_ROUTING:
            chosen_experts = config.ROUTING[all_y].cuda().long()
        else:
            chosen_experts = torch.remainder(all_y, other=num_experts)
        true_dist = torch.ones(b*(num_experts - 1), num_classes).cuda()/num_classes
        mask = torch.ones_like(preds).scatter(1, chosen_experts.unsqueeze(1).unsqueeze(2), 0.)[:, :, 0][:, :, None].repeat(1, 1, num_classes)
        removed_preds = preds[mask.bool()].view(b, num_experts-1, num_classes).reshape(-1, num_classes)

        assert true_dist.shape == removed_preds.shape

        log_softmax_removed_preds = F.log_softmax(removed_preds, dim=1)
        preds_flatten = preds.reshape(preds.shape[0], -1)
        gt = all_y + chosen_experts*num_classes
        loss = F.cross_entropy(preds_flatten, gt)
        loss += lmd*(self.kl_loss(log_softmax_removed_preds, true_dist))

        return loss
    # for ablation studies
    def only_cross_entropy(self, preds, all_y):
        b, num_experts, num_classes = preds.shape

        if USE_ROUTING:
            chosen_experts = config.ROUTING[all_y].cuda().long()
        else:
            chosen_experts = torch.remainder(all_y, other=num_experts)
        loss = F.cross_entropy(preds[range(preds.shape[0]), chosen_experts, :], all_y)

        return loss

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        config.current_y = all_y # htg
        preds = self.predict(all_x)
        #loss = self.only_cross_entropy(preds, all_y)
        loss = self.cross_entropy_and_kl(preds, all_y)
        #loss = self.cross_entropyAll_and_kl(preds, all_y)
        loss_aux_list = []
        for block in self.model.blocks:
            if getattr(block, 'aux_loss') is not None:
                loss_aux_list.append(block.aux_loss)

        loss_aux = 0
        for layer_loss in loss_aux_list:
            loss_aux += layer_loss

        if len(loss_aux_list) != 0:
            loss += loss_aux

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if len(loss_aux_list) != 0:
            return {'loss': loss.item(), 'loss_aux': loss_aux.item()}
        else:
            return {'loss': loss.item()}

    def predict(self, x):
        prediction = self.model(x)
        if type(prediction) is tuple:
            return (prediction[0] + prediction[1]) / 2
        else:
            return prediction

class Oracle(Algorithm):
    """
    Uses class labels for routing
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Oracle, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.model = vision_transformer.oracle_deit_small_patch16_224(pretrained=True, num_classes=num_classes, moe_layers=['F'] * 8 + ['S', 'F'] * 2, mlp_ratio=4., num_experts=6, drop_path_rate=0.1, router='cosine_top').cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams['weight_decay'])

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        #print("all_x", all_x.shape)
        #print("all_y", all_y.shape)
        config.current_y = all_y
        loss = F.cross_entropy(self.predict(all_x), all_y)
        loss_aux_list = []
        for block in self.model.blocks:
            if getattr(block, 'aux_loss') is not None:
                loss_aux_list.append(block.aux_loss)

        loss_aux = 0
        for layer_loss in loss_aux_list:
            loss_aux += layer_loss

        if len(loss_aux_list) != 0:
            loss += loss_aux

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #print("Update done")
        if len(loss_aux_list) != 0:
            return {'loss': loss.item(), 'loss_aux': loss_aux.item()}
        else:
            return {'loss': loss.item()}

    def predict(self, x, forward_feature=False):
        if forward_feature:
            return self.model.forward_features(x)
        else:
            prediction = self.model(x)
            if type(prediction) is tuple:
                return (prediction[0] + prediction[1]) / 2
            else:
                return prediction

class GMOE(Algorithm):
    """
    SFMOE
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GMOE, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.model = vision_transformer.deit_small_patch16_224(pretrained=True, num_classes=num_classes, moe_layers=['F'] * 8 + ['S', 'F'] * 2, mlp_ratio=4., num_experts=6, drop_path_rate=0.1, router='cosine_top').cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams['weight_decay'])

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches]).cuda()
        all_y = torch.cat([y for x, y in minibatches]).cuda()
        loss = F.cross_entropy(self.predict(all_x), all_y)
        loss_aux_list = []
        for block in self.model.blocks:
            if getattr(block, 'aux_loss') is not None:
                loss_aux_list.append(block.aux_loss)

        loss_aux = 0
        for layer_loss in loss_aux_list:
            loss_aux += layer_loss

        loss += loss_aux
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'loss_aux': loss_aux.item()}

    def predict(self, x, forward_feature=False):
        if forward_feature:
            return self.model.forward_features(x)
        else:
            prediction = self.model(x)
            if type(prediction) is tuple:
                return (prediction[0] + prediction[1]) / 2
            else:
                return prediction



class Fish(Algorithm):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain
    Generalization, Shi et al. 2021.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Fish, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.network = networks.WholeFish(input_shape, num_classes, hparams)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_inner_state = None

    def create_clone(self, device):
        self.network_inner = networks.WholeFish(self.input_shape, self.num_classes, self.hparams,
                                                weights=self.network.state_dict()).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, unlabeled=None):
        self.create_clone(minibatches[0][0].device)

        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams["meta_lr"]
        )
        self.network.reset_weights(meta_weights)

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains,
                                           hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.discriminator = networks.MLP(self.featurizer.n_outputs,
                                          num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes,
                                             self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) +
             list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) +
             list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9))

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(),
                                   [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad ** 2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = (classifier_loss +
                        (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class DANN(AbstractDANN):
    """Unconditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(input_shape, num_classes, num_domains,
                                   hparams, conditional=False, class_balance=False)


#
#
# class CDANN(AbstractDANN):
#     """Conditional DANN"""
#
#     def __init__(self, input_shape, num_classes, num_domains, hparams):
#         super(CDANN, self).__init__(input_shape, num_classes, num_domains,
#                                     hparams, conditional=True, class_balance=True)
#
#
class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                                                        >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}


class Fishr(Algorithm):
    "Invariant Gradients variances for Out-of-distribution Generalization"

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert backpack is not None, "Install backpack with: 'pip install backpack-for-pytorch==1.3.0'"
        super(Fishr, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.num_domains = num_domains

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = extend(
            networks.Classifier(
                self.featurizer.n_outputs,
                num_classes,
                self.hparams['nonlinear_classifier'],
            )
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)

        self.register_buffer("update_count", torch.tensor([0]))
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))
        self.ema_per_domain = [
            MovingAverage(ema=self.hparams["ema"], oneminusema_correction=True)
            for _ in range(self.num_domains)
        ]
        self._init_optimizer()

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, minibatches, unlabeled=False):
        assert len(minibatches) == self.num_domains
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        len_minibatches = [x.shape[0] for x, y in minibatches]

        all_z = self.featurizer(all_x)
        all_logits = self.classifier(all_z)

        penalty = self.compute_fishr_penalty(all_logits, all_y, len_minibatches)
        all_nll = F.cross_entropy(all_logits, all_y)

        penalty_weight = 0
        if self.update_count >= self.hparams["penalty_anneal_iters"]:
            penalty_weight = self.hparams["lambda"]
            if self.update_count == self.hparams["penalty_anneal_iters"] != 0:
                # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
                # gradient magnitudes that happens at this step.
                self._init_optimizer()
        self.update_count += 1

        objective = all_nll + penalty_weight * penalty
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item(), 'nll': all_nll.item(), 'penalty': penalty.item()}

    def compute_fishr_penalty(self, all_logits, all_y, len_minibatches):
        dict_grads = self._get_grads(all_logits, all_y)
        grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, len_minibatches)
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads(self, logits, y):
        self.optimizer.zero_grad()
        loss = self.bce_extended(logits, y).sum()
        with backpack(BatchGrad()):
            loss.backward(
                inputs=list(self.classifier.parameters()), retain_graph=True, create_graph=True
            )

        # compute individual grads for all samples across all domains simultaneously
        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in self.classifier.named_parameters()
            ]
        )
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(self.num_domains)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

        # moving average
        for domain_id in range(self.num_domains):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):

        # compute gradient variances averaged across domains
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.num_domains)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.num_domains):
            penalty += l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_domains

    def predict(self, x):
        return self.network(x)
