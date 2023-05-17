import math
import sys
from inspect import isfunction

import torch
import torch.nn.functional as F
from torch import nn

sys.path.append("/mnt/lustre/bli/projects/EIL/domainbed")

from domainbed import config
import os
import random 

ROUTING_ANALYSIS = os.getenv("ROUTING_ANALYSIS", default=False)
SELECT_ONE_EXPERT = os.getenv("SELECT_ONE_EXPERT", default=False)
USE_ROUTING = os.getenv("USE_ROUTING", default=False)
# constants

MIN_EXPERT_CAPACITY = 4
# helper functions

def default(val, default_val):
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val


def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)


# tensor related helper functions

def top1(t):
    values, index = t.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index


def cumsum_exclusive(t, dim=-1):
    num_dims = len(t.shape)
    num_pad_dims = - dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice = (slice(None),) * num_pad_dims
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]


# pytorch one hot throws an error if there are out of bound indices.
# tensorflow, in contrast, does not throw an error
def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]


def init_(t):
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)


# activations

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

class MLPExperts(nn.Module):
    def __init__(self, dim, num_experts=16):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(num_experts, dim, dim * 4))
        self.w2 = nn.Parameter(torch.randn(num_experts, dim * 4, dim * 4))
        self.w3 = nn.Parameter(torch.randn(num_experts, dim * 4, dim))
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        hidden1 = self.act(torch.einsum('end,edh->enh', x, self.w1))
        hidden2 = self.act(torch.einsum('end,edh->enh', hidden1, self.w2))
        out = torch.einsum('end,edh->enh', hidden2, self.w3)
        return out


class SimpleExperts(nn.Module):
    def __init__(self,
                 dim,
                 num_experts=16,
                 hidden_dim=None,
                 activation=GELU):
        super().__init__()

        hidden_dim = default(hidden_dim, dim * 4)
        num_experts = cast_tuple(num_experts)

        w1 = torch.zeros(*num_experts, dim, hidden_dim)
        w2 = torch.zeros(*num_experts, hidden_dim, dim)

        w1 = init_(w1)
        w2 = init_(w2)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.act = activation()

    def forward(self, x):
        hidden = torch.einsum('...nd,...dh->...nh', x, self.w1)
        hidden = self.act(hidden)
        out = torch.einsum('...nh,...hd->...nd', hidden, self.w2)
        return out

class SimpleAllExperts(nn.Module):
    def __init__(self,
                 dim,
                 num_experts=16,
                 hidden_dim=None,
                 activation=GELU):
        super().__init__()

        hidden_dim = default(hidden_dim, dim * 4)
        num_experts = cast_tuple(num_experts)

        w1 = torch.zeros(*num_experts, dim, hidden_dim)
        w2 = torch.zeros(*num_experts, hidden_dim, dim)

        w1 = init_(w1)
        w2 = init_(w2)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        # self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.act = activation()
        self.drop1 = nn.Dropout(p=0.6)
        self.drop2 = nn.Dropout(p=0.6)

    def forward(self, x):
        hidden = torch.einsum('...nd,...edh->...neh', x, self.w1)
        hidden = self.act(hidden)
        # hidden = self.drop1(hidden)
        
        # hidden = self.linear(hidden)
        hidden = self.act(hidden)
        # hidden = self.drop2(hidden)

        out = torch.einsum('...neh,...ehd->...ned', hidden, self.w2)
        return out

class ClassBasedGating(nn.Module):
    def __init__(self, num_gates, index_hook, fixed_expert=0, capacity_factor_train=1.25, capacity_factor_eval=2., **kwargs):
        super().__init__()
        self.num_gates = num_gates
        self.index_hook = index_hook
        self.capacity_factor_eval = capacity_factor_eval
        self.capacity_factor_train = capacity_factor_train

    def forward(self, x, importance=None):
        # b, group_size, dim = x.shape
        *_, b, group_size, dim = x.shape
        num_gates = self.num_gates

        if self.training:
            capacity_factor = self.capacity_factor_train
        else:
            capacity_factor = self.capacity_factor_eval

        # FIND TOP 2 EXPERTS PER POSITON
        # Find the top expert for each position. shape=[batch, group]
        index_1 = torch.remainder(config.current_y[:, None], other=self.num_gates).repeat(1, x.shape[1])
        #print("index_1", index_1.shape)
        mask_1 = F.one_hot(index_1, num_gates).float()
        # Each sequence sends (at most?) expert_capacity positions to each expert.
        # Static expert_capacity dimension is needed for expert batch sizes
        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)

        # COMPUTE ASSIGNMENT TO EXPERTS
        # [batch, group, experts]
        # This is the position within the expert's mini-batch for this sequence
        position_in_expert_1 = cumsum_exclusive(mask_1, dim=-2) * mask_1
        # Remove the elements that don't fit. [batch, group, experts]
        mask_1 *= (position_in_expert_1 < expert_capacity_f).float()
        # [batch, experts]
        # How many examples in this sequence go to this expert
        mask_1_count = mask_1.sum(dim=-2, keepdim=True)
        # [batch, group] - mostly ones, but zeros where something didn't fit
        mask_1_flat = mask_1.sum(dim=-1)
        # [batch, group]
        position_in_expert_1 = position_in_expert_1.sum(dim=-1)
        # Weight assigned to first expert.  [batch, group
        # [batch, group, experts, expert_capacity]
        #print("mask_flat", mask_1_flat.shape)
        #print("index1 one hot", F.one_hot(index_1, num_gates).shape)
        #print("safe_one_hot", safe_one_hot(position_in_expert_1.long(), expert_capacity)[..., None, :].shape)
        combine_tensor = (
                mask_1_flat[..., None, None]
                * F.one_hot(index_1, num_gates)[..., None]
                * safe_one_hot(position_in_expert_1.long(), expert_capacity)[..., None, :]
        )
        #print("combine_tensor", combine_tensor.shape)

        dispatch_tensor = combine_tensor.bool().to(combine_tensor)
        loss = None
        if self.index_hook is True:
            return x, index_1, None, dispatch_tensor, combine_tensor, loss
        else:
            return dispatch_tensor, combine_tensor, loss

class SimpleGating(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_hidden_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.layers = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), 
            *[nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_hidden_layers)],
            nn.Linear(self.hidden_dim, self.num_classes)
        )

    def forward(self, x):
        print("x shape : ", x.shape)
        preds = self.layers(x)
        # print("preds shape : ", preds.shape)
        # print("current_y shape : ", config.current_y)
        # print("ROUTING shape : ", config.ROUTING)
        # print("true shape : ", config.ROUTING[config.current_y].cuda().long().shape)
        loss = F.cross_entropy(preds, config.ROUTING[config.current_y].cuda().long())
        chosen_exp = torch.argmax(preds, dim = 1)

        return chosen_exp, loss

class Top2Gating(nn.Module):
    def __init__(
            self,
            dim,
            num_gates,
            eps=1e-9,
            index_hook=True,
            outer_expert_dims=tuple(),
            second_policy_train='random',
            second_policy_eval='random',
            second_threshold_train=0.2,
            second_threshold_eval=0.2,
            capacity_factor_train=1.25,
            capacity_factor_eval=2.):
        super().__init__()

        self.eps = eps
        self.index_hook = index_hook
        self.num_gates = num_gates
        self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))
        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

        self.eval_batch_num = 0

    def forward(self, x, importance=None):
        # b, group_size, dim = x.shape
        *_, b, group_size, dim = x.shape
        # print("Shape of input to Top2Gating :", x.shape)
        num_gates = self.num_gates

        if self.training:
            policy = self.second_policy_train
            threshold = self.second_threshold_train
            capacity_factor = self.capacity_factor_train
        else:
            policy = self.second_policy_eval
            threshold = self.second_threshold_eval
            capacity_factor = self.capacity_factor_eval

        raw_gates = torch.einsum('...bnd,...de->...bne', x, self.w_gating)
        raw_gates = raw_gates.softmax(dim=-1)

        if ROUTING_ANALYSIS:
            self.raw_gates = raw_gates
        # FIND TOP 2 EXPERTS PER POSITON
        # Find the top expert for each position. shape=[batch, group]

        gate_1, index_1 = top1(raw_gates)

        # print("Index 1", index_1.shape)

        mask_1 = F.one_hot(index_1, num_gates).float()
        density_1_proxy = raw_gates

        if importance is not None:
            equals_one_mask = (importance == 1.).float()
            mask_1 *= equals_one_mask[..., None]
            gate_1 *= equals_one_mask
            density_1_proxy = density_1_proxy * equals_one_mask[..., None]
            del equals_one_mask

        gates_without_top_1 = raw_gates * (1. - mask_1)

        gate_2, index_2 = top1(gates_without_top_1)
        mask_2 = F.one_hot(index_2, num_gates).float()

        if importance is not None:
            greater_zero_mask = (importance > 0.).float()
            mask_2 *= greater_zero_mask[..., None]
            del greater_zero_mask

        # normalize top2 gate scores
        denom = gate_1 + gate_2 + self.eps
        gate_1 /= denom
        gate_2 /= denom

        # BALANCING LOSSES
        # shape = [batch, experts]
        # We want to equalize the fraction of the batch assigned to each expert
        density_1 = mask_1.mean(dim=-2)
        # Something continuous that is correlated with what we want to equalize.
        density_1_proxy = density_1_proxy.mean(dim=-2)
        loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)

        # Depending on the policy in the hparams, we may drop out some of the
        # second-place experts.
        if policy == "all":
            pass
        elif policy == "none":
            mask_2 = torch.zeros_like(mask_2)
        elif policy == "threshold":
            mask_2 *= (gate_2 > threshold).float()
        elif policy == "random":
            probs = torch.zeros_like(gate_2).uniform_(0., 1.)
            mask_2 *= (probs < (gate_2 / max(threshold, self.eps))).float().unsqueeze(-1)
        else:
            raise ValueError(f"Unknown policy {policy}")

        # Each sequence sends (at most?) expert_capacity positions to each expert.
        # Static expert_capacity dimension is needed for expert batch sizes
        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)

        print("Expert capacity :", expert_capacity)

        # COMPUTE ASSIGNMENT TO EXPERTS
        # [batch, group, experts]
        # This is the position within the expert's mini-batch for this sequence
        position_in_expert_1 = cumsum_exclusive(mask_1, dim=-2) * mask_1
        # Remove the elements that don't fit. [batch, group, experts]
        mask_1 *= (position_in_expert_1 < expert_capacity_f).float()
        # [batch, experts]
        # How many examples in this sequence go to this expert
        mask_1_count = mask_1.sum(dim=-2, keepdim=True)

        # [batch, group] - mostly ones, but zeros where something didn't fit
        mask_1_flat = mask_1.sum(dim=-1)
        # print("Mask_1_flat", mask_1_flat)
        # [batch, group]
        position_in_expert_1 = position_in_expert_1.sum(dim=-1)
        # Weight assigned to first expert.  [batch, group]
        gate_1 *= mask_1_flat

        position_in_expert_2 = cumsum_exclusive(mask_2, dim=-2) + mask_1_count
        position_in_expert_2 *= mask_2
        mask_2 *= (position_in_expert_2 < expert_capacity_f).float()
        mask_2_flat = mask_2.sum(dim=-1)

        # print("position_in_expert_2", position_in_expert_2)

        position_in_expert_2 = position_in_expert_2.sum(dim=-1)
        gate_2 *= mask_2_flat

        # print("Safe one hot :", safe_one_hot(position_in_expert_1.long(), expert_capacity)[..., None, :].shape)
        # [batch, group, experts, expert_capacity]
        combine_tensor = (
                gate_1[..., None, None]
                * mask_1_flat[..., None, None]
                * F.one_hot(index_1, num_gates)[..., None]
                * safe_one_hot(position_in_expert_1.long(), expert_capacity)[..., None, :] +
                gate_2[..., None, None]
                * mask_2_flat[..., None, None]
                * F.one_hot(index_2, num_gates)[..., None]
                * safe_one_hot(position_in_expert_2.long(), expert_capacity)[..., None, :]
        )


        dispatch_tensor = combine_tensor.bool().to(combine_tensor)
        if self.index_hook is True:
            return x, index_1, index_2, dispatch_tensor, combine_tensor, loss
        else:
            return dispatch_tensor, combine_tensor, loss

class OracleMoE(nn.Module):
    def __init__(self,
                 dim,
                 num_experts=16,
                 hidden_dim=None,
                 activation=nn.ReLU,
                 second_policy_train='random',
                 second_policy_eval='random',
                 second_threshold_train=0.2,
                 second_threshold_eval=0.2,
                 capacity_factor_train=1.25,
                 capacity_factor_eval=2.,
                 loss_coef=1e-2,
                 experts=None,
                 index_hook=False):
        super().__init__()

        self.num_experts = num_experts
        self.index_hook = index_hook

        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train,
                         'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}
        self.gate = ClassBasedGating(num_gates=num_experts, index_hook=self.index_hook, **gating_kwargs)

        # SimpleExperts : 2 FC layers
        self.experts = default(experts, lambda: SimpleExperts(dim, num_experts=num_experts))
        self.loss_coef = loss_coef
        self.train_used_experts = [0]*num_experts
        self.eval_used_experts = [0]*num_experts
        self.train_batch_count = 0
        self.eval_batch_count = 0

    def forward(self, inputs, **kwargs):
        if self.training:
            self.train_batch_count += 1
        else:
            self.eval_batch_count += 1
        #print("inputs", inputs.shape)
        b, n, d = inputs.shape
        e = self.num_experts

        if self.index_hook:
            x, index1, index2, dispatch_tensor, combine_tensor, loss = self.gate(inputs)
            if ROUTING_ANALYSIS:
                for i, elem1 in enumerate(index1):
                    curr_label = config.current_y[i]
                    for elem2 in elem1:
                        if self.training:
                            config.expert_usage_dict_sparse_train[config.curr_sparse_layer][curr_label.item()][elem2.item()] += 1
                        else:
                            config.expert_usage_dict_sparse_eval[config.curr_sparse_layer][curr_label.item()][elem2.item()] += 1

                for i, elem1 in enumerate(index2):
                    curr_label = config.current_y[i]
                    for elem2 in elem1:
                        if self.training:
                            config.expert_usage_dict_sparse_train[config.curr_sparse_layer][curr_label.item()][elem2.item()] += 1
                        else:
                            config.expert_usage_dict_sparse_eval[config.curr_sparse_layer][curr_label.item()][elem2.item()] += 1

        else:
            dispatch_tensor, combine_tensor, loss = self.gate(inputs)
        inputs = inputs.cuda()
        dispatch_tensor = dispatch_tensor.cuda()
        combine_tensor = combine_tensor.cuda()

        expert_inputs = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor)

        # Now feed the expert inputs through the experts.
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(e, -1, d)
        # [e, bs * comb, d]
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        output = torch.einsum('ebcd,bnec->bnd', expert_outputs, combine_tensor)
        # loss = loss * self.loss_coef
        return output, loss

class HierarchicalClassExperts(nn.Module):
    def __init__(self,
                 dim,
                 num_experts=16,
                 hidden_dim=None,
                 activation=nn.ReLU,
                 second_policy_train='random',
                 second_policy_eval='random',
                 second_threshold_train=0.2,
                 second_threshold_eval=0.2,
                 capacity_factor_train=1.25,
                 capacity_factor_eval=2.,
                 loss_coef=1e-1,
                 experts=None,
                 index_hook=False,
                 teacher_forcing=0.3):
        super().__init__()
        self.num_experts = num_experts
        self.index_hook = index_hook
        self.dim = dim
        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train,
                         'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}
        self.hidden_dim = hidden_dim if hidden_dim is not None else 2*dim
        self.gate = SimpleGating(input_dim=self.dim, hidden_dim=self.hidden_dim, num_classes=self.num_experts)

        # SimpleExperts : 2 FC layers
        self.experts = [
            nn.Sequential(nn.Linear(self.dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.dim))
            for _ in range(self.num_experts)
        ]
        self.loss_coef = loss_coef
        self.tf = teacher_forcing

    def forward(self, inputs, **kwargs):

        b, n, d = inputs.shape
        e = self.num_experts

        true_experts = config.ROUTING[config.current_y]
        classification_inputs = inputs[:,0,:].cuda()
        
        chosen_exp, aux_loss = self.gate(classification_inputs)
        if self.training:
            tf_list = random.sample(range(b), int(b*self.tf))
        else:
            tf_list = []
        
        out_list = []
        for i in range(b):
            if i in tf_list:
                out_list.append(self.experts[true_experts[i]].cuda()(classification_inputs[i]))
            else:
                out_list.append(self.experts[chosen_exp[i]].cuda()(classification_inputs[i]))

        output = torch.stack(out_list, dim=0)
        # print("Output dim :", output.shape)

        # Now feed the expert inputs through the experts.
        loss = aux_loss * self.loss_coef
        return output, loss


# plain mixture of experts

class MoE(nn.Module):
    def __init__(self,
                 dim,
                 num_experts=16,
                 hidden_dim=None,
                 activation=nn.ReLU,
                 second_policy_train='random',
                 second_policy_eval='random',
                 second_threshold_train=0.2,
                 second_threshold_eval=0.2,
                 capacity_factor_train=1.25,
                 capacity_factor_eval=2.,
                 loss_coef=1e-2,
                 experts=None,
                 index_hook=False):
        super().__init__()

        self.num_experts = num_experts
        self.index_hook = index_hook

        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train,
                         'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}
        self.gate = Top2Gating(dim, num_gates=num_experts, index_hook=self.index_hook, **gating_kwargs)

        # SimpleExperts : 2 FC layers
        self.experts = default(experts, lambda: SimpleExperts(dim, num_experts=num_experts))
        self.loss_coef = loss_coef
        self.train_used_experts = [0]*num_experts
        self.eval_used_experts = [0]*num_experts
        self.train_batch_count = 0
        self.eval_batch_count = 0

    def forward(self, inputs, **kwargs):
        if self.training:
            self.train_batch_count += 1
        else:
            self.eval_batch_count += 1

        b, n, d = inputs.shape
        e = self.num_experts

        if self.index_hook:
            x, index1, index2, dispatch_tensor, combine_tensor, loss = self.gate(inputs)
            if ROUTING_ANALYSIS:
                for i, elem1 in enumerate(index1):
                    curr_label = config.current_y[i]
                    for elem2 in elem1:
                        if self.training:
                            config.expert_usage_dict_sparse_train[config.curr_sparse_layer][curr_label.item()][elem2.item()] += 1
                        else:
                            config.expert_usage_dict_sparse_eval[config.curr_sparse_layer][curr_label.item()][elem2.item()] += 1

                for i, elem1 in enumerate(index2):
                    curr_label = config.current_y[i]
                    for elem2 in elem1:
                        if self.training:
                            config.expert_usage_dict_sparse_train[config.curr_sparse_layer][curr_label.item()][elem2.item()] += 1
                        else:
                            config.expert_usage_dict_sparse_eval[config.curr_sparse_layer][curr_label.item()][elem2.item()] += 1

        else:
            dispatch_tensor, combine_tensor, loss = self.gate(inputs)
        inputs = inputs.cuda()
        dispatch_tensor = dispatch_tensor.cuda()
        combine_tensor = combine_tensor.cuda()

        expert_inputs = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor)

        # Now feed the expert inputs through the experts.
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(e, -1, d)
        # [e, bs * comb, d]
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        output = torch.einsum('ebcd,bnec->bnd', expert_outputs, combine_tensor)
        # loss = loss * self.loss_coef
        return output, loss

class AllExperts(nn.Module):
    def __init__(self,
                 dim,
                 num_experts=16,
                 hidden_dim=None,
                 activation=nn.ReLU,
                 second_policy_train='random',
                 second_policy_eval='random',
                 second_threshold_train=0.2,
                 second_threshold_eval=0.2,
                 capacity_factor_train=1.25,
                 capacity_factor_eval=2.,
                 loss_coef=1e-2,
                 experts=None,
                 index_hook=False):
        super().__init__()

        self.num_experts = num_experts
        self.index_hook = index_hook

        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train,
                         'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}
        self.experts = default(experts, lambda: SimpleAllExperts(dim, num_experts=num_experts))
        self.loss_coef = loss_coef

    def forward(self, inputs, **kwargs):

        b, n, d = inputs.shape
        e = self.num_experts

        inputs = inputs.cuda()
        output = self.experts(inputs)

        return output


# 2-level heirarchical mixture of experts

class HierarchicalMoE(nn.Module):
    def __init__(self,
                 dim,
                 num_experts=(4, 4),
                 hidden_dim=None,
                 activation=nn.ReLU,
                 second_policy_train='random',
                 second_policy_eval='random',
                 second_threshold_train=0.2,
                 second_threshold_eval=0.2,
                 capacity_factor_train=1.25,
                 capacity_factor_eval=2.,
                 loss_coef=1e-2,
                 experts=None):
        super().__init__()

        assert len(num_experts) == 2, 'only 2 levels of heirarchy for experts allowed for now'
        num_experts_outer, num_experts_inner = num_experts
        self.num_experts_outer = num_experts_outer
        self.num_experts_inner = num_experts_inner

        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train,
                         'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}

        self.gate_outer = Top2Gating(dim, num_gates=num_experts_outer, **gating_kwargs)
        self.gate_inner = Top2Gating(dim, num_gates=num_experts_inner, outer_expert_dims=(num_experts_outer,), **gating_kwargs)

        self.experts = default(experts, lambda: SimpleExperts(dim, num_experts=num_experts, hidden_dim=hidden_dim, activation=activation))
        self.loss_coef = loss_coef

    def forward(self, inputs, **kwargs):
        b, n, d, eo, ei = *inputs.shape, self.num_experts_outer, self.num_experts_inner
        dispatch_tensor_outer, combine_tensor_outer, loss_outer = self.gate_outer(inputs)
        expert_inputs_outer = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor_outer)

        # we construct an "importance" Tensor for the inputs to the second-level
        # gating.  The importance of an input is 1.0 if it represents the
        # first-choice expert-group and 0.5 if it represents the second-choice expert
        # group.  This is used by the second-level gating.
        importance = combine_tensor_outer.permute(2, 0, 3, 1).sum(dim=-1)
        importance = 0.5 * ((importance > 0.5).float() + (importance > 0.).float())

        dispatch_tensor_inner, combine_tensor_inner, loss_inner = self.gate_inner(expert_inputs_outer, importance=importance)
        expert_inputs = torch.einsum('ebnd,ebnfc->efbcd', expert_inputs_outer, dispatch_tensor_inner)

        # Now feed the expert inputs through the experts.
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(eo, ei, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        # NOW COMBINE EXPERT OUTPUTS (reversing everything we have done)
        # expert_output has shape [y0, x1, h, d, n]

        expert_outputs_outer = torch.einsum('efbcd,ebnfc->ebnd', expert_outputs, combine_tensor_inner)
        output = torch.einsum('ebcd,bnec->bnd', expert_outputs_outer, combine_tensor_outer)
        return output, (loss_outer + loss_inner) * self.loss_coef
