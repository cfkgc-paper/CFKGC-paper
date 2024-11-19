import torch
import torch.nn as nn
import torch.nn.functional as F
import math




class SubnetLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, sparsity=0.5, trainable=True):
        super(SubnetLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.sparsity = sparsity
        self.trainable = trainable

        # Mask Parameters of Weights and Bias
        self.w_m = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_mask = None
        self.zeros_weight, self.ones_weight = torch.zeros(self.w_m.shape), torch.ones(self.w_m.shape)
        if bias:
            self.b_m = nn.Parameter(torch.empty(out_features))
            self.bias_mask = None
            self.zeros_bias, self.ones_bias = torch.zeros(self.b_m.shape), torch.ones(self.b_m.shape)
        else:
            self.register_parameter('bias', None)

        # Init Mask Parameters
        self.init_mask_parameters()

        if not trainable:
            raise Exception("Non-trainable version is not yet implemented")

    def forward(self, x, weight_mask=None, bias_mask=None, mode="train", is_base=True):
        self.sparsity = 0.5 if is_base else 0.85  # TODO:
        w_pruned, b_pruned = None, None
        # If training, Get the subnet by sorting the scores
        if mode == "train" or mode == "val":
            self.weight_mask = GetSubnetFaster.apply(self.w_m.abs(),
                                                     self.zeros_weight,
                                                     self.ones_weight,
                                                     self.sparsity) if weight_mask is None else weight_mask
            w_pruned = self.weight_mask * self.weight 
            b_pruned = None
            if self.bias is not None:
                self.bias_mask = GetSubnetFaster.apply(self.b_m.abs(),
                                                       self.zeros_bias,
                                                       self.ones_bias,
                                                       self.sparsity)
                b_pruned = self.bias_mask * self.bias
        # If inference/valid, use the last compute masks/subnetworks

        # elif mode == "val": # TODO: problem is ?
        #     # self.weight_mask.grad = self.weight_mask_grad
        #     # self.weight.grad = self.weight_grad
        #     w_pruned = self.weight_mask * self.weight
        #     b_pruned = None
        #     if self.bias is not None:
        #         b_pruned = self.bias_mask * self.bias

        return F.linear(input=x, weight=w_pruned, bias=b_pruned)

    def init_mask_parameters(self):
        nn.init.kaiming_uniform_(self.w_m, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_m)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b_m, -bound, bound)

class EntityMask(nn.Module):
    def __init__(self, relation, few, in_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_m = nn.Parameter(torch.empty(relation, few, in_features))
        self.init_mask_parameters()

    def forward(self, x, weight_mask=None, bias_mask=None, mode="train"):
        return F.sigmoid(self.w_m) * x

    def init_mask_parameters(self):
        nn.init.kaiming_uniform_(self.w_m, a=math.sqrt(5))


