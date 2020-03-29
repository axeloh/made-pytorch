import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinearMasked(nn.Linear):
    """
    Class implementing nn.Linear with mask
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.mask = None
        # self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask = torch.from_numpy(mask.astype(np.uint8))
        # self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8)))

    def forward(self, x):
        return F.linear(x.cuda(), self.mask.cuda() * self.weight.cuda(), self.bias)

