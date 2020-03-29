import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from linear_masked import LinearMasked


class MADE(nn.Module):
    """ MADE model for binary image dataset. """
    def __init__(self, input_dim, use_cuda=True):
        super().__init__()
        self.input_dim = input_dim
        self.device = torch.device('cuda') if use_cuda else None

        self.net = nn.Sequential(
            LinearMasked(input_dim, input_dim), nn.ReLU(),
            LinearMasked(input_dim, input_dim), nn.ReLU(),
            LinearMasked(input_dim, input_dim)
        )

        self.apply_masks()

    def forward(self, x):
        return self.net(x)

    def set_mask(self, mask):
        self.mask = torch.from_numpy(mask.astype(np.uint8)).to(self.device)
        # self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8)))

    def apply_masks(self):
        # Set order of masks, i.e. who can make which edges
        # Using natural ordering
        order1 = np.arange(self.input_dim)
        order2 = np.arange(self.input_dim)
        order3 = np.arange(self.input_dim)

        # Construct the mask matrices
        masks = []
        m1 = (order1[:, None] <= order2[None,:]).T
        m2 = (order2[:, None] <= order3[None,:]).T
        m3 = (order2[:,None] < order3[None,:]).T
        masks.append(m1)
        masks.append(m2)
        masks.append(m3)

        # Set the masks in all LinearMasked layers
        layers = [l for l in self.net.modules() if isinstance(l, LinearMasked)]

        for l, m in zip(layers, masks):
            l.set_mask(m)

