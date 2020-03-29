# pytorch-made

PyTorch implementation of Masked Autoencoder Distribution Estimation (MADE) for binary image dataset. 
Based on [MADE: Masked Autoencoder for Distribution Estimation](https://arxiv.org/abs/1502.03509) by Germain et. al., and inspired by [karpathy](https://github.com/karpathy/pytorch-made).

## MADE
MADE takes a regular autoencoder and tweaks it so that its output units predict the n conditional distributions instead of reconstructing the n inputs (as in regular autoencoders). It does this by masking the MLP connections in an appropriate way, see figure below. 
![](https://i.imgur.com/Eq9A8Hz.png)

