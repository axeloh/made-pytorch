# pytorch-made

PyTorch implementation of Masked Autoencoder Distribution Estimation (MADE) for binary image dataset. 
Based on [MADE: Masked Autoencoder for Distribution Estimation](https://arxiv.org/abs/1502.03509) by Germain et. al., and inspired by [karpathy](https://github.com/karpathy/pytorch-made).

## MADE

MADE takes a regular autoencoder and tweaks it so that its output units predict the n conditional distributions instead of reconstructing the n inputs (as in regular autoencoders). It does this by masking the MLP connections in an appropriate way, see figure below. As with other autoregressive models, evaluating likelihood is cheap (requires a single forward pass), while sampling must be done iteratively for every pixel, and is thus linear in the number of pixels. 


![](https://i.imgur.com/Eq9A8Hz.png)

## Results
Shape dataset | MADE samples| 
:--- | :---
![](https://i.imgur.com/4iU3eDY.png) | ![](https://i.imgur.com/x7tZ3H2.png)

MNIST dataset | MADE samples| 
:--- | :---
![](https://i.imgur.com/mlO1TuB.png) | ![](https://i.imgur.com/d9kQWV7.png)
