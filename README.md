# made-pytorch

A simple PyTorch implementation of Masked Autoencoder Distribution Estimation (MADE) for binary image dataset. 
Based on [MADE: Masked Autoencoder for Distribution Estimation](https://arxiv.org/abs/1502.03509) by Germain et. al., and inspired by [karpathy](https://github.com/karpathy/pytorch-made).

## MADE
MADE is a type of autoregressive generative models which try to model the generation of images as a sequence of generation of pixels.
MADE takes a regular autoencoder and tweaks it so that its output units predict the n conditional distributions instead of reconstructing the n inputs (as in regular autoencoders). It does this by masking the MLP connections such that the k-th output unit only depends on the previous k-1 inputs, see figure below. As with most autoregressive models, evaluating likelihood is cheap (requires a single forward pass), while sampling must be done iteratively for every pixel, and is thus linear in the number of pixels. 

![](https://i.imgur.com/Eq9A8Hz.png)


## Generating samples
Below are samples generated by MADE (for shape and mnist dataset) after training for 10 epochs:
Shape dataset | MADE samples| 
:--- | :---
![](https://i.imgur.com/4iU3eDY.png) | ![](https://i.imgur.com/x7tZ3H2.png)

MNIST dataset | MADE samples| 
:--- | :---
![](https://i.imgur.com/mlO1TuB.png) | ![](https://i.imgur.com/d9kQWV7.png)
