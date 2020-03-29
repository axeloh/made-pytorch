# pytorch-made

PyTorch implementation of Masked Autoencoder Distribution Estimation (MADE) for binary image dataset. 
Based on [MADE: Masked Autoencoder for Distribution Estimation](https://arxiv.org/abs/1502.03509) by Germain et. al., and inspired by [karpathy](https://github.com/karpathy/pytorch-made).

## MADE

MADE takes a regular autoencoder and tweaks it so that its output units predict the n conditional distributions instead of reconstructing the n inputs (as in regular autoencoders). It does this by masking the MLP connections in an appropriate way, see figure below. As with other autoregressive models, evaluating likelihood is cheap (requires a single forward pass), while sampling must be done iteratively for every pixel, and is thus linear in the number of pixels. 

Given some binary image of height H and width W, we can represent image $$x\in \{0, 1\}^{H\times W}$$ as a flattened binary vector $x\in \{0, 1\}^{HW}$ to input into MADE to model $p_\theta(x) = \prod_{i=1}^{HW} p_\theta(x_i|x_{<i})$.

![](https://i.imgur.com/Eq9A8Hz.png)

## Results
Shape dataset | MADE samples| 
:--- | :---
![](https://i.imgur.com/SqHZ80C.png) | ![](https://i.imgur.com/TJJC5F3.png)
![](https://i.imgur.com/K0fpM4L.png) | ![](https://i.imgur.com/Xvlgs7w.png)
![](https://i.imgur.com/KZTDxzS.png) | ![](https://i.imgur.com/eIvILXE.png)
