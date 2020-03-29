
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

from made import MADE



def train(train_data, test_data, image_shape):
    """ Trains MADE model on binary image dataset.
        Arguments:
        train_data: A (n_train, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
        test_data: An (n_test, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
        image_shape: (H, W), height and width of the image

        Returns:
        - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
        - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
        - a numpy array of size (100, H, W, 1) of samples with values in {0, 1}
    """

    use_cuda = True
    device = torch.device('cuda') if use_cuda else None

    train_data = torch.from_numpy(
        train_data.reshape((train_data.shape[0], train_data.shape[1] * train_data.shape[2]))).float().to(device)
    test_data = torch.from_numpy(
        test_data.reshape((test_data.shape[0], test_data.shape[1] * test_data.shape[2]))).float().to(device)

    def nll_loss(batch, output):
        return F.binary_cross_entropy(torch.sigmoid(output), batch)

    H, W = image_shape
    input_dim = H * W

    made = MADE(input_dim)
    epochs = 10
    lr = 0.005
    batch_size = 32

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(made.parameters(), lr=lr)

    init_test_loss = nll_loss(test_data, made(test_data))
    train_losses = []
    test_losses = [init_test_loss.item()]

    # Training
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            output = made(batch)
            loss = nll_loss(batch, output)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        test_loss = nll_loss(test_data, made(test_data))
        test_losses.append(test_loss.item())
        print(f'{epoch + 1}/{epochs} epochs')

    # Generate samples
    made.eval()
    samples = torch.zeros(size=(100, H * W)).to(device)
    with torch.no_grad():
        for i in range(H * W):
            logits = made(samples)
            probas = torch.sigmoid(logits)
            pixel_i_samples = torch.bernoulli(probas[:, i])
            samples[:, i] = pixel_i_samples

    return np.array(train_losses), np.array(test_losses), samples.reshape((100, H, W, 1)).detach().cpu().numpy()