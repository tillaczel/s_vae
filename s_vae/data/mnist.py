from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import random_split

import requests
import os
import matplotlib.pyplot as plt
import numpy as np


def create_MNIST(path: str, train_ratio):
    # ------------------------------------------------------------------------------------------------------------ #
    # torchtext newest update (04.03.2021) broke pytorch-lightning, but older torch version can not download MNIST
    # because of a new redirect. Temporary solve to download MNIST manually
    raw_path = f'{path}/MNIST/raw/'
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
        files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
        for file in files:
            url = f'http://yann.lecun.com/exdb/mnist/{file}'
            r = requests.get(url, allow_redirects=True)
            open(f'{raw_path}/{file}', 'wb').write(r.content)
    # ------------------------------------------------------------------------------------------------------------ #

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST(root=path, train=True, transform=transform, download=True)
    train_size = int(train_ratio * len(dataset))
    train_set, valid_set = random_split(dataset, [train_size, len(dataset) - train_size])
    test_set = MNIST(root=path, train=False, transform=transform, download=True)
    return train_set, valid_set, test_set


def vis_mnist(folder_path, x, x_hat, n=5):
    idx = np.random.choice(x.shape[0], n, replace=False)
    x, x_hat = x.numpy()[idx, 0], x_hat.numpy()[idx, 0]

    fig, axs = plt.subplots(n, 2, figsize=(8, 4*n))
    for i, (img, recon) in enumerate(zip(x, x_hat)):
        axs[i, 0].imshow(img, cmap='gray')
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])
        axs[i, 1].imshow(recon, cmap='gray')
        axs[i, 1].set_xticks([])
        axs[i, 1].set_yticks([])
    plt.tight_layout()
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    path = os.path.join(folder_path, 'data_reconstruction.png')
    plt.savefig(path)
    return path







