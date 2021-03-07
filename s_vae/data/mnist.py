from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import random_split

import requests
import os


def create_MNIST(path: str):
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
    train_set, valid_set = random_split(dataset, [50000, 10000])
    test_set = MNIST(root=path, train=False, transform=transform, download=True)
    return train_set, valid_set, test_set






