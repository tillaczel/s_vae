import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
import os


def generate_latent_data(n, dim=3, R=1, datadist=None, seed=0):
    """
    A function for generating n randomly distributed
    points on the surface of a hypersphere.
    args:
    n : number of points generated
    dim : dimension of the hypersphere
    R : radius of the hypersphere
    datadist : a list of strings that may include 'uniform'
               and 'skew'. 'uniform' is a random uniform
               distribution. 'skew' is another n/2 data
               points distributed in one m'th quadrant
               (where all vectors have positive values)
               of the hypersphere.
    """
    if datadist is None:
        datadist = ['uniform', 'skew']

    np.random.seed(seed)

    dim_arr = [[] for _ in range(dim)]

    if 'uniform' in datadist:
        for _ in range(n):
            r = np.random.random(dim)**2
            r = r/(r.sum())
            coords = np.sqrt(r)
            for i in range(dim):
                if bool(random.getrandbits(1)):
                    x = -1 * coords[i]
                else:
                    x = coords[i]
                dim_arr[i].append(x*R)

    if 'skew' in datadist:
        for _ in range(int(n/2)):
            rp = np.random.random(dim)
            rp = rp/(rp.sum())
            coords = np.sqrt(rp)
            for i in range(dim):
                dim_arr[i].append(coords[i]*R)
    
    return np.array(dim_arr).T


class Nonlinearity(nn.Module):
    def __init__(self, input_dim, output_dim, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(  # sequential operation
            nn.Linear(input_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, output_dim))

    def forward(self, X):
        return self.net(X)


def generate_observed_data(latent_data, dim=256, seed=0):
    nonlinearity = Nonlinearity(latent_data.shape[1], dim, seed=seed)
    _latent_data = torch.Tensor(latent_data)
    return nonlinearity(_latent_data).detach().numpy()


def create_synthetic_hypersphere(path: str, latent_dim, observed_dim, n_dev_samples, n_test_samples, train_ratio=0.8,
                                 seed=0):
    raw_path = f'{path}/synthetic_hypersphere/raw/'
    tag = f'{latent_dim}_{observed_dim}_{n_dev_samples}_{n_test_samples}_{seed}'
    if os.path.isfile(f'{path}/observed_data_dev_{tag}.csv'):
        observed_data_dev = np.loadtxt(f'{path}/observed_data_dev_{tag}.csv', dtype=np.float32, delimiter=',')
        observed_data_test = np.loadtxt(f'{path}/observed_data_test_{tag}.csv', dtype=np.float32, delimiter=',')
    else:
        if not os.path.exists(raw_path):
            os.makedirs(raw_path)

        n_samples = n_dev_samples+n_test_samples
        latent_data = generate_latent_data(n_samples, dim=latent_dim, datadist=['uniform'], seed=seed)
        observed_data = generate_observed_data(latent_data, dim=observed_dim, seed=seed)

        latent_data_dev, latent_data_test = latent_data[:n_dev_samples], latent_data[-n_test_samples:]
        observed_data_dev, observed_data_test = observed_data[:n_dev_samples], observed_data[-n_test_samples:]

        np.savetxt(f'{raw_path}/latent_data_dev_{tag}.csv', latent_data_dev.astype(float), delimiter=',')
        np.savetxt(f'{raw_path}/latent_data_test_{tag}.csv', latent_data_test.astype(float), delimiter=',')
        np.savetxt(f'{raw_path}/observed_data_dev_{tag}.csv', observed_data_dev.astype(float), delimiter=',')
        np.savetxt(f'{raw_path}/observed_data_test_{tag}.csv', observed_data_test.astype(float), delimiter=',')

    dataset = SyntheticHypersphereDataset(observed_data_dev)
    train_size = int(train_ratio * len(dataset))
    train_set, valid_set = random_split(dataset, [train_size, len(dataset)-train_size])
    test_set = SyntheticHypersphereDataset(observed_data_test)

    return train_set, valid_set, test_set


class SyntheticHypersphereDataset(Dataset):
    def __init__(self, x: np.array):
        self.x = torch.Tensor(x)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x_i = self.x[idx]
        return x_i, torch.Tensor(np.zeros(x_i.shape))






