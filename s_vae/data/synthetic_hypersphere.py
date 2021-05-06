import sys
sys.path.append('../../')

import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
import os
import math 

from s_vae.models.s_vae.unif_on_sphere import UnifOnSphere
from s_vae.models.s_vae.vMF import vMF

def generate_latent_data(n, dim=2, R=1, datadist=None, vMF_centroids = 5 , seed=0, device = 'cpu'):
    """
    A function for generating n randomly distributed
    points on the surface of a hypersphere.
    args:
    n : number of points generated
    dim : dimension of the hypersphere
    R : radius of the hypersphere
    datadist : a list of strings that may include 'uniform' and 'vMF'. 
            'uniform' is a uniform distribution on a hypersphere. 
            'vMF' is the von Mises-Fischer distribution on a hypersphere.
    vMF_centroids: number of centroids for the vMF distributions. Essentially means we sample from vMF_centroids vMF distributions with 
                   vMF_centroids number of mean-vectors.
    """
    if datadist is None:
        datadist = ['uniform', 'skew']

    np.random.seed(seed)

    dim_arr = [[] for _ in range(dim)]
            
    if 'uniform' in datadist:
        distribution = UnifOnSphere(dim, device)
        samples = distribution.sample(torch.Size((n,)))

    if 'vMF' in datadist:
        # Only works for a circle right now
        assert(dim == 2)

        mean_vector_angles = np.arange(0, 360, 360/vMF_centroids)
        centroids = []
        for angle in mean_vector_angles:
            x1 =  math.cos(math.radians(angle+5))
            x2 =  math.sin(math.radians(angle+5))
            centroids.append(torch.tensor([x1,x2]))

        mu = torch.stack(centroids)
        kappa = torch.ones(torch.Size((vMF_centroids,1)))
        distribution = vMF(mu, kappa)
    
    samples = []
    for sample in range(int(n/vMF_centroids)):
        samples.append(distribution.rsample(sample_shape=mu.shape))
    
    samples = torch.stack(samples).view(-1,2)
    return samples


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
        self.x = torch.tensor(x)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x_i = self.x[idx]
        return x_i, torch.tensor(np.zeros(x_i.shape))