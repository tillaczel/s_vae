import sys
sys.path.append('../../')

import torch
import yaml
import pickle
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd.functional import jacobian 

from s_vae.models import build_model
from s_vae.models.s_vae.unif_on_sphere import UnifOnSphere
from s_vae.models.s_vae.vMF import vMF
from s_vae.data.mnist import create_MNIST


class Model(torch.nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.model = build_model(config['model'],device, lambda: False)
        self.latent_dim = config['model']['latent_dim']

    def decode(self,x):
        return self.model.decode_exp(x)

    def encode(self,x):
        return self.model.encode(x)

    def forward(self, x):
        return self.model(x)
    
    def sample(self,mu, kappa):
        return self.model.sample(mu, kappa)

    def __call__(self, x):
        return self.forward(x)


class Jacobian_experiment():
    
    def __init__(self, config):
        self.data = config['data']['name']
        self.name = config['model']['name']
        
        self.device = config['jacobian']['device']
        self.sampling = config['jacobian']['sampling']
        self.data_split = config['jacobian']['data_split']
        self.checkpoint_path = config['jacobian']['checkpoint_path']
        self.num_samples = config['jacobian']['num_samples']
        
        self.model = Model(config, self.device)
        self.checkpoint = torch.load(self.checkpoint_path,map_location=torch.device(self.device))
        self.model.load_state_dict(self.checkpoint['state_dict'])
        self.config = config
        
    
    def __call__(self):
        return self.run_experiment()

    def run_experiment(self):
        
        z = self.sample_latent_points()
        jacobians = self.calc_jacobian(z)
        determinants = self.calc_determinants(jacobians)

        return determinants



    def sample_latent_points(self):
        if self.sampling == 'prior':
            if self.name == 's_vae':
                distribution = UnifOnSphere(model.latent_dim, model._device)
            else:
                distribution = torch.distributions.Normal(torch.zeros(self.model.latent_dim), torch.ones(self.model.latent_dim))

            z = distribution.sample(torch.Size((self.num_samples,)))
            
        else:
            x = self.get_data()
            # kappa_or_log_var because it depends on the incoder
            mu_param_vector = []
            kappa_or_log_var_param_vector = []
            for batch,label in x:
                mu, kappa_or_log_var = self.model.encode(batch)
                mu_param_vector.append(mu)
                kappa_or_log_var_param_vector.append(mu)
            mu = torch.stack(mu_param_vector)
            kappa_or_log_var = torch.stack(kappa_or_log_var_param_vector)

            p, q, z = self.model.sample(mu, kappa_or_log_var)
            
        return z

    def calc_jacobian(self,samples):
        jacobians = []
        for sample in range(len(samples)):
            jacobians.append(jacobian(self.model.decode, samples[sample]).squeeze().view(784,-1))
        return jacobians

    def calc_determinants(self,jacobians):
        determinants = []
        for jc in range(len(jacobians)):
            tnsr = jacobians[jc]
            M = torch.matmul(torch.transpose(tnsr, 0, 1),tnsr)
            detmnt = torch.linalg.det(M)
            determinants.append(torch.sqrt(torch.abs(detmnt)))
        return determinants
            
    
    def get_data(self):
        if self.data == 'MNIST':
            train_set, val_set, test_set = create_MNIST(self.config) # with some paramters. These are dataloaders.

            if self.data_split == 'train':
                loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
            elif self.data_split == 'val':
                loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
            else:
                loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
        else:
            loader = None
        return loader



def main(config_path: str):
    with open(config_path) as fd:
        config = yaml.load(fd, yaml.FullLoader)
    experiment = Jacobian_experiment(config)
    output = experiment.run_experiment()
    
    with open('../../local/jacobian/experiment_outcome.pickle', 'wb') as handle:
        pickle.dump(output, handle)


if __name__ == '__main__':
    config_path = '../config.yaml'
    main(config_path)
