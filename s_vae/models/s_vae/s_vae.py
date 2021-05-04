import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from s_vae.models.s_vae.unif_on_sphere import UnifOnSphere
from s_vae.models.s_vae.vMF import vMF


class SVAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, recon_shape: int, latent_dim: int, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.recon_shape = recon_shape

        self._device = device

        self.fc_mu = nn.Linear(encoder.out_dim, latent_dim)
        self.fc_kappa = nn.Linear(encoder.out_dim, 1)  # concentration parameter is just a scalar

    @property
    def device(self):
        return self._device()

    def encode(self, x):
        encoded = self.encoder(x)

        mu = self.fc_mu(encoded)
        mu = mu / torch.linalg.norm(mu, ord=2, keepdim=True, dim=-1)  # make mu a vector on the sphere

        kappa = F.softplus(self.fc_kappa(encoded)) + 1

        return mu, kappa

    def decode(self, z):
        x_hat, log_var = self.decoder(z)

        return x_hat, log_var

    def _forward(self, x):
        mu, kappa = self.encode(x)
        p, q, z = self.sample(mu, kappa)
        x_hat, log_var = self.decode(z)
        return x_hat, log_var, mu, kappa, p, q, z

    def forward(self, x):
        x_hat, log_var, mu, kappa, p, q, z = self._forward(x)
        return x_hat, z

    def step(self, x):
        x_hat, log_var, mu, kappa, p, q, z = self._forward(x)

        loss_recon = F.mse_loss(x_hat, x, reduction='none').view(-1, np.prod(self.recon_shape)).sum(axis=1, keepdim=True)
        loss_recon = 1 / (2 * torch.exp(log_var)) * loss_recon + (1 / 2) * log_var
        loss_recon = loss_recon.mean(axis=0)

        loss_kl = torch.distributions.kl.kl_divergence(q, p).mean()
        loss = loss_kl + loss_recon

        return {'loss': loss, 'loss_recon': loss_recon, 'loss_kl': loss_kl}

    def sample(self, mu, kappa):
        p = UnifOnSphere(self.latent_dim, self._device)
        q = vMF(mu, kappa, device=self._device)
        z = q.rsample(sample_shape=mu.shape)

        return p, q, z
    
    def decode_exp(self, z):
        x_hat, log_var = self.decoder(z)

        return x_hat
