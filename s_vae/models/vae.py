import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from s_vae.models.backbone.utils import Reshape


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, recon_shape: int, latent_dim: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.recon_shape = recon_shape

        self.fc_mu = nn.Linear(encoder.out_dim, latent_dim)
        self.fc_var = nn.Linear(encoder.out_dim, latent_dim)

    def encode(self, x):
        encoded = self.encoder(x)

        mu = self.fc_mu(encoded)
        log_var = self.fc_var(encoded)

        return mu, log_var

    def decode(self, z):
        x_hat, log_var = self.decoder(z)

        return x_hat, log_var

    def _forward(self, x):
        mu, log_var = self.encode(x)
        p, q, z = self.sample(mu, log_var)
        x_hat, log_var = self.decode(z)
        return x_hat, mu, log_var, p, q, z

    def forward(self, x):
        x_hat, mu, log_var, p, q, z = self._forward(x)
        return x_hat, z

    def step(self, x):
        x_hat, mu, log_var, p, q, z = self._forward(x)

        loss_recon = F.mse_loss(x_hat, x, reduction='none').view(-1, np.prod(self.recon_shape)).sum(axis=1, keepdim=True)
        loss_recon = 1 / (2 * torch.exp(log_var)) * loss_recon + (1 / 2) * log_var
        loss_recon = loss_recon.mean(axis=0)

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        loss_kl = log_qz - log_pz
        loss_kl = loss_kl.mean()

        loss = loss_kl + loss_recon

        return {'loss': loss, 'loss_recon': loss_recon, 'loss_kl': loss_kl}

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = p.rsample() * std + mu
        return p, q, z

    def decode_exp(self, z):
        x_hat, log_var = self.decoder(z)

        return x_hat