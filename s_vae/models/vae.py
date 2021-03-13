import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, encoder_out_dim: int, latent_dim: int, kl_coeff: float):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

        self.kl_coeff = kl_coeff

        self.fc_mu = nn.Linear(encoder_out_dim, latent_dim)
        self.fc_var = nn.Linear(encoder_out_dim, latent_dim)

    def encode(self, x):
        encoded = self.encoder(x)

        mu = self.fc_mu(encoded)
        log_var = self.fc_var(encoded)

        return mu, log_var

    def decode(self, z):
        reconstructed = self.decoder(z)
        return reconstructed

    def forward(self, x):
        mu, log_var = self.encode(x)
        p, q, z = self.sample(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var, p, q, z

    def step(self, x):
        x_hat, mu, log_var, p, q, z = self.forward(x)

        loss_recon = F.mse_loss(x_hat, x, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        loss_kl = log_qz - log_pz
        loss_kl = loss_kl.mean()
        loss_kl *= self.kl_coeff

        loss = loss_kl + loss_recon

        return {'loss': loss, 'loss_recon': loss_recon, 'loss_kl': loss_kl}

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = p.rsample() * std + mu
        return p, q, z
