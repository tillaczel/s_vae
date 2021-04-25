import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, encoder_out_dim: int, 
    decoder_out_dim: int, recon_shape: int,latent_dim: int, kl_coeff: float):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.recon_shape = recon_shape

        self.kl_coeff = kl_coeff

        self.fc_mu = nn.Linear(encoder_out_dim, latent_dim)
        self.fc_var = nn.Linear(encoder_out_dim, latent_dim)

        self.fc_xhat = nn.Linear(decoder_out_dim, recon_shape)
        self.fc_log_var = nn.Linear(decoder_out_dim, 1)

    def encode(self, x):
        encoded = self.encoder(x)

        mu = self.fc_mu(encoded)
        log_var = self.fc_var(encoded)

        return mu, log_var

    def decode(self, z):
        reconstructed = self.decoder(z)
        xhat = self.fc_xhat(reconstructed)
        log_var = self.fc_log_var(reconstructed)

        return xhat, log_var

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

        loss_recon = F.mse_loss(x_hat, x, reduction='none').mean(axis=0)

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        loss_kl = log_qz - log_pz
        loss_kl = loss_kl.mean()
        loss_kl *= self.kl_coeff

        loss = loss_kl + 1/(2 * torch.exp(log_var)) * loss_recon + (self.recon_shape/2)*log_var

        return {'loss': loss, 'loss_recon': loss_recon, 'loss_kl': loss_kl}

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = p.rsample() * std + mu
        return p, q, z
