import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, encoder_out_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

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

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return [x, x_hat, mu, log_var]

    def loss_function(self, x, x_hat, mu, log_var, kld_weight=1):
        recons_loss = F.mse_loss(x, x_hat)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'reconstruction_ross': recons_loss, 'KLD': -kld_loss}

    def sample(self, num_samples: int, current_device: int):
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x):
        return self.forward(x)[0]
