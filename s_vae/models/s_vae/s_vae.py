import torch
from torch import nn
from torch.nn import functional as F
from s_vae.models.s_vae.unif_on_sphere import UnifOnSphere
from s_vae.models.s_vae.vMF import vMF


class SVAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, encoder_out_dim: int, latent_dim: int, kl_coeff: float,
                 device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

        self.kl_coeff = kl_coeff
        self._device = device

        self.fc_mu = nn.Linear(encoder_out_dim, latent_dim)
        self.fc_kappa = nn.Linear(encoder_out_dim, 1)  # concentration parameter is just a scalar

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
        reconstructed = self.decoder(z)
        return reconstructed

    def _forward(self, x):
        mu, kappa = self.encode(x)
        p, q, z = self.sample(mu, kappa)
        x_hat = self.decode(z)
        return x_hat, mu, kappa, p, q, z

    def forward(self, x):
        x_hat, mu, log_var, p, q, z = self._forward(x)
        return x_hat, z # z-latent vector

    def step(self, x):
        x_hat, mu, kappa, p, q, z = self._forward(x)

        loss_recon = F.mse_loss(x_hat, x, reduction='mean')

        loss_kl = self.kl_coeff * torch.distributions.kl.kl_divergence(q, p).mean()

        loss = loss_kl + loss_recon

        return {'loss': loss, 'loss_recon': loss_recon, 'loss_kl': loss_kl}

    def sample(self, mu, kappa):
        p = UnifOnSphere(self.latent_dim, self._device)
        q = vMF(mu, kappa, device=self._device)
        z = q.rsample(sample_shape=mu.shape)

        return p, q, z
