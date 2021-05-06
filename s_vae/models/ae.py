from torch import nn
from torch.nn import functional as F


class AE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_dim: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.latent = nn.Linear(in_features=encoder.out_dim, out_features=latent_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        z = self.latent(encoded)
        x_hat, _ = self.decoder(z)
        return x_hat, z

    def step(self, x):
        x_hat, z = self.forward(x)
        return {'loss': F.mse_loss(x_hat, x, reduction='mean')}
