from torch import nn
from torch.nn import functional as F
from torch.nn import Module


class AE(nn.Module):
    def __init__(self, encoder: Module, decoder: Module, encoder_out_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.latent = nn.Linear(in_features=encoder_out_dim, out_features=latent_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        z = self.latent(encoded)
        x_hat = self.decoder(z)
        return [x, x_hat]

    def loss_function(self, x, x_hat):
        return {'loss': F.mse_loss(x_hat, x)}
