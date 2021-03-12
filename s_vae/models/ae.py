import torch
from torch import nn


class AE(nn.Module):
    def __init__(self, encoder, decoder, encoder_out_dim, latent_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.latent = nn.Linear(in_features=encoder_out_dim, out_features=latent_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        latent = self.latent(encoded)
        reconstructed = self.decoder(latent)
        return reconstructed
