import numpy as np
from torch import nn

from s_vae.models.backbone.utils import Reshape


class LinearEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dims):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims

        modules = list()
        if len(in_dim) > 1:
            modules.append(nn.Flatten())
        in_dim = np.prod(np.array(in_dim))
        self.encoder, self.out_dim = construct_linear_layers(modules, in_dim, hidden_dims)

    def forward(self, x):
        return self.encoder(x)


class LinearDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dims, recon_shape):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims

        modules = list()
        self.decoder, self.out_dim = construct_linear_layers(modules, in_dim, hidden_dims)

        self.fc_mu = nn.Sequential(nn.Linear(self.out_dim, np.prod(recon_shape)),
                                   Reshape(recon_shape))
        self.fc_log_var = nn.Linear(self.out_dim, 1)

    def forward(self, x):
        x = self.decoder(x)
        return self.fc_mu(x), self.fc_log_var(x)


def construct_linear_layers(modules, in_dim, hidden_dims):
    for h_dim in hidden_dims:
        modules.append(nn.Linear(in_features=in_dim, out_features=h_dim))
        modules.append(nn.LeakyReLU())
        in_dim = h_dim

    modules = modules[:-1]
    model = nn.Sequential(*modules)

    return model, hidden_dims[-1]



