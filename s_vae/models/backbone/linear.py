import numpy as np
from torch import nn

from s_vae.models.backbone.utils import Reshape


def linear_encoder(in_dim, hidden_dims):
    modules = list()
    if len(in_dim) > 1:
        modules.append(nn.Flatten())
    in_dim = np.prod(np.array(in_dim))

    for h_dim in hidden_dims:
        modules.append(nn.Linear(in_features=in_dim, out_features=h_dim))
        modules.append(nn.LeakyReLU())
        in_dim = h_dim
    modules = modules[:-1]
    encoder = nn.Sequential(*modules)

    return encoder, hidden_dims[-1]


def linear_decoder(in_dim, hidden_dims, out_shape):
    modules = list()
    for h_dim in hidden_dims:
        modules.append(
            nn.Sequential(
                nn.Linear(in_features=in_dim,
                          out_features=h_dim),
                nn.LeakyReLU())
        )
        in_dim = h_dim

    modules = modules[:-1]
    decoder = nn.Sequential(*modules)

    return decoder, hidden_dims[-1], out_shape


