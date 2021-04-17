from torch import nn

from s_vae.models.backbone.utils import Reshape


def conv_encoder(in_channels, hidden_dims):
    modules = list()
    for h_dim in hidden_dims:
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=h_dim,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU())
        )
        in_channels = h_dim

    encoder = nn.Sequential(*modules)
    return encoder


def conv_decoder(latent_dim, hidden_dims, data_shape):
    modules = list()
    input_shape = (data_shape[0], data_shape[1])
    # H = (H-1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
    for _ in hidden_dims:
        input_shape[0] = (input_shape[0]+1)/3
    # H=(H−1)×3+2
    modules.append(Reshape())
    for h_dim in hidden_dims:
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels,
                                   out_channels=h_dim,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU())
        )
        in_channels = h_dim

    modules.append(nn.Conv2d(hidden_dims[-1],
                             out_channels=out_channels,
                             kernel_size=3,
                             padding=1))
    modules.append(nn.Sigmoid())
    modules.append(nn.Flatten())

    decoder = nn.Sequential(*modules)
    return decoder