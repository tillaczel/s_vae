from torch import nn

from s_vae.models.backbone.utils import Reshape


def conv_encoder(data_shape, hidden_dims):
    in_channels = data_shape[0]
    modules = list()
    for h_dim in hidden_dims:
        modules.append(nn.Conv2d(in_channels=in_channels,
                                out_channels=h_dim,
                                kernel_size=3,
                                stride=1,
                                padding=1))
        modules.append(nn.BatchNorm2d(h_dim))
        modules.append(nn.LeakyReLU())
        in_channels = h_dim
    modules = modules[:-2]
    modules.append(nn.Flatten())
    encoder = nn.Sequential(*modules)
    return encoder, hidden_dims[-1]*data_shape[1]*data_shape[2]


def conv_decoder(latent_dim, hidden_dims, data_shape):
    kernel_size, stride, output_padding = 3, 1, 0

    layer_shapes = [(data_shape[1], data_shape[2])]
    paddings = list()
    for _ in hidden_dims:
        h_ = layer_shapes[-1][0] - (kernel_size-1) - 1 - output_padding
        h_padding = int(stride-(stride % h_)/2)
        h = (h_ + h_padding*2) / stride + 1
        w_ = layer_shapes[-1][1] - (kernel_size-1) - 1 - output_padding
        w_padding = int(stride-(stride % w_)/2)
        w = (w_ + w_padding*2) / stride + 1
        layer_shapes.append((int(h), int(w)))
        paddings.append((h_padding, w_padding))
    output_paddings = paddings[::-1]

    in_channels = data_shape[0]

    modules = list()
    modules.append(nn.Linear(in_features=latent_dim,
                             out_features=in_channels * layer_shapes[-1][0] * layer_shapes[-1][1]))
    modules.append(Reshape((1, *layer_shapes[-1])))
    for h_dim, padding in zip(hidden_dims, paddings):
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels,
                                   out_channels=h_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   output_padding=output_padding),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU())
        )
        in_channels = h_dim

    modules.append(nn.Conv2d(hidden_dims[-1],
                             out_channels=data_shape[0],
                             kernel_size=3,
                             padding=1))
    modules.append(nn.Sigmoid())

    decoder = nn.Sequential(*modules)
    return decoder
