from torch import nn
import numpy as np

from s_vae.models.backbone.utils import Reshape


class ConvEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dims):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.encoder, self.out_dim = self.build_encoder()

    def forward(self, x):
        return self.encoder(x)

    def build_encoder(self):
        in_channels = self.in_dim[0]
        modules = list()
        for h_dim in self.hidden_dims:
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
        return encoder, self.hidden_dims[-1]*self.in_dim[1]*self.in_dim[2]


class ConvDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dims, data_shape, fix_var):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.fix_var = fix_var
        if fix_var():
            self._fix_var = fix_var()
        else:
            self._fix_var = 1e-4
        self.decoder, self.out_dim = self.build_decoder(data_shape)

        self.fc_mu = nn.Conv2d(self.hidden_dims[-1],
                                 out_channels=data_shape[0],
                                 kernel_size=3,
                                 padding=1)
        self.fc_log_var = nn.Linear(np.prod(self.out_dim), 1)

    def forward(self, x):
        x = self.decoder(x)
        mu, log_var = self.fc_mu(x), self.fc_log_var(nn.Flatten()(x))
        if self.fix_var():
            log_var = log_var*0+self.fix_var()
        else:
            log_var = log_var*self._fix_var
        return mu, log_var

    def build_decoder(self, data_shape):
        kernel_size, stride, output_padding = 3, 1, 0

        layer_shapes = [(data_shape[1], data_shape[2])]
        paddings = list()
        for _ in self.hidden_dims:
            h_ = layer_shapes[-1][0] - (kernel_size-1) - 1 - output_padding
            h_padding = int(stride-(stride % h_)/2)
            h = (h_ + h_padding*2) / stride + 1
            w_ = layer_shapes[-1][1] - (kernel_size-1) - 1 - output_padding
            w_padding = int(stride-(stride % w_)/2)
            w = (w_ + w_padding*2) / stride + 1
            layer_shapes.append((int(h), int(w)))
            paddings.append((h_padding, w_padding))
        paddings = paddings[::-1]

        in_channels = data_shape[0]

        modules = list()
        modules.append(nn.Linear(in_features=self.in_dim,
                                 out_features=in_channels * layer_shapes[-1][0] * layer_shapes[-1][1]))
        modules.append(Reshape((1, *layer_shapes[-1])))
        for h_dim, padding in zip(self.hidden_dims, paddings):
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

        decoder = nn.Sequential(*modules)
        return decoder, (self.hidden_dims[-1], np.prod(layer_shapes[0]))
