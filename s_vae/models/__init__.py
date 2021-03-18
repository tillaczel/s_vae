from s_vae.models.ae import AE
from s_vae.models.vae import VAE
from s_vae.models.backbone.linear import linear_encoder, linear_decoder
from s_vae.models.backbone.conv import conv_encoder, conv_decoder


def build_backbone(model_config: dict):
    latent_dim = model_config['latent_dim']
    name = model_config['backbone']['name']
    hidden_dims = model_config['backbone']['hidden_dims']
    data_shape = model_config['backbone']['data_shape']
    if name == 'linear':
        return *linear_encoder(data_shape, hidden_dims), linear_decoder(latent_dim, hidden_dims[::-1], data_shape)
    if name == 'conv':
        # Todo: it's not working yet
        return *conv_encoder(data_shape, hidden_dims), conv_decoder(latent_dim, hidden_dims[::-1], data_shape)
    else:
        raise ValueError(f'{name} not in models')


def build_model(model_config: dict):
    encoder, encoder_out_dim, decoder = build_backbone(model_config)

    name = model_config['name']
    latent_dim = model_config['latent_dim']
    if name == 'ae':
        return AE(encoder, decoder, encoder_out_dim, latent_dim)
    if name == 'vae':
        kl_coeff = model_config['kl_coeff']
        return VAE(encoder, decoder, encoder_out_dim, latent_dim, kl_coeff)
    else:
        raise ValueError(f'{name} not in models')
