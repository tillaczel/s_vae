from s_vae.models.ae import AE
from s_vae.models.vae import VAE
from s_vae.models.s_vae import SVAE
from s_vae.models.backbone.linear import LinearEncoder, LinearDecoder
from s_vae.models.backbone.conv import ConvEncoder, ConvDecoder


def build_backbone(model_config: dict):
    latent_dim = model_config['latent_dim']
    name = model_config['backbone']['name']
    hidden_dims = model_config['backbone']['hidden_dims']
    data_shape = model_config['backbone']['data_shape']
    if name == 'linear':
        return LinearEncoder(data_shape, hidden_dims), LinearDecoder(latent_dim, hidden_dims[::-1], data_shape)
    elif name == 'conv':
        return ConvEncoder(data_shape, hidden_dims), ConvDecoder(latent_dim, hidden_dims[::-1], data_shape)
    else:
        raise ValueError(f'{name} not in models')


def build_model(model_config: dict, device):
    recon_shape = model_config['backbone']['data_shape']
    encoder, decoder = build_backbone(model_config)

    name = model_config['name']
    latent_dim = model_config['latent_dim']
    if name == 'ae':
        return AE(encoder, decoder, latent_dim)
    elif name == 'vae':
        kl_coeff = model_config['kl_coeff']
        return VAE(encoder, decoder, recon_shape, latent_dim, kl_coeff)
    elif name == 's_vae':
        kl_coeff = model_config['kl_coeff']
        return SVAE(encoder, decoder, recon_shape, latent_dim, kl_coeff, device)
    else:
        raise ValueError(f'{name} not in models')
