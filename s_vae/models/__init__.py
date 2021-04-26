from s_vae.models.ae import AE
from s_vae.models.vae import VAE
from s_vae.models.s_vae import SVAE
from s_vae.models.backbone.linear import LinearEncoder, LinearDecoder
from s_vae.models.backbone.conv import ConvEncoder, ConvDecoder


def build_backbone(model_config: dict, fix_var):
    latent_dim = model_config['latent_dim']
    name = model_config['backbone']['name']
    hidden_dims = model_config['backbone']['hidden_dims']
    data_shape = model_config['backbone']['data_shape']
    if name == 'linear':
        return LinearEncoder(data_shape, hidden_dims), LinearDecoder(latent_dim, hidden_dims[::-1], data_shape, fix_var)
    elif name == 'conv':
        return ConvEncoder(data_shape, hidden_dims), ConvDecoder(latent_dim, hidden_dims[::-1], data_shape, fix_var)
    else:
        raise ValueError(f'{name} not in models')


def build_model(model_config: dict, device, fix_var):
    recon_shape = model_config['backbone']['data_shape']
    encoder, decoder = build_backbone(model_config, fix_var)

    name = model_config['name']
    latent_dim = model_config['latent_dim']
    if name == 'ae':
        return AE(encoder, decoder, latent_dim)
    elif name == 'vae':
        return VAE(encoder, decoder, recon_shape, latent_dim)
    elif name == 's_vae':
        return SVAE(encoder, decoder, recon_shape, latent_dim, device)
    else:
        raise ValueError(f'{name} not in models')
