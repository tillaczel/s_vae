from s_vae.models.AE import AE


def build_model(model_config):
    name = model_config['name']
    input_shape = model_config['input_shape']
    if name == 'ae':
        return AE(input_shape=input_shape)
    else:
        raise ValueError(f'{name} not in models')
