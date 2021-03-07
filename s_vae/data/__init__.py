from s_vae.data.mnist import create_MNIST


def create_dataset(config: dict):
    name = config['data']['name']
    path = config['data']['path']

    if name == 'MNIST':
        return create_MNIST(path)
    else:
        raise ValueError(f'{name} is not in datasets')


