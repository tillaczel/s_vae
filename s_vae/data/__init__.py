from s_vae.data.mnist import create_MNIST, vis_mnist
from s_vae.data.synthetic_hypersphere import create_synthetic_hypersphere


def create_dataset(config: dict, seed=0):
    data_config = config['data']
    name = data_config['name']
    path = data_config['path']
    train_ratio = data_config['train_ratio']

    if name == 'MNIST':
        return create_MNIST(config)
    elif name == 'synth':
        latent_dim = data_config['latent_dim']
        observed_dim = data_config['observed_dim']
        n_dev_samples = data_config['n_dev_samples']
        n_test_samples = data_config['n_test_samples']
        return create_synthetic_hypersphere(path, latent_dim, observed_dim, n_dev_samples, n_test_samples, train_ratio,
                                            seed=seed)
    else:
        raise ValueError(f'{name} is not in datasets')


def dataset_vis_factory(name):
    if name == 'MNIST':
        return vis_mnist
    elif name == 'synth':
        return None
    else:
        raise ValueError(f'{name} is not in datasets')


