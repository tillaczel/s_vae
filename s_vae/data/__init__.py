from s_vae.data.mnist import create_MNIST
from s_vae.data.synthetic_hypersphere import create_synthetic_hypersphere


def create_dataset(data_config: dict, seed=0):
    name = data_config['name']
    path = data_config['path']
    train_ratio = data_config['train_ratio']

    if name == 'MNIST':
        return create_MNIST(path, train_ratio)
    elif name == 'synth':
        latent_dim = data_config['latent_dim']
        observed_dim = data_config['observed_dim']
        n_dev_samples = data_config['n_dev_samples']
        n_test_samples = data_config['n_test_samples']
        return create_synthetic_hypersphere(path, latent_dim, observed_dim, n_dev_samples, n_test_samples, train_ratio,
                                            seed=seed)
    else:
        raise ValueError(f'{name} is not in datasets')


