import sys
sys.path.append('../')

import yaml

from s_vae.training import train


def main(config_path: str):
    with open(config_path) as fd:
        config = yaml.load(fd, yaml.FullLoader)
    # Run training
    train(config)


if __name__ == '__main__':
    config_path = 'config.yaml'
    main(config_path)

