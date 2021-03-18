import sys

sys.path.append('../../')

import yaml

from s_vae.embedding_projector import run


ep_config = {'version': 6,
             'epoch': 99
             }

def main(config_path: str):
    with open(config_path) as fd:
        config = yaml.load(fd, yaml.FullLoader)
    config['embedding_projector'] = ep_config
    run(config)


if __name__ == '__main__':
    config_path = '../config.yaml'
    main(config_path)


