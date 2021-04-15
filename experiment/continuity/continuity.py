import sys
sys.path.append('../../')

import torch
import yaml
import numpy as np

from s_vae.models import build_model


class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = build_model(config['model'])

    def forward(self, x):
        return self.model(x)

    def __call__(self, x):
        return self.forward(x)


def continuity(config):
    checkpoint_path = config['continuity']['checkpoint_path']

    model = Model(config)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(model(torch.Tensor(np.zeros((1, 28, 28)))))


def main(config_path: str):
    with open(config_path) as fd:
        config = yaml.load(fd, yaml.FullLoader)
    continuity(config)


if __name__ == '__main__':
    config_path = '../config.yaml'
    main(config_path)
