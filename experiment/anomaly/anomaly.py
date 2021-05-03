import sys
sys.path.append('../../')

import torch
import yaml
import numpy as np
#import matplotlib as plt
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from s_vae.models import build_model                        # ??? Where is this ?? 
from s_vae.data.mnist import create_MNIST


class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = build_model(config['model'], 'CPU')  # is this right?

    def forward(self, x):
        return self.model(x)

    def __call__(self, x):
        return self.forward(x)


def anomaly(config):
    checkpoint_path = config['anomaly']['checkpoint_path']

    model = Model(config) #architecture of the model
    checkpoint = torch.load(checkpoint_path) # WEIGHTS AND BIASES
    model.load_state_dict(checkpoint['state_dict']) # update model weights and biases
    model.eval() # evaluation mode

    # for data loader
    batch_size = config['training'].get('batch_size', 32)
    num_workers = config['data'].get('num_workers', 1)

    # create data loader, load in images, do test on them
    train_set, val_set, test_set = create_MNIST(config['data']['path'], config['data']['train_ratio']) # with some paramters. These are dataloaders.
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # train_set -> train_loader -> train_image:  google

    # image = test_loader[0][0]
    # label = test_loader[0][1]
    # x, y, x_hat, z are for each bath. test_loader = batch     (?)
    
    z_total = torch.empty(1,3)
    y_total = torch.empty(1)

    z_reg = []#torch.empty(1,3) 0 - 8
    z_anomaly = []#torch.empty(1,3) 9 

    y_reg = []#torch.empty(1)
    y_anomaly = []#torch.empty(1)

    for i, batch in enumerate(test_loader, 0):
        x, y = batch # x: 32 * 3 | y: 32 * 1 
        x_hat, z = model.forward(x)
        
        if i == 0:
            y_total = y
            z_total = z
        else:
           y_total = torch.cat((y_total, y), 0)
           z_total = torch.cat((z_total, z), 0)

    # convert tensors to numpy arrays
    y_total = y_total.cpu().detach().numpy()
    z_total = z_total.cpu().detach().numpy()
    #z_total = z_total.reshape()

    print('y_total size:', np.shape(y_total))
    print('z_total size:', np.shape(z_total))
    print('\n')


    for i in range(len(y_total)):

        if y_total[i] == 9:
            y_anomaly = np.append(y_anomaly, y_total[i])
            
            z_anomaly = np.append(z_anomaly, z_total[i], axis = 0)
            #z_anomaly = z_anomaly.reshape(-1, 3)

            #print(z_total[i])
        else:
            y_reg = np.append(y_reg, y_total[i])
            z_reg = np.append(z_reg, z_total[i], axis = 0)
    
    z_anomaly = z_anomaly.reshape(-1, 3)
    z_reg = z_reg.reshape(-1, 3)
    #print(type(z_total))

    print('y_regular size:', np.shape(y_reg))
    print('z_regular size:', np.shape(z_reg))
    print('\n')
    print('y_anomaly size:', np.shape(y_anomaly))
    print('z_anomaly size:', np.shape(z_anomaly))
    print('\n')

    
    # Plotting Anomaly
    fig, m_axs = plt.subplots(config['model']['latent_dim'] ,config['model']['latent_dim'], figsize=(config['model']['latent_dim']*5, config['model']['latent_dim']*5))
    if config['model']['latent_dim'] == 1:
        m_axs = [[m_axs]]
    for i, n_axs in enumerate(m_axs, 0):
        for j, c_ax in enumerate(n_axs, 0):
            c_ax.scatter(np.concatenate([z_reg[:, i], z_anomaly[:,i]],0), 
                           np.concatenate([z_reg[:, j], z_anomaly[:,j]],0),
            c=(['g'] * z_reg.shape[0]) + ['r'] * z_anomaly.shape[0], alpha = 0.5)
    # save it locally
    plt.savefig('anomaly_MNIST_2.png')


    # can also do the KNN on the z. 
    # pytorch tensorboard - see how to set up . google
    # or some other. 


def main(config_path: str):
    with open(config_path) as fd:
        config = yaml.load(fd, yaml.FullLoader)
    anomaly(config)


if __name__ == '__main__':
    config_path = '../config.yaml'
    main(config_path)
