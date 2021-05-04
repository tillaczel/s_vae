import sys
sys.path.append('../../')

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

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


def knn(config):
    checkpoint_path = config['knn']['checkpoint_path']

    model = Model(config) #architecture of the model
    checkpoint = torch.load(checkpoint_path) # WEIGHTS AND BIASES
    model.load_state_dict(checkpoint['state_dict']) # update model weights and biases
    model.eval() # evaluation mode

    # for data loader
    batch_size = config['training'].get('batch_size', 32)
    num_workers = config['data'].get('num_workers', 1)

    # create data loader, load in images, do test on them
    train_set, val_set, test_set = create_MNIST(config) # with some paramters. These are dataloaders.
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    z_total = torch.empty(1,3)
    y_total = torch.empty(1)

    z_reg = []#torch.empty(1,3)
    z_anomaly = []#torch.empty(1,3)

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

    print('y_total size:', np.shape(y_total))
    print('z_total size:', np.shape(z_total))
    print('\n')


    # Visualize the latent space in 2D: Should I standardize data? 
    # Do PCA
    pca = PCA(n_components = 2)
    z_total_2d = pca.fit_transform(z_total)
    print('z_total_2d size:', np.shape(z_total_2d))

    # plt.scatter(z_total_2d[:,0], z_total_2d[:,1])
    # plt.savefig('MNIST_latentspace_2D.png')

    # 2D Visualization
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal component 1', fontsize=14)
    ax.set_ylabel('Principal component 2', fontsize=14)
    ax.set_title('2 component PCA', fontsize=14)

    targets = [0,1,2,3,4,5,6,7,8,9]
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive']
    for target, color in zip(targets,colors):
        #for i in y_total:
        indicesToKeep = y_total == target
        ax.scatter(z_total_2d[indicesToKeep, 0],
                   z_total_2d[indicesToKeep, 1],
                   c = color,
                   s = 10,
                   alpha = 0.5)
    ax.legend(targets)
    #ax.grid()
    plt.savefig('MNIST_latentspace_2D.png')


    # 3D Visulization
    x = z_total[:,0]
    y = z_total[:,1]
    z = z_total[:,2]

    # #col_array = ["" for x in range(len(y_total))]
    # col_array = np.empty([len(y_total), 1],  dtype=str)
    # colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive']
    # for i in y_total:
    #     for j in range(len(colors)):
    #         #print(j)
    #         if y_total[i] == j:
    #             #print('NIGGA!')
    #             col_array[i] = colors[j]
    # #print('Colors array:', col_array)



    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    
    for target, color in zip(targets,colors):
        #for i in y_total:
        indicesToKeep = y_total == target
        ax.scatter(x,y,z,
                   c = y_total,
                   s = 2,
                   alpha = 0.5)

    #ax.scatter(x,y,z)
    plt.savefig('MNIST_latentspace_3D.png')



    ############ 2d on dimensions ###############

    # 2D Visualization: dim 0 and 1 
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Dimension 0', fontsize=14)
    ax.set_ylabel('Dimension 1', fontsize=14)
    ax.set_title('2 component PCA', fontsize=14)
    targets = [0,1,2,3,4,5,6,7,8,9]
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive']
    for target, color in zip(targets,colors):
        #for i in y_total:
        indicesToKeep = y_total == target
        ax.scatter(x[indicesToKeep],
                   y[indicesToKeep],
                   c = color,
                   s = 10,
                   alpha = 0.5)
    ax.legend(targets)
    #ax.grid()
    plt.savefig('MNIST_latentspace_dim_0_and_1.png')

    # 2D Visualization: dim 0 and 2 
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Dimension 0', fontsize=14)
    ax.set_ylabel('Dimension 1', fontsize=14)
    ax.set_title('2 component PCA', fontsize=14)
    targets = [0,1,2,3,4,5,6,7,8,9]
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive']
    for target, color in zip(targets,colors):
        #for i in y_total:
        indicesToKeep = y_total == target
        ax.scatter(x[indicesToKeep],
                   z[indicesToKeep],
                   c = color,
                   s = 10,
                   alpha = 0.5)
    ax.legend(targets)
    #ax.grid()
    plt.savefig('MNIST_latentspace_dim_0_and_2.png')

    # 2D Visualization: dim 1 and 2 
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Dimension 0', fontsize=14)
    ax.set_ylabel('Dimension 1', fontsize=14)
    ax.set_title('2 component PCA', fontsize=14)
    targets = [0,1,2,3,4,5,6,7,8,9]
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive']
    for target, color in zip(targets,colors):
        #for i in y_total:
        indicesToKeep = y_total == target
        ax.scatter(y[indicesToKeep],
                   z[indicesToKeep],
                   c = color,
                   s = 10,
                   alpha = 0.5)
    ax.legend(targets)
    #ax.grid()
    plt.savefig('MNIST_latentspace_dim_1_and_2.png')






    # KNN: Choose number of neighbors and fit on the whole latent space
    knn_model = KNeighborsClassifier(config['knn']['n_neighbors'])


    # cross-val ?: get the best k . 
    # F1 score for goodness!!!  Accuracy or F1. 

    z_train, z_test, y_train, y_test = train_test_split(  # split and train and test
        z_total, y_total, test_size = 0.2, random_state = 1
    )

    knn_model.fit(z_train, y_train)
    y_preds = knn_model.predict(z_test)

    conf_matrix = metrics.confusion_matrix(y_test, y_preds)
    classif_report = metrics.classification_report(y_test, y_preds, digits = 3)
    print('Classification report for Nearest Neighbor', '\n')
    print(classif_report)

def main(config_path: str):
    with open(config_path) as fd:
        config = yaml.load(fd, yaml.FullLoader)
    knn(config)


if __name__ == '__main__':
    config_path = '../config.yaml'
    main(config_path)
