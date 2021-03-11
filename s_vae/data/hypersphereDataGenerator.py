import numpy as np
import matplotlib.pyplot as plt
import random

def dataGen(n, dim=3, R=1, datadist=['uniform', 'skew']):
    '''
    A function for generating n randomly distributed
    points on the surface of a hypersphere.
    args:
    n : number of points generated
    dim : dimension of the hypersphere
    R : radius of the hypersphere
    datadist : a list of strings that may include 'uniform'
               and 'skew'. 'uniform' is a random uniform
               distribution. 'skew' is another n/2 data
               points distributed in one m'th quadrant
               (where all vectors have positive values)
               of the hypersphere.
    '''

    dim_arr = [[] for _ in range(dim)]

    if 'uniform' in datadist:
        for _ in range(n):
            r = np.random.random(dim)**2
            r = r/(r.sum())
            coords = np.sqrt(r)
            for i in range(dim):
                if bool(random.getrandbits(1)):
                    x = -1 * coords[i]
                else:
                    x = coords[i]
                dim_arr[i].append(x*R)

    if 'skew' in datadist:
        for _ in range(int(n/2)):
            rp = np.random.random(dim)
            rp = rp/(rp.sum())
            coords = np.sqrt(rp)
            for i in range(dim):
                dim_arr[i].append(coords[i]*R)
    
    return np.array(dim_arr).T

 
# Creating and saving two test-datasets:
#sphereData = dataGen(1000, dim=3, datadist=['uniform'])
#sphereDataNon = dataGen(1000, dim=3, datadist=['uniform','skew'])

#np.savetxt('sphereDataUniform.csv', sphereData.astype(float), delimiter=',')
#np.savetxt('sphereDataNonUniform.csv', sphereData.astype(float), delimiter=',')