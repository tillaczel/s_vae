import torch

class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape)


def sample_hypersphere(n, dim, R = 1):
    """ 
    This function will sample from uniform distribution on the *dim*-dimensional unit sphere. 
    The method used comes from  "Computer Generation of Distributions on
    the m-sphere" By GARY ULRICH. 
    n: number of points to sample
    dim: the dimension of the sphere.
    R: radius of the sphere
    """

    distrib = torch.distributions.MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    dim_arr = torch.empty(dim, n)

    for i in range(n):
        values = distrib.sample()
        assert(R>0)
        dim_arr[:,i] = (values*R)/torch.linalg.norm(values, ord= 2)
    
    return  dim_arr