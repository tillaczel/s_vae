import torch 
import math
from torch.distributions.distribution import Distribution
from s_vae.models.backbone.BesselFunc import Bessel
from s_vae.models.backbone.unif_on_sphere import UnifOnSphere

class vMF(Distribution):
    """
    Creates a von Mises-Fisher distribution on a m-dimensional hypersphere

    Args:
    mu (Number, Tensor): Mean direction of the distribution
    kappa (Number, Tensor): Concentration parameter
    ndim (Number, Tensor): Dimension of the hypersphere   
    """

    def __init__(self, mu, kappa, validate_args=None):
        
        self.loc = mu # The mean direction vector
        self.scale = kappa # The concentration parameter    
        self.__ndim = mu.shape[-1]

        self.__e1 = (torch.Tensor([1.0] + [0] * (mu.shape[-1] - 1)))   # Standard Modal vector
        super().__init__(self.loc.size(), validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        pass
    

    def rsample(self, sample_shape= torch.Size()):
        
        shape = sample_shape if isinstance(sample_shape, torch.Size) else torch.Size([sample_shape])