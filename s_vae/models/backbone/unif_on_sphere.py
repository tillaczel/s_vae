import torch 
import math
from torch.distributions.distribution import Distribution

class UnifOnSphere(Distribution):
    """
    Creates a uniform distribution on a m-dimensional hypersphere

    Args:
    ndim (Number, Tensor): Dimension of the hypersphere   
    device "cpu"/"gpu": device to have the torch.tensor stored on 
    """

    has_rsample = False
    support = torch.distributions.constraints.real
    _mean_carrier_measure = 0

    def __init__(self, ndim):
        super(UnifOnSphere, self).__init__(
            torch.Size([ndim]))
        self._ndim = ndim

    @property
    def dim(self):
        return self._ndim


    def sample(self, sample_shape=torch.Size(), R = 1):
        """
        To sample n-observations pass sample_shape = torch.Size((n,)) to the sampling method. 
        """
        shape = sample_shape if isinstance(sample_shape, torch.Size) else torch.Size([sample_shape])
        MultNorm = torch.distributions.MultivariateNormal(torch.zeros(self._ndim), torch.eye(self._ndim)).sample(shape)

        assert(R>0)
        return (MultNorm*R)/torch.linalg.norm(MultNorm, dim = -1,  ord= 2, keepdim = True)

    def entropy(self):
        lgamma = torch.lgamma(torch.tensor([(self._ndim + 1) / 2]).to(self.device))
        return math.log(2) + ((self._ndim + 1) / 2) * math.log(math.pi) - lgamma
