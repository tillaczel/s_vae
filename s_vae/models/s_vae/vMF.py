import torch 
import math
from torch.distributions.distribution import Distribution
from torch.distributions.beta import Beta
from torch.distributions.uniform import Uniform
from torch.distributions.kl import register_kl
from BesselFunc import Bessel
from unif_on_sphere import UnifOnSphere

class vMF(Distribution):
    """
    Creates a von Mises-Fisher distribution on a m-dimensional hypersphere

    Args:
    mu (Number, Tensor): Mean direction of the distribution
    kappa (Number, Tensor): Concentration parameter 
    """
    arg_constraints = {'mu': torch.distributions.constraints.real_vector,
                       'kappa': torch.distributions.constraints.positive}

    support = torch.distributions.constraints.real_vector
    has_rsample = True

    def __init__(self, mu, kappa, validate_args=None):
        
        self.loc = mu # The mean direction vector
        self.scale = kappa # The concentration parameter    
        self.ndim = torch.tensor(mu.shape[-1],dtype=torch.float64)

        self.__H = self.__Householder()

        super().__init__(self.loc.size(), validate_args=validate_args)



    def rsample(self, sample_shape= torch.Size()):
        
        shape = sample_shape if isinstance(sample_shape, torch.Size) else torch.Size([sample_shape])

        UnifSphere = UnifOnSphere(self.loc.shape[-1]-1)
        sample_rslt = torch.empty(shape)

        for i in range(shape[0]):
            w = UnifSphere.sample(torch.Size((1,)))
            omega = self.sample_omega()
            w= torch.sqrt((1-torch.pow(omega,2)))*w
            z = torch.cat((omega, w.view(-1)))
            sample_rslt[i,:] = self.__Reflect(z)
        
        return sample_rslt


    def sample_omega(self):
        
        # Init:         
        b = (-2*self.scale + torch.sqrt(4*torch.pow(self.scale,2)+torch.pow(self.ndim-1,2)))/(self.ndim-1)
        a = ((self.ndim-1)+2*self.scale+torch.sqrt(4*torch.pow(self.scale,2)+torch.pow(self.ndim-1,2)))/4
        d = 4*a*b/(1+b)-(self.ndim-1)*torch.log(self.ndim-1)

        # Acceptance/Rejection sampling:
        while True:
            epsilon = Beta(0.5*(self.ndim-1),0.5*(self.ndim-1)).sample(torch.Size((1,)))

            omega = (1-(1+b)*epsilon)/(1+(1-b)*epsilon)

            T = 2*a*b/(1+(1-b)*epsilon)

            u = Uniform(torch.tensor([0.0]), torch.tensor([1.0])).sample(torch.Size((1,)))

            if (self.ndim-1)*torch.log(T)-T+d >= torch.log(u):
                break
        
        return omega


    def __Householder(self):
        """
        Defines the Householder transformation matrix H. 

        The matrix is constructed using the initial unit vector and the mean direction vecor.
        """

        ksi = (torch.Tensor([1.0] + [0] * (self.loc.shape[-1] - 1)))
        nu = ksi-self.loc
        nu = nu/torch.linalg.norm(nu,ord =2, dim = -1)
        
        return torch.eye(self.loc.shape[-1])- 2*torch.outer(nu,nu)


    def __Reflect(self, x):
        """
        Applies the Householder reflection of x to a direction parallel to the mean direction vector
        
        Args:
        x (Tensor): vector to transform
        """
        return torch.mv(self.__H,x.view(-1).float())


    def entropy(self):
        output = (
            -self.scale
            * Bessel(self.ndim/ 2, self.scale)
            / Bessel((self.ndim/ 2) - 1, self.scale)
        )

        return output.view(*(output.shape[:-1])) + self._log_normalization()

    def _log_normalization(self):
        output = -(
            (self.ndim/ 2 - 1) * torch.log(self.scale)
            - (self.ndim/ 2) * math.log(2 * math.pi)
            - torch.log(Bessel(self.ndim/ 2 - 1, self.scale))
        )

        return output.view(*(output.shape[:-1]))


@register_kl(vMF, UnifOnSphere)
def _kl_vmf_uniform(vmf, unisphere):
    return -vmf.entropy() + unisphere.entropy()


if __name__ == "__main__":
    hyp = UnifOnSphere(2)
    
    mu = hyp.sample()
    
    kappa = torch.tensor([2.5])
    
    
    test_vmf = vMF(mu, kappa)
    
    sample = test_vmf.rsample(torch.Size((10,2)))
    print(torch.linalg.norm(sample, ord = 2, dim = -1))