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


        super().__init__(self.loc.size(), validate_args=validate_args)



    def rsample(self, sample_shape= torch.Size()):
        
        shape = sample_shape if isinstance(sample_shape, torch.Size) else torch.Size([sample_shape])

        sample_rslt = torch.empty(shape)

        # Sample V~Unif(S)
        UnifSphere = UnifOnSphere(self.loc.shape[-1]-1)
        w = UnifSphere.sample(torch.Size((shape[0],)))


        # Sample Acceptance/rejection for vMF in 1 dimension: 
        omega = self.sample_omega(shape)

        omega_factor = torch.sqrt((1-torch.pow(omega,2)))

        w =  w * omega_factor[:, None]

        z = torch.cat((omega[:, None],w),1)

        
        for i in range(shape[0]):
            H = self.__Householder(i)
            sample_rslt[i,:] = self.__Reflect(H,z[i,:])
        
        return sample_rslt


    def sample_omega(self,sample_shape):
        
        # Init:      
        b = (-2*self.scale + torch.sqrt(4*torch.pow(self.scale,2)+torch.pow(self.ndim-1,2)))/(self.ndim-1)
        a = ((self.ndim-1)+2*self.scale+torch.sqrt(4*torch.pow(self.scale,2)+torch.pow(self.ndim-1,2)))/4
        d = 4*a*b/(1+b)-(self.ndim-1)*torch.log(self.ndim-1)

        # Acceptance/Rejection sampling:
        accpt = torch.zeros_like(b,dtype=torch.bool)
        indx_rejected = (accpt == 0).nonzero(as_tuple=False).view(-1)
        indx_accepted = (accpt == 1).nonzero(as_tuple=False).view(-1)
        number_rejected = (accpt == 0).count_nonzero()
        omega = torch.empty(size = (sample_shape[0],1), dtype=torch.float64).view(-1)
        _omega = torch.empty(size = (sample_shape[0],1), dtype=torch.float64).view(-1)
        while True:
            _a = a[indx_rejected]
            _b = b[indx_rejected]
            _d = d[indx_rejected]

            epsilon = Beta(0.5*(self.ndim-1),0.5*(self.ndim-1)).sample(torch.Size((number_rejected,)))

            _omega[indx_rejected] = (1-(1+_b)*epsilon)/(1+(1-_b)*epsilon).view(-1)

            T = 2*_a*_b/(1+(1-_b)*epsilon)

            u = Uniform(torch.tensor([0.0]), torch.tensor([1.0])).sample(torch.Size((number_rejected,)))

            accpt[indx_rejected] = (self.ndim-1)*torch.log(T)-T+_d >= torch.log(u).view(-1)

            indx_rejected = (accpt == 0).nonzero(as_tuple=False).view(-1)
            indx_accepted = (accpt == 1).nonzero(as_tuple=False).view(-1)

            omega[indx_accepted] = _omega[indx_accepted]
            number_rejected = (accpt == 0).count_nonzero()

            if number_rejected == 0:
                break
        
        return omega


    def __Householder(self, i):
        """
        Defines the Householder transformation matrix H. 

        The matrix is constructed using the initial unit vector and the mean direction vecor.
        """

        ksi = (torch.Tensor([1.0] + [0] * (self.loc.shape[1] - 1)))
        nu = ksi-self.loc[i,:]
        nu = nu/torch.linalg.norm(nu,ord =2, dim = -1)
        
        return torch.eye(self.loc.shape[-1])- 2*torch.outer(nu,nu)


    def __Reflect(self, H, x):
        """
        Applies the Householder reflection of x to a direction parallel to the mean direction vector
        
        Args:
        x (Tensor): vector to transform
        """

        return torch.mv(H,x.view(-1).float())


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
    hyp = UnifOnSphere(5)
    
    mu = hyp.sample(torch.Size((5,)))
    print(mu)
    kappa = torch.tensor([2.5]*5)
    print(kappa)
    
    test_vmf = vMF(mu, kappa)
    sample = test_vmf.rsample(torch.Size((5,5)))
    print(torch.linalg.norm(sample, ord = 2, dim = -1))