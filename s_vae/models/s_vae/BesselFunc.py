import torch
import math
import numpy
import scipy.special

class BesselFunc(torch.autograd.Function):
    """
    Inspired by: https://discuss.pytorch.org/t/modified-bessel-function-of-order-0/18609
    """
    @staticmethod
    def forward(ctx, nu, kappa):
        # Save for the backward pass
        ctx._nu = nu
        ctx.save_for_backward(kappa)

        if math.isclose(nu, 0.0):
            return torch.from_numpy(scipy.special.i0e(kappa.detach().numpy()))
        elif math.isclose(nu, 1.0):
            return torch.from_numpy(scipy.special.i1e(kappa.detach().numpy()))
        else:
            return torch.from_numpy(scipy.special.ive(nu.detach().numpy(), kappa.detach().numpy()))
    
    @staticmethod
    def backward(ctx, grad_out):
        kappa, = ctx.saved_tensors
        nu = ctx._nu
        # formula is from Wolfram: https://functions.wolfram.com/Bessel-TypeFunctions/BesselI/20/01/02/0003/
        return None, 0.5* grad_out *(BesselFunc.apply(nu - 1.0, kappa)+BesselFunc.apply(nu + 1.0, kappa))



Bessel = BesselFunc.apply
