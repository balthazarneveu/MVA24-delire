"""
    Plug and Play ADMM and DRS
    Authors: Andrés Almansa (2020)
             Based on Ryu (2019) and on
             Jialin Liu (danny19921123@gmail.com)
"""
import numpy as np

def drs(proxF,proxG,init,**opts):
    """
    Douglas Rachford Splitting (notation differs from Ryu's paper as follows)
    zk     = x_old + u_old
    xk+1/2 = v
    xk+1   = x
    """

    """ Process parameters. """
    maxitr = opts.get('maxitr', 50)
    verbose = opts.get('verbose', 1)
    monitor  = opts.get('monitor', None)  # debugging messages


    """ Initialization. """

    n = init.size  # == np.prod(init.shape), i.e. number of elements in ndarray

    x = np.copy(init)
    v = np.zeros_like(init, dtype=np.float64)
    u = np.zeros_like(init, dtype=np.float64)

    """ Main loop. """

    for i in range(maxitr):

        # record the variables in the current iteration
        x_old = np.copy(x)
        v_old = np.copy(v)
        u_old = np.copy(u)

        """ proximal step. """

        v = proxF(x+u)
        # prox_poisson_datafit(x+u,noisy,lam)

        """ denoising step. """

        xtilde = np.copy(2*v - x_old - u_old)
        #x = denoiser(np.reshape(xtilde, (m,n)))
        #x = np.reshape(x, -1)
        x = proxG(xtilde)

        """ dual update """

        u = np.copy(u_old + x_old - v)

        """ Monitors """

        if verbose and not (monitor is None) :
            monitor.drs_iter(i,x,v,u,x_old,v_old,u_old)

    """ Get restored image. """
    #x = np.reshape((x) , (m, n))
    return x


def admm(proxF,proxG,init,**opts):
    """
    ADMM - min_x F(x) + G(x) by ADMM splitting
    Inputs:
        :proxF      Proximal operator of F
        :proxG      Proximal operator of G
        :init       Initial value of x
    Optional inputs
        :maxitr     default 50
        :verbose    default 1 (show messages)
        :monitor    default None
    Outputs
        :x          minimizer
    """

    """ Process parameters. """
    maxitr = opts.get('maxitr', 50)
    verbose = opts.get('verbose', 1)
    monitor  = opts.get('monitor', None)  # debugging messages


    """ Initialization. """

    n = init.size  # == np.prod(init.shape), i.e. number of elements in ndarray

    y = np.copy(init)
    x = np.zeros_like(init, dtype=np.float64)
    u = np.zeros_like(init, dtype=np.float64)

    """ Main loop. """

    for i in range(maxitr):

        # record the variables in the current iteration
        x_old = np.copy(x)
        y_old = np.copy(y)
        u_old = np.copy(u)

        """ denoising step. """

        # COMPLETE THIS CODE

        """ proximal step. """

        # COMPLETE THIS CODE

        """ dual update """

        # COMPLETE THIS CODE

        """ Monitors """

        if verbose and not (monitor is None) :
            monitor.admm_iter(i,x,y,u,x_old,y_old,u_old)

    """ Get restored image. """
    return x

#%% ---- define problem-specific monitor ----

# define psnr in a way compatible to skimage
from utils.psnr import psnr
def compare_psnr(im_true,im_test,data_range=None):
  return psnr(im_test,im_true,data_range)

class GaussianMonitor:
    def __init__(self,clean):
        self.clean = clean
    def drs_iter(self,i,x,v,u,x_old,v_old,u_old):
        fpr = np.sqrt(np.sum(np.square((x + u - x_old - u_old)))/x.size)
        psnrs = compare_psnr(im_true=self.clean,
                            im_test=x,
                            # data_range=1.0
                            )
        print("i = {},\t psnr = {},\t fpr = {}".format(i+1, psnrs, fpr))
    admm_iter = drs_iter
