"""
    simplified interface to the Chambolle-Pock TV denoising algorithm
    Requires: numnpy
    author: Julie Delon & Andrés Almansa
    date: 2020-02-06
"""
import numpy as np
import matplotlib.pyplot as plt

def div(cx,cy):
    #cy and cy are coordonates of a vector field.
    #the function compute the discrete divergence of this vector field

    nr,nc=cx.shape

    ddx=np.zeros((nr,nc))
    ddy=np.zeros((nr,nc))

    ddx[:,1:-1]=cx[:,1:-1]-cx[:,0:-2]
    ddx[:,0]=cx[:,0]
    ddx[:,-1]=-cx[:,-2]

    ddy[1:-1,:]=cy[1:-1,:]-cy[0:-2,:]
    ddy[0,:]=cy[0,:]
    ddy[-1,:]=-cy[-2,:]

    d=ddx+ddy

    return d

def grad(im):
    #compute the gradient of the image 'im'
    # image size
    nr,nc=im.shape

    gx = im[:,1:]-im[:,0:-1]
    gx = np.block([gx,np.zeros((nr,1))])

    gy =im[1:,:]-im[0:-1,:]
    gy=np.block([[gy],[np.zeros((1,nc))]])
    return gx,gy



def chambolle_pock_prox_TV(ub,lambd,niter):

    nr,nc = ub.shape
    ut = np.copy(ub)
    ubar = np.copy(ut)
    p = np.zeros((nr,nc,2))
    tau   = 0.9/np.sqrt(8*lambd**2)
    sigma = 0.9/np.sqrt(8*lambd**2)
    theta = 1

    for k in range(niter):
        # calcul de proxF
        ux,uy  = grad(ubar)
        p = p + sigma*lambd*np.stack((ux,uy),axis=2)
        normep = np.sqrt(p[:,:,0]**2+p[:,:,1]**2)
        normep = normep*(normep>1) + (normep<=1)
        p[:,:,0] = p[:,:,0]/normep
        p[:,:,1] = p[:,:,1]/normep

        # calcul de proxG
        d=div(p[:,:,0],p[:,:,1])
        #TVL2
        unew = 1/(1+tau)*(ut+tau*lambd*d+tau*ub)

        #extragradient step
        ubar = unew+theta*(unew-ut)
        ut = unew
    return ut

class TVDenoiser:
    def __init__(self,
        lamb = 0.1,                   # Any positive real value
        niter = 100):                 # Positive integer
        self.lamb = lamb
        self.niter = niter
    def denoise(self,xtilde):
        """
        Inputs:
            :xtilde     noisy image
        Outputs:
            :x          denoised image
        """
        x = chambolle_pock_prox_TV(xtilde,self.lamb,self.niter)
        return x
