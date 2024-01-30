"""
    simplified interface to the pretrained denoisers in Ryu (2019)
    Requires: numnpy, torch, utils/utils.py and Pretrained_models/*
    author: A. Almansa
    date: 2020-01-20
"""

import numpy as np
import torch
from utils.utils import load_model

class PyTorchDenoiser:
    def __init__(self,
        model_type = "RealSN_DnCNN", # Alternatives: DnCNN/ SimpleCNN / RealSN_DnCNN / RealSN_SimpleCNN
        sigma = 40,                  # Alternatives: 5, 15, 40
        cuda = False,                # If true use pytorch cuda optimisations
        rescale = True):             # If true rescale [min,max] to [0,1] before applying denoiser
        self.model = load_model(model_type, sigma, cuda)
        self.sigma = sigma/255.0
        self.cuda = cuda
        self.rescale = rescale
    def denoise(self,xtilde):
        """
        Inputs:
            :xtilde     noisy image
        Outputs:
            :x          denoised image
        """
        # scale xtilde to be in range of [0,1] (for the clean image)
        if self.rescale:
            mintmp = np.min(xtilde)
            #mintmp = 0.0
            maxtmp = np.max(xtilde)
            #maxtmp = 255.0
            xtilde = (xtilde - mintmp) / (maxtmp - mintmp)


        # image size
        m, n = np.shape(xtilde)

        with torch.no_grad():
            # load to torch
            xtilde_torch = np.reshape(xtilde, (1,1,m,n))
            if self.cuda:
                xtilde_torch = torch.from_numpy(xtilde_torch).type(torch.cuda.FloatTensor)
            else:
                xtilde_torch = torch.from_numpy(xtilde_torch).type(torch.FloatTensor)

            # denoise
            r = self.model(xtilde_torch).cpu().numpy()
            #r = np.reshape(r, -1)
            x = xtilde - r

        # rescale the denoised v back to original scale
        if self.rescale:
            x = x * (maxtmp - mintmp) + mintmp

        # Reshape back
        x = np.reshape(x,(m,n))

        return x
