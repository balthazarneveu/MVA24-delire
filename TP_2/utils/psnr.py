# PSNR and RMSE error measures
# Authors: Andrés Almansa

import numpy as np

# ---- calculating PSNR (dB) of x -----
def psnr(x,im_orig,peak=None):
    # psnr(x,x0) = 10*log10( ||x0||^2 / ||x-x0||^2 )
    # psnr(x,x0,peak) = 10*log10( peak^2 * size(x0) / ||x-x0||^2 )
    if peak==None:
        norm1 = np.sum((np.absolute(im_orig)) ** 2)
    else:
        norm1 = (peak**2)*np.prod(np.shape(im_orig))
    norm2 = np.sum((np.absolute(x - im_orig)) ** 2)
    psnr = 10 * np.log10( norm1 / norm2 )
    return psnr

# ---- calculating RMSE of x -----
def rmse(x,im_orig):
    # rmse(x,x0) = sqrt( || x - x0 ||^2 / size(x0) )
    N = np.prod(np.shape(im_orig))
    norm2 = np.sum((np.absolute(x - im_orig)) ** 2)
    rmse = np.sqrt(norm2/N)
    return rmse
