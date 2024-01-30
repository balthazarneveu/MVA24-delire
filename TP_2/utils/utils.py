# Utils used in this project.
# Authors: Jialin Liu (UCLA math, danny19921123@gmail.com)

import torch
import torch.nn as nn
import numpy as np

# ---- load the model based on the type and sigma (noise level) ----
def load_model(model_type, sigma, cuda=True):
    path = "Pretrained_models/" + model_type + "_noise" + str(sigma) + ".pth"
    if model_type == "DnCNN":
        from model.models import DnCNN
        net = DnCNN(channels=1, num_of_layers=17)
        if cuda:
            model = nn.DataParallel(net).cuda()
        else:
            model = nn.DataParallel(net)
    elif model_type == "SimpleCNN":
        from model.SimpleCNN_models import DnCNN
        if cuda:
            model = DnCNN(1, num_of_layers = 4, lip = 0.0, no_bn = True).cuda()
        else:
            model = DnCNN(1, num_of_layers = 4, lip = 0.0, no_bn = True)
    elif model_type == "RealSN_DnCNN":
        from model.realSN_models import DnCNN
        net = DnCNN(channels=1, num_of_layers=17)
        if cuda:
            model = nn.DataParallel(net).cuda()
        else:
            model = nn.DataParallel(net)
    elif model_type == "RealSN_SimpleCNN":
        from model.SimpleCNN_models import DnCNN
        if cuda:
            model = DnCNN(1, num_of_layers = 4, lip = 1.0, no_bn = True).cuda()
        else:
            model = DnCNN(1, num_of_layers = 4, lip = 1.0, no_bn = True)
    else:
        from model.realSN_models import DnCNN
        net = DnCNN(channels=1, num_of_layers=17)
        if cuda:
            model = nn.DataParallel(net).cuda()
        else:
            model = nn.DataParallel(net)

    if cuda:
        model.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
    model.eval()
    return model

# # ---- calculating PSNR (dB) of x -----
# def psnr(x,im_orig,peak=None):
#     #xout = (x - np.min(x)) / (np.max(x) - np.min(x))
#     if peak == None :
#         norm1 = np.sum((np.absolute(im_orig)) ** 2)
#     else
#         norm1 = (peak**2)*np.prod(np.shape(im_orig))
#     norm2 = np.sum((np.absolute(x - im_orig)) ** 2)
#     psnr = 10 * np.log10( norm1 / norm2 )
#     return psnr
# def rmse(x,im_orig):
#     N = np.prod(np.shape(im_orig))
#     norm2 = np.sum((np.absolute(x - im_orig)) ** 2)
#     rmse = np.sqrt(norm2/N)
#     return rmse
