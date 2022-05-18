###############################################################################
# Image loss and tonemapper Adapted from https://github.com/NVlabs/nvdiffrec
###############################################################################

import torch
from torch.nn import functional

def _tonemap_srgb(f):
    return torch.where(f > 0.0031308, torch.pow(torch.clamp(f, min=0.0031308), 1.0/2.4)*1.055 - 0.055, 12.92*f)
    
def _SMAPE(img, target, eps=0.01):
    nom = torch.abs(img - target)
    denom = torch.abs(img) + torch.abs(target) + 0.01
    return torch.mean(nom / denom)

def _RELMSE(img, target, eps=0.1):
    nom = (img - target) * (img - target)
    denom = img * img + target * target + 0.1 
    return torch.mean(nom / denom)

def get_loss_fn(loss_fn, tonemap=False):
    if tonemap:
        tonemapper = _tonemap_srgb
    else:
        tonemapper = lambda x: x
        
    if loss_fn == 'l1':
        loss_fn_ = functional.l1_loss
    elif loss_fn == 'l2':
        loss_fn_ = functional.mse_loss
    elif loss_fn == 'smape':
        loss_fn_ = _SMAPE
    elif loss_fn == 'relmse':
        loss_fn_ = _RELMSE
    else:
        raise NotImplementedError(f'Loss function {loss_fn} not implemented yet')
    
    return lambda x, y: loss_fn_(tonemapper(x), tonemapper(y))