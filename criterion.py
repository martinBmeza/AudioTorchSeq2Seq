import torch
import numpy as np
import torch.nn.functional as F

def mse_loss(output, target, ignored_index, reduction='mean'):
    mask = target == ignored_index
    out = (output[~mask]-target[~mask])**2
    if reduction=='mean':
        return out.mean()
    elif reduction==None:
        return out
