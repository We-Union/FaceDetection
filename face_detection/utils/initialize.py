from torch import nn
import torch

# initialise a module with normal distribution
def init_with_normal(module : nn.Module, mean : float, std : float, truncated : bool = False):
    if truncated:
        module.weight.data.normal_().fmod_(2).mul_(std).add_(mean)  
    else:
        module.weight.data.normal_(mean, std)
        module.bias.data.zero_()

# this is a decroator, create an environment doing the forward only
def without_gradient(f):
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return f(*args, **kwargs)
    return wrapper
