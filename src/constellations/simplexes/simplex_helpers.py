import math
import torch
import gpytorch
from .utils import flatten, unflatten_like
import time

def volume_loss(model):
    """Computes the volume of the simplex specified by the (model-)weights
    of model. See https://en.wikipedia.org/wiki/Cayley%E2%80%93Menger_determinant for 
    further description of how volume is calculated.
    Args:
        model:  A SimplexNet (see simplex/models/simplex_models.py) instance.
    """
    cdist = gpytorch.kernels.Kernel().covar_dist
    n_vert = model.n_vert
        
    mat = torch.ones(n_vert+1, n_vert+1) - torch.eye(n_vert + 1)
    
    ## compute distance between parameters ##
    n_par = int(sum([p.numel() for p in model.parameters()])/n_vert)
    par_vecs = torch.zeros(n_vert, n_par)
    for ii in range(n_vert):
        par_vecs[ii, :] = flatten([p for p in model.net.parameters()][ii::n_vert])
    
    dist_mat = cdist(par_vecs, par_vecs).pow(2)
    mat[:n_vert, :n_vert] = dist_mat
    
    norm = (math.factorial(n_vert-1)**2) * (2. ** (n_vert-1))
    return torch.abs(torch.det(mat)).div(norm)


def complex_volume(model, ind: int) -> torch.Tensor:
    """Computes the volume of the simplex at index ind in the simplicial_complex
    of model. See https://en.wikipedia.org/wiki/Cayley%E2%80%93Menger_determinant for 
    further description of how volume is calculated.
    Args:
        model:  A SimplexNet (see simplex/models/simplex_models.py) instance.
    """
    cdist = gpytorch.kernels.Kernel().covar_dist
    n_vert = len(model.simplicial_complex[ind])
    total_vert = model.n_vert
        
    mat = torch.ones(n_vert+1, n_vert+1) - torch.eye(n_vert + 1)
    
    ## compute distance between parameters ##
    temp_pars = [p for p in model.net.parameters()][0::total_vert]
    n_par = int(sum([p.numel() for p in temp_pars]))
    par_vecs = torch.zeros(n_vert, n_par).to(temp_pars[0].device)
    for ii, vv in enumerate(model.simplicial_complex[ind]):
        par_vecs[ii, :] = flatten([p for p in model.net.parameters()][vv::total_vert])

    dist_mat = cdist(par_vecs, par_vecs).pow(2)
    mat[:n_vert, :n_vert] = dist_mat

    norm = (math.factorial(n_vert-1)**2) * (2. ** (n_vert-1))
    return torch.abs(torch.det(mat)).div(norm)