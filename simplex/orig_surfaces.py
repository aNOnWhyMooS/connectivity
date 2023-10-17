import sys
import collections
import torch
import torch.nn as nn
import numpy as np
import utils
import warnings

from typing import Any, Callable, Iterable, Tuple, Union, OrderedDict

sys.path.append("../../simplex/models/")
from orig_basic_simplex import BasicSimplex
from simplex_models import SimplexNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gram_schmidt(basis1: torch.Tensor, 
                 basis2: torch.Tensor) -> Tuple[torch.Tensor, 
                                                torch.Tensor,
                                                float]:
    """Performs gram-schmidt orthogonalization for basis1 and basis2.
    Returns:
        1. basis1, just normalized to unit length and,
        2. component of basis2, perpendicular to basis1, normalized 
           to unit length
        3. The length of the component of basis2 removed while orthogonalizing
           it, divided by the length of basis1. 
    [[NOTE: No normalization in current implementation.]]
    """
    vu = basis2.squeeze().dot(basis1.squeeze())
    uu = basis1.squeeze().dot(basis1.squeeze())

    basis2 = basis2 - basis1.mul(vu).div(uu)

    #basis1 = basis1.div(basis1.norm())
    #basis2 = basis2.div(basis2.norm())
    return basis1, basis2, vu.div(uu).item()

def check_big_comp(vec1: torch.Tensor, vec2: torch.Tensor,) -> bool:
    """Returns True if component of vec1 along vec2 is greater than or equal to 
    the norm of vec2"""
    dot_prod = vec1.squeeze().dot(vec2.squeeze())
    return dot_prod.div(vec2.norm()).item() >= vec2.norm().item()

def get_sd_basis(anchor: nn.Module,
                 base1: nn.Module,
                 base2: nn.Module) -> Tuple[OrderedDict[str, torch.Tensor], 
                                            OrderedDict[str, torch.Tensor],
                                            float, bool]:
    """Returns two state-dicts corresponding to basis vectors for perturbations around 
    anchor. The returned state dicts DO-NOT contain buffers.
    Args:
        anchor, base1, base2:   The models constituting the anchor and two other points
                                determining the plane for which to plot loss.
    PRE-CONDITION:
        1. All inputs must have be instances of the same model.

    Process:
        1. Generate two vectors corresponding to basis1.parameters()-anchor.parameters() and
           basis2.parameters()-anchor.parameters(), say v1 and v2.
        2. If component of v2 along v1 is greater than or equal to the norm of v1, then
               2.1. v2 is kept as it is.
               2.2  Component of v1 along v2 is removed from v1 to get a a vector perpendicular
                    to v2.
            Else:
                2.1. v1 is kept as it is.
                2.2  Component of v2 along v1 is removed from v2 to get a a vector perpendicular
                     to v1.
        Let o denote the orthogonalized vector and o' denote the other one.
    
    Returns:
        A tuple of two state dicts, corresponding to basis vectors for perturbations around anchor, and some additional
        quantities. A description is given below. 
            basis1_state_dict: A state dict holding parameters of the un-modified vector of parmeters.
            basis2_state_dict: A state dict holding parameters of the orthogonalized vector of parameters.
            removed_comp_len:  The length of component removed while orthogonalizing o, divided by the length 
                            of o'.
            swapped_basis:       True if v1 was orthogonalized to v2, else False.
            perp_scaling_factor: Scaling factor to make sure scale on X-axis and Y-axis are equal. 
                                  Multiply y-axis by this factor.
    """
    if not type(anchor)==type(base1)==type(base2):
        raise NotImplementedError("All models: anchor, base1 and base2, must have same type.")
    
    buffers = [name for (name, _) in anchor.named_buffers()]
    
    basis1_pars = [v1-v for (_,v1), (k,v) in zip(base1.state_dict().items(), 
                                                 anchor.state_dict().items()) if k not in buffers]
    basis2_pars = [v2-v for (_,v2), (k,v) in zip(base2.state_dict().items(), 
                                                 anchor.state_dict().items()) if k not in buffers]

    basis1 = utils.flatten(basis1_pars)
    basis2 = utils.flatten(basis2_pars)
    
    swapped_basis = False
    
    if check_big_comp(basis2, basis1):
        swapped_basis = True
        basis1, basis2 = basis2, basis1
    
    basis1, basis2, removed_comp_len = gram_schmidt(basis1, basis2)
    perp_scaling_factor = basis2.norm().div(basis1.norm()).item()
    
    anchor_pars = [v for k,v in anchor.state_dict().items() if k not in buffers]
    basis1 = utils.unflatten_like(basis1.unsqueeze(0), anchor_pars)
    basis2 = utils.unflatten_like(basis2.unsqueeze(0), anchor_pars)

    basis1_state_dict = collections.OrderedDict()
    basis2_state_dict = collections.OrderedDict()

    i=0
    for (k, _) in anchor.state_dict().items():
        if k not in buffers:
            basis1_state_dict[k] = basis1[i]
            basis2_state_dict[k] = basis2[i]
            i += 1
    
    return basis1_state_dict, basis2_state_dict, removed_comp_len, swapped_basis, perp_scaling_factor

def get_basis(model: Union[SimplexNet, BasicSimplex], 
              anchor: int=0, 
              base1: int=1, 
              base2: int=2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs Gram-Schmidt normalization to obtain basis vectors to span the plane 
    passing through anchor, base1, and base2 vertices of the simplex stored in model.
    Args:
        model: Any model maintaining a "n_vert" state, and whose parameters() iterator
               yields the each parameter for all vertices.
        anchor: The index of the anchor vertex. w_1 in paper.
        base1:  The index of the first base vertex. w_2 in paper.
        base2:  The index of the second base vertex. w_3 in paper. 
    Computes:
        dir1 = w_2 - w_1
        dir2 = w_3 - w_1
        component of dir2 along dir1 = dir1 * <dir1, dir2> / ||dir1||^2
        dir2 = dir2 - component of dir2 along dir1
        Normalize dir1 and dir2 to unit length.
    Returns:
        Return two torch.Tensor, dir1 and dir2, of shape (n_par, 1) where n_par is the
        number of parameters in a single vertex of simplex. dir1 and dir2 are perpendicular
        and of unit length, can server as basis vectors for a plane passing through w_1, w_2, w_3.
    """
    n_vert = model.n_vert
    n_par = int(sum([p.numel() for p in model.parameters()])/n_vert)
    
    if n_vert <= 2:
        warnings.warn("Not enough vertices in simplex to span a plane, returning default random basis vectors.")
        return torch.randn(n_par, 1), torch.randn(n_par, 1)
    else:
        par_vecs = torch.zeros(n_vert, n_par)
        if torch.has_cuda:
            par_vecs = par_vecs.to(device)
        for ii in range(n_vert):
            temp_pars = [p for p in model.net.parameters()][ii::n_vert]
            par_vecs[ii, :] = utils.flatten(temp_pars)
            
        
        first_pars = torch.cat((n_vert * [par_vecs[anchor, :].unsqueeze(0)]))
        diffs = (par_vecs - first_pars)
        dir1 = diffs[base1, :]
        dir2 = diffs[base2, :]
        
        ## now gram schmidt these guys ##
        vu = dir2.squeeze().dot(dir1.squeeze())
        uu = dir1.squeeze().dot(dir1.squeeze())

        dir2 = dir2 - dir1.mul(vu).div(uu)

        ## normalize ##
        dir1 = dir1.div(dir1.norm())
        dir2 = dir2.div(dir2.norm())

        return dir1.unsqueeze(-1), dir2.unsqueeze(-1)

def compute_loss_surface(model: nn.Module, 
                         train_x: Any, train_y: Any, 
                         v1: torch.Tensor, v2: torch.Tensor,
                         loss: Callable, n_pts: int=50, 
                         range_: np.ndarray=np.array(10.)) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Computes loss(calculated on train_x) surface for the model. The surface axes are
    specified by v1 and v2. Each axis is divided into n_pts points. And loss is calculated for a 
    total of n_ptsXn_pts model weights.
    PRE-CONDITION:
        model.state_dict() must return the parameters of the anchor used for obtaining v1 and v2.
    Args:
        model:   A PyTorch model that accepts train_x in its forward() method.
        train_x: The argument to pass to the forward() method of the model.
        train_y: The argument to pass as the labels in the loss() function.
        loss:    A function that takes in the output of model.forward() and train_y as the arguments
                 and returns the loss that is incurred in by the model on train_x.
        v1, v2:  The basis vectors for the plane.
        n_pts:   The number of points to sample on each axis of the loss surface.
        range_:  Each axis extends from -range_ to range_ in the loss surface.
    
    Returns:
        X,Y,loss_surf: The X coords, Y coords and the losses at those points. All arrays are of shape
                        (n_pts, n_pts).
    
    [DOES STORING AND RE-STORING THE MODEL STATE_DICT MAKE ANY DIFFERENCE?]
    Yes, it does. We perturb in directions of v1 and v2 from start_pars. So once we have perturbed, we 
    need to come back to starting parameters to perturb again with a different linear combination of 
    v1 and v2.
    """
    start_pars = model.state_dict()
    vec_len = torch.linspace(-range_.item(), range_.item(), n_pts)
    ## init loss surface and the vector multipliers ##
    loss_surf = torch.zeros(n_pts, n_pts)
    with torch.no_grad():
        ## loop and get loss at each point ##
        for ii in range(n_pts):
            for jj in range(n_pts):
                perturb = v1.mul(vec_len[ii]) + v2.mul(vec_len[jj])
                # print(perturb.shape)
                perturb = utils.unflatten_like(perturb.t(), model.parameters())
                for i, par in enumerate(model.parameters()):
                    par.data = par.data + perturb[i].to(par.device)

                loss_surf[ii, jj] = loss(model(train_x), train_y)

                model.load_state_dict(start_pars)

    X, Y = np.meshgrid(vec_len, vec_len)
    return X, Y, loss_surf


def compute_loader_loss(model: nn.Module, 
                        loader: Iterable[Tuple[Any, Any]], 
                        loss: Callable, n_batch: int,
                        device = torch.device("cuda:0")) -> torch.Tensor:
    """Sums up the loss of all elements in the loader.
    Args:
        model:   A PyTorch model that accepts input data yielded by loader 
                 in its forward() method.
        loader:  An iterable that returns a tuple of input data, and label.
        loss:    A function that takes in the output of model.forward() and label as the arguments
                 and returns the loss that is incurred by the model on input data.
    Returns:
        Total loss incurred by the model on all elements in the loader.
    """
    total_loss = torch.tensor([0.])
    for i, data in enumerate(loader):
        if i < n_batch:
            x, y = data
            x, y = x.to(device), y.to(device)

            preds = model(x)
            total_loss += loss(preds, y).item()
        else:
            break

    return total_loss

def compute_loss_surface_loader(model: nn.Module, 
                                loader: Iterable[Tuple[Any, Any]], 
                                v1: torch.Tensor, v2: torch.Tensor,
                                loss: Callable=torch.nn.CrossEntropyLoss(),
                                n_batch: int =10, n_pts: int =50, 
                                range_: np.ndarray=np.array(10.),
                                device=torch.device("cuda:0")) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Compute the loss surface for the model, where the loss is calculated as the 
    sum of loss over all elements in the loader.
    
    PRE-CONDITION:
        model.state_dict() must return the parameters of the anchor used for obtaining v1 and v2.
    
    Args:
        model:   A PyTorch model that accepts input data yielded by loader
                 in its forward() method.
        loader:  An iterable that returns a tuple of input data, and label.
        loss:    A function that takes in the output of model.forward() and label as the arguments
                 and returns the loss that is incurred by the model on input data.
        v1, v2:  The basis vectors for the plane of (model-)weights to consider.
        n_pts:   The number of points to sample on each axis of the loss surface.
        range_:  Each axis extends from -range_ to range_ in the loss surface.

    Returns:
        X,Y,loss_surf: The X coords, Y coords and the losses at those points. All arrays are of shape
                        (n_pts, n_pts).
    """
    start_pars = model.state_dict()
    vec_len = torch.linspace(-range_.item(), range_.item(), n_pts)
    ## init loss surface and the vector multipliers ##
    loss_surf = torch.zeros(n_pts, n_pts)
    with torch.no_grad():
        ## loop and get loss at each point ##
        for ii in range(n_pts):
            for jj in range(n_pts):
                perturb = v1.mul(vec_len[ii]) + v2.mul(vec_len[jj])
                # print(perturb.shape)
                perturb = utils.unflatten_like(perturb.t(), model.parameters())
                for i, par in enumerate(model.parameters()):
                    par.data = par.data + perturb[i].to(par.device)
                    
                loss_surf[ii, jj] = compute_loader_loss(model, loader,
                                                        loss, n_batch,
                                                        device=device)

                model.load_state_dict(start_pars)

    X, Y = np.meshgrid(vec_len, vec_len)
    return X, Y, loss_surf