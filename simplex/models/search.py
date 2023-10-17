import collections
from typing import Callable, List, Optional, Tuple, Union
import warnings

import zarr
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Function

import orig_utils as utils

def params_to_tensors(module: nn.Module,
                      params: List[Tuple[nn.Module, str]],) -> None:
    """Function to be called recursively on a module to convert all parameters
    to normal tensors.
    NOTE: 1. The new tensors have the same values as the old parameters.
          2. You can access all the added tensors using the entries in params list.
    """
    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            continue
        data = module._parameters[name].data.detach_()
        module._parameters.pop(name)
        setattr(module, name, data)
        params.append((module, name))

class LimitToSimplex(Function):
    """A function that limits the :
    
    1. parameters to stay in the simplex specified by 
    {Σx_i=1: x_i ∈ [0, 1]}.} in the forward pass. And,
    
    2. gradients to stay in the plane specified by
    Σx_i =1 in the backward pass.
    """
    @staticmethod
    def forward(ctx, x):
        with torch.no_grad():
            x.clamp_(0, 1)
            x.div_(torch.sum(x))
        return x
    
    @staticmethod
    def backward(ctx, grad):
        """Take component of gradient along the plane Σx_i =1."""
        perp_vec = torch.ones_like(grad)
        squared_norm = perp_vec.dot(perp_vec)
        grad_perp = grad.dot(perp_vec).div(squared_norm)*perp_vec
        final_grad = grad-grad_perp
        grad_sum = torch.sum(final_grad)
        
        if torch.is_nonzero(grad_sum):
            per_term_diff = -grad_sum/torch.numel(final_grad)
            warnings.warn(f"Gradients {final_grad} don't sum to 0. Making them sum to 0, by adding \
                            {per_term_diff} to each gradient.")
            final_grad = final_grad + per_term_diff
        
        return final_grad


limit_to_simplex = LimitToSimplex.apply

class EffLinearComb(Function):
    """Efficiently computes the forward() and backward() of a linear combination.
    Features:
        1. Only one model weights vector loaded into GPU, at a time.
        2. Only the gradients of the weights vector are calculated.
    """
    @staticmethod
    def forward(ctx, wts: torch.Tensor, elems: zarr.Array) -> torch.Tensor:
        """
        Args:
            wts:    weights of the linear combination. Preferrably on GPU.
            elems:  Array of elements to be combined. elems.shape[0] must be
                    equal to wts.shape[0].
        Returns:
            The linear combination of the elements, on same device as wts.
        """
        ctx.save_for_backward(wts)
        ctx.elems = elems
        out = torch.zeros(elems[0].shape, device=wts.device,)
        for i, elem in enumerate(elems):
            out += wts[i] * torch.tensor(elem[:], device=wts.device)
        return out
    
    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, None]:
        wts = ctx.saved_tensors[0]
        grad_wts = torch.zeros_like(wts)
        for i, elem in enumerate(ctx.elems):
            grad_wts[i] = grad_out.dot(torch.tensor(elem[:], device=wts.device))
        return grad_wts, None

eff_linear_comb = EffLinearComb.apply

class SimplexOptimizedModel(nn.Module):
    """Wrapper for a model which is to be optimized within a simplex."""
    def __init__(self, model: nn.Module, 
                 models_per_chunk: int=3, 
                 paths: Optional[List[str]]=[]):
        """Initializes the parameters of a SimplexOptimizedModel.
        Args:
            model:            A nn.Module instance that can be used to load weights in the simplex,
                              and run forward and backward passes.
            models_per_chunk: Number of models that will be loaded into CPU at a time. 
            paths:            A list of paths to the state dicts of the model.
        NOTE: The weights of the model passed are not used anywhere.
        """
        super().__init__()
        self.num_vertices = 0
        
        self.all_model_keys = [k for k in model.state_dict().keys()]
        self.orig_buffer_keys = [name for name, _ in model.named_buffers()]
        self.orig_param_keys = [name for name, _ in model.named_parameters()]
        self.vertex_model_shapes = [v.shape for k, v in model.state_dict().items() if k not in self.orig_buffer_keys]
        
        self.params = []
        self.model = model
        self.model.apply(
            lambda module: params_to_tensors(module, self.params)
        )
        
        self.models_per_chunk = models_per_chunk
        self.vertex_scaling_weights = nn.Parameter()
        for path in paths:
            self.add_vertex(path)
        
        if len(paths)>0:
            self.init_params()

    def init_params(self) -> None:
        """Initialize parameters to mean of all vertices."""
        with torch.no_grad():
            self.vertex_scaling_weights = nn.Parameter(data=torch.tensor([1/self.num_vertices]*self.num_vertices))
        
        warnings.warn("Re-initialized parameters of model, make sure to update parameters passed to optimizer etc.")

    def _get_path_params(self, path: str) -> np.ndarray:
        """Returns the parameters of a model as a flat np.ndarray, loaded
        from the state dict in the location specified by path.Checks for matching keys 
        and shapes of parameters.
        """
        state_dict = torch.load(path, map_location=torch.device('cpu'))    
        
        params_lis = []
        
        for i, (k, v) in enumerate(state_dict.items()):
            if k in self.orig_buffer_keys:
                continue

            if k != self.orig_param_keys[len(params_lis)]:
                raise ValueError(f"Key number {i+1}: {k} in state dict for vertex {self.num_vertices} does not\
                    match the key number {len(params_lis)} : {self.orig_param_keys[i]} in the state dict for vertex 1.")
            
            if v.shape != self.vertex_model_shapes[len(params_lis)]:
                raise ValueError(f"Shape {v.shape} of parameter {k} in vertex {self.num_vertices} does not \
                    match the shape {self.vertex_model_shapes[len(params_lis)]} of same parameter in vertex 1.")
            
            params_lis.append(v)
        
        return utils.flatten(params_lis).detach().numpy()
    
    
    def _get_model_params(self, model: nn.Module) -> np.ndarray:
        """Returns the parameters of a model as a flat np.ndarray, loaded
        from the parameters of the specified model. Checks for matching keys 
        and shapes of parameters.
        """

        params_lis = []
        for i, (k, v) in enumerate(model.named_parameters()):
            if k != self.orig_param_keys[i]:
                raise ValueError(f"Key number {i+1}: {k} in state dict for vertex {self.num_vertices} does not\
                    match the key {self.orig_param_keys[i]} in the state dict for vertex 1.")
            
            if v.shape != self.vertex_model_shapes[i]:
                raise ValueError(f"Shape {v.shape} of parameter {k} in vertex {self.num_vertices} does not \
                    match the shape {self.vertex_model_shapes[i]} of same parameter in vertex 1.")
            
            params_lis.append(v)

        return utils.flatten(params_lis).detach().numpy()
    
    def import_params(self, model: nn.Module, vertex_index: int) -> None:
        """Imports the parameters of a model into the specified vertex."""
        model_params = self._get_model_params(model)
        self.vertex_params[vertex_index] = model_params
    
    def add_vertex(self, path_or_model: Union[nn.Module, str],
                   reinitialize: bool=False) -> None:
        """Adds a vertex to the model.
        Args:
            path_or_model:  Either a path to a state dict of a model, or a model.
            reinitialize:   Whether to reinitialize scaling weights of all the vertices 
                            of the simplex, to 1/n. If False, the scaling weights of earlier
                            vertices are kept as it is and the weight of new vertex is set to 0.
        """
        self.num_vertices += 1
        
        if type(path_or_model) == str:
            model_params = self._get_path_params(path_or_model)
        else:
            model_params = self._get_model_params(path_or_model)
        
        if not hasattr(self, "vertex_params"):
            self.vertex_params = zarr.array([model_params],
                                            chunks=(self.models_per_chunk,
                                                    len(model_params)))
        else:
            self.vertex_params.append(np.expand_dims(model_params, axis=0),
                                      axis=0)
        
        if reinitialize:
            self.init_params()
        else:
            self.vertex_scaling_weights.data = torch.cat([self.vertex_scaling_weights.data, 
                                                          torch.tensor([0.])])

    def set_model_params(self, params: torch.Tensor):
        """Loads the parameters specified in params, into self.model's attributes.
        The device and dtype are same as that of the original value in the attribute.
        """
        new_params = utils.unflatten_like(params.unsqueeze(0), 
                                         [getattr(module, name) for module, name in self.params])
        
        if len(new_params)!=len(self.params):
            raise ValueError(f"Parameters to be loaded({len(new_params)}) do not match the \
                number of parameters({len(self.params)}) in the model.")
        
        for new_param, (module, name) in zip(new_params, self.params):
            old_param = getattr(module, name)
            
            if old_param.shape != new_param.shape:
                raise ValueError(f"Can't load parameters of shape {new_param.shape} into \
                    parameter of shape {old_param.shape}.")
            
            setattr(module, name, new_param.to(old_param.device).type(old_param.dtype))
        return self
    
    def params_to_state_dict(self, params: torch.Tensor) -> collections.OrderedDict :
        """Converts the params to a state dict. The state dict contains parameters as well as 
        buffers. The buffer values are recovered from the ones stored in the original buffers of 
        self.model, at the time of call to this function.
        """
        unflattened_params = utils.unflatten_like(params.unsqueeze(0), 
                                                  [getattr(module, name) for module, name in self.params])
        
        if len(unflattened_params)!=len(self.params):
            raise ValueError(f"Parameters to be converted({len(unflattened_params)}) to state dict do not match the \
                number of parameters({len(self.params)}) in the model.")
        
        state_dict = collections.OrderedDict()
        
        i=0
        for k in self.all_model_keys:
            if k in self.orig_param_keys:
                state_dict[k] = unflattened_params[i]
                if unflattened_params[i].shape!=self.vertex_model_shapes[i]:
                    raise ValueError(f"Shape({unflattened_params[i].shape}) of parameter {k}: does not match \
                        the required shape({self.vertex_model_shapes[i]}).")
                i+=1
            else:
                state_dict[k] = self.model.state_dict()[k]
        
        return state_dict
    
    def _apply(self, fn: Callable):
        """Re-define nn.Module._apply() to apply fn to the parameters of self.model
        that have been converted to simple tensors. This will ensure that these tensors are also
        considered in .cuda(), .to(device), .float() etc. calls.
        """
        with torch.no_grad():
            for module, name in self.params:
                setattr(module, name, fn(getattr(module, name)))
        super()._apply(fn)
        return self            
    
    def forward(self, x):
        """Linear-ly combines weights of the vertices, loads those weights into the model, and 
        runs the forward pass of self.model() on the input x.
        """
        vertex_scaling_weights = limit_to_simplex(self.vertex_scaling_weights)
        params = eff_linear_comb(vertex_scaling_weights, self.vertex_params)
        self.set_model_params(params)
        return self.model(x)
    
    def save_params(self, path: str) -> None:
        """Saves the parameters of the optimized model, as a state dict, into a file."""
        
        print("Vertex Scaling weights: ", self.vertex_scaling_weights.data.cpu().numpy(), flush=True)
        print("Saving parameters..", flush=True)
        with torch.no_grad():
            params = eff_linear_comb(self.vertex_scaling_weights, self.vertex_params).cpu()
        torch.save(self.params_to_state_dict(params), path)
        print(f"Saved all parameters in: {path}!", flush=True)
    
    def total_volume(self):
        return torch.tensor(1e-4)
