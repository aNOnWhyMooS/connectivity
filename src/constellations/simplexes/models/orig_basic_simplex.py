import abc, ast
import collections
import os
import random
import warnings
import torch, copy
import torch.nn as nn
import math, re

from gpytorch.kernels import Kernel

from typing import FrozenSet, Iterable, List, Set, Tuple, Optional

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def simplex_parameters(module: nn.Module, 
                       params: List[Tuple[nn.Module, str]], 
                       num_vertices: int):
    """Function to be recursively called on all sub-modules of a module.
    Process:
        We do the following steps for each parameter in the module:
            1. Reads the data from an existing parameter(with name <param_name>) of the module.
            2. Removes that parameter from that module's parameters. 
            3. Add a parameter with name <param_name>_<vertex_no> for each vertex of the simplex.
            4. Set the value of all new parameters to be the same as the one read in 1.
            5. Append (module, <param_name>) to the params list. 
    NOTE:
        Using the entries of params list, we can access any parameter of any vertex of any sub-module.
    """
    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            continue
        data = module._parameters[name].data
        module._parameters.pop(name)

        for i in range(num_vertices):
            module.register_parameter(name + "_vertex_" + str(i),
                                      nn.Parameter(data.clone().detach_().requires_grad_()))

        params.append((module, name))

def extract_parameters(module: nn.Module, 
                       params: List[Tuple[nn.Module, str]],):
    """Function to be recursively called on all sub-modules of a module.
    To extract all parameters with their names from a module.
    Process:
        We do the following steps for each parameter in the module:
            1. Doesn't remove that parameter from that module's parameters. 
            2. Append (module, <param_name>) to the params list. 
    NOTE:
        Using the entries of params list, we can access any parameter of any vertex of any sub-module.
    """
    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            continue
        params.append((module, name))

cdist = Kernel().covar_dist

class BaseSimplex(nn.Module, abc.ABC):
    """Base class for maintaining a set of vertices in a model. Main purpose:
    1. Take care of the devices the vertices of the model are on, 
    2. Maintain the num_vertices and basic utility for adding, importing, saving 
    new vertices.
    3. Maintain which vertices are fixed, and provide utility for fixing, un-fixing 
    vertices.
    """

    def __init__(self, base: nn.Module, 
                 num_vertices: int = 2, 
                 fixed_points: List[bool] = [True, False], 
                 max_gpu_verts: int = 1,
                 *args, **kwargs):
        
        super().__init__()

        if len(fixed_points)!=num_vertices:
            raise ValueError(f"Not enough values in fixed_points: {fixed_points} \
                              to decide whether each of the {num_vertices} vertices is fixed or not.")

        self.params = list()
        # self.base = base(*args, **kwargs)
        self.base = base
        self.base.apply(
            lambda module: simplex_parameters(
                module=module, params=self.params, num_vertices=num_vertices
            )
        )
        self.num_vertices = num_vertices
        self.fixed_points = fixed_points
        self._fix_points()
        self.n_vert = num_vertices
        self.shift_sample_to_gpu = False
        self.use_gpu = False
        self.max_gpu_verts = max_gpu_verts

    def _fix_points(self) -> None:
        """To be called after replicating each of the parameters of base,
        self.num_vertices time. This function detaches all the parameters 
        belonging to the vertices for which fixed_points[vertex_no] is True.
        
        [WHEN/WHY IS IT NECESSARY TO DO THIS, THOUGH?]
        """
        for (module, name) in self.params:
            for vertex in range(self.num_vertices):
                module.__getattr__(name + "_vertex_" + str(vertex)).detach_()
                module.__getattr__(name + "_vertex_" + str(vertex)).requires_grad_(not self.fixed_points[vertex])
    
    def parameters(self):
        """Yields all the parameters of the model, vertex wise,
        in the order in which they appear in self.params.
        
        NOTE: This order must be maintained if you want to use [i::n_vert] 
        slices of parameters.
        """
        for module, name in self.params:
            for vertex in range(self.num_vertices):
                yield module.__getattr__(name + "_vertex_" + str(vertex))
        
    def sample(self, coeffs_t) -> None:
        """Weights all the vertices of the simplex by the (sclaing-)weights specified 
        in coeffs_t to obtain a sample from the simplex, and then re-sets all the parameters 
        of self.base that were popped in simplex_parameters() to the weights of the model sampled 
        from the simplex.
        
        NOTE:   
            1. If base.L1._parameters.pop('weight') was done before, a nn.Parameter was removed from base.L1;
               where L1 is say a linear layer in the self.base model. But, when re-setting, we only set 
               base.L1.weight to a torch.Tensor. 

            2. Also, although pop-ing from _parameters, removed 'weight' from the _parameters dict, and the attribute
               base.L1.weight. We only add the attribute base.L1.weight back. And it is not added to _parameters dict.

            3. These new added 'weight' will be passed to F.linear(self.weight, self.bias, input) call within the linear 
               layer, and hence used for computation.
        """
        for (module, name) in self.params:
            new_par = 0.
            for vertex in range(self.num_vertices):
                vert = module.__getattr__(name + "_vertex_" + str(vertex))
                new_par = new_par + vert * coeffs_t[vertex]
            if self.shift_sample_to_gpu and self.use_gpu:
                new_par = new_par.cuda()
            module.__setattr__(name, new_par)

    def forward(self, X, coeffs_t=None):
        """Samples model from the simplex based on coeffs_t and compute its output on X."""
        if coeffs_t is None:
            coeffs_t = self.vertex_weights()

        self.sample(coeffs_t)
        return self.base(X)

    def adjust_device(self):
        if self.num_vertices>self.max_gpu_verts:
            with torch.no_grad():
                for (module, name) in self.params:
                    for vertex in range(self.num_vertices):
                        param_name = name + "_vertex_" + str(vertex)
                        param = module._parameters[param_name]
                        param.data = param.cpu()
                        if param.grad is not None:
                            param.grad.data = param.grad.cpu()
            self.shift_sample_to_gpu=True
    
    def to(self, device: torch.device):
        super().to(device)
        self.use_gpu = "cuda" in str(device.type)
        if self.num_vertices>self.max_gpu_verts:    
            self.adjust_device()
        return self
    
    def cuda(self):
        return self.to(torch.device("cuda:0"))

    def _after_adding_vertex(self, fix_last_vertex: bool):
        self.num_vertices+=1
        if fix_last_vertex and self.num_vertices>1:
            self.fixed_points[-1] = True
        self.fixed_points.append(False)
        self._fix_points()
        self.adjust_device()
    
    def add_vertex(self, fix_last_vertex: bool, coeffs: Optional[List[float]]=None,):
        """Registers new parameters with the modules within self.base for a new vertex.
        The new vertex's parameters are initialized to the centre of the simplex formed by
        the existing vertices. Additionally, self.num_vertices is incremented by 1."""
        new_vertex = self.num_vertices
        
        if coeffs is None:
            coeffs = [1./self.num_vertices]*self.num_vertices
        
        for (module, name) in self.params:
            data = 0.
            for vertex in range(self.num_vertices):
                with torch.no_grad():
                    data += module.__getattr__(name + "_vertex_" + str(vertex))*coeffs[vertex]

            module.register_parameter(name + "_vertex_" + str(new_vertex),
                                      nn.Parameter(data.clone().detach_().requires_grad_()))
        
        self._after_adding_vertex(fix_last_vertex)
    
    def par_vector(self, vertex_no: int) -> torch.Tensor:
        """Returns a PyTorch tensor of shape (n_pars,) where n_pars is the total
        number of parameters in a vertex. This tensor contains (model-)weights of vertex_no 
        model in self.base."""
        pars = []
        for (module, name) in self.params:
            val = module.__getattr__(name + "_vertex_" + str(vertex_no))
            pars.append(val)
        return flatten(pars)

    def par_vectors(self) -> torch.Tensor:
        """Returns a PyTorch tensor of shape (num_vertices, n_pars) where n_pars is the total
        number of parameters in a vertex. This tensor contains (model-)weights of all models
        at the vertices of the current simplex in self.base. """
        all_vertices_list = []
        for vertex in range(self.num_vertices):
            all_vertices_list.append(self.par_vector(vertex))
        return torch.stack(all_vertices_list)
    
    def simplex_volume(self, vertices: Iterable[int]) -> torch.Tensor:
        """Returns the volume of simplex formed by the vertices specified."""
        n_vert = len(vertices)
        mat = torch.ones(n_vert+1, n_vert+1) - torch.eye(n_vert + 1)
        par_vecs = torch.stack([self.par_vector(vertex) for vertex in vertices])
        dist_mat = cdist(par_vecs, par_vecs).pow(2)
        mat[:n_vert, :n_vert] = dist_mat
        norm = (math.factorial(n_vert-1)**2) * (2. ** (n_vert-1))
        simplex_vol = torch.abs(torch.det(mat)).div(norm)
        return simplex_vol
    
    def import_params(self, model: nn.Module, vertex_no: int, fix_last_vertex: bool=False):
        """Imports parameters from model, into a new vertex, if vertex_no==self.num_vertices,
        else, into the vertex specified by vertex_no. The registered parameters are trainable. 
        """
        params= []
        model.apply(
            lambda module: extract_parameters(module, params)
        )
        
        for (from_module, from_name), (to_module, to_name) in zip(params, self.params):
            if from_name!=to_name:
                raise ValueError("Mis-matched parameter names encountered while importing parameters.")
            if type(from_module) != type(to_module):
                raise ValueError("Mis-matched parameter types encountered while importing parameters.")
            
            from_tensor = from_module.__getattr__(from_name)
            sample_to_tensor = to_module.__getattr__(to_name + "_vertex_" + str(vertex_no-1))
            
            if from_tensor.shape!=sample_to_tensor.shape:
                raise ValueError(f"Shape {from_tensor.shape} of parameter {from_name} from {type(from_module)}, \
                                   to be copied doesn't match the destination shape {sample_to_tensor.shape}.")
            
            if vertex_no==self.num_vertices:
                to_module.register_parameter(to_name + "_vertex_" + str(vertex_no),
                                             nn.Parameter(from_tensor.clone().detach_().requires_grad_()))
            else:
                to_module[to_name + "_vertex_" + str(vertex_no)].data = from_tensor.data
        
        if vertex_no==self.num_vertices:
            self._after_adding_vertex(fix_last_vertex)
    
    def save_params(self, save_file_prefix: str) -> None:
        print("Saving parameters...", flush=True)
        params_dicts = [collections.OrderedDict() for _ in range(self.num_vertices)]
        for k, v in self.base.state_dict().items():
            search_vertex = re.search(r"(_vertex_(\d+))$", k)
            if search_vertex is None:
                for i in range(self.num_vertices):
                    params_dicts[i][k] = v
            else:
                vertex_number = int(search_vertex[2])
                original_key = k[:search_vertex.span(1)[0]]
                params_dicts[vertex_number][original_key] = v
        
        for vertex in range(self.num_vertices):
            filename = save_file_prefix + ("0" if vertex<10 else "") + str(vertex) + ".pt"
            torch.save(params_dicts[vertex], filename)
            print(f"Saved Parameters for vertex {vertex} in {filename}", flush=True)

    def load_params(self, base_model: nn.Module, save_file_prefix: str,
                    fix_points: Optional[List[bool]]=None,
                    verts_to_load: Iterable[int]=None,) -> None:
        """
        Args:
            base_model:       An instance of the base model used for any vertex of the simplex.
                              This is used to provide an interface between the state_dict and (module, name)
                              structure maintained by the simplex.
            save_file_prefix: The save_file_prefix sent to self.save_params() while saving the params.
            fix_points:       A list of bools, where a true at i-th index corresponds to the (i+1)-th 
                              loaded vertex being fixed. By default all vertices are trainable.
            verts_to_load:    A list of vertices to load. By default all vertices are loaded.
        NOTE:
            1. Works fine even if existing vertices are there in the simplex.
        """
        print("Loading parameters...", flush=True)
        if verts_to_load is None:
            verts_to_load = []
            i=0
            while os.path.isfile(save_file_prefix + ("0" if i<10 else "") + str(i) + ".pt"):
                verts_to_load.append(i)
        
        for vertex in verts_to_load:
            filename = save_file_prefix + ("0" if vertex<10 else "") + str(vertex) + ".pt"
            params_dict = torch.load(filename)
            base_model.load_state_dict(params_dict)
            self.import_params(base_model, vertex)
            print(f"Loaded Parameters for vertex {vertex} from {filename}", flush=True)
        
        if fix_points is not None:
            if len(fix_points) != len(verts_to_load):
                raise ValueError(f"{fix_points} must be of length {len(verts_to_load)}.")
            self.fixed_points = fix_points
            self._fix_points()
        
class BasicSimplex(BaseSimplex):
    
    def __init__(self, base: nn.Module, 
                 num_vertices: int = 1, 
                 fixed_points: List[bool] = [True], 
                 *args, **kwargs):
        super().__init__(base, num_vertices, fixed_points, *args, **kwargs)
    
    def vertex_weights(self)-> torch.Tensor:
        """Samples random numbers from exponential distribution and normalizes 
        them to provide (sclaing-)weights for vertices of the simplex."""
        exps = -torch.rand(self.num_vertices).log()
        return exps / exps.sum()

    def total_volume(self) -> torch.Tensor:
        """Computes the volume of the simplex specified by the parameters of self.base. 
        See https://en.wikipedia.org/wiki/Cayley%E2%80%93Menger_determinant for 
        further description of how volume is calculated.
        """
        n_vert = self.num_vertices

        dist_mat = 0.
        for (module, name) in self.params:
            all_vertices = [] #* self.num_vertices
            for vertex in range(self.num_vertices):
                par = module.__getattr__(name + "_vertex_" + str(vertex))
                all_vertices.append(flatten(par))
            par_vecs = torch.stack(all_vertices)
            dist_mat = dist_mat + cdist(par_vecs, par_vecs).pow(2)

        mat = torch.ones(n_vert+1, n_vert+1) - torch.eye(n_vert + 1)
        # dist_mat = cdist(par_vecs, par_vecs).pow(2)
        mat[:n_vert, :n_vert] = dist_mat

        norm = (math.factorial(n_vert-1)**2) * (2. ** (n_vert-1))
        return torch.abs(torch.det(mat)).div(norm)

    def add_vert(self, fix_last_vertex: bool = False):
        return self.add_vertex(fix_last_vertex)
        
class SimplicialComplex(BaseSimplex):
    def __init__(self, base: nn.Module, 
                 num_vertices: int = 1, 
                 fixed_points: List[bool] = [True], 
                 *args, **kwargs):
        """To design a simplicial complex. 
        NOTE:
            All vertices are divided among two categories:
                - base vertices, which are pre-defined pre-trained
                models that are used to initialize each simplex in 
                the complex.
                - connecting vertices, which are trained/'adjusted'
                as a part of the complex, and are essentially used to 
                form the simplices.
        STATE-MAINTAINED:
            adj_simplicial_complex: Adjacency matrix of the simplicial complex.
            _pc_connected_comps:    A list of connected components of the simplicial complex. 
                                    Re-computed in self._connected_comps(), whenever new vertex
                                    is added.
        """
        super().__init__(base, num_vertices, fixed_points, *args, **kwargs)
                
        self.adj_simplicial_complex = {i : [0]*(i)+[1]+[0]*(num_vertices-1-i)
                                       for i in range(num_vertices) }
        # ^ each simplex in the simplicial complex is meant to be stored as a  
        # bit mask over all the vertices, including base vertices and connectors.

    def vertex_weights(self)-> torch.Tensor:
        """Picks a random simplex from the ones specified from simplicial complex
        and samples a random model from this simplex.
        Process:
            The model is sampled by specifying random (scaling-)weights for each of the vertices of 
            the selected simplex. Random numbers are sampled from an exponential distribution with 
            mean parameter(λ) 1 and normalized to obtain the (scaling-)weights.
        Returns:
            The list of (scaling-)weights is returned.
        
        NOTE: This function assumes that there exists a bijection between each fixed vertex and 
        the simplexes in the simplicial complex.
        """
        ## pick a simplex to sample from ##
        simplex = random.choice(list(self._connected_comps()))
        ## assign a weight to each vertex in this simplex ##
        vert_weights = [0] * self.num_vertices
        for vertex in simplex:
            vert_weights[vertex] = -(torch.rand(1)).log().item()
        total = sum(vert_weights)
        vert_weights = [exp/total for exp in vert_weights]
        return vert_weights
    
    def add_vert(self):
        raise NotImplementedError("Use add_base_vertex() or add_conn_vertex() instead.")
    
    def add_base_vertex(self, model: nn.Module, fix_last_vertex: bool = False):
        """Initializes a new simplex within the similicial complex.
        Registers new parameters with the modules within self.base for a new base vertex.
        A given model is initialized as the base model of a new simplex which is created with
        this function. Additionally, self.num_vertices is incremented by 1.
        Args:
            model:    the model which needs to be the base vertex of the new simplex.
        """
        new_vertex = self.num_vertices
        
        for i in range(self.num_vertices):
            self.adj_simplicial_complex[i].append(not self.fixed_points[i])
        
        self.adj_simplicial_complex[new_vertex] = [0]*self.num_vertices + [1]

        self.import_params(model, new_vertex, fix_last_vertex)
        
        #Fix Base Vertex
        self.fixed_points[-1] = True
        self._fix_points()

    def add_conn_vertex(self, linking: Optional[List[bool]]=None,
                        fix_last_vertex: bool = False,
                        coeffs: Optional[List[float]]=None):
        """Registers new parameters with the modules within self.base for a new connecting vertex.
        The new vertex's parameters are initialized to the mean of all existing vertices.
        Additionally, self.num_vertices is incremented by 1.
        Args:
            linking:    specifies which of the existing vertices the new vertex will be connected with.
        """
        new_vertex = self.num_vertices
        if not linking:
            linking = [1]*len(self.adj_simplicial_complex)

        for vi in self.adj_simplicial_complex:
            self.adj_simplicial_complex[vi].append(linking[vi])

        self.adj_simplicial_complex[new_vertex] = linking+[1]

        self.add_vertex(fix_last_vertex, coeffs)

    def _connected_comps(self) -> Set[FrozenSet[int]]:
        """Returns Pre-Computed completely connected components of the simplicial complex,
        if num_vertices is same as the number of vertices in the simplicial complex when it 
        was last computed, otherwise computes them again.
        """
        if hasattr(self, "_pc_connected_comps_for_vertices") and hasattr(self, "_pc_connected_comps"):
            if self._pc_connected_comps_for_vertices==self.num_vertices:
                return self._pc_connected_comps
        
        final_sets = set()
        sets = frozenset({frozenset([i]) for i in range(self.num_vertices)})
        
        while len(sets)!=0:
            new_sets = set()
            for sett in sets:
                expanded = False
                for i in range(self.num_vertices):
                    if i in sett:
                        continue
                    for elem in sett:
                        if not self.adj_simplicial_complex[i][elem]:
                            break
                    else:
                        expanded = True
                        new_sets.add(sett.union([i]))
                if not expanded:
                    final_sets.add(sett)
            sets = frozenset(new_sets)
        
        self._pc_connected_comps = final_sets
        self._pc_connected_comps_for_vertices = self.num_vertices
        return final_sets

    def total_volume(self) -> torch.Tensor:
        """Computes the total volume of the simplicial complex.
        See https://en.wikipedia.org/wiki/Cayley%E2%80%93Menger_determinant for 
        further description of how volume is calculated.

        NOTE: This function assumes that there exists a bijection between each fixed vertex and 
        the simplexes in the simplicial complex.
        """
        total_vol = 0
        simplices = self._connected_comps()
        max_dim  = max([len(simplex) for simplex in simplices])
        for simplex in simplices:
            if len(simplex)==max_dim:
                total_vol += self.simplex_volume(simplex)
            else:
                warnings.warn(f"A simplex: {simplex} in the simplicial complex has lower dimension than some other\
                    simplices in the complex. This simplex will not contribute to total volume of simplex, but will be\
                    sampled from. Kindly check adjacency matrix: {self.adj_simplicial_complex}. And the computed\
                    simplices: {simplices}.")
        return total_vol
    
    def save_params(self, save_file_prefix: str) -> None:
        """Saves the parameters and adjacency matrix of the simplicial-complex vertex-by-vertex"""
        super().save_params(save_file_prefix)
        print("Saving adjacency matrix..", flush=True)
        with open(save_file_prefix + "_adjacency.txt", "w") as f:
            for i in range(self.num_vertices):
                f.write(str(self.adj_simplicial_complex[i]) + "\n")
    
    @classmethod
    def from_pretrained(cls, base_model:nn.Module, save_file_prefix: str, 
                        fix_points: Optional[List[bool]]=None) -> "SimplicialComplex":
        """Loads the parameters and adjacency matrix of the simplicial-complex vertex-by-vertex.
        Assumes no existing vertices in the simplicial complex. Reads the vertex numbered 
        state dict files and the adjacency matrix."""
        base_model_cp = copy.deepcopy(base_model)
        complex_model = cls(base_model, num_vertices=0, fixed_points=[])
        super().load_params(base_model_cp, save_file_prefix, fix_points)
        with open(save_file_prefix + "_adjacency.txt", "w") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            complex_model.adj_simplicial_complex[i] = ast.literal_eval(line.strip())
        return complex_model