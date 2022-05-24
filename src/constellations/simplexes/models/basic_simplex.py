import torch
import math

from gpytorch.kernels import Kernel
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def simplex_parameters(module, params, num_vertices):
    state_dict = copy.deepcopy(module.state_dict())
    for name in list(state_dict.keys()):
        # print(name, end=' ')
        if state_dict[name] is None:
            continue
        data = state_dict[name].data
        state_dict.pop(name)

        # name = '*'.join(name.split('.'))
        for i in range(num_vertices):
            state_dict[name + "_vertex_" + str(i)] = data.detach_().clone().float().requires_grad_()
            # module.register_parameter(name + "_vertex_" + str(i),
            #                            torch.nn.Parameter(data.clone().detach_().requires_grad_()))
        
        params.append((state_dict, name))
        # params.append((module, name))


cdist = Kernel().covar_dist

class BasicSimplex(torch.nn.Module):
    def __init__(self, base, num_vertices = 2, fixed_points = [True, False], *args, **kwargs):
        super().__init__()
        self.params = list()
        # self.base = base(*args, **kwargs)
        self.base = base.to(device)
        # self.base.apply(
        #     lambda module: simplex_parameters(
        #         module=module, params=self.params, num_vertices=num_vertices
        #     )
        # )
        simplex_parameters(self.base, params=self.params, num_vertices=num_vertices)
        self.num_vertices = num_vertices
        self._fix_points(fixed_points)
        self.n_vert = num_vertices

    def _fix_points(self, fixed_points):
        for (module, name) in self.params:
            for vertex in range(self.num_vertices):
                if fixed_points[vertex]:
                    module[name + "_vertex_" + str(vertex)] = module[name + "_vertex_" + str(vertex)].detach_()
                    # module.__getattr__(name + "_vertex_" + str(vertex)).detach_()

    def sample(self, coeffs_t):
        base_s_dir = self.base.state_dict()
        # print(self.base.state_dict().keys())
        # print(base_s_dir.keys())
        for (module, name) in self.params:
            new_par = 0.
            for vertex in range(self.num_vertices):
                vert = module[name + "_vertex_" + str(vertex)]
                # vert = module.__getattr__(name + "_vertex_" + str(vertex))
                new_par = new_par + vert * coeffs_t[vertex]
            # name = '.'.join(name.split('*'))
            base_s_dir[name] = new_par
        self.base.load_state_dict(base_s_dir)

    def vertex_weights(self):
        exps = -torch.rand(self.num_vertices).log()
        return exps / exps.sum()

    def forward(self, X_ids, X_attn_mask, X_tok_ids=None, target=None, coeffs_t=None):
        if coeffs_t is None:
            coeffs_t = self.vertex_weights()

        self.sample(coeffs_t)
        if target != None:
            if X_tok_ids != None:
                return self.base(input_ids=X_ids, 
                                attention_mask=X_attn_mask, 
                                token_type_ids=X_tok_ids,
                                labels=target)
            else:
                return self.base(input_ids=X_ids, 
                                attention_mask=X_attn_mask, 
                                labels=target)
        else:
            if X_tok_ids != None:
                return self.base(input_ids=X_ids, 
                                attention_mask=X_attn_mask, 
                                token_type_ids=X_tok_ids)
            else:
                return self.base(input_ids=X_ids, 
                                attention_mask=X_attn_mask)

    def add_vert(self):
        return self.add_vertex()

    def add_vertex(self):
        new_vertex = self.num_vertices

        for (module, name) in self.params:
            data = 0.
            for vertex in range(self.num_vertices):
                with torch.no_grad():
                    data += module[name + "_vertex_" + str(vertex)]
                    # data += module.__getattr__(name + "_vertex_" + str(vertex))
            data = data / self.num_vertices
            data = data / ((0.8-1.2) * torch.rand(1) + 1.2).item()      # to add some noise to the new vertex weights
            data = data.detach_().clone().float().requires_grad_()
            # with torch.no_grad():
            #     data.add_((abs(torch.mean(data))**0.5)*torch.randn(*tuple(data.shape)).float().to(device))
            module[name + "_vertex_" + str(new_vertex)] = data
            # module.register_parameter(name + "_vertex_" + str(new_vertex),
            #                           torch.nn.Parameter(data.clone().detach_().requires_grad_()))
            
        self.num_vertices += 1

    def total_volume(self):
        n_vert = self.num_vertices

        dist_mat = 0.
        for (module, name) in self.params:
            all_vertices = [] #* self.num_vertices
            for vertex in range(self.num_vertices):
                par = module[name + "_vertex_" + str(vertex)]
                # par = module.__getattr__(name + "_vertex_" + str(vertex))
                all_vertices.append(flatten(par))
            par_vecs = torch.stack(all_vertices)
            dist_mat = dist_mat + cdist(par_vecs, par_vecs).pow(2)

        mat = torch.ones(n_vert+1, n_vert+1) - torch.eye(n_vert + 1)
        # dist_mat = cdist(par_vecs, par_vecs).pow(2)
        mat[:n_vert, :n_vert] = dist_mat

        norm = (math.factorial(n_vert-1)**2) * (2. ** (n_vert-1))
        return torch.abs(torch.det(mat)).div(norm)
    
    def par_vectors(self):
        all_vertices_list = []
        for vertex in range(self.num_vertices):
            vertex_list = []
            for (module, name) in self.params:
                val = module[name + "_vertex_" + str(vertex)].detach()
                # val = module.__getattr__(name + "_vertex_" + str(vertex)).detach()
                vertex_list.append(val)
            all_vertices_list.append(flatten(vertex_list))
        return torch.stack(all_vertices_list)


class PreDefSimplex(torch.nn.Module):
    def __init__(self, base, num_vertices = 1, fixed_points = [True], *args, **kwargs):
        super().__init__()
        self.params = list()
        # self.base = base(*args, **kwargs)
        self.base = base
        # self.base.apply(
        #    lambda module: simplex_parameters(
        #        module=module, params=self.params, num_vertices=num_vertices
        #    )
        # )
        simplex_parameters(self.base, params=self.params, num_vertices=num_vertices)
        self.num_vertices = num_vertices
        self._fix_points(fixed_points)
        self.n_vert = num_vertices

    def _fix_points(self, fixed_points):
        for (module, name) in self.params:
            for vertex in range(self.num_vertices):
                if fixed_points[vertex]:
                    module[name + "_vertex_" + str(vertex)] = module[name + "_vertex_" + str(vertex)].detach_()
                    # module.__getattr__(name + "_vertex_" + str(vertex)).detach_()

    def sample(self, coeffs_t):
        base_s_dir = self.base.state_dict()
        # print(self.base.state_dict().keys())
        # print(base_s_dir.keys())
        for (module, name) in self.params:
            new_par = torch.tensor([0.]).to(device)
            for vertex in range(self.num_vertices):
                vert = module[name + "_vertex_" + str(vertex)].to(device)
                # vert = module.__getattr__(name + "_vertex_" + str(vertex))
                new_par = new_par + vert * coeffs_t[vertex]
            # name = '.'.join(name.split('*'))
            base_s_dir[name] = new_par
        self.base.load_state_dict(base_s_dir)

    def vertex_weights(self):
        exps = -torch.rand(self.num_vertices).log()
        return exps / exps.sum()

    def forward(self, X_ids, X_attn_mask, X_tok_ids=None, target=None, coeffs_t=None):
        if coeffs_t is None:
            coeffs_t = self.vertex_weights()

        self.sample(coeffs_t)
        if target != None:
            if X_tok_ids != None:
                return self.base(input_ids=X_ids, 
                                attention_mask=X_attn_mask, 
                                token_type_ids=X_tok_ids,
                                labels=target)
            else:
                return self.base(input_ids=X_ids, 
                                attention_mask=X_attn_mask, 
                                labels=target)
        else:
            if X_tok_ids != None:
                return self.base(input_ids=X_ids, 
                                attention_mask=X_attn_mask, 
                                token_type_ids=X_tok_ids)
            else:
                return self.base(input_ids=X_ids, 
                                attention_mask=X_attn_mask)

    def add_vert(self, model):
        return self.add_vertex(model)

    def add_vertex(self, model):
        new_vertex = self.num_vertices

        for (module, name) in self.params:
            data = model.state_dict()[name].data
            
            module[name + "_vertex_" + str(new_vertex)] = data.clone().detach_().float().requires_grad_()
            # module.register_parameter(name + "_vertex_" + str(new_vertex),
            #                           torch.nn.Parameter(data.clone().detach_().requires_grad_()))
        self.num_vertices += 1

    def total_volume(self):
        n_vert = self.num_vertices

        dist_mat = 0.
        for (module, name) in self.params:
            all_vertices = [] #* self.num_vertices
            for vertex in range(self.num_vertices):
                par = module[name + "_vertex_" + str(vertex)].to(device)
                # par = module.__getattr__(name + "_vertex_" + str(vertex))
                all_vertices.append(flatten(par))
            par_vecs = torch.stack(all_vertices)
            dist_mat = dist_mat + cdist(par_vecs, par_vecs).pow(2)

        mat = torch.ones(n_vert+1, n_vert+1) - torch.eye(n_vert + 1)
        # dist_mat = cdist(par_vecs, par_vecs).pow(2)
        mat[:n_vert, :n_vert] = dist_mat

        norm = (math.factorial(n_vert-1)**2) * (2. ** (n_vert-1))
        return torch.abs(torch.det(mat)).div(norm)
    
    def par_vectors(self):
        all_vertices_list = []
        for vertex in range(self.num_vertices):
            vertex_list = []
            for (module, name) in self.params:
                val = module[name + "_vertex_" + str(vertex)].detach()
                # val = module.__getattr__(name + "_vertex_" + str(vertex)).detach()
                vertex_list.append(val)
            all_vertices_list.append(flatten(vertex_list))
        return torch.stack(all_vertices_list)
