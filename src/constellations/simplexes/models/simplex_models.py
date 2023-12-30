import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch.nn.modules.utils import _pair
from scipy.special import binom
import sys

from .. import utils
from ..simplex_helpers import complex_volume

from typing import Any, Dict, Optional, Type, List, Tuple, Iterable


class SimplicialComplex(Module):
    def __init__(self, n_simplex):
        super(SimplicialComplex, self).__init__()
        self.n_simplex = n_simplex

    def forward(self, complex_model) -> List[float]:
        """Samples a random simplex from the complex model, the probability of
        sampling a given simplex being proportional to its volume. Then, Samples a
        (scaling-)weight from exponential distribution, for each vertex of this simplex.
        The (scaling-)weights are normalized and a list of (scaling-)weights is returned.

        Args:
            complex_model:   A SimplexNet instance. See below for SimplexNet defintion.

        Returns:
            A list of (sclaing-)weights for the vertices of the simplex at index simp_ind
            in complex_model.simplexes.

        NOTE:
            1. Not sure where it is used. Probably should change simplexes to simplicial_complex
               to use with SimplexNet.
        """
        ## first need to pick a simplex to sample from ##
        vols = []
        n_verts = []
        for ii in range(self.n_simplex):
            vols.append(complex_volume(complex_model, ii))
            n_verts.append(len(complex_model.simplexes[ii]))

        norm = sum(vols)
        vol_cumsum = np.cumsum([vv / norm for vv in vols])
        simp_ind = np.min(np.where(np.random.rand(1) < vol_cumsum)[0])

        ## sample (scaling-)weights for simplex
        exps = [-(torch.rand(1)).log().item() for _ in range(n_verts[simp_ind])]
        total = sum(exps)
        exps = [exp / total for exp in exps]

        ## now assign vertex weights out
        vert_weights = [0] * complex_model.n_vert
        for ii, vert in enumerate(complex_model.simplexes[simp_ind]):
            vert_weights[vert] = exps[ii]

        return vert_weights


class Simplex(Module):
    def __init__(self, n_vert):
        super(Simplex, self).__init__()
        self.n_vert = n_vert
        self.register_buffer("range", torch.arange(0, float(n_vert)))

    def forward(self, t) -> List[float]:
        """Samples a (scaling-)weight from exponential distribution, for each vertex.
        The (scaling-)weights are normalized and a list of (scaling-)weights is returned.
        """
        exps = [-torch.log(torch.rand(1)).item() for _ in range(self.n_vert)]
        total = sum(exps)

        return [exp / total for exp in exps]


class PolyChain(Module):
    def __init__(self, num_bends):
        """
        Args:
            num_bends:  Number of bends in the poly-chain, including end vertices[TO CONFIRM].
        """
        super(PolyChain, self).__init__()
        self.num_bends = num_bends
        self.register_buffer("range", torch.arange(0, float(num_bends)))

    def forward(self, t: float) -> torch.Tensor:
        """Re-scales (t)raversal-parameter of poly-chain to traversal parameter
        for a segment of poly-chain.
        Args:
            t:  float in [0, 1]; the parameter for traversing along polyChain.
        Returns:
            tensor of shape (num_bends,).

            Note: If the point corresponding to t, lies between i-th and j-th bend the
            i-th position in tensor will encode distance from i-th bend to the point.
            Similarly for j-th position. All other positions will be zero.

            For e.g.: if t=0.1 and num_bends=10. This t lies in the first segment and
            output will be torch.Tensor([0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).
        """
        t_n = t * (self.num_bends - 1)
        return torch.max(self.range.new([0.0]), 1.0 - torch.abs(t_n - self.range))


class SimplexModule(Module):
    """Super-class for various layers like Linear, Conv2d etc. defined below.
    Each of the sub-classes will maintain a simplex(equivalently, the vertices
    of the simplex) of models, rather than just a single model."""

    def __init__(self, fix_points: List[bool], parameter_names: Tuple[str] = ()):
        """
        Args:
            fix_points:       List of booleans, one for each vertex, denoting whether that
                              vertex's (model-)weights are trainable of fixed.
            parameter_names:  Tuple of strings, corresponding to names of various parameters
                              in the layer class sub-classing SimplexModule class.
        """
        super(SimplexModule, self).__init__()
        self.fix_points = fix_points
        self.num_bends = len(self.fix_points)
        self.parameter_names = parameter_names
        self.l2 = 0.0

    def compute_weights_t(self, coeffs_t: Iterable[float]) -> List[torch.Tensor]:
        """Scales parameters of each vertex of a simplex of models, by the provided
        coefficients, to draw a sample model from the simplex.

        PRE-CONDITION:  parameter names of the layer class sub-classing SimplexModule class
                        must be specified in the format <parameter_name>_<vertex_index>.

        Args:
            coeffs_t: An iterable of floats, equal in length to the number of vertices of the
                      simplex maintained by the sub-class.

        Side-Effects:
            self.l2:  Stores the L2 norm of the (model-)weights of the sampled model.

        Returns:
            A list of tensor, each of which specifies the value for a parameter of the layer class
            sub-classing SimplexModule class.
        """
        w_t = [None] * len(self.parameter_names)
        self.l2 = 0.0
        for i, parameter_name in enumerate(self.parameter_names):
            for j, coeff in enumerate(coeffs_t):
                parameter = getattr(self, "%s_%d" % (parameter_name, j))
                if parameter is not None:
                    if w_t[i] is None:
                        w_t[i] = parameter * coeff
                    else:
                        w_t[i] += parameter * coeff
            if w_t[i] is not None:
                self.l2 += torch.sum(w_t[i] ** 2)
        return w_t


class Linear(SimplexModule):
    """A class implementing functionality for a simplex of Linear layers."""

    def __init__(
        self, in_features: int, out_features: int, fix_points: List[bool], bias=True
    ):
        super(Linear, self).__init__(fix_points, ("weight", "bias"))
        self.in_features = in_features
        self.out_features = out_features

        self.l2 = 0.0

        # Register (model-)weights and biases for each vertex of simplex
        for i, fixed in enumerate(self.fix_points):
            self.register_parameter(
                "weight_%d" % i,
                Parameter(
                    torch.Tensor(out_features, in_features), requires_grad=not fixed
                ),
            )
        for i, fixed in enumerate(self.fix_points):
            if bias:
                self.register_parameter(
                    "bias_%d" % i,
                    Parameter(torch.Tensor(out_features), requires_grad=not fixed),
                )
            else:
                self.register_parameter("bias_%d" % i, None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes all parameters of all vertices of simplex, with uniform
        distribution between -stdv to stdv."""
        stdv = 1.0 / math.sqrt(self.in_features)
        for i in range(self.num_bends):
            getattr(self, "weight_%d" % i).data.uniform_(-stdv, stdv)
            bias = getattr(self, "bias_%d" % i)
            if bias is not None:
                bias.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor, coeffs_t: Iterable[float]) -> torch.Tensor:
        """Samples a linear layer model from the simplex, as specified by
        coeffs_t and computes the output of the model on the specified input.
        Args:
            inputs:   Input for the linear layer. Last dimension must be equal to in_features.
            coeffs_t: An iterable of floats, equal in length to the number of vertices of the
                      simplex maintained by the sub-class.
        """
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        return F.linear(input, weight_t, bias_t)


class Conv2d(SimplexModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        fix_points,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv2d, self).__init__(fix_points, ("weight", "bias"))
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        for i, fixed in enumerate(self.fix_points):
            self.register_parameter(
                "weight_%d" % i,
                Parameter(
                    torch.Tensor(out_channels, in_channels // groups, *kernel_size),
                    requires_grad=not fixed,
                ),
            )
        for i, fixed in enumerate(self.fix_points):
            if bias:
                self.register_parameter(
                    "bias_%d" % i,
                    Parameter(torch.Tensor(out_channels), requires_grad=not fixed),
                )
            else:
                self.register_parameter("bias_%d" % i, None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        for i in range(self.num_bends):
            getattr(self, "weight_%d" % i).data.uniform_(-stdv, stdv)
            bias = getattr(self, "bias_%d" % i)
            if bias is not None:
                bias.data.uniform_(-stdv, stdv)

    def forward(self, input, coeffs_t):
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        return F.conv2d(
            input,
            weight_t,
            bias_t,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class _BatchNorm(SimplexModule):
    _version = 2

    def __init__(
        self,
        num_features,
        fix_points,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(_BatchNorm, self).__init__(fix_points, ("weight", "bias"))
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.l2 = 0.0
        for i, fixed in enumerate(self.fix_points):
            if self.affine:
                self.register_parameter(
                    "weight_%d" % i,
                    Parameter(torch.Tensor(num_features), requires_grad=not fixed),
                )
            else:
                self.register_parameter("weight_%d" % i, None)
        for i, fixed in enumerate(self.fix_points):
            if self.affine:
                self.register_parameter(
                    "bias_%d" % i,
                    Parameter(torch.Tensor(num_features), requires_grad=not fixed),
                )
            else:
                self.register_parameter("bias_%d" % i, None)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            for i in range(self.num_bends):
                getattr(self, "weight_%d" % i).data.uniform_()
                getattr(self, "bias_%d" % i).data.zero_()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input, coeffs_t):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            weight_t,
            bias_t,
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps,
        )

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = metadata.get("version", None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_BatchNorm, self)._load_from_state_dict(
            state_dict,
            prefix,
            metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class BatchNorm2d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))


class SimplexNet(Module):
    """Wrapper class for models composed of simple "simplex of models" classes,
    like Linear or Conv2D etc. aboce. See VGG16Simplex in vgg_noBN.py, for an example.

    It allows for efficient management of simplex of such models.
    """

    def __init__(
        self,
        n_output: int,
        architecture: Type[nn.Module],
        n_vert: int,
        fix_points: List[bool] = None,
        architecture_kwargs: Optional[Dict[str, Any]] = {},
        simplicial_complex: Optional[Dict[int, List[int]]] = None,
        config: Optional[Any] = None,
    ):
        """
        Args:
            n_output:             number of outputs of the network for which the simplex
                                  is to be maintained.

            architecture:         a subclass of nn.Module that implements a model using only simple
                                  "simplex of models" classes like Linear or Conv2D above. See
                                  VGG16Simplex in vgg_noBN.py, for an example.

            n_vert:               Number of vertices in a single simplex.

            fix_points:           A list of boolean values indicating which vertices of a simplex
                                  are trainable.

            architecture_kwargs:  A dictionary of keyword arguments to be passed when instantiating
                                  the architecture.

            simplicial_complex:   A dictionary mapping from index of a simplex in the simplicial complex,
                                  to the vertices in that simplex. [NOT CLEAR HOW THIS IS USED, FULLY]

            config:               config object to be used to instantiate the simplexes, by passing the
                                  config to from_config() function of architecture.
        """
        super(SimplexNet, self).__init__()
        self.n_output = n_output
        self.n_vert = n_vert

        if fix_points is not None:
            self.fix_points = fix_points
        else:
            self.fix_points = n_vert * [False]

        if simplicial_complex is None:
            simplicial_complex = {0: [ii for ii in range(n_vert)]}

        self.simplicial_complex = simplicial_complex
        self.n_simplex = len(simplicial_complex)
        self.config = config
        self.architecture = architecture
        self.architecture_kwargs = architecture_kwargs

        if not self.config:
            self.net = self.architecture(
                n_output, fix_points=self.fix_points, **architecture_kwargs
            )
        else:
            self.net = self.architecture.from_config(self.config)

        # Keep record of all sub-modules(upto 1 level) in an instance of architecture.
        self.simplex_modules = []
        for module in self.net.modules():
            if issubclass(module.__class__, SimplexModule):
                self.simplex_modules.append(module)

    def import_base_parameters(self, base_model, index):
        """Copies the parameters of base_model to the vertex at index in the simplex.

        PRE-CONDITIONS:
        1. The .parameters() iterator on the instance of architecture, must yield a
           parameter for all vertices in the simplex, before beginning to yield another parameter.
           For e.g., a Linear layer should yield (model-)weights for all vertices in the simplex, and then
           yield the biases for all vertices of the simplex, one-by-one.

        2. base_model.parameters() must yield parameters in same order as in the order of parameters
           of different types in the instance of architecture. That is, in the previous example,
           base_model.parameters() must yield its weight first, then its bias.
        """
        parameters = list(self.net.parameters())[index :: self.n_vert]
        base_parameters = base_model.parameters()
        for parameter, base_parameter in zip(parameters, base_parameters):
            parameter.data.copy_(base_parameter.data)

    def import_base_buffers(self, base_model):
        """Copies the data in buffers of base_model, to the buffers in
        architecture instance. Same pre-conditions as import_base_parameters."""
        for buffer, base_buffer in zip(self.net.buffers(), base_model.buffers()):
            buffer.data.copy_(base_buffer.data)

    def export_base_parameters(self, base_model, index):
        """Copies the parameters of index no. vertex of the simplex into the base_model.
        Same pre-conditions as import_base_parameters."""
        parameters = list(self.net.parameters())[index :: self.n_vert]
        base_parameters = base_model.parameters()
        for parameter, base_parameter in zip(parameters, base_parameters):
            base_parameter.data.copy_(parameter.data)

    def init_linear(self):
        """For initializing parameters of an architecture, when using a poly-chain
        simplex. This function is to be used by a base-class defining self.num_bends.
        The bends of the polychain are initialised to be equally spaced on the line
        joining the two vertices(i & j) composing the 1-D simplex.

        PRE-CONDITIONS:

        1. The .parameters() iterator on the instance of architecture, must yield a
           parameter for all vertices & bends in the simplex, before beginning to yield another parameter.

        2. When yielding a parameter for the 1-D simplex, the first value must correspond to a parameter of
           vertex-i of simplex, and the last parameter value must correspond to vertex j.

        For e.g., a Linear layer should yield "weight" parameters for vertex-i first, then random "weight"
        parameters for bends, and at last the "weight" param for vertex-j. And then only, start to yield
        the "bias" parameters for vertex-i, bends, and vertex-j, in that order.
        """
        parameters = list(self.net.parameters())
        for i in range(0, len(parameters), self.num_bends):
            weights = parameters[i : i + self.num_bends]
            for j in range(1, self.num_bends - 1):
                alpha = j * 1.0 / (self.num_bends - 1)
                weights[j].data.copy_(
                    alpha * weights[-1].data + (1.0 - alpha) * weights[0].data
                )

    def weights(self, t) -> np.ndarray:
        """Samples random (model-)weights for all the vertices of a simplex, and scales all parameters of
        "simplex of models" classes in the architecture by these (model-)weights.

        Returns:
            A single 1-D numpy array containing all the (model-)weights, for all the "simplex of models"
            modules in the architecture.

        NOTE:
            The returned (model-)weights are in the order of the occurance of the "simplex of models" modules
            in the architecture. Within a "simplex of models" module, the (model-)weights are in the order
            of the "different" parameters yielded by the .parameters() iterator of the "simplex of models" module.
        """
        coeffs_t = self.vertex_weights()
        weights = []
        for module in self.simplex_modules:
            weights.extend(
                [w for w in module.compute_weights_t(coeffs_t) if w is not None]
            )  # Should raise error if w is None?
        return np.concatenate([w.detach().cpu().numpy().ravel() for w in weights])

    def forward(self, input, t=None):
        """Computes the output of the model in the simplex, specified by t, at the input.
        Args:
            t:  The (t)raversal parameter along the simplex. If None, it is randomly sampled from [0,1].
                [CURRENTLY NOT USED].
        Returns:
            The output of the running self.net(an instance of architecture) on the input, with the
            (model-)weights of self.net are set to that of a model, randomly sampled from any of the
            simplexes in self.simplicial_complex.
        """
        if t is None:
            t = input.data.new(1).uniform_()
        coeffs_t = self.vertex_weights()
        output = self.net(input, coeffs_t)
        return output

    def compute_center_weights(self) -> torch.Tensor:
        """
        Returns:
            A 1-D tensor containing the (model-)weights of the center of the simplex stored
            currently in self.net.
            The (model-)weights are in the order of the "different" parameters yielded
            by the .parameters() iterator self.net.
        """
        temp = [p for p in self.net.parameters()][0 :: self.n_vert]  # NOT USED ANYWHERE
        n_par = sum([p.numel() for p in temp])  # NOT USED ANYWHERE
        ## assign mean of old pars to new vertex ##
        par_vecs = self.par_vectors()

        return par_vecs.mean(0).unsqueeze(0)

    def par_vectors(self) -> torch.Tensor:
        """
        Returns:
            A tensor of shape (n_vert, n_par) containing the parameters of the vertices of a simplex.
            n_par is the total number of (model-)weights in a vertex of the simplex.
        """
        temp = [p for p in self.net.parameters()][0 :: self.n_vert]
        n_par = sum([p.numel() for p in temp])

        par_vecs = torch.zeros(self.n_vert, n_par).to(temp[0].device)

        for ii in range(self.n_vert):
            temp = [p for p in self.net.parameters()][ii :: self.n_vert]
            par_vecs[ii, :] = utils.flatten(temp)

        return par_vecs

    def add_vert(self, to_simplexes: int = [0]) -> None:
        """Adds a new vertex, to all the simplexes specified in to_simplexes.
        Args:
            to_simplexes: A list of indices of the simplexes in the simplicial complex,
                          to which we want to add a new vertex.
        Process:
            1. If config is not specified, a new instance of architecture is instantiated with one
               extra vertex, otherwise a new instance is instantiated with the self.config.
            2. All vertices except the last one, of the new instance of architecture, are fixed, in case
               config is not provided.
            3. (Model-)weights for the learnt vertices, are copied from self.net to the new instance,
               for all but the last vertex of the new instance.
            4. The last vertex's (model-)weights are initialized to the centre of the simplex formed by the
               original vertices.
            5. self.n_vert, self.net, self.simplex_modules and self.simplicial_complex are updated to include
               the new vertices added.

        NOTE:
            Currently, the function only supports len(to_simplexes) == 1.
        """
        self.fix_points = [True] * self.n_vert + [False]
        if not self.config:
            new_model = self.architecture(
                self.n_output, fix_points=self.fix_points, **self.architecture_kwargs
            )
        else:
            new_model = self.architecture.from_config(self.config)
        ## assign old pars to new model ##
        for index in range(self.n_vert):
            old_parameters = list(self.net.parameters())[index :: self.n_vert]
            new_parameters = list(new_model.parameters())[index :: (self.n_vert + 1)]
            for old_par, new_par in zip(old_parameters, new_parameters):
                new_par.data.copy_(old_par.data)

        new_parameters = list(new_model.parameters())
        new_parameters = new_parameters[(self.n_vert) :: (self.n_vert + 1)]
        n_par = sum([p.numel() for p in new_parameters])
        ## assign mean of old pars to new vertex ##
        par_vecs = torch.zeros(self.n_vert, n_par).to(new_parameters[0].device)
        for ii in range(self.n_vert):
            temp = [p for p in self.net.parameters()][ii :: self.n_vert]
            par_vecs[ii, :] = utils.flatten(temp)

        center_pars = torch.mean(par_vecs, 0).unsqueeze(0)
        center_pars = utils.unflatten_like(center_pars, new_parameters)
        for cntr, par in zip(center_pars, new_parameters):
            par.data = cntr.to(par.device)

        ## update self values ##
        self.n_vert += 1
        self.net = new_model
        self.simplex_modules = []
        for module in self.net.modules():
            if issubclass(module.__class__, SimplexModule):
                self.simplex_modules.append(module)

        for cc in to_simplexes:
            self.simplicial_complex[cc].append(self.n_vert - 1)

        return

    def vertex_weights(self) -> List[float]:
        """Picks a random simplex from the ones specified in self.simplicial_complex
        and samples a random model from this simplex.
        Process:
            The model is sampled by specifying random (scaling-)weights for each of the vertices of
            the selected simplex. Random numbers are sampled from an exponential distribution with
            mean parameter(λ) 1 and normalized to obtain the (scaling-)weights.
        Returns:
            The list of (scaling-)weights is returned.
        """
        ## first need to pick a simplex to sample from ##
        simp_ind = np.random.randint(self.n_simplex)
        vols = []
        n_verts = []
        for ii in range(self.n_simplex):
            # vols.append(complex_volume(self, ii))
            n_verts.append(len(self.simplicial_complex[ii]))

        ## sample weights for simplex
        exps = [-(torch.rand(1)).log().item() for _ in range(n_verts[simp_ind])]
        total = sum(exps)
        exps = [exp / total for exp in exps]

        ## now assign vertex weights out
        vert_weights = [0] * self.n_vert
        for ii, vert in enumerate(self.simplicial_complex[simp_ind]):
            vert_weights[vert] = exps[ii]

        return vert_weights

    def total_volume(self, vol_function=complex_volume):
        """Returns the volume of a smiplex at index 0 of self.simplicial_complex."""
        vol = 0
        #         for simp in range(self.n_simplex):
        #             vol += complex_volume(self, simp)
        vol = complex_volume(self, 0)
        return vol
