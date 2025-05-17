from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct, Linear
from typing import Dict, Union
import torch_geometric
import torch_scatter
import torch

class equivGNN(torch.nn.Module):
    def __init__(
        self,
        irreps_node_inputs: str = '32x0e',
        irreps_node_attr: str = '16x0e',
        irreps_in: str = '128x0e',
        irreps_out: str = '128x0e',
        irreps_node_hidden: str = None,
        irreps_linear_hidden: str = '128x0e',
        number_of_basis: int = 8,
        radial_layers: int = 2,
        radial_neurons: int = 64,
        num_neighbors: int = 10,
        max_radius: int = 4.25,
        layers: int = 3,
        lmax: int = 2,
        mul: int = 64,
        use_sc: bool = True,
        pool_nodes: bool = True,
    ) -> None:
        super().__init__()

        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.pool_nodes = pool_nodes

        self.atom_embedding = Linear(irreps_node_inputs,irreps_in)
        if irreps_node_hidden is not None:
            irreps_node_hidden = o3.Irreps(irreps_node_hidden)
        else:
            irreps_node_hidden = o3.Irreps([(mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]])

        self.mp = MessagePassing(
            irreps_node_sequence=[irreps_in] + layers * [irreps_node_hidden] + [irreps_out],
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=o3.Irreps.spherical_harmonics(lmax),
            fc_neurons=[self.number_of_basis]+radial_layers*[radial_neurons],
            num_neighbors=num_neighbors,
            use_sc=use_sc,
        )

        self.irreps_in = self.mp.irreps_node_features
        self.irreps_out = self.mp.irreps_node_output

        self.readout = torch.nn.Sequential(
            Linear(self.irreps_out, irreps_linear_hidden), torch.nn.SiLU(),
            Linear(irreps_linear_hidden, '0e')
            )

    def forward(self, data: Union[torch_geometric.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        batch, node_inputs, node_attr, edge_src, edge_dst, edge_vec = data['batch'], data['x'], data['n'],\
                    data['edge_index'][0], data['edge_index'][1], data['edge_vec']
        del data
        node_features = self.atom_embedding(node_inputs)

        # Edge length embedding
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = soft_one_hot_linspace(
            edge_length,
            0.25,
            self.max_radius,
            self.number_of_basis,
            basis="bessel",
            cutoff=True,
        ).mul(self.number_of_basis**0.5)

        edge_sh = o3.spherical_harmonics(range(self.lmax + 1), edge_vec, True, normalization="component")
        node_outputs = self.mp(node_features, node_attr, edge_src, edge_dst, edge_sh, edge_length_embedding)
        if self.pool_nodes:
            x = torch_scatter.scatter_mean(node_outputs, batch, dim=0)  # Take mean over atoms per example
            x = self.readout(x)
            return x.reshape(1, -1)[0]
        else:
            return node_outputs


class MessagePassing(torch.nn.Module):
    r"""

    Parameters
    ----------
    irreps_node_sequence : list of `e3nn.o3.Irreps`
        representation of the input/hidden/output features

    irreps_node_attr : `e3nn.o3.Irreps`
        representation of the nodes attributes

    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes

    layers : int
        number of gates (non linearities)

    fc_neurons : list of int
        number of neurons per layers in the fully connected network
        first layer and hidden layers but not the output layer
    """

    def __init__(
        self,
        irreps_node_sequence,
        irreps_node_attr,
        irreps_edge_attr,
        num_neighbors,
        fc_neurons,
        use_sc,
    ) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors

        irreps_node_sequence = [o3.Irreps(irreps) for irreps in irreps_node_sequence]
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)

        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        self.layers = torch.nn.ModuleList()

        self.irreps_node_sequence = [irreps_node_sequence[0]]
        irreps_node = irreps_node_sequence[0]

        for irreps_node_hidden in irreps_node_sequence[1:-1]:
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_node_hidden
                    if ir.l == 0 and tp_path_exists(irreps_node, self.irreps_edge_attr, ir)
                ]
            ).simplify()
            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in irreps_node_hidden
                    if ir.l > 0 and tp_path_exists(irreps_node, self.irreps_edge_attr, ir)
                ]
            )
            if irreps_gated.dim > 0:
                if tp_path_exists(irreps_node, self.irreps_edge_attr, "0e"):
                    ir = "0e"
                elif tp_path_exists(irreps_node, self.irreps_edge_attr, "0o"):
                    ir = "0o"
                else:
                    raise ValueError(
                        f"irreps_node={irreps_node} times irreps_edge_attr={self.irreps_edge_attr} is unable to produce gates "
                        f"needed for irreps_gated={irreps_gated}"
                    )
            else:
                ir = None
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
            conv = Convolution(
                irreps_node, self.irreps_node_attr, self.irreps_edge_attr, gate.irreps_in, fc_neurons, num_neighbors, use_sc
            )
            self.layers.append(Compose(conv, gate))
            irreps_node = gate.irreps_out
            self.irreps_node_sequence.append(irreps_node)

        irreps_node_output = irreps_node_sequence[-1]
        self.layers.append(
            Convolution(
                irreps_node, self.irreps_node_attr, self.irreps_edge_attr, irreps_node_output, fc_neurons, num_neighbors, use_sc
            )
        )
        self.irreps_node_sequence.append(irreps_node_output)

        self.irreps_node_features = self.irreps_node_sequence[0]
        self.irreps_node_output = self.irreps_node_sequence[-1]

    def forward(self, node_features, node_attr, edge_src, edge_dst, edge_attr, edge_scalars) -> torch.Tensor:
        for lay in self.layers:
            node_features = lay(node_features, node_attr, edge_src, edge_dst, edge_attr, edge_scalars)

        return node_features


@compile_mode("script")
class Convolution(torch.nn.Module):
    r"""equivariant convolution

    Parameters
    ----------
    irreps_node_input : `e3nn.o3.Irreps`
        representation of the input node features

    irreps_node_attr : `e3nn.o3.Irreps`
        representation of the node attributes

    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes

    irreps_node_output : `e3nn.o3.Irreps` or None
        representation of the output node features

    fc_neurons : list of int
        number of neurons per layers in the fully connected network
        first layer and hidden layers but not the output layer

    num_neighbors : float
        typical number of nodes convolved over
    """

    def __init__(
        self, irreps_node_input, irreps_node_attr, irreps_edge_attr, irreps_node_output, fc_neurons, num_neighbors, use_sc
    ) -> None:
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.num_neighbors = num_neighbors
        self.use_sc = use_sc

        self.linear_1 = Linear(
            irreps_in=self.irreps_node_input,
            irreps_out=self.irreps_node_input,
            internal_weights=True,
            shared_weights=True,
        )

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_node_input):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_node_output or ir_out == o3.Irrep(0, 1):
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        assert irreps_mid.dim > 0, (
            f"irreps_node_input={self.irreps_node_input} time irreps_edge_attr={self.irreps_edge_attr} produces nothing "
            f"in irreps_node_output={self.irreps_node_output}"
        )
        instructions = [(i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instructions]

        tp = TensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet(fc_neurons + [tp.weight_numel], torch.nn.functional.silu)
        self.tp = tp

        self.linear_2 = Linear(
            irreps_in=irreps_mid.simplify(),
            irreps_out=self.irreps_node_output,
            internal_weights=True,
            shared_weights=True,
        )

        self.sc = None
        if self.use_sc:
            self.sc = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr, self.irreps_node_output)

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars) -> torch.Tensor:
        node_features = self.linear_1(node_input)
        weight = self.fc(edge_scalars)
        edge_features = self.tp(node_features[edge_src], edge_attr, weight)
        node_features = torch_scatter.scatter(edge_features, edge_dst, dim_size=node_input.shape[0], dim=0).div(self.num_neighbors**0.5)
        node_features = self.linear_2(node_features)

        if self.sc is not None:
            node_self_connection = self.sc(node_input, node_attr)
            # node_features += node_self_connection
            node_features = node_features + node_self_connection

        return node_features

def tp_path_exists(irreps_in1, irreps_in2, ir_out) -> bool:
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class Compose(torch.nn.Module):
    def __init__(self, first, second) -> None:
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, *input):
        x = self.first(*input)
        return self.second(x)