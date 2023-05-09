import math
import itertools
from functools import partial
from typing import List, Callable, Tuple
import dataclasses

import haiku as hk
import numpy as np
import jax
import jax.numpy as jnp

import e3nn_jax as e3nn

# from e3nn_jax.experimental.transformer import Transformer, _index_max
from e3nn_jax.experimental.transformer import _index_max

from .misc import timestep_embedding, get_activation, scatter
from neural_diffusion_processes.utils import dotdict


class Constant(hk.initializers.Initializer):
    """Initializes with a constant."""

    # def __init__(self, std=None, **kwargs):
    def __init__(self, constant=0.0, **kwargs):
        """Constructs a Constant initializer.

        Args:
        constant: Constant to initialize with.
        """
        # self.constant = 0.0
        self.constant = constant

    def __call__(self, shape, dtype) -> jnp.ndarray:
        return jnp.broadcast_to(jnp.asarray(self.constant), shape).astype(dtype)


# def get_constant_init(constant):

#     return Constant


@dataclasses.dataclass
class Convolution(hk.Module):
    irreps_node_output: e3nn.Irreps
    list_neurons: List[int]
    act: Callable[[jnp.ndarray], jnp.ndarray]
    num_heads: int = 1
    zero_init: bool = False
    residual: bool = False
    batch_norm: bool = False

    def __call__(
        self,
        edge_src: jnp.ndarray,
        edge_dst: jnp.ndarray,
        edge_weight_cutoff: jnp.ndarray,
        edge_attr: e3nn.IrrepsArray,
        node_feat: e3nn.IrrepsArray,
        edge_sh: e3nn.IrrepsArray = None,
    ) -> e3nn.IrrepsArray:
        # node_feat = e3nn.haiku.Linear(node_feat.irreps)(
        #     node_feat
        # )  # [num_nodes, irreps]

        if edge_sh is not None:
            edge_equivariant = edge_sh.filter(drop="0e")
        else:
            edge_invariant = edge_attr.filter(keep="0e")
            edge_equivariant = edge_attr.filter(drop="0e")

        messages = e3nn.tensor_product(node_feat[edge_src], edge_equivariant)
        messages = e3nn.concatenate([messages, node_feat[edge_src]])
        messages = messages.regroup().filter(keep=self.irreps_node_output)
        factors = e3nn.haiku.MultiLayerPerceptron(
            self.list_neurons + [messages.irreps.num_irreps],
            act=self.act,
            output_activation=False,
        )(edge_invariant)
        messages = e3nn.elementwise_tensor_product(messages, factors)
        messages = edge_weight_cutoff[:, None] * messages

        # node_output = e3nn.IrrepsArray.zeros(
        #     messages.irreps, (node_feat.shape[0],), dtype=messages.dtype
        # )
        # node_output = node_output.at[edge_dst].add(messages) / jnp.sqrt(
        #     self.num_neighbors
        # )
        node_output = e3nn.scatter_sum(
            messages, dst=edge_dst, output_size=node_feat.shape[0]
        )

        def get_parameter(
            name: str,
            path_shape: Tuple[int, ...],
            weight_std: float,
            dtype: jnp.dtype = jnp.float32,
        ):
            init = (
                # get_constant_init(0.0)()
                Constant(0.0)
                if self.zero_init
                else hk.initializers.RandomNormal(stddev=weight_std)
            )
            return hk.get_parameter(
                name,
                shape=path_shape,
                dtype=dtype,
                init=init,
            )

        node_output = e3nn.haiku.Linear(
            self.irreps_node_output, get_parameter=get_parameter
        )(node_output)
        return node_output


@dataclasses.dataclass
class Transformer(hk.Module):
    irreps_node_output: e3nn.Irreps
    list_neurons: List[int]
    act: Callable[[jnp.ndarray], jnp.ndarray]
    num_heads: int = 1
    zero_init: bool = False
    residual: bool = False
    batch_norm: bool = False

    def __call__(
        self,
        edge_src: jnp.ndarray,  # [E] dtype=int32
        edge_dst: jnp.ndarray,  # [E] dtype=int32
        edge_weight_cutoff: jnp.ndarray,  # [E] dtype=float
        edge_attr: e3nn.IrrepsArray,  # [E, D] dtype=float
        node_feat: e3nn.IrrepsArray,  # [N, D] dtype=float
        edge_sh: e3nn.IrrepsArray = None,
    ) -> e3nn.IrrepsArray:
        r"""Equivariant Transformer.

        Args:
            edge_src (array of int32): source index of the edges
            edge_dst (array of int32): destination index of the edges
            edge_weight_cutoff (array of float): cutoff weight for the edges (typically given by ``soft_envelope``)
            edge_attr (e3nn.IrrepsArray): attributes of the edges (typically given by ``spherical_harmonics``)
            node_f (e3nn.IrrepsArray): features of the nodes

        Returns:
            e3nn.IrrepsArray: output features of the nodes
        """
        edge_sh = edge_attr if edge_sh is None else edge_sh

        def f(x, y, edge_sh, filter_ir_out=None, name=None):
            out1 = (
                e3nn.concatenate([x, e3nn.tensor_product(x, edge_sh.filter(drop="0e"))])
                .regroup()
                .filter(keep=filter_ir_out)
            )
            y = y.filter(keep="0e") if isinstance(y, e3nn.IrrepsArray) else y
            out2 = e3nn.haiku.MultiLayerPerceptron(
                self.list_neurons + [out1.irreps.num_irreps],
                self.act,
                output_activation=False,
                name=name,
            )(y)
            return out1 * out2

        edge_key = f(
            node_feat[edge_src], edge_attr, edge_sh, node_feat.irreps, name="mlp_key"
        )
        edge_logit = e3nn.haiku.Linear(f"{self.num_heads}x0e", name="linear_logit")(
            e3nn.tensor_product(node_feat[edge_dst], edge_key, filter_ir_out="0e")
        ).array  # [E, H]
        node_logit_max = _index_max(edge_dst, edge_logit, node_feat.shape[0])  # [N, H]
        exp = edge_weight_cutoff[:, None] * jnp.exp(
            edge_logit - node_logit_max[edge_dst]
        )  # [E, H]
        z = e3nn.scatter_sum(
            exp, dst=edge_dst, output_size=node_feat.shape[0]
        )  # [N, H]
        z = jnp.where(z == 0.0, 1.0, z)
        alpha = exp / z[edge_dst]  # [E, H]

        edge_v = f(
            node_feat[edge_src], edge_attr, edge_sh, self.irreps_node_output, "mlp_val"
        )  # [E, D]
        edge_v = edge_v.mul_to_axis(self.num_heads)  # [E, H, D]
        edge_v = edge_v * jnp.sqrt(jax.nn.relu(alpha))[:, :, None]  # [E, H, D]
        edge_v = edge_v.axis_to_mul()  # [E, D]

        node_out = e3nn.scatter_sum(
            edge_v, dst=edge_dst, output_size=node_feat.shape[0]
        )  # [N, D]

        def get_parameter(
            name: str,
            path_shape: Tuple[int, ...],
            weight_std: float,
            dtype: jnp.dtype = jnp.float32,
        ):
            init = (
                # get_constant_init(0.0)
                Constant(0.0)
                if self.zero_init
                else hk.initializers.RandomNormal(stddev=weight_std)
            )
            return hk.get_parameter(
                name,
                shape=path_shape,
                dtype=dtype,
                init=init,
            )

        node_out = e3nn.haiku.Linear(
            self.irreps_node_output, get_parameter=get_parameter, name="linear_out"
        )(
            node_out
        )  # [N, D]

        if self.residual:
            # # padded = jnp.pad(y.array, (0, node_attr.shape[-1] - y.shape[-1]))
            padded = jnp.pad(
                node_feat.array, (0, node_out.shape[-1] - node_feat.shape[-1])
            )
            # NOTE: note sure to understand why the 1st index is padded
            padded = e3nn.IrrepsArray(node_out.irreps, padded[: node_feat.shape[0]])
            # padded = node_feat
            node_out = node_out + padded

        if self.batch_norm:
            node_out = e3nn.haiku.BatchNorm(irreps=self.irreps_node_output)(node_out)
        return node_out


class TensorProductConvLayer(hk.Module):
    def __init__(
        self,
        in_irreps,
        sh_irreps,
        out_irreps,
        n_edge_features,
        residual=False,
        batch_norm=False,
    ):
        super().__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual

        self.tp = tp = e3nn.haiku.FullyConnectedTensorProduct(
            out_irreps,
            irreps_in1=in_irreps,
            irreps_in2=sh_irreps,  # , shared_weights=False
        )

        self.fc = hk.Sequential(
            [hk.Linear(n_edge_features), jax.nn.relu, hk.Linear(len(tp.instructions))]
        )
        self.batch_norm = (
            e3nn.haiku.BatchNorm(irreps=out_irreps) if batch_norm else None
        )

    def __call___(
        self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce="mean"
    ):
        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        # out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)
        out = scatter(tp, 0, edge_src, dim_size=out_nodes, reduce=reduce)
        # z = e3nn.scatter_sum(exp, dst=edge_dst, output_size=node_feat.shape[0])
        if self.residual:
            padded = jnp.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))[
                : node_attr.shape[0]
            ]
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)

        return out


def stat(text, z):
    print(
        f"{text} ({z.shape}) = {jax.tree_util.tree_map(lambda x: jnp.sqrt(jnp.mean(x.astype(float)**2)), z)}"
    )


def fill_diagonal(a, val):
    assert a.ndim >= 2
    i, j = jnp.diag_indices(min(a.shape[-2:]))
    return a.at[..., i, j].set(val)


@partial(jax.jit, static_argnums=1)
def nearest_neighbors_jax(X, k):
    pdist = jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    pdist = fill_diagonal(pdist, jnp.inf * jnp.ones((X.shape[0])))
    return jax.lax.top_k(-pdist, k)[1]
    # return jnp.argsort(distance_matrix, axis=-1)[:, :k]


@partial(jax.jit, static_argnums=1)
def get_edges_knn(x, k):
    senders = nearest_neighbors_jax(x, k=k).reshape(-1)
    receivers = jnp.arange(0, x.shape[-2], 1)
    receivers = jnp.repeat(receivers, k, 0).reshape(-1)
    return senders, receivers


def get_edge_attr(edge_length, max_radius, n_basis, radial_basis):
    edge_attr = (
        e3nn.soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=max_radius,
            number=n_basis,
            basis=radial_basis,
            cutoff=False,
        )
        * n_basis**0.5
        * 0.95
    )
    edge_weight_cutoff = 1.4 * e3nn.sus(10 * (1 - edge_length / max_radius))
    # edge_weight_cutoff = e3nn.sus(3.0 * (2.0 - edge_length))
    edge_attr *= edge_weight_cutoff[:, None]
    return edge_attr, edge_weight_cutoff


class TransformerModule(hk.Module):
    """from https://github1s.com/e3nn/e3nn-jax/blob/0.8.0/examples/qm9_transformer.py#L106-L107"""

    def __init__(
        self,
        output_shape=None,
        **config,
    ):
        super().__init__()
        # print(config)
        config = dotdict(config)
        self.config = config
        self.config.irreps_out = e3nn.Irreps("1x1e")  # TODO:

        # if use_second_order_repr:
        #     irrep_seq = [
        #         f"{ns}x0e",
        #         f"{ns}x0e + {nv}x1o + {nv}x2e",
        #         f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o",
        #         f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o",
        #     ]
        # else:
        #     irrep_seq = [
        #         f"{ns}x0e",
        #         f"{ns}x0e + {nv}x1o",
        #         f"{ns}x0e + {nv}x1o + {nv}x1e",
        #         f"{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o",
        #     ]

        mul0 = config.mul0
        mul1 = config.mul1
        mul2 = config.mul2
        self.irreps_features = e3nn.Irreps(
            f"{mul0}x0e + {mul0}x0o + {mul1}x1e + {mul1}x1o + {mul2}x2e + {mul2}x2o"
        ).simplify()
        print("irreps_features", self.irreps_features.dim, self.irreps_features)

        def act(x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
            x = e3nn.scalar_activation(
                x, [self.act_fn, jnp.tanh] + [None] * (len(x.irreps) - 2)
            )
            if config.sq_nl:
                y = e3nn.tensor_square(x)
                # y = jax.vmap(tp)(x)
                y = jax.vmap(e3nn.haiku.Linear(self.irreps_features))(y)
                # y = y._convert(self.irreps_features)
                return x + y
            else:
                return x

        self.act = act
        self.act_fn = get_activation(config.act_fn)
        self.kw = dict(
            # list_neurons=[config.radial_n_neurons * config.radial_n_layers],
            list_neurons=[config.radial_n_neurons] * config.radial_n_layers,
            act=self.act_fn,
            num_heads=config.num_heads,
            zero_init=config.zero_init,
            residual=config.residual,
            batch_norm=config.batch_norm,
        )
        self.layer = Transformer if config.attention else Convolution

    def __call__(self, x, y, t):
        # hk.vmap.require_split_rng = False
        # return hk.vmap(self.score, in_axes=0, out_axes=0)(x, y, t)
        # add empty 3rd coordinate for 2d data
        x_dim, y_dim = x.shape[-1], y.shape[-1]
        if x_dim == 2:
            x = jnp.concatenate([x, jnp.zeros((*x.shape[:-1], 1), dtype=x.dtype)], -1)
        if y_dim == 2:
            y = jnp.concatenate([y, jnp.zeros((*y.shape[:-1], 1), dtype=y.dtype)], -1)
        out = jax.vmap(self.score)(x, y, t)
        if y_dim == 2:
            out = out[..., :-1]
        return out

    def score(self, x, y, t):
        config = self.config
        t = timestep_embedding(t[None], 16).squeeze(0)
        t_emb = jnp.repeat(t.reshape(-1)[None, :], y.shape[-2], -2)
        t_emb = e3nn.IrrepsArray(f"{t.shape[-1]}x0e", t_emb)
        node_attr = y  # node_attr = e3nn.IrrepsArray("1x1e", y)
        pos = x

        senders, receivers = get_edges_knn(pos, config.k)
        print("senders / receivers", senders.shape, receivers.shape)

        edge_vec = pos[receivers] - pos[senders]
        sh_irreps = e3nn.Irreps.spherical_harmonics(config.shlmax)
        edge_sh = e3nn.spherical_harmonics(
            sh_irreps,
            edge_vec,
            normalize=True,
            normalization="component",
        )

        edge_length = jnp.linalg.norm(edge_vec, axis=-1)
        edge_attr, edge_weight_cutoff = get_edge_attr(
            edge_length,
            config.max_radius,
            config.n_basis,
            config.radial_basis,
        )

        # edge_attr = jnp.concatenate(
        #     [edge_attr, node_attr[senders], node_attr[receivers]], axis=-1
        # )
        edge_t_emb = t_emb[senders]
        edge_attr = jnp.concatenate([edge_attr, edge_t_emb.array], axis=-1)
        self.edge_embedding = hk.Sequential([hk.Linear(64), self.act_fn, hk.Linear(64)])
        edge_attr = self.edge_embedding(edge_attr)

        # irreps = e3nn.Irreps(f'{config.mul0}x0e')
        # irreps = e3nn.Irreps(f'{config.mul1}x1e')
        irreps = self.irreps_features
        # print("t", type(t), t.shape)
        node_attr = e3nn.IrrepsArray("1x1e", node_attr)
        ns = node_attr.irreps.count("0e") + node_attr.irreps.count("0o")

        # for _ in range(config.n_layers - 1):
        for _ in range(config.n_layers):
            node_attr = e3nn.concatenate([node_attr, t_emb])
            node_attr = jax.vmap(e3nn.haiku.Linear(irreps))(node_attr)
            _edge_attr = jnp.concatenate(
                [
                    edge_attr,
                    node_attr.array[senders, :ns],
                    node_attr.array[receivers, :ns],
                ],
                axis=-1,
            )
            _edge_attr = e3nn.concatenate([_edge_attr, edge_sh])
            node_attr = self.layer(irreps, **self.kw)(
                senders,
                receivers,
                edge_weight_cutoff,
                _edge_attr,
                node_attr,
                # edge_sh=edge_sh,
            )
            node_attr = self.act(node_attr)

        # _edge_attr = jnp.concatenate(
        #     [
        #         edge_attr,
        #         node_attr.array[senders, :ns],
        #         node_attr.array[receivers, :ns],
        #     ],
        #     axis=-1,
        # )
        # _edge_attr = e3nn.concatenate([_edge_attr, edge_sh])
        # # node_attr = self.layer(config.irreps_out, **self.kw)(
        # node_attr = self.layer(irreps, **self.kw)(
        #     senders,
        #     receivers,
        #     edge_weight_cutoff,
        #     _edge_attr,
        #     node_attr,
        #     # edge_sh=edge_sh,
        # )
        node_attr = e3nn.haiku.Linear(config.irreps_out)(node_attr)
        out = node_attr.array
        if self.config.residual:
            out = out - y
        return out
