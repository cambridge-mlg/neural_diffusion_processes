import math
import itertools
from functools import partial

import haiku as hk
import numpy as np
import jax
import jax.numpy as jnp

# from scipy.spatial import cKDTree

import e3nn_jax as e3nn

# from e3nn_jax import (
#     Irreps,
#     IrrepsArray,
#     index_add,
#     Linear,
#     FullyConnectedTensorProduct,
#     BatchNorm,
# )
# from e3nn_jax.experimental.transformer import _index_max

from e3nn_jax.experimental.transformer import Transformer


def get_constant_init(constant):
    class Constant(hk.initializers.Initializer):
        """Initializes with a constant."""

        def __init__(self, std=None, **kwargs):
            """Constructs a Constant initializer.

            Args:
            constant: Constant to initialize with.
            """
            self.constant = constant

        def __call__(self, shape, dtype) -> jnp.ndarray:
            return jnp.broadcast_to(jnp.asarray(self.constant), shape).astype(dtype)

    return Constant


def stat(text, z):
    print(
        f"{text} ({z.shape}) = {jax.tree_util.tree_map(lambda x: jnp.sqrt(jnp.mean(x.astype(float)**2)), z)}"
    )


@partial(jax.jit, static_argnums=1)
def nearest_neighbors_jax(X, k):
    distance_matrix = jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    return jax.lax.top_k(-distance_matrix, k)[1]
    # return jnp.argsort(distance_matrix, axis=-1)[:, :k]


@partial(jax.jit, static_argnums=1)
def get_edges_knn(x, k):
    edge_src = nearest_neighbors_jax(x, k=k).reshape(-1)
    edge_dst = jnp.arange(0, x.shape[-2], 1)
    edge_dst = jnp.repeat(edge_dst, k, 0).reshape(-1)
    edges = [edge_src, edge_dst]
    return edges


def get_edge_attr(edge_length, maximum_radius, num_basis, radial_basis):
    edge_attr = (
        e3nn.soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=maximum_radius,
            number=num_basis,
            basis=radial_basis,
            cutoff=False,
        )
        * num_basis**0.5
        * 0.95
    )
    edge_weight_cutoff = 1.4 * e3nn.sus(10 * (1 - edge_length / maximum_radius))
    edge_attr *= edge_weight_cutoff[:, None]
    return edge_attr, edge_weight_cutoff


# class Transformer(hk.Module):
#     def __init__(
#         self,
#         irreps_node_output,
#         list_neurons,
#         act,
#         residual=False,
#         batch_norm=False,
#         num_heads=1,
#     ):
#         super().__init__()

#         self.irreps_node_output = Irreps(irreps_node_output)
#         self.list_neurons = list_neurons
#         self.act = act
#         self.num_heads = num_heads
#         self.residual = residual
#         self.batch_norm = (
#             BatchNorm(irreps=self.irreps_node_output) if batch_norm else None
#         )

#     def __call__(
#         self,
#         edge_src,
#         edge_dst,
#         edge_scalar_attr,
#         edge_weight_cutoff,
#         edge_attr: IrrepsArray,
#         node_feat: IrrepsArray,
#     ) -> IrrepsArray:
#         r"""
#         Args:
#             edge_src (array of int32): source index of the edges
#             edge_dst (array of int32): destination index of the edges
#             edge_scalar_attr (array of float): scalar attributes of the edges (typically given by ``soft_one_hot_linspace``)
#             edge_weight_cutoff (array of float): cutoff weight for the edges (typically given by ``sus``)
#             edge_attr (IrrepsArray): attributes of the edges (typically given by ``spherical_harmonics``)
#             node_f (IrrepsArray): features of the nodes

#         Returns:
#             IrrepsArray: output features of the nodes
#         """
#         edge_src_feat = jax.tree_util.tree_map(lambda x: x[edge_src], node_feat)
#         edge_dst_feat = jax.tree_util.tree_map(lambda x: x[edge_dst], node_feat)

#         kw = dict(list_neurons=self.list_neurons, act=self.act)
#         edge_k = jax.vmap(
#             lambda w, x, y: _tp_mlp_uvu(w, x, y, edge_dst_feat.irreps, **kw)
#         )(
#             edge_scalar_attr, edge_src_feat, edge_attr
#         )  # IrrepData[edge, irreps]
#         edge_v = jax.vmap(
#             lambda w, x, y: _tp_mlp_uvu(w, x, y, self.irreps_node_output, **kw)
#         )(
#             edge_scalar_attr, edge_src_feat, edge_attr
#         )  # IrrepData[edge, irreps]
#         # print("edge_k", edge_k.shape)
#         # print("edge_v", edge_v.shape)

#         edge_logit = jax.vmap(FullyConnectedTensorProduct(f"{self.num_heads}x0e"))(
#             edge_dst_feat, edge_k
#         ).array  # array[edge, head]
#         node_logit_max = _index_max(
#             edge_dst, edge_logit, node_feat.shape[0]
#         )  # array[node, head]
#         exp = edge_weight_cutoff[:, None] * jnp.exp(
#             edge_logit - node_logit_max[edge_dst]
#         )  # array[edge, head]
#         z = index_add(edge_dst, exp, out_dim=node_feat.shape[0])  # array[node, head]
#         z = jnp.where(z == 0.0, 1.0, z)
#         alpha = exp / z[edge_dst]  # array[edge, head]

#         edge_v = edge_v.factor_mul_to_last_axis(
#             self.num_heads
#         )  # IrrepsArray[edge, head, irreps_out]
#         edge_v = (
#             edge_v * jnp.sqrt(jax.nn.relu(alpha))[:, :, None]
#         )  # IrrepsArray[edge, head, irreps_out]
#         edge_v = edge_v.repeat_mul_by_last_axis()  # IrrepsArray[edge, irreps_out]

#         node_out = index_add(
#             edge_dst, edge_v, out_dim=node_feat.shape[0]
#         )  # IrrepsArray[node, irreps_out]
#         out = jax.vmap(Linear(self.irreps_node_output))(
#             node_out
#         )  # IrrepsArray[edge, head, irreps_out]

#         # if self.residual:
#         #     padded = jnp.pad(node_feat.array, (0, out.shape[-1] - node_feat.shape[-1]))
#         #     out = out + e3nn.IrrepsArray(out.irreps, padded)
#         #     # out = e3nn.IrrepsArray.cat([out, node_feat])

#         # if self.batch_norm:
#         #     out = self.batch_norm(out)
#         return out


class TransformerModule(hk.Module):
    """from https://github1s.com/e3nn/e3nn-jax/blob/0.8.0/examples/qm9_transformer.py#L106-L107"""

    def __init__(
        self,
        output_shape=None,
        **config,
    ):
        super().__init__()
        # print(config)
        self.config = config
        self.config["irreps_out"] = e3nn.Irreps("1x1e")  # TODO:

        # if use_second_order_repr:
        #     irrep_seq = [
        #         f'{ns}x0e',
        #         f'{ns}x0e + {nv}x1o + {nv}x2e',
        #         f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
        #         f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
        #     ]
        # else:
        #     irrep_seq = [
        #         f'{ns}x0e',
        #         f'{ns}x0e + {nv}x1o',
        #         f'{ns}x0e + {nv}x1o + {nv}x1e',
        #         f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
        #     ]

        mul0 = config["mul0"]
        mul1 = config["mul1"]
        mul2 = config["mul2"]
        self.irreps_features = e3nn.Irreps(
            f"{mul0}x0e + {mul0}x0o + {mul1}x1e + {mul1}x1o + {mul2}x2e + {mul2}x2o"
        ).simplify()
        print("irreps_features", self.irreps_features.dim, self.irreps_features)

        def act(x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
            x = e3nn.scalar_activation(
                x, [jax.nn.gelu, jnp.tanh] + [None] * (len(x.irreps) - 2)
            )
            if config["sq_nl"]:
                tp = e3nn.TensorSquare(
                    self.irreps_features, init=get_constant_init(0.0)
                )
                y = jax.vmap(tp)(x)
                return x + y
            else:
                return x

        self.act = act
        self.kw = dict(
            list_neurons=[config["radial_num_neurons"] * config["radial_num_layers"]],
            act=jax.nn.gelu,
            num_heads=config["num_heads"],
            residual=config["residual"],
            batch_norm=config["batch_norm"],
        )

    def __call__(self, x, y, t):
        # hk.vmap.require_split_rng = False
        # return hk.vmap(self.score, in_axes=0, out_axes=0)(x, y, t)
        # add empty 3rd coordinate for 2d data
        x_dim, y_dim = x.shape[-1], y.shape[-1]
        if x_dim == 2:
            x = jnp.concatenate([x, jnp.zeros((*x.shape[:-1], 1))], -1)
        if y_dim == 2:
            y = jnp.concatenate([y, jnp.zeros((*y.shape[:-1], 1))], -1)
        out = jax.vmap(self.score)(x, y, t)
        if y_dim == 2:
            out = out[..., :-1]
        return out

    def score(self, x, y, t):
        config = self.config

        t_emb = jnp.repeat(t.reshape(-1)[None, :], y.shape[-2], -2)
        t_emb = e3nn.IrrepsArray(f"{t.shape[-1]}x0e", t_emb)
        node_attr = y  # node_attr = e3nn.IrrepsArray("1x1e", y)
        pos = x

        # edge_src, edge_dst = a["edge_index"]
        ## https://github.com/google/jax/discussions/9813
        ## radius graph -> does not work with jit...
        # edge_src, edge_dst = e3nn.radius_graph(pos, config["maximum_radius"])
        ## complete graph -> scales quadratically...
        # edge_src, edge_dst = zip(*itertools.permutations(range(x.shape[-2]), 2))
        # edge_src, edge_dst = jnp.array(edge_src), jnp.array(edge_dst)
        edges = get_edges_knn(pos, config["k"])
        edge_src, edge_dst = edges[0], edges[1]

        edge_vec = pos[edge_dst] - pos[edge_src]
        edge_sh = e3nn.spherical_harmonics(
            e3nn.Irreps.spherical_harmonics(config["shlmax"]),
            edge_vec,
            normalize=True,
            normalization="component",
        )

        edge_length = jnp.linalg.norm(edge_vec, axis=-1)
        edge_attr, edge_weight_cutoff = get_edge_attr(
            edge_length,
            config["maximum_radius"],
            config["num_basis"],
            config["radial_basis"],
        )

        # edge_attr = jnp.concatenate(
        #     [edge_attr, node_attr[edge_src], node_attr[edge_dst]], axis=-1
        # )
        edge_t_emb = t_emb[edge_src]
        edge_attr = jnp.concatenate([edge_attr, edge_t_emb.array], axis=-1)

        # irreps = e3nn.Irreps(f'{config["mul0"]}x0e')
        # irreps = e3nn.Irreps(f'{config["mul1"]}x1e')
        irreps = self.irreps_features
        # print("t", type(t), t.shape)
        # print("node_attr", type(node_attr), x.shape)
        node_attr = e3nn.IrrepsArray("1x1e", node_attr)
        node_attr = e3nn.IrrepsArray.cat([node_attr, t_emb])
        node_attr = jax.vmap(e3nn.Linear(irreps))(node_attr)

        ns = node_attr.irreps.count("0e") + node_attr.irreps.count("0o")
        for _ in range(config["num_layers"]):
            _edge_attr = jnp.concatenate(
                [
                    edge_attr,
                    node_attr.array[edge_src, :ns],
                    node_attr.array[edge_dst, :ns],
                ],
                axis=-1,
            )
            node_attr = Transformer(self.irreps_features, **self.kw)(
                edge_src, edge_dst, _edge_attr, edge_weight_cutoff, edge_sh, node_attr
            )

            node_attr = self.act(node_attr)

        _edge_attr = jnp.concatenate(
            [
                edge_attr,
                node_attr.array[edge_src, :ns],
                node_attr.array[edge_dst, :ns],
            ],
            axis=-1,
        )
        node_attr = Transformer(config["irreps_out"], **self.kw)(
            edge_src, edge_dst, _edge_attr, edge_weight_cutoff, edge_sh, node_attr
        )
        out = node_attr.array
        # return e3nn.index_add(a["batch"], out, a["y"].shape[0])
        return out
