from typing import NamedTuple, Callable, Sequence, Tuple
from functools import partial
import dataclasses

import numpy as np
import jax
from jax import vmap
import jax.numpy as jnp
import haiku as hk
import chex
from check_shapes import check_shape as cs, check_shapes

from .misc import timestep_embedding, get_activation
import e3nn_jax as e3nn


from .egnn import get_edges_knn, get_edge_attr, get_edges_batch
from neural_diffusion_processes.utils.misc import flatten, unflatten
from .stable_mlp import StableMLP


def safe_norm(x: jnp.ndarray, axis: int = None, keepdims=False) -> jnp.ndarray:
    """nan-safe norm. Copied from mace-jax"""
    x2 = jnp.sum(x**2, axis=axis, keepdims=keepdims)
    return jnp.where(x2 == 0.0, 0.0, jnp.where(x2 == 0, 1.0, x2) ** 0.5)


def get_senders_and_receivers_fully_connected(
    n_nodes: int,
) -> Tuple[chex.Array, chex.Array]:
    """Get senders and receivers for fully connected graph of `n_nodes`."""
    receivers = []
    senders = []
    for i in range(n_nodes):
        for j in range(n_nodes - 1):
            receivers.append(i)
            senders.append((i + 1 + j) % n_nodes)
    return jnp.array(senders, dtype=int), jnp.array(receivers, dtype=int)


@dataclasses.dataclass
class EGCL(hk.Module):
    """A version of EGCL coded only with haiku (not e3nn) so works for arbitary dimension of inputs.
    Follows notation of https://arxiv.org/abs/2105.09016."""

    name: str
    mlp_units: Sequence[int]
    n_invariant_feat_hidden: int
    activation_fn: Callable
    residual_h: bool
    residual_x: bool
    residual_y: bool
    normalize: bool
    norm_constant: float
    attention: float
    variance_scaling_init: float
    cross_multiplicty_node_feat: bool
    cross_multiplicity_shifts: bool
    norm_wrt_centre_feat: bool

    """
    Args:
        name (str)
        mlp_units (Sequence[int]): sizes of hidden layers for all MLPs
        residual_h (bool): whether to use a residual connectio probability density for scalars
        residual_x (bool): whether to use a residual connectio probability density for vectors.
        norm_constant (float): Value to normalize the output of MLP multiplying message vectors.
            C in the en normalizing flows paper (https://arxiv.org/abs/2105.09016).
        variance_scaling_init (float): Value to scale the output variance of MLP multiplying message vectors
        cross_multiplicty_node_feat (bool): Whether to use cross multiplicity for node features.
        cross_multiplicity_shifts (bool): Whether to use cross multiplicity for shifts.
        norm_wrt_centre_feat (bool): Whether to include the norm of `node_positions` as a feature.
    """

    # def coord2radial(self, edge_index, coord):
    #     # receivers, senders = edge_index
    #     # coord_diff = coord[receivers] - coord[senders]
    #     # radial = jnp.expand_dims(jnp.sum(coord_diff**2, 1), 1)

    #     # if self.normalize:
    #     #     norm = jax.lax.stop_gradient(jnp.sqrt(radial)) + self.norm_constant
    #     #     coord_diff = coord_diff / norm

    #     coord_diff_y = node_vectors[receivers] - node_vectors[senders]
    #     lengths_y = safe_norm(coord_diff_y, axis=-1, keepdims=False)
    #     sq_lengths_y = lengths_y**2

    #     if self.normalize:
    #         norm = self.norm_constant + jax.lax.stop_gradient(lengths_y)[:, :, None]
    #         coord_diff_y /= norm

    #     return radial, coord_diff

    def __call__(
        self,
        node_positions: chex.Array,
        node_vectors: chex.Array,
        node_features: chex.Array,
        senders: chex.Array,
        receivers: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """E(N)GNN layer implementation.
        Args:
            node_positions [n_nodes, self.n_vectors_hidden, 3]-ndarray: augmented set of euclidean coodinates for each node
            node_features [n_nodes, self.n_invariant_feat_hidden]-ndarray: scalar features at each node
            senders: [n_edges]-ndarray: sender nodes for each edge
            receivers: [n_edges]-ndarray: receiver nodes for each edge
        Returns:
            vectors_out [n_nodes, self.n_vectors_hidden, 3]-ndarray: augmented set of euclidean coodinates for each node
            features_out [n_nodes, self.n_invariant_feat_hidden]-ndarray: scalar features at each node
        """
        chex.assert_rank(node_positions, 3)
        chex.assert_rank(node_features, 2)
        chex.assert_rank(senders, 1)
        chex.assert_equal_shape([senders, receivers])
        n_nodes, n_vectors, dim = node_positions.shape
        # avg_num_neighbours = n_nodes - 1
        avg_num_neighbours = int(np.sqrt(receivers.shape[0]))
        chex.assert_tree_shape_suffix(node_features, (self.n_invariant_feat_hidden,))

        # Prepare the edge attributes.
        coord_diff_x = node_positions[receivers] - node_positions[senders]
        lengths_x = safe_norm(coord_diff_x, axis=-1, keepdims=False)
        sq_lengths_x = lengths_x**2

        coord_diff_y = node_vectors[receivers] - node_vectors[senders]
        lengths_y = safe_norm(coord_diff_y, axis=-1, keepdims=False)
        sq_lengths_y = lengths_y**2

        edge_feat_in = jnp.concatenate(
            [
                node_features[senders],
                node_features[receivers],
                sq_lengths_x,
                sq_lengths_y,
            ],
            axis=-1,
        )

        MLP = hk.nets.MLP
        # MLP = StableMLP

        # if self.cross_multiplicity_shifts:
        #     self.phi_x_cross_torso = MLP(
        #         mlp_units, activate_final=True, activation=activation_fn
        #     )

        # build messages
        self.phi_e = MLP(
            self.mlp_units, activation=self.activation_fn, activate_final=True
        )
        self.phi_inf = lambda x: jax.nn.sigmoid(hk.Linear(1)(x))
        m_ij = self.phi_e(edge_feat_in)

        # Get positional output
        w_init = hk.initializers.VarianceScaling(
            self.variance_scaling_init, "fan_avg", "uniform"
        )
        # self.phi_x_torso = MLP(
        #     self.mlp_units, activate_final=True, activation=self.activation_fn
        # )
        # phi_x_out = self.phi_x_torso(m_ij)
        # phi_x_out = hk.Linear(n_vectors, w_init=w_init)(phi_x_out)
        # if self.normalize:
        #     norm = self.norm_constant + jax.lax.stop_gradient(lengths_x)[:, :, None]
        #     coord_diff_x /= norm
        # shifts_ij = coord_diff_x * phi_x_out[:, :, None]
        # shifts_i = e3nn.scatter_sum(data=shifts_ij, dst=receivers, output_size=n_nodes)
        # residual_positions = shifts_i / avg_num_neighbours
        # chex.assert_equal_shape((residual_positions, node_positions))

        self.phi_y_torso = MLP(
            self.mlp_units, activate_final=True, activation=self.activation_fn
        )
        w_init = jnp.zeros
        phi_y_out = self.phi_y_torso(m_ij)
        phi_y_out = hk.Linear(n_vectors, w_init=w_init)(phi_y_out)
        if self.normalize:
            norm = self.norm_constant + jax.lax.stop_gradient(lengths_y)[:, :, None]
            coord_diff_y /= norm
        shifts_ij = coord_diff_y * phi_y_out[:, :, None]
        shifts_i = e3nn.scatter_sum(data=shifts_ij, dst=receivers, output_size=n_nodes)
        residual_vectors = shifts_i / avg_num_neighbours

        # Get feature output
        if self.attention:
            e_ij = self.phi_inf(m_ij)
            m_ij *= e_ij
        m_i = e3nn.scatter_sum(data=m_ij, dst=receivers, output_size=n_nodes)
        m_i /= avg_num_neighbours
        # m_i /= jnp.sqrt(avg_num_neighbours)
        phi_h_in = jnp.concatenate([m_i, node_features], axis=-1)
        self.phi_h = MLP(
            (*self.mlp_units, self.n_invariant_feat_hidden),
            activate_final=False,
            activation=self.activation_fn,
        )
        residual_features = self.phi_h(phi_h_in)
        chex.assert_equal_shape((residual_features, node_features))

        # Final processing and conversion into plain arrays.

        if self.residual_h:
            features_out = node_features + residual_features
        skip = residual_vectors
        # skip = residual_vectors + residual_positions
        if self.residual_y:
            vectors_out = node_vectors + skip
        # vectors_out = node_vectors + vectors_out + positions_out
        if self.residual_x:
            positions_out = node_positions
            # positions_out = node_positions + residual_positions
        return positions_out, vectors_out, features_out, skip


@dataclasses.dataclass
class EGNNScore:
    hidden_dim: int
    # out_node_dim: int
    act_fn: str = "silu"
    n_layers: int = 4
    residual_x: bool = True
    residual_y: bool = True
    residual_h: bool = True
    attention: bool = False
    normalize: bool = False
    # tanh: bool = False
    # coords_agg: str = "mean"
    norm_constant: int = 0
    num_heads: int = 0
    k: int = 0

    @check_shapes(
        "t: [batch_size]",
        "x: [batch_size, num_points, input_dim]",
        "y: [batch_size, num_points, output_dim]",
        "return: [batch_size, num_points, output_dim]",
    )
    def __call__(self, x, y, t):
        # return jax.vmap(self.score)(x, y)
        node_features = y
        vectors = y[:, :, None, :]

        # positions = x[:, :, None, :]
        positions = x[:, :, None, :]

        n_nodes, vec_multiplicity_in, dim = positions.shape[-3:]

        if self.k > 0:
            senders, receivers = get_edges_knn(x[0], self.k)
            print("senders / receivers", senders.shape, receivers.shape)
        elif self.k == 0:
            # edges = get_edges_knn(x, 20)
            # print("edges", edges[0].shape, edges[1].shape)
            senders, receivers = get_senders_and_receivers_fully_connected(n_nodes)
            # senders, receivers = get_edges_batch(x.shape[-2], batch_size=1)[0]
            print("senders / receivers", senders.shape, receivers.shape)
        elif self.k < 0:
            receivers_k, senders_k = get_edges_knn(x, abs(self.k))
            edges = get_edges_batch(x.shape[-2], batch_size=1)[0]
            receivers, senders = edges
            key = jax.random.PRNGKey(0)
            points_perm = jax.random.permutation(key, jnp.arange(len(edges[0])))[
                :10_000
            ]
            receivers = jnp.concatenate((receivers[points_perm], receivers_k), axis=0)
            senders = jnp.concatenate((senders[points_perm], senders_k), axis=0)
            print("senders / receivers", senders.shape, receivers.shape)
        else:
            raise NotImplementedError()

        x, y, scalars = hk.vmap(
            self.call_single, split_rng=False, in_axes=(0, 0, 0, 0, None, None)
        )(positions, vectors, node_features, t, senders, receivers)
        # print("x, y, scalars", x.shape, y.shape, scalars.shape)
        # return x.squeeze(-2)
        return y.squeeze(-2)
        # return scalars

    @check_shapes(
        "t: []",
        "x: [num_points, mul, input_dim]",
        "y: [num_points, mul, output_dim]",
        # "return: [num_points, output_dim]",
    )
    def call_single(self, x, y, h, t, senders, receivers):
        chex.assert_rank(x, 3)
        chex.assert_rank(h, 2)

        # self.n_invariant_feat_out = y.shape[-1]
        self.n_invariant_feat_out = self.hidden_dim
        # self.n_invariant_feat_out = self.hidden_dim // 2

        node_features = h
        # positions = x[:, None, :]
        act_fn = get_activation(self.act_fn)
        n_vectors_hidden_per_vec_in = 1

        # Create n-multiplicity copies of h and vectors.
        x = jnp.repeat(x, n_vectors_hidden_per_vec_in, axis=1)
        y = jnp.repeat(y, n_vectors_hidden_per_vec_in, axis=1)
        initial_x = x
        initial_y = y
        h = hk.Linear(self.n_invariant_feat_out)(node_features)

        t = jnp.broadcast_to(t[None, ...], (x.shape[0]))
        t_embedding = timestep_embedding(t, self.hidden_dim).squeeze()

        # Loop through torso layers.
        skip = None
        for i in range(self.n_layers):
            h = h + hk.Linear(self.n_invariant_feat_out)(t_embedding)
            x, y, h, skip_connection = EGCL(
                name=f"EGCL_{i}_",
                mlp_units=[self.hidden_dim] * 2,
                n_invariant_feat_hidden=self.n_invariant_feat_out,
                activation_fn=act_fn,
                residual_h=self.residual_h,
                residual_x=self.residual_x,
                residual_y=self.residual_y,
                normalize=self.normalize,
                norm_constant=self.norm_constant,
                attention=self.attention,
                variance_scaling_init=0.001,
                cross_multiplicty_node_feat=False,
                cross_multiplicity_shifts=False,
                norm_wrt_centre_feat=False,
            )(x, y, h, senders, receivers)
            skip = skip_connection if skip is None else skip_connection + skip

        # torso = build_torso(self.name, self.nets_config, self.n_equivariant_vectors_out, vec_multiplicity_in)
        # vectors, h = torso(x, h, senders, receivers)

        if self.residual_x:
            x = x - initial_x
        # if self.residual_y:
        #     y = y - initial_y
        y = skip

        # final_layer_h = hk.Linear(
        #     self.n_invariant_feat_out, w_init=jnp.zeros, b_init=jnp.zeros
        # )
        # final_layer_h = hk.Linear(self.n_invariant_feat_out)
        # h = final_layer_h(h)
        return x, y, h


# @dataclasses.dataclass
# class MLPScore:
#     hidden_dim: int
#     # out_node_dim: int
#     act_fn: str = "silu"
#     n_layers: int = 4
#     residual_x: bool = True
#     residual_h: bool = True
#     attention: bool = False
#     normalize: bool = False
#     tanh: bool = False
#     coords_agg: str = "mean"
#     norm_constant: int = 0
#     num_heads: int = 0
#     k: int = 0

#     @check_shapes(
#         "x: [batch_size, num_points, input_dim]",
#         "y: [batch_size, num_points, output_dim]",
#         "return: [batch_size, num_points, output_dim]",
#     )
#     def __call__(self, x, y):
#         return jax.vmap(self.score)(x, y)

#     @check_shapes(
#         "x: [num_points, input_dim]",
#         "y: [num_points, output_dim]",
#         "return: [num_points, output_dim]",
#     )
#     def score(self, x, y):
#         n, y_dim = y.shape
#         act_fn = get_activation(self.act_fn)
#         mlp = hk.nets.MLP(
#             [self.hidden_dim] * self.n_layers + [n * y_dim],
#             activation=act_fn,
#             activate_final=False,
#         )
#         input = jnp.concatenate([flatten(x), flatten(y)], axis=-1)
#         out = mlp(input)
#         return unflatten(out, y_dim)
