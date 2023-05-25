import functools
from functools import partial
from typing import Callable, NamedTuple, Optional, Sequence, Tuple

import chex
import jax
import jax.nn as jnn
import jax.numpy as jnp
from check_shapes import check_shape as cs
from check_shapes import check_shapes

from neural_diffusion_processes.utils import register_category


@check_shapes(
    "t: [batch_size]",
    "return: [batch_size, embedding_dim]",
)
def timestep_embedding(t: jnp.ndarray, embedding_dim: int, max_positions: int = 10_000):
    """Sinusoidal embedding"""
    half_dim = embedding_dim // 2
    emb = jnp.log(max_positions) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jnp.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape == (t.shape[0], embedding_dim)
    return emb


def scatter(input, dim, index, src, reduce=None):
    # Works like PyTorch's scatter. See https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html

    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )

    if reduce is None:
        _scatter = jax.lax.scatter
    elif reduce == "add":
        _scatter = jax.lax.scatter_add
    elif reduce == "multiply":
        _scatter = jax.lax.scatter_mul

    _scatter = partial(_scatter, dimension_numbers=dnums)
    vmap_inner = partial(jax.vmap, in_axes=(0, 0, 0), out_axes=0)
    vmap_outer = partial(jax.vmap, in_axes=(1, 1, 1), out_axes=1)

    for idx in range(len(input.shape)):
        if idx == dim:
            pass
        elif idx < dim:
            _scatter = vmap_inner(_scatter)
        else:
            _scatter = vmap_outer(_scatter)

    return _scatter(input, jnp.expand_dims(index, axis=-1), src)


get_activation, register_activation = register_category("activation")

register_activation(jnn.gelu, name="gelu")
register_activation(jnn.elu, name="elu")
register_activation(jnn.relu, name="relu")
register_activation(
    functools.partial(jnn.leaky_relu, negative_slope=0.01), name="lrelu"
)
register_activation(jnn.swish, name="swish")
register_activation(jnp.sin, name="sin")
register_activation(jax.nn.silu, name="silu")



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


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = jnp.ones((len(edges[0]) * batch_size, 1))
    # edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    edges = [jnp.array(edges[0]), jnp.array(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [jnp.concatenate(rows), jnp.concatenate(cols)]
    return edges, edge_attr



