import functools
from functools import partial

from check_shapes import check_shape as cs, check_shapes

import jax
import jax.numpy as jnp
import jax.nn as jnn

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
