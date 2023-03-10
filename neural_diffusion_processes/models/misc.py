import functools

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
