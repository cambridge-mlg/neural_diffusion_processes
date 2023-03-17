from __future__ import annotations
from typing import Tuple
from dataclasses import dataclass

import math
import haiku as hk
from einops import rearrange, reduce
from check_shapes import check_shape as cs, check_shapes

import jax
import jax.numpy as jnp

from .misc import timestep_embedding


@check_shapes(
    "q: [batch..., seq_len_q, depth]",
    "k: [batch..., seq_len_k, depth]",
    "v: [batch..., seq_len_k, depth_v]",
    "return[0]: [batch..., seq_len_q, depth_v]",
    "return[1]: [batch..., seq_len_q, seq_len_k]",
)
def scaled_dot_product_attention(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate the attention weights.
    Returns:
      output, attention_weights
    """

    matmul_qk = cs(
        jnp.einsum("...qd,...kd->...qk", q, k), "[batch..., seq_len_q, seq_len_k]"
    )

    # scale matmul_qk
    depth = jnp.shape(k)[-1] * 1.0
    scaled_attention_logits = matmul_qk / jnp.sqrt(depth)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += mask * -1e9

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = jax.nn.softmax(
        scaled_attention_logits, axis=-1
    )  # (..., seq_len_q, seq_len_k)

    output = cs(
        jnp.einsum("...qk,...kd->...qd", attention_weights, v),
        "[batch..., seq_len_q, depth_v]",
    )

    return output, attention_weights


class MultiHeadAttention(hk.Module):
    def __init__(self, d_model: int, num_heads: int, name: str | None = None):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

    @check_shapes(
        "v: [batch..., seq_len_k, dim_v]",
        "k: [batch..., seq_len_k, dim_k]",
        "q: [batch..., seq_len_q, dim_q]",
        "return: [batch..., seq_len_q, hidden_dim] if not return_attention_weights",
        "return[0]: [batch..., seq_len_q, hidden_dim] if return_attention_weights",
        "return[1]: [batch..., num_heads, seq_len_q, seq_len_k] if return_attention_weights",
    )
    def __call__(self, v, k, q, mask=None, return_attention_weights: bool = False):
        q = hk.Linear(output_size=self.d_model)(q)  # (batch_size, seq_len, d_model)
        k = hk.Linear(output_size=self.d_model)(k)  # (batch_size, seq_len, d_model)
        v = hk.Linear(output_size=self.d_model)(v)  # (batch_size, seq_len, d_model)

        rearrange_arg = "... seq_len (num_heads depth) -> ... num_heads seq_len depth"
        q = rearrange(q, rearrange_arg, num_heads=self.num_heads, depth=self.depth)
        k = rearrange(k, rearrange_arg, num_heads=self.num_heads, depth=self.depth)
        v = rearrange(v, rearrange_arg, num_heads=self.num_heads, depth=self.depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )

        scaled_attention = rearrange(
            scaled_attention,
            "... num_heads seq_len depth -> ... seq_len (num_heads depth)",
        )
        output = hk.Linear(output_size=self.d_model)(
            scaled_attention
        )  # (batch_size, seq_len_q, d_model)

        if return_attention_weights:
            return output, attention_weights
        else:
            return output


@dataclass
class BiDimensionalAttentionBlock(hk.Module):
    hidden_dim: int
    num_heads: int

    @check_shapes(
        "s: [batch_size, num_points, input_dim, hidden_dim]",
        "t: [batch_size, hidden_dim]",
        "return[0]: [batch_size, num_points, input_dim, hidden_dim]",
        "return[1]: [batch_size, num_points, input_dim, hidden_dim]",
    )
    def __call__(
        self, s: jnp.ndarray, t: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Bi-dimensional attention block. Main computation block in the NDP noise model.
        """
        t = cs(
            hk.Linear(self.hidden_dim)(t)[:, None, None, :],
            "[batch_size, 1, 1, hidden_dim]",
        )
        y = cs(s + t, "[batch_size, num_points, input_dim, hidden_dim]")

        y_att_d = MultiHeadAttention(2 * self.hidden_dim, self.num_heads)(y, y, y)
        y_att_d = cs(y_att_d, "[batch_size, num_points, input_dim, hidden_dim_x2]")

        y_r = cs(
            jnp.swapaxes(y, 1, 2), "[batch_size, input_dim, num_points, hidden_dim]"
        )
        y_att_n = MultiHeadAttention(2 * self.hidden_dim, self.num_heads)(y_r, y_r, y_r)
        y_att_n = cs(y_att_n, "[batch_size, input_dim, num_points, hidden_dim_x2]")
        y_att_n = cs(
            jnp.swapaxes(y_att_n, 1, 2),
            "[batch_size, num_points, input_dim, hidden_dim_x2]",
        )

        y = y_att_n + y_att_d

        residual, skip = jnp.split(y, 2, axis=-1)
        residual = jax.nn.gelu(residual)
        skip = jax.nn.gelu(skip)
        return (s + residual) / math.sqrt(2.0), skip


@dataclass
class BiDimensionalAttentionModel(hk.Module):
    n_layers: int
    """Number of bi-dimensional attention blocks."""
    hidden_dim: int
    num_heads: int

    @check_shapes(
        "x: [batch_size, seq_len, input_dim]",
        "y: [batch_size, seq_len, 1]",
        "return: [batch_size, seq_len, input_dim, 2]",
    )
    def process_inputs(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Transform inputs to split out the x dimensions for dimension-agnostic processing.
        """
        num_x_dims = jnp.shape(x)[-1]
        x = jnp.expand_dims(x, axis=-1)
        y = jnp.repeat(jnp.expand_dims(y, axis=-1), num_x_dims, axis=2)
        return jnp.concatenate([x, y], axis=-1)

    @check_shapes(
        "x: [batch_size, num_points, input_dim]",
        "y: [batch_size, num_points, output_dim]",
        "t: [batch_size]",
        "return: [batch_size, num_points, 1]",
    )
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the additive noise that was added to `y_0` to obtain `y_t`
        based on `x_t` and `y_t` and `t`
        """
        x = cs(self.process_inputs(x, y), "[batch_size, num_points, input_dim, 2]")

        x = cs(
            hk.Linear(self.hidden_dim)(x),
            "[batch_size, num_points, input_dim, hidden_dim]",
        )
        x = jax.nn.gelu(x)

        t_embedding = timestep_embedding(t, self.hidden_dim)

        skip = None
        for _ in range(self.n_layers):
            layer = BiDimensionalAttentionBlock(self.hidden_dim, self.num_heads)
            x, skip_connection = layer(x, t_embedding)
            skip = skip_connection if skip is None else skip_connection + skip

        x = cs(x, "[batch_size, num_points, input_dim, hidden_dim]")
        skip = cs(skip, "[batch_size, num_points, input_dim, hidden_dim]")

        skip = cs(
            reduce(skip, "b n d h -> b n h", "mean"), "[batch, num_points, hidden_dim]"
        )

        eps = skip / math.sqrt(self.n_layers * 1.0)
        eps = jax.nn.gelu(hk.Linear(self.hidden_dim)(eps))
        eps = hk.Linear(1, w_init=jnp.zeros)(eps)
        return eps


@dataclass
class MultiOutputAttentionModel(hk.Module):
    def __init__(
        self,
        n_layers: int,  # Number of bi-dimensional attention blocks
        hidden_dim: int,
        num_heads: int,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = n_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

    def __post_init__(self):
        print(">>>>>>>>>> AttentionModel")

    @check_shapes(
        "x: [batch_size, seq_len, x_dim]",
        "y: [batch_size, seq_len, y_dim]",
        "return: [batch_size, seq_len, x_dim__times__y_dim, 2]",
    )
    def process_inputs(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Transform inputs to split out the x dimensions for dimension-agnostic processing.
        """
        x_dim, y_dim = x.shape[-1], y.shape[-1]
        x = cs(x[..., None], "[batch_size, seq_len, x_dim, 1]")
        x = cs(jnp.repeat(x, y_dim, axis=-1), "[batch_size, seq_len, x_dim, y_dim]")
        y = cs(y[..., None, :], "[batch_size, seq_len, 1, y_dim]")
        y = cs(jnp.repeat(y, x_dim, axis=-2), "[batch_size, seq_len, x_dim, y_dim]")
        out = jnp.concatenate([x[..., None], y[..., None]], axis=-1)
        out = cs(out, "[batch_size, seq_len, x_dim, y_dim, 2]")
        out = rearrange(out, "... n x_dim y_dim h -> ... n (x_dim y_dim) h")
        return out

    @check_shapes(
        "x: [batch_size, num_points, x_dim]",
        "y: [batch_size, num_points, y_dim]",
        "t: [batch_size]",
        "return: [batch_size, num_points, y_dim]",
    )
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the additive noise that was added to `y_0` to obtain `y_t`
        based on `x_t` and `y_t` and `t`
        """
        x_dim = x.shape[-1]
        x = cs(
            self.process_inputs(x, y),
            "[batch_size, num_points, x_dim__times__y_dim, 2]",
        )

        x = cs(
            hk.Linear(self.hidden_dim)(x),
            "[batch_size, num_points, x_dim__times__y_dim, hidden_dim]",
        )
        x = jax.nn.gelu(x)

        t_embedding = timestep_embedding(t, self.hidden_dim)

        skip = None
        for _ in range(self.num_layers):
            layer = BiDimensionalAttentionBlock(self.hidden_dim, self.num_heads)
            x, skip_connection = layer(x, t_embedding)
            skip = skip_connection if skip is None else skip_connection + skip

        x = cs(x, "[batch_size, num_points, x_dim__times__y_dim, hidden_dim]")
        skip = cs(skip, "[batch_size, num_points, x_dim__times__y_dim, hidden_dim]")
        skip = rearrange(
            skip, "... n (x_dim y_dim) h -> ... n x_dim y_dim h", x_dim=x_dim
        )

        skip = cs(
            reduce(skip, "b n x_dim y_dim h -> b n y_dim h", "mean"),
            "[batch, num_points, y_dim, hidden_dim]",
        )
        skip = skip / math.sqrt(self.num_layers * 1.0)
        eps = jax.nn.gelu(hk.Linear(self.hidden_dim)(skip))
        eps = hk.Linear(1, w_init=jnp.zeros)(eps)
        eps = cs(jnp.squeeze(eps, -1), "[batch, num_points, y_dim]")
        return eps
