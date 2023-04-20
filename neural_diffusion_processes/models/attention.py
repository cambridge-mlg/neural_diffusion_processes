from __future__ import annotations
from typing import Tuple
from dataclasses import dataclass
import functools

import math
import haiku as hk
from einops import rearrange, reduce
from check_shapes import (
    check_shape as cs,
    check_shapes,
    set_enable_function_call_precompute,
)

set_enable_function_call_precompute(True)
import jax
import jax.numpy as jnp

from .misc import timestep_embedding


def _query_chunk_attention(
    query_idx,
    query,
    key,
    value,
    mask,
    bias,
    precision,
    key_chunk_size=4096,
    mask_calc_fn=None,
    bias_calc_fn=None,
    weights_calc_fn=None,
    calc_fn_data=None,
):
    num_kv, num_heads, k_features = key.shape[-3:]
    v_features = value.shape[-1]
    num_q = query.shape[-3]
    key_chunk_size = min(key_chunk_size, num_kv)
    query = query / jnp.sqrt(k_features)

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk(chunk_idx, query, key, value, mask, bias):
        attn_weights = jnp.einsum(
            "...qhd,...khd->...qhk", query, key, precision=precision
        )
        if bias_calc_fn is not None:
            bias = bias_calc_fn(query_idx, chunk_idx, bias, attn_weights, calc_fn_data)
        if bias is not None:
            bias = jnp.einsum("...hqk->...qhk", bias)
            attn_weights = attn_weights + bias
        if mask_calc_fn is not None:
            mask = mask_calc_fn(query_idx, chunk_idx, mask, attn_weights, calc_fn_data)
        if mask is not None:
            big_neg = jnp.finfo(attn_weights.dtype).min
            mask = jnp.einsum("...hqk->...qhk", mask)
            attn_weights = jnp.where(mask, attn_weights, big_neg)
        if weights_calc_fn is not None:
            attn_weights = weights_calc_fn(
                query_idx, chunk_idx, attn_weights, calc_fn_data
            )
        max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
        max_score = jax.lax.stop_gradient(max_score)
        exp_weights = jnp.exp(attn_weights - max_score)
        exp_values = jnp.einsum(
            "...vhf,...qhv->...qhf", value, exp_weights, precision=precision
        )
        max_score = jnp.einsum("...qhk->...qh", max_score)
        return exp_values, exp_weights.sum(axis=-1), max_score

    def chunk_scanner(chunk_idx):
        key_chunk = jax.lax.dynamic_slice(
            key,
            tuple([0] * (key.ndim - 3)) + (chunk_idx, 0, 0),
            slice_sizes=tuple(key.shape[:-3]) + (key_chunk_size, num_heads, k_features),
        )
        value_chunk = jax.lax.dynamic_slice(
            value,
            tuple([0] * (value.ndim - 3)) + (chunk_idx, 0, 0),
            slice_sizes=tuple(value.shape[:-3])
            + (key_chunk_size, num_heads, v_features),
        )

        if bias is None:
            bias_chunk = None
        elif bias.shape[-1] == 1:
            bias_chunk = bias
        elif bias.shape[-1] == num_kv:
            bias_chunk = jax.lax.dynamic_slice(
                bias,
                tuple([0] * (bias.ndim - 3)) + (0, 0, chunk_idx),
                slice_sizes=tuple(bias.shape[:-3])
                + (bias.shape[-3], bias.shape[-2], key_chunk_size),
            )
        else:
            raise TypeError(
                f"bias.shape[-1] == {bias.shape[-1]} must broadcast with key.shape[-3] == {num_kv}"
            )

        if mask is None:
            mask_chunk = None
        elif bias.shape[-1] == 1:
            mask_chunk = mask
        elif mask.shape[-1] == num_kv:
            mask_chunk = jax.lax.dynamic_slice(
                mask,
                tuple([0] * (mask.ndim - 3)) + (0, 0, chunk_idx),
                slice_sizes=tuple(mask.shape[:-3])
                + (mask.shape[-3], mask.shape[-2], key_chunk_size),
            )
        else:
            raise TypeError(
                f"mask.shape[-1] == {mask.shape[-1]} must broadcast with key.shape[-3] == {num_kv}"
            )

        return summarize_chunk(
            chunk_idx, query, key_chunk, value_chunk, mask_chunk, bias_chunk
        )

    chunk_values, chunk_weights, chunk_max = jax.lax.map(
        chunk_scanner, xs=jnp.arange(0, num_kv, key_chunk_size)
    )

    global_max = jnp.max(chunk_max, axis=0, keepdims=True)
    max_diffs = jnp.exp(chunk_max - global_max)
    chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(axis=0)
    all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis=0)
    return all_values / all_weights


def efficient_dot_product_attention(
    query,
    key,
    value,
    mask=None,
    bias=None,
    precision=jax.lax.Precision.HIGHEST,
    query_chunk_size=1024,
    key_chunk_size=4096,
    bias_calc_fn=None,
    mask_calc_fn=None,
    weights_calc_fn=None,
    calc_fn_data=None,
):
    """Computes efficient dot-product attention given query, key, and value.
    This is efficient version of attention presented in
    https://arxiv.org/abs/2112.05682v2 which comes with O(sqrt(n)) memory requirements.
    Note: query, key, value needn't have any batch dimensions.
    Args:
      query: queries for calculating attention with shape of
        `[batch..., q_length, num_heads, qk_depth_per_head]`.
      key: keys for calculating attention with shape of
        `[batch..., kv_length, num_heads, qk_depth_per_head]`.
      value: values to be used in attention with shape of
        `[batch..., kv_length, num_heads, v_depth_per_head]`.
      bias: bias for the attention weights. This should be broadcastable to the
        shape `[batch..., num_heads, q_length, kv_length]`.
        This can be used for incorporating padding masks, proximity bias, etc.
      mask: mask for the attention weights. This should be broadcastable to the
        shape `[batch..., num_heads, q_length, kv_length]`.
        Attention weights are masked out if their corresponding mask value
        is `False`.
      query_chunk_size: int: query chunks size
      key_chunk_size: int: key chunks size
      bias_calc_fn: a bias calculation callback for each chunk, of form
        `(q_offset, k_offset, bias_chunk, attn_weights, calc_fn_data) -> bias`.
        This can be used for incorporating causal masks, padding masks,
        proximity bias, etc.
      mask_calc_fn: a mask calculation callback for each chunk, of form
        `(q_offset, k_offset, mask_chunk, attn_weights, calc_fn_data) -> mask`.
        This can be used for incorporating causal or other large masks.
        Attention weights are masked out if their corresponding mask value
        is `False`.
      weights_calc_fn: a general attn_weights callback for each chunk, of form
        `(q_offset, k_offset, attn_weights, calc_fn_data) -> attn_weights`.
        attn_weights has shape of
        `[batch..., q_chunk_size, num_heads, k_chunk_size]`.
        This can be used to implement complex weights processing in a memory
        efficient way.
      calc_fn_data: optional pure data to pass to each per-chunk call of
        bias_calc_fn, mask_calc_fn, and weights_calc_fn.
      precision: numerical precision of the computation see `jax.lax.Precision`
              for details.
    Returns:
      Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
    """
    num_q, num_heads, q_features = query.shape[-3:]
    num_kv = key.shape[-3]

    def chunk_scanner(chunk_idx, _):
        query_chunk = jax.lax.dynamic_slice(
            query,
            tuple([0] * (query.ndim - 3)) + (chunk_idx, 0, 0),
            slice_sizes=tuple(query.shape[:-3])
            + (min(query_chunk_size, num_q), num_heads, q_features),
        )

        if mask is None:
            mask_chunk = None
        elif mask.shape[-2] == 1:
            mask_chunk = mask
        elif mask.shape[-2] == num_q:
            mask_chunk = jax.lax.dynamic_slice(
                mask,
                tuple([0] * (mask.ndim - 3)) + (0, chunk_idx, 0),
                slice_sizes=tuple(mask.shape[:-3])
                + (mask.shape[-3], min(query_chunk_size, num_q), mask.shape[-1]),
            )
        else:
            raise TypeError(
                f"mask.shape[-2] == {mask.shape[-2]} must broadcast with query.shape[-3] == {num_q}"
            )

        if bias is None:
            bias_chunk = None
        elif mask.shape[-2] == 1:
            bias_chunk = bias
        elif bias.shape[-2] == num_q:
            bias_chunk = jax.lax.dynamic_slice(
                bias,
                tuple([0] * (bias.ndim - 3)) + (0, chunk_idx, 0),
                slice_sizes=tuple(bias.shape[:-3])
                + (bias.shape[-3], min(query_chunk_size, num_q), bias.shape[-1]),
            )
        else:
            raise TypeError(
                f"bias.shape[-2] == {bias.shape[-2]} must broadcast with query.shape[-3] == {num_q}"
            )

        return (
            chunk_idx + query_chunk_size,
            _query_chunk_attention(
                chunk_idx,
                query_chunk,
                key,
                value,
                mask_chunk,
                bias_chunk,
                precision=precision,
                key_chunk_size=key_chunk_size,
                bias_calc_fn=bias_calc_fn,
                mask_calc_fn=mask_calc_fn,
                weights_calc_fn=weights_calc_fn,
                calc_fn_data=calc_fn_data,
            ),
        )

    _, res = jax.lax.scan(
        chunk_scanner, init=0, xs=None, length=math.ceil(num_q / query_chunk_size)
    )
    return jnp.concatenate(res, axis=-3)


@check_shapes(
    "q: [batch..., seq_len_q, depth]",
    "k: [batch..., seq_len_k, depth]",
    "v: [batch..., seq_len_k, depth_v]",
    # "return[0]: [batch..., seq_len_q, depth_v]",
    # "return[1]: [batch..., seq_len_q, seq_len_k]",
    "return: [batch..., seq_len_q, depth_v]",
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

    # return output, attention_weights
    return output


class MultiHeadAttention(hk.Module):
    def __init__(
        self, d_model: int, num_heads: int, sparse: bool = False, name: str = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        self.attention = (
            efficient_dot_product_attention if sparse else scaled_dot_product_attention
        )

    @check_shapes(
        "v: [batch..., seq_len_k, dim_v]",
        "k: [batch..., seq_len_k, dim_k]",
        "q: [batch..., seq_len_q, dim_q]",
        "return: [batch..., seq_len_q, hidden_dim]"
        # "return: [batch..., seq_len_q, hidden_dim] if not return_attention_weights",
        # "return[0]: [batch..., seq_len_q, hidden_dim] if return_attention_weights",
        # "return[1]: [batch..., num_heads, seq_len_q, seq_len_k] if return_attention_weights",
    )
    # def __call__(self, v, k, q, mask=None, return_attention_weights: bool = False):
    def __call__(self, v, k, q, mask=None):
        q = hk.Linear(output_size=self.d_model)(q)  # (batch_size, seq_len, d_model)
        k = hk.Linear(output_size=self.d_model)(k)  # (batch_size, seq_len, d_model)
        v = hk.Linear(output_size=self.d_model)(v)  # (batch_size, seq_len, d_model)

        rearrange_arg = "... seq_len (num_heads depth) -> ... num_heads seq_len depth"
        q = rearrange(q, rearrange_arg, num_heads=self.num_heads, depth=self.depth)
        k = rearrange(k, rearrange_arg, num_heads=self.num_heads, depth=self.depth)
        v = rearrange(v, rearrange_arg, num_heads=self.num_heads, depth=self.depth)

        # scaled_attention, attention_weights = scaled_dot_product_attention(
        scaled_attention = self.attention(q, k, v, mask=mask)

        scaled_attention = rearrange(
            scaled_attention,
            "... num_heads seq_len depth -> ... seq_len (num_heads depth)",
        )
        output = hk.Linear(output_size=self.d_model)(
            scaled_attention
        )  # (batch_size, seq_len_q, d_model)

        return output

        # if return_attention_weights:
        #     return output, attention_weights
        # else:
        #     return output


@dataclass
class BiDimensionalAttentionBlock(hk.Module):
    hidden_dim: int
    num_heads: int
    sparse: bool

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

        y_att_d = MultiHeadAttention(2 * self.hidden_dim, self.num_heads, self.sparse)(
            y, y, y
        )
        y_att_d = cs(y_att_d, "[batch_size, num_points, input_dim, hidden_dim_x2]")

        y_r = cs(
            jnp.swapaxes(y, 1, 2), "[batch_size, input_dim, num_points, hidden_dim]"
        )
        y_att_n = MultiHeadAttention(2 * self.hidden_dim, self.num_heads, self.sparse)(
            y_r, y_r, y_r
        )
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
class AttentionBlock(hk.Module):
    hidden_dim: int
    num_heads: int
    sparse: bool

    @check_shapes(
        "s: [batch_size, num_points, hidden_dim]",
        "t: [batch_size, hidden_dim]",
        "return[0]: [batch_size, num_points, hidden_dim]",
        "return[1]: [batch_size, num_points, hidden_dim]",
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
        # y = cs(s + t, "[batch_size, num_points, hidden_dim]")
        y = cs(s + t.squeeze(1), "[batch_size, num_points, hidden_dim]")

        y_att_d = MultiHeadAttention(2 * self.hidden_dim, self.num_heads, self.sparse)(
            y, y, y
        )
        y_att_d = cs(y_att_d, "[batch_size, num_points, hidden_dim_x2]")

        # y = y_att_n + y_att_d
        y = y_att_d

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
class MultiOutputBiAttentionModel(hk.Module):
    def __init__(
        self,
        n_layers: int,  # Number of bi-dimensional attention blocks
        hidden_dim: int,
        num_heads: int,
        sparse: bool,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = n_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.sparse = sparse

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
            layer = BiDimensionalAttentionBlock(
                self.hidden_dim, self.num_heads, self.sparse
            )
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


@dataclass
class MultiOutputAttentionModel(hk.Module):
    def __init__(
        self,
        n_layers: int,  # Number of bi-dimensional attention blocks
        hidden_dim: int,
        num_heads: int,
        sparse: bool,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = n_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.sparse = sparse

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
        y_dim = y.shape[-1]
        # x = cs(
        #     self.process_inputs(x, y),
        #     "[batch_size, num_points, x_dim__times__y_dim, 2]",
        # )
        x = cs(
            jnp.concatenate([x, y], axis=-1),
            "[batch_size, num_points, x_dim__plus__y_dim]",
        )

        x = cs(
            hk.Linear(self.hidden_dim)(x),
            # "[batch_size, num_points, x_dim__times__y_dim, hidden_dim]",
            "[batch_size, num_points, hidden_dim]",
        )
        x = jax.nn.gelu(x)

        t_embedding = timestep_embedding(t, self.hidden_dim)

        skip = None
        for _ in range(self.num_layers):
            # layer = BiDimensionalAttentionBlock(self.hidden_dim, self.num_heads)
            layer = AttentionBlock(self.hidden_dim, self.num_heads, self.sparse)
            x, skip_connection = layer(x, t_embedding)
            skip = skip_connection if skip is None else skip_connection + skip

        x = cs(x, "[batch_size, num_points, hidden_dim]")
        skip = cs(skip, "[batch_size, num_points, hidden_dim]")
        # skip = rearrange(
        #     skip, "... n (x_dim y_dim) h -> ... n x_dim y_dim h", x_dim=x_dim
        # )

        # skip = cs(
        #     reduce(skip, "b n x_dim y_dim h -> b n y_dim h", "mean"),
        #     "[batch, num_points, y_dim, hidden_dim]",
        # )
        skip = skip / math.sqrt(self.num_layers * 1.0)
        eps = jax.nn.gelu(hk.Linear(self.hidden_dim)(skip))
        # eps = hk.Linear(1, w_init=jnp.zeros)(eps)
        eps = hk.Linear(y_dim, w_init=jnp.zeros)(eps)
        # eps = cs(jnp.squeeze(eps, -1), "[batch, num_points, y_dim]")
        eps = cs(eps, "[batch, num_points, y_dim]")
        return eps
