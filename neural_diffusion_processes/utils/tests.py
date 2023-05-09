from __future__ import annotations

from typing import Any, Callable, Tuple

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from einops import rearrange


def permute(
    key: Any | jr.PRNGKey,
    a: jnp.ndarray,
    axis: int,
    shuffled_inds: jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Permutes a tensor `a` along dimension `axis`. Returns the permuted tensor as well
    as the indices. When `shuffled_inds` is None, generates a new permutation,
    otherwise reuses the indices.
    """
    if shuffled_inds is None:
        len_axis = jnp.shape(a)[axis]
        inds = jnp.arange(len_axis, dtype=int)
        shuffled_inds = jr.shuffle(key, inds)  # Random shuffle

    return tfp.tf2jax.gather(a, shuffled_inds, axis=axis), shuffled_inds


def _check_permutation_invariance(
    key: jr.PRNGKey, f: Callable, axis: int, *args: jnp.ndarray
):
    """
    Tests that shuffling the input `args` along `axis` does not change the output of `f`.
    """
    permuted_args = []
    indices = None
    for arg in args:
        arg_p, indices = permute(key, arg, axis=axis, shuffled_inds=indices)
        permuted_args.append(arg_p)

    outputs_original = f(*args)
    outputs_permuted_inputs = f(*permuted_args)

    if not isinstance(outputs_original, tuple):
        outputs_original = (outputs_original,)
        outputs_permuted_inputs = (outputs_permuted_inputs,)

    for output_orginal, output_permuted_inputs in zip(
        outputs_original, outputs_permuted_inputs
    ):
        # we assume invariance: the output should not be affected by shuffled inputs
        np.testing.assert_array_almost_equal(output_orginal, output_permuted_inputs)


def _check_permutation_equivariance(
    key: jr.PRNGKey, f: Callable, axis: int, output_axis: int, *args: jnp.ndarray
):
    """
    Tests that shuffling the input `args` along `axis` changes the output of `f`. The output
    of `f` will be shuffled along `output_axis` in the same way as the shuffled inputs.
    """

    key1, key2 = jr.split(key)
    permuted_args = []
    indices = None
    for arg in args:
        arg_p, indices = permute(key1, arg, axis=axis, shuffled_inds=indices)
        permuted_args.append(arg_p)

    outputs_original = f(*args)
    outputs_permuted_inputs = f(*permuted_args)

    if not isinstance(outputs_original, tuple):
        outputs_original = (outputs_original,)
        outputs_permuted_inputs = (outputs_permuted_inputs,)

    for output_orginal, output_permuted_inputs in zip(
        outputs_original, outputs_permuted_inputs
    ):
        output_orginal_permuted, _ = permute(
            key2, output_orginal, axis=output_axis, shuffled_inds=indices
        )
        # we assume equivariance: the output should affected by shuffled inputs in the same
        # way as shuffling the output itself.
        np.testing.assert_allclose(
            actual=output_permuted_inputs,
            desired=output_orginal_permuted,
            atol=3e-5,
            rtol=1e-2,
        )


def _check_e3_equivariance(
    key: jr.PRNGKey,
    f: Callable,
    x: jnp.ndarray,
    y: jnp.ndarray,
    t,
    irreps_input=e3nn.Irreps("1e"),
    irreps_output=e3nn.Irreps("1e"),
):
    """ """
    y_dim = irreps_input.dim

    key, next_rng = jax.random.split(key)
    R = e3nn.rand_matrix(next_rng, ())
    if irreps_input == e3nn.Irreps("1e"):
        D_in = R
    else:
        D_in = irreps_input.D_from_matrix(R)
    if irreps_output == e3nn.Irreps("1e"):
        D_out = R
    else:
        D_out = irreps_output.D_from_matrix(R)

    x_transformed = x @ D_in.T
    y_transformed = rearrange(
        rearrange(y, "... (n d) -> ... n d", d=y_dim) @ D_in.T, "... n d -> ... (n d)"
    )

    out_transformed = f(x, y, t) @ D_out.T
    out = f(x_transformed, y_transformed, t)
    print(out_transformed.shape, out_transformed[0][:2])
    print(out.shape, out[0][:2])
    assert jnp.allclose(out_transformed, out)


def _check_e2_equivariance(
    key: jr.PRNGKey,
    f: Callable,
    x: jnp.ndarray,
    y: jnp.ndarray,
    t,
    irreps_input=e3nn.Irreps("1e"),
    irreps_output=e3nn.Irreps("1e"),
):
    """ """
    x_dim, y_dim = x.shape[-1], y.shape[-1]
    if x_dim == 2:
        x = jnp.concatenate([x, jnp.zeros((*x.shape[:-1], 1), dtype=x.dtype)], -1)
    if y_dim == 2:
        y = jnp.concatenate([y, jnp.zeros((*y.shape[:-1], 1), dtype=y.dtype)], -1)
    y_dim = irreps_output.dim
    # set last component to 0 to have '2d vector'
    # x = x.at[:, :, -1].set(jnp.zeros_like(x[:, :, -1]))
    # y = y.at[:, :, -1].set(jnp.zeros_like(y[:, :, -1]))

    # sample element of O(2) embedded in O(3)
    key, next_rng = jax.random.split(key)
    theta = jax.random.uniform(next_rng, (), maxval=2 * jnp.pi)
    R = jnp.array(
        [
            [jnp.cos(theta), -jnp.sin(theta), 0],
            [jnp.sin(theta), jnp.cos(theta), 0],
            [0, 0, 1],
        ],
    )
    # NOTE: calling e3nn.Irreps is not preserving the off diagonal zeros...
    if irreps_input == e3nn.Irreps("1e"):
        D_in = R
    else:
        D_in = irreps_input.D_from_matrix(R)
    if irreps_output == e3nn.Irreps("1e"):
        D_out = R
    else:
        D_out = irreps_output.D_from_matrix(R)

    print(R.shape, R)
    # print(D_in.shape, D_in)
    # print(D_out.shape, D_out)
    print(x.shape, x[0][:2])
    x_transformed = x @ D_in.T
    y_transformed = rearrange(
        rearrange(y, "... (n d) -> ... n d", d=y_dim) @ D_in.T, "... n d -> ... (n d)"
    )
    print(x_transformed.shape, x_transformed[0][:2])
    print(y_transformed.shape, y_transformed[0][:2])
    assert jnp.allclose(x_transformed[..., -1], jnp.zeros_like(x[..., -1]))
    assert jnp.allclose(y_transformed[..., -1], jnp.zeros_like(y[..., -1]))

    out = f(x, y, t)
    out_transformed = f(x, y, t) @ D_out.T
    out_pre_transformed = f(x_transformed, y_transformed, t)
    print("out", out.shape, out[0][:2])
    print("out_transformed", out_transformed.shape, out_transformed[0][:2])
    print("out_pre_transformed", out_pre_transformed.shape, out_pre_transformed[0][:2])
    assert jnp.allclose(out_transformed, out_pre_transformed, atol=1e-5, rtol=1e-5)
