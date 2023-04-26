from __future__ import annotations

from typing import Any, Callable, Tuple

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import tensorflow_probability.substrates.jax as tfp


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
