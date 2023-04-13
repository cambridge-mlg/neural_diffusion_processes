from typing import Optional, Iterator

import math
import jax
import jax.numpy as jnp
from jax import lax

from check_shapes import check_shapes
from einops import rearrange

from .types import Array
from ..config import get_config

from jaxtyping import Float, jaxtyped
from typeguard import typechecked as typechecker

def check_shape(func):
    return typechecker(jaxtyped(func))

@check_shapes(
    "mean: [num_points, 1]",
    "cov: [num_points, num_points]",
    "return: [num_samples, num_points, 1] if num_samples",
    "return: [num_points, 1] if not num_samples",
)
def sample_mvn(key, mean: Array, cov: Array, num_samples: Optional[int] = None, noise_var=None):
    """Returns samples from a GP(mean, kernel) at x."""
    num_samples_was_none = num_samples is None
    num_samples = num_samples or 1
    diag = noise_var or get_config().jitter
    L = jnp.linalg.cholesky(cov + noise_var * jnp.eye(len(mean)))
    eps = jax.random.normal(key, (len(mean), num_samples), dtype=mean.dtype)
    s = mean + L @ eps
    s = jnp.transpose(s)[..., None]
    if num_samples_was_none:
        return s[0]
    else:
        return s


def flatten(y):
    return rearrange(y, "... n d -> ... (n d)")


def unflatten(y, d):
    return rearrange(y, "... (n d) -> ... n d", d=d)


def get_key_iter(init_key) -> Iterator["jax.random.PRNGKey"]:
    while True:
        init_key, next_key = jax.random.split(init_key)
        yield next_key
    

def generate_logarithmic_sequence(end, L):
    base = math.exp(math.log(end) / (L-1))
    sequence = [0]
    current = base
    while len(sequence) < L:
        current = int(current)
        sequence.append(current)
        current *= base
    sequence.append(end - 1)
    return sequence

def jax_unstack(x, axis=0):
  return [lax.index_in_dim(x, i, axis, keepdims=False) for i in range(x.shape[axis])]
