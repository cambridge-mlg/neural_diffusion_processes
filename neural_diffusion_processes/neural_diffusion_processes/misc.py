import jax
import jax.numpy as jnp

from check_shapes import check_shapes

from .types import Array
from .constants import JITTER


@check_shapes(
    "mean: [num_points, 1]",
    "cov: [num_points, num_points]",
    "return: [num_samples, num_points, 1]"
)
def sample_mvn(key, mean: Array, cov: Array, num_samples: int = 10):
    """Returns samples from a GP(mean, kernel) at x."""
    L = jnp.linalg.cholesky(cov + JITTER * jnp.eye(len(mean)))
    eps = jax.random.normal(key, (len(mean), num_samples))
    s = mean + L @ eps
    return jnp.transpose(s)[..., None]