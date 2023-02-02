import dataclasses

from functools import partial
import jax
import jax.numpy as jnp
from check_shapes import check_shapes

from .types import Array, Scalar


@dataclasses.dataclass(frozen=True)
class SquaredExpontialKernel:
    """Radial basis functions (RBF) kernel."""
    variance: Scalar = 1.0
    lengthscale: Scalar = 1.0

    @partial(jax.jit, static_argnums=0)
    @check_shapes("x: [num_points, input_dim]", "return: [num_points, num_points]")
    def __call__(self, x: Array) -> Array:

        def se(x1, x2):
            c = (x1 - x2) / self.lengthscale
            return self.variance * jnp.exp(- c**2 / 2)

        return self.variance * jax.vmap(lambda x1: jax.vmap(lambda x2: se(x1, x2))(x))(x)[..., 0]


@dataclasses.dataclass(frozen=True)
class WhiteKernel:
    variance: Scalar = 1.0

    @partial(jax.jit, static_argnums=0)
    @check_shapes("x: [num_points, input_dim]", "return: [num_points, num_points]")
    def __call__(self, x: Array) -> Array:
        return jnp.eye(jnp.shape(x)[0]) * self.variance
