import dataclasses

from functools import partial
import jax
import jax.numpy as jnp

from check_shapes import check_shapes
from jaxtyping import Float as f, jaxtyped
from typeguard import typechecked as typechecker

from .types import Array, Optional, Union, Tuple, Int


def check_shape(func):
    return typechecker(jaxtyped(func))


from .types import Array


@dataclasses.dataclass(frozen=True)
class SquaredExpontialKernel:
    """Radial basis functions (RBF) kernel."""

    variance: float = 1.0
    lengthscale: float = 1.0

    @partial(jax.jit, static_argnums=0)
    @check_shapes(
        "x1: [input_dim]",
        "x2: [input_dim]",
        "return: []"
    )
    def __call__(self, x1: Array, x2: Array) -> Array:
        x1 = x1 / self.lengthscale
        x2 = x2 / self.lengthscale
        c = jnp.sum((x1 - x2)**2)
        return self.variance * jnp.exp(-0.5 * c)


@dataclasses.dataclass(frozen=True)
class WhiteKernel:
    variance: float = 1.0

    @partial(jax.jit, static_argnums=0)
    @check_shapes(
        "x1: [input_dim]",
        "x2: [input_dim]",
        "return: []"
    )
    def __call__(self, x1: Array, x2: Array) -> Array:
        return jnp.all(jnp.equal(x1, x2)).squeeze() * self.variance


# @check_shape
def gram(
    kernel, x: f[jax.Array, "n d"], y: Optional[f[jax.Array, "m d"]] = None
) -> Union[f[jax.Array, "n m d d"], f[jax.Array, "n m"]]:
    y = x if y is None else y
    return jax.vmap(lambda x1: jax.vmap(lambda y1: kernel(x1, y1))(y))(x)


# @dataclasses.dataclass(frozen=True)
# class WhiteVecKernel:
#     variance: float = 1.0
#     output_dim: Int = 1

#     @partial(jax.jit, static_argnums=0)
#     def __call__(
#         self, x: f[jax.Array, "1 D"], y: f[jax.Array, "1 D"]
#     ) -> f[jax.Array, "D D"]:
#         return jnp.eye(self.output_dim) * self.variance


@dataclasses.dataclass(frozen=True)
class RBFKernel:
    """Radial basis functions (RBF) kernel."""

    variance: float = 1.0
    lengthscale: float = 1.0

    @partial(jax.jit, static_argnums=0)
    def __call__(
        self, x: f[jax.Array, "1 D"], y: f[jax.Array, "1 D"]
    ) -> f[jax.Array, ""]:
        diff = x - y
        sq_dist = (diff**2).sum(axis=-1)
        sq_lengthscale = self.lengthscale**2
        K = self.variance * jnp.exp(-0.5 * sq_dist / sq_lengthscale)
        return K


@dataclasses.dataclass(frozen=True)
class RBFVec:
    variance: float = 1.0
    lengthscale: float = 1.0

    # @check_shape
    @partial(jax.jit, static_argnums=0)
    def __call__(
        self, x: f[jax.Array, "D"], y: f[jax.Array, "D"]
    ) -> f[jax.Array, "D D"]:
        dim = x.shape[-1]
        diff = x - y

        sq_dist = (diff**2).sum(axis=-1)

        I = jnp.eye(dim)
        sq_lengthscale = self.lengthscale**2
        K = self.variance * jnp.exp(-0.5 * sq_dist / sq_lengthscale)

        A = I
        K = A * K

        return K


@dataclasses.dataclass(frozen=True)
class RBFDivFree:
    """Based on the kernels defined in equation (24) in
    "Kernels for Vector-Valued Functions: a Review" by Alvarez et al
    """

    variance: float = 1.0
    lengthscale: float = 1.0

    # @check_shape
    @partial(jax.jit, static_argnums=0)
    def __call__(
        self, x: f[jax.Array, "D"], y: f[jax.Array, "D"]
    ) -> f[jax.Array, "D D"]:
        dim = x.shape[-1]
        diff = x - y

        sq_dist = (diff**2).sum(axis=-1)

        outer_product = jnp.einsum("i,j->ij", diff, diff)
        I = jnp.eye(dim)
        sq_lengthscale = self.lengthscale**2
        K = self.variance * jnp.exp(-0.5 * sq_dist / sq_lengthscale)

        A = I - (outer_product / sq_lengthscale)
        K = A * K

        return K


@dataclasses.dataclass(frozen=True)
class RBFCurlFree:
    """Based on the kernels defined in equation (24) in
    "Kernels for Vector-Valued Functions: a Review" by Alvarez et al
    """

    variance: float = 1.0
    lengthscale: float = 1.0

    # @check_shape
    @partial(jax.jit, static_argnums=0)
    def __call__(
        self, x: f[jax.Array, "D"], y: f[jax.Array, "D"]
    ) -> f[jax.Array, "D D"]:
        dim = x.shape[-1]
        diff = x - y

        sq_dist = (diff**2).sum(axis=-1)

        outer_product = jnp.einsum("i,j->ij", diff, diff)
        I = jnp.eye(dim)
        sq_lengthscale = self.lengthscale**2

        A = outer_product / sq_lengthscale
        A += (dim - 1 - sq_dist / sq_lengthscale)[..., None, None] * I

        K = self.variance * jnp.exp(-0.5 * sq_dist / sq_lengthscale)

        K = A * K

        return K


## Mean functions


@dataclasses.dataclass(frozen=True)
class Constant:
    value: float = 0.0
    output_dim: Optional[Tuple] = None

    def __call__(self, x):
        output_dim = x.shape[-1] if self.output_dim is None else self.output_dim
        return self.value * jnp.ones((*x.shape[:-1], output_dim))


@dataclasses.dataclass(frozen=True)
class Zero(Constant):
    value: float = 0.0
    output_dim: Optional[Tuple] = None


@dataclasses.dataclass(frozen=True)
class Quadratic:
    output_dim: Optional[Tuple] = None
    a: float = 1.0
    b: float = 0.0

    def __call__(self, x):
        # output_dim = x.shape[-1] if self.output_dim is None else self.output_dim
        return self.a * x**2 + self.b
