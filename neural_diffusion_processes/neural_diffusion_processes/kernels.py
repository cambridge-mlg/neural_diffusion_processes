import dataclasses

from functools import partial
import jax
import jax.numpy as jnp
from jax import vmap

from check_shapes import check_shapes
from jaxtyping import Float, jaxtyped
from typeguard import typechecked as typechecker
from einops import rearrange

from jaxkern.base import AbstractKernel
import jaxkern
from jaxkern.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
    ConstantDiagonalKernelComputation,
)
from gpjax.mean_functions import AbstractMeanFunction
from jaxlinop import (
    ConstantDiagonalLinearOperator,
    DiagonalLinearOperator,
)

from .types import Array, Optional, Union, Tuple, Int, Dict, List, Mapping, Callable


def check_shape(func):
    return typechecker(jaxtyped(func))


class MultiOutputDenseKernelComputation(DenseKernelComputation):
    """Dense kernel computation class. Operations with the kernel assume
    a dense gram matrix structure.
    """

    def __init__(
        self,
        kernel_fn: Callable[
            [Dict, Float[jax.Array, "1 D"], Float[jax.Array, "1 D"]], jax.Array
        ] = None,
    ) -> None:
        super().__init__(kernel_fn)

    def cross_covariance(
        self, params: Dict, x: Float[jax.Array, "N D"], y: Float[jax.Array, "M D"]
    ) -> Float[Array, "N M"]:
        """For a given kernel, compute the NxM covariance matrix on a pair of input
        matrices of shape NxD and MxD.
        Args:
            kernel (AbstractKernel): The kernel for which the Gram
                matrix should be computed for.
            params (Dict): The kernel's parameter set.
            x (Float[Array,"N D"]): The input matrix.
            y (Float[Array,"M D"]): The input matrix.
        Returns:
            CovarianceOperator: The computed square Gram matrix.
        """
        cross_cov = super().cross_covariance(params, x, y)
        cross_cov = rearrange(cross_cov, "n1 n2 p1 p2 -> (n1 p1) (n2 p2)")
        return cross_cov


class MultiOutputConstantDiagonalKernelComputation(MultiOutputDenseKernelComputation):
    """Dense kernel computation class. Operations with the kernel assume
    a dense gram matrix structure.
    """

    def __init__(
        self,
        kernel_fn: Callable[
            [Dict, Float[jax.Array, "1 D"], Float[jax.Array, "1 D"]], jax.Array
        ] = None,
    ) -> None:
        super().__init__(kernel_fn)

    def gram(
        self,
        params: Dict,
        inputs: Float[Array, "N D"],
    ) -> ConstantDiagonalLinearOperator:
        """For a kernel with diagonal structure, compute the NxN gram matrix on
        an input matrix of shape NxD.

        Args:
            kernel (AbstractKernel): The kernel for which the Gram matrix
                should be computed for.
            params (Dict): The kernel's parameter set.
            inputs (Float[Array, "N D"]): The input matrix.

        Returns:
            CovarianceOperator: The computed square Gram matrix.
        """
        matrix = self.kernel_fn(params, inputs[0], inputs[0])[0, 0]
        value = matrix[0, 0]
        output_dim = matrix.shape[0]
        input_dim = inputs.shape[0]

        return ConstantDiagonalLinearOperator(
            value=jnp.atleast_1d(value), size=input_dim * output_dim
        )


@dataclasses.dataclass
class DiagMultiOutputKernel(AbstractKernel):
    def __init__(
        self,
        output_dim,
        scalar_kernel=None,
        active_dims: Optional[List[int]] = None,
    ) -> None:
        if scalar_kernel is None:
            scalar_kernel = jaxkern.stationary.RBF(active_dims=active_dims)
        if isinstance(scalar_kernel.compute_engine, ConstantDiagonalKernelComputation):
            compute_engine = MultiOutputConstantDiagonalKernelComputation
        else:
            # TODO: use black diagonal structure
            compute_engine = MultiOutputDenseKernelComputation

        super().__init__(
            compute_engine=compute_engine,
            active_dims=active_dims,
            name=f"{scalar_kernel.name} diag multi output kernel",
        )
        self.output_dim = output_dim
        self.scalar_kernel = scalar_kernel
        self._stationary = scalar_kernel.stationary

    def init_params(self, key: jnp.DeviceArray) -> Dict:
        return self.scalar_kernel.init_params(key)

    @check_shape
    def __call__(
        self,
        params: Mapping[str, float],
        x: Float[jax.Array, "D"],
        y: Float[jax.Array, "D"],
    ) -> Float[jax.Array, "output_dim output_dim"]:
        I = jnp.eye(self.output_dim)
        K = self.scalar_kernel(params, x, y)
        return I * K


@dataclasses.dataclass
class RBFVec(DiagMultiOutputKernel):
    # TODO: RFF?
    def __init__(self, output_dim, active_dims: Optional[List[int]] = None):
        super().__init__(
            output_dim=output_dim,
            scalar_kernel=jaxkern.stationary.RBF(active_dims=active_dims),
            active_dims=active_dims,
        )


class WhiteVec(DiagMultiOutputKernel):
    def __init__(self, output_dim, active_dims: Optional[List[int]] = None):
        super().__init__(
            output_dim=output_dim,
            scalar_kernel=jaxkern.stationary.White(active_dims=active_dims),
            active_dims=active_dims,
        )


@dataclasses.dataclass
class RBFDivFree(AbstractKernel):
    """Based on the kernels defined in equation (24) in
    "Kernels for Vector-Valued Functions: a Review" by Alvarez et al
    """

    def __init__(
        self,
        active_dims: Optional[List[int]] = None,
        name: Optional[str] = "RBFDivFree kernel",
    ) -> None:
        super().__init__(
            compute_engine=MultiOutputDenseKernelComputation,
            active_dims=active_dims,
            name=name,
        )
        self._stationary = True

    def init_params(self, key: jnp.DeviceArray) -> Dict:
        return {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
        }

    @check_shape
    def __call__(
        self,
        params: Mapping[str, float],
        x: Float[jax.Array, "D"],
        y: Float[jax.Array, "D"],
    ) -> Float[jax.Array, "D D"]:
        dim = x.shape[-1]
        diff = (x - y) / params["lengthscale"]

        sq_dist = (diff**2).sum(axis=-1)

        outer_product = jnp.einsum("i,j->ij", diff, diff)
        I = jnp.eye(dim)
        K = params["variance"] * jnp.exp(-0.5 * sq_dist)

        A = I - (outer_product)
        K = A * K

        return K


@dataclasses.dataclass
class RBFCurlFree(AbstractKernel):
    """Based on the kernels defined in equation (24) in
    "Kernels for Vector-Valued Functions: a Review" by Alvarez et al
    """

    def __init__(
        self,
        active_dims: Optional[List[int]] = None,
        name: Optional[str] = "RBFCurlFree kernel",
    ) -> None:
        super().__init__(
            compute_engine=MultiOutputDenseKernelComputation,
            active_dims=active_dims,
            name=name,
        )
        self._stationary = True

    def init_params(self, key: jnp.DeviceArray) -> Dict:
        return {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
        }

    @check_shape
    def __call__(
        self,
        params: Mapping[str, float],
        x: Float[jax.Array, "D"],
        y: Float[jax.Array, "D"],
    ) -> Float[jax.Array, "D D"]:
        dim = x.shape[-1]
        diff = (x - y) / params["lengthscale"]

        sq_dist = (diff**2).sum(axis=-1)

        outer_product = jnp.einsum("i,j->ij", diff, diff)
        I = jnp.eye(dim)

        A = outer_product
        A += (dim - 1 - sq_dist)[..., None, None] * I

        K = params["variance"] * jnp.exp(-0.5 * sq_dist)

        K = A * K

        return K


## Mean functions


# @dataclasses.dataclass(frozen=True)
# class Constant:
#     value: float = 0.0
#     output_dim: Optional[Tuple] = None

#     def __call__(self, x):
#         output_dim = x.shape[-1] if self.output_dim is None else self.output_dim
#         return self.value * jnp.ones((*x.shape[:-1], output_dim))


# @dataclasses.dataclass(frozen=True)
# class Zero(Constant):
#     value: float = 0.0
#     output_dim: Optional[Tuple] = None


# @dataclasses.dataclass(frozen=True)
# class Quadratic:
#     output_dim: Optional[Tuple] = None
#     a: float = 1.0
#     b: float = 0.0

#     def __call__(self, x):
#         # output_dim = x.shape[-1] if self.output_dim is None else self.output_dim
#         return self.a * x**2 + self.b


# Multi-dimensional


# class MultiOutputMeanFunction:
#     def __init__(self, output_dim: int):
#         self.output_dim = output_dim

#     @check_shapes("x: [input_dim]", "return: [output_dim]")
#     def __call__(self, x):
#         ...


# class ZeroMultiOutputMeanFunction(MultiOutputMeanFunction):
#     @check_shapes("x: [input_dim]", "return: [output_dim]")
#     def __call__(self, x):
#         return jnp.zeros((self.output_dim,))


# @check_shapes("x: [N, input_dim]", "return: [N, output_dim]")
# def eval_meanfunction(mf: MultiOutputMeanFunction, x):
#     return jax.vmap(lambda x_: mf.__call__(x_))(x)


# @check_shapes(
#     "x: [N, D]", "return: [S, N, P] if num_samples", "return: [N, P] if not num_samples"
# )
# def sample_gp(
#     key,
#     kernel,
#     mean_function,
#     x,
#     num_samples: Optional[int] = 1,
#     obs_noise: float = 0.0,
# ):
#     # kxx = kernel.gram(kernel_params, x).to_dense()
#     # kxx = rearrange(kxx, "n1 n2 p1 p2 -> (n1 p1) (n2 p2)")
#     kxx = kernel(x)
#     kxx += obs_noise * jnp.eye(kxx.shape[-1])
#     mu = eval_meanfunction(mean_function, x)  # [N, P]
#     p = mu.shape[-1]
#     mu = rearrange(mu, "n p -> (n p) 1")
#     samples = sample_mvn(key, mu, kxx, num_samples)
#     if num_samples is not None:
#         return rearrange(samples, "s (n p) 1 -> s n p", p=p)
#     else:
#         return rearrange(samples, "(n p) 1 -> n p", p=p)


from .constants import JITTER
from gpjax.gaussian_distribution import GaussianDistribution
from jaxlinop import identity


@check_shapes(
    "x: [N, D]", "return: [S, N, P] if num_samples", "return: [N, P] if not num_samples"
)
def sample_prior_gp(
    key,
    kernel: AbstractKernel,
    mean_function: AbstractMeanFunction,
    x,
    params: Mapping,
    num_samples: Optional[int] = None,
    obs_noise: float = 0.0,
):
    μt = mean_function(params["mean_fn"], x)
    n_test = μt.shape[0] * μt.shape[1]
    p = μt.shape[-1]
    μt = rearrange(μt, "n p -> (n p)")  # jnp.atleast_1d(μt.squeeze())
    Ktt = kernel.gram(params["kernel"], x)
    Ktt += identity(n_test) * (JITTER + obs_noise)
    dist = GaussianDistribution(μt, Ktt)
    samples = dist.sample(seed=key, sample_shape=(num_samples or 1,))
    if num_samples is not None:
        samples = rearrange(samples, "s (n p) -> s n p", p=p)
    else:
        samples = rearrange(samples, "1 (n p) -> n p", p=p)
    return samples
