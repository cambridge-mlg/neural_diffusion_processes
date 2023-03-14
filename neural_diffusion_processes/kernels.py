import dataclasses

from functools import partial
import jax
import jax.numpy as jnp

from check_shapes import check_shapes
from jaxtyping import Float, jaxtyped
from typeguard import typechecked as typechecker
from einops import rearrange

import gpjax
from gpjax.mean_functions import AbstractMeanFunction
from gpjax.gaussian_distribution import GaussianDistribution
from jaxkern.base import AbstractKernel
import jaxkern
from jaxkern.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
    ConstantDiagonalKernelComputation,
)
from jaxlinop import ConstantDiagonalLinearOperator, DiagonalLinearOperator, identity

from .constants import JITTER
from .types import Array, Optional, Union, Tuple, Int, Dict, List, Mapping, Callable
from .misc import flatten, unflatten


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


# inspired by GPjax but handling mutli-dimensional output
def prior_gp(
    mean_function: AbstractMeanFunction,
    kernel: AbstractKernel,
    params: Mapping,
    obs_noise: float = 0.0,
) -> Callable[[Float[Array, "N x_dim"]], GaussianDistribution]:
    @check_shapes("x_test: [N, x_dim]")
    def predict(x_test) -> GaussianDistribution:
        μt = mean_function(params["mean_fn"], x_test)
        n_test = μt.shape[0] * μt.shape[1]
        # p = μt.shape[-1]
        μt = flatten(μt)  # jnp.atleast_1d(μt.squeeze())
        Ktt = kernel.gram(params["kernel"], x_test)
        Ktt += identity(n_test) * (JITTER + obs_noise)
        dist = GaussianDistribution(μt, Ktt)
        return dist

    return predict


@check_shapes(
    "x: [N, D]", "return: [S, N, P] if num_samples", "return: [N, P] if not num_samples"
)
def sample_prior_gp(
    key,
    mean_function: AbstractMeanFunction,
    kernel: AbstractKernel,
    params: Mapping,
    x,
    num_samples: Optional[int] = None,
    obs_noise: float = 0.0,
):
    p = mean_function(params["mean_fn"], x).shape[-1]
    dist = prior_gp(mean_function, kernel, params, obs_noise)(x)
    samples = dist.sample(seed=key, sample_shape=(num_samples or 1,))
    if num_samples is not None:
        samples = rearrange(samples, "s (n p) -> s n p", p=p)
    else:
        samples = rearrange(samples, "1 (n p) -> n p", p=p)
    return samples


@check_shapes("x: [N, x_dim]", "y: [N, y_dim]", "return: []")
def log_prob_prior_gp(
    mean_function: AbstractMeanFunction,
    kernel: AbstractKernel,
    params: Mapping,
    x,
    y,
    obs_noise: float = 0.0
):
    return prior_gp(mean_function, kernel, params, obs_noise)(x).log_prob(flatten(y))


@check_shapes("x: [M, x_dim]", "y: [M, y_dim]")
def posterior_gp(
    mean_function: AbstractMeanFunction,
    kernel: AbstractKernel,
    params: Mapping,
    x,
    y,
    obs_noise: float = 0.0,
):
    μx = mean_function(params["mean_fn"], x)
    n = μx.shape[0] * μx.shape[1]
    μx = flatten(μx)  # jnp.atleast_1d(μt.squeeze())
    Kxx = kernel.gram(params["kernel"], x)
    Sigma = Kxx + identity(n) * (JITTER + obs_noise)

    @check_shapes("x_test: [N, x_dim]")
    def predict(x_test):
        
        μt = mean_function(params["mean_fn"], x_test)
        n_test = μt.shape[0] * μt.shape[1]
        Ktt = kernel.gram(params["kernel"], x_test)
        Kxt = kernel.cross_covariance(params["kernel"], x, x_test)

        # Σ⁻¹ Kxt
        Sigma_inv_Kxt = Sigma.solve(Kxt)

        # μt  +  Ktx (Kxx + Iσ²)⁻¹ (y  -  μx)
        mean = flatten(μt) + jnp.matmul(Sigma_inv_Kxt.T, flatten(y) - μx)

        # Ktt  -  Ktx (Kxx + Iσ²)⁻¹ Kxt, TODO: Take advantage of covariance structure to compute Schur complement more efficiently.
        covariance = Ktt - jnp.matmul(Kxt.T, Sigma_inv_Kxt)
        covariance += identity(n_test) * JITTER

        dist = GaussianDistribution(mean, covariance)
        return dist

    return predict


def get_kernel(kernel_type: str, active_dims = Optional[List[int]]) -> jaxkern.base.AbstractKernel:
    if kernel_type.lower() == "matern12":
        return jaxkern.stationary.Matern12(active_dims=active_dims)
    elif kernel_type.lower() == "matern32":
        return jaxkern.stationary.Matern32(active_dims=active_dims)
    elif kernel_type.lower() == "matern52":
        return jaxkern.stationary.Matern52(active_dims=active_dims)
    elif kernel_type.lower() in ["rbf", "se", "squared_exponential"]:
        return jaxkern.stationary.RBF(active_dims=active_dims)
    elif kernel_type.lower() == "white":
        return jaxkern.stationary.White(active_dims=active_dims)
    else:
        raise NotImplementedError("Unknown kernel: %s" % kernel_type)


def get_mean_fn(mean_fn_type: str) -> gpjax.mean_functions.AbstractMeanFunction:
    if mean_fn_type.lower() == "zero":
        return gpjax.mean_functions.Zero()
    elif mean_fn_type.lower() == "constant":
        return gpjax.mean_functions.Constant()
    else:
        raise NotImplementedError("Unknown mean function type %s" % mean_fn_type)