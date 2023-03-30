import dataclasses

from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

from check_shapes import check_shapes
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
from jaxlinop import LinearOperator, DenseLinearOperator, ConstantDiagonalLinearOperator, DiagonalLinearOperator, identity
from jaxlinop.dense_linear_operator import _check_matrix

from .constants import JITTER
from .types import Array, Optional, Union, Tuple, Int, Dict, List, Mapping, Callable, Float, Type
from .misc import flatten, unflatten, check_shape, jax_unstack


class BlockDiagonalLinearOperator(DenseLinearOperator):
    """Block diagonal matrix."""

    def __init__(self, matrices: List[Float[Array, "N N"]]):
        """Initialize the covariance operator.

        Args:
            matrix (Float[Array, "N N"]): Dense matrix.
        """
        [_check_matrix(matrix) for matrix in matrices]
        self.linops = [matrix if isinstance(matrix, LinearOperator) else DenseLinearOperator(matrix) for matrix in matrices]

    @property
    def matrix(self):
        return self.to_dense()

    @property
    def T(self) -> LinearOperator:
        return BlockDiagonalLinearOperator([linop.T for linop in self.linops])

    def trace(self) -> Float[Array, "1"]:
        """Trace of the linear matrix.

        Returns:
            Float[Array, "1"]: Trace of the linear matrix.
        """
        return jnp.sum(jnp.array([linop.trace() for linop in self.linops]))

    def log_det(self) -> Float[Array, "1"]:
        """Trace of the linear matrix.

        Returns:
            Float[Array, "1"]: Trace of the linear matrix.
        """
        return jnp.sum(jnp.array([linop.log_det() for linop in self.linops]))

    def to_root(self) -> LinearOperator:
        return BlockDiagonalLinearOperator([linop.to_root() for linop in self.linops])

    def inverse(self) -> LinearOperator:
        return BlockDiagonalLinearOperator([linop.inverse() for linop in self.linops])
    
    def solve(self, rhs: Float[Array, "N"]) -> Float[Array, "N"]:
        """Solve linear system. Default implementation uses dense Cholesky decomposition.

        Args:
            rhs (Float[Array, "N"]): Right hand side of the linear system.

        Returns:
            Float[Array, "N]: Solution of the linear system.
        """
        # NOTE: use jax.scipy.sparse.linalg.cg?
        shapes_0, _ = zip(*[linop.shape for linop in self.linops])
        def f(m, rhs):
            root = m.to_root()
            rootT = root.T
            return rootT.solve(root.solve(rhs))
        return jnp.concatenate([f(m, rhs[shapes_0[0]*i:shapes_0[0]*(i+1)]) for i, m in enumerate(self.linops)], axis=0)

    def _add_diagonal(self, other: DiagonalLinearOperator) -> LinearOperator:
        """Add diagonal to the covariance operator,  useful for computing, Kxx + Iσ².

        Args:
            other (DiagonalLinearOperator): Diagonal covariance operator to add to the covariance operator.

        Returns:
            LinearOperator: Sum of the two covariance operators.
        """
        shapes_0, _ = zip(*[linop.shape for linop in self.linops])
        diag = other.diagonal()
        diag_linops = [DiagonalLinearOperator(diag[shapes_0[0]*i:shapes_0[0]*(i+1)]) for i in range(len(self.linops))]
        linops = [linop._add_diagonal(diag_linops[i]) for i, linop in enumerate(self.linops)]

        return BlockDiagonalLinearOperator(linops)

    @property
    def shape(self) -> Tuple[int, int]:
        """Covaraince matrix shape.

        Returns:
            Tuple[int, int]: shape of the covariance operator.
        """
        shapes_0, shapes_1 = zip(*[linop.shape for linop in self.linops])
        return (sum(shapes_0), sum(shapes_1))


    def __mul__(self, other: float) -> LinearOperator:
        """Multiply covariance operator by scalar.

        Args:
            other (LinearOperator): Scalar.

        Returns:
            LinearOperator: Covariance operator multiplied by a scalar.
        """

        return BlockDiagonalLinearOperator([linop * other for linop in self.linops])


    def to_dense(self) -> Float[Array, "N N"]:
        """Construct dense Covaraince matrix from the covariance operator.

        Returns:
            Float[Array, "N N"]: Dense covariance matrix.
        """
        return jax.scipy.linalg.block_diag(*[linop.to_dense() for linop in self.linops])


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
    """
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
        matrix = self.kernel_fn(params, inputs[0], inputs[0])
        value = matrix[0, 0]
        output_dim = matrix.shape[0]
        input_dim = inputs.shape[0]

        return ConstantDiagonalLinearOperator(
            value=jnp.atleast_1d(value), size=input_dim * output_dim
        )

class MultiOutputDiagonalKernelComputation(MultiOutputDenseKernelComputation):
    """
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
    ) -> BlockDiagonalLinearOperator:
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
        gram = jax.vmap(lambda x: jax.vmap(lambda y: self.kernel_fn(params, x, y))(inputs))(inputs)
        # matrices = jax.numpy.diagonal(gram, axis1=-2, axis2=-1)
        matrices = jax.vmap(jax.vmap(jax.numpy.diagonal))(gram)
        matrices = jax_unstack(matrices, axis=-1)

        return BlockDiagonalLinearOperator(matrices)


def promote_compute_engines(engine1: Type[jaxkern.computations.AbstractKernelComputation], engine2: Type[jaxkern.computations.AbstractKernelComputation]) -> Type[jaxkern.computations.AbstractKernelComputation]:
    if engine1 ==MultiOutputDiagonalKernelComputation and engine2 == jaxkern.computations.ConstantDiagonalKernelComputation:
        return jaxkern.computations.ConstantDiagonalKernelComputation

    if engine1 == jaxkern.computations.ConstantDiagonalKernelComputation and engine2 == jaxkern.computations.DiagonalKernelComputation:
        return jaxkern.computations.DiagonalKernelComputation

    if engine2 == jaxkern.computations.ConstantDiagonalKernelComputation and engine1 == jaxkern.computations.DiagonalKernelComputation:
        return jaxkern.computations.DiagonalKernelComputation

    if engine2 ==MultiOutputDiagonalKernelComputation and engine1 == MultiOutputDiagonalKernelComputation:
        return MultiOutputDiagonalKernelComputation

    if engine1 == jaxkern.computations.DenseKernelComputation or engine2 == jaxkern.computations.DenseKernelComputation:
        return jaxkern.computations.DenseKernelComputation

    raise NotImplementedError(
        "Add rule for optimal compute engine sum kernel for types %s and %s." % (
            engine1, engine2
        ))

@dataclasses.dataclass
class DiagMultiOutputKernel(AbstractKernel):
    def __init__(
        self,
        output_dim,
        scalar_kernel: AbstractKernel = None,
        active_dims: Optional[List[int]] = None,
    ) -> None:
        if scalar_kernel is None:
            scalar_kernel = jaxkern.stationary.RBF(active_dims=active_dims)
        if scalar_kernel.compute_engine == ConstantDiagonalKernelComputation:
            compute_engine = MultiOutputConstantDiagonalKernelComputation
        else:
            # compute_engine = MultiOutputDiagonalKernelComputation
            compute_engine = MultiOutputDenseKernelComputation
            # TODO: seems to be an issue with MultiOutputDiagonalKernelComputation...
        print("compute_engine", compute_engine)

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
        μt = mean_function(params["mean_function"], x_test)
        n_test = μt.shape[0] * μt.shape[1]
        # p = μt.shape[-1]
        μt = flatten(μt)  # jnp.atleast_1d(μt.squeeze())
        Ktt = kernel.gram(params["kernel"], x_test)
        # Ktt += identity(n_test) * (JITTER + obs_noise)
        Ktt = Ktt._add_diagonal(identity(n_test) * (JITTER + obs_noise))
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
    p = mean_function(params["mean_function"], x).shape[-1]
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
    dist = prior_gp(mean_function, kernel, params, obs_noise)(x)
    return dist.log_prob(flatten(y)).squeeze()


@check_shapes("x: [M, x_dim]", "y: [M, y_dim]")
def posterior_gp(
    mean_function: AbstractMeanFunction,
    kernel: AbstractKernel,
    params: Mapping,
    x,
    y,
    obs_noise: float = 0.0,
):
    μx = mean_function(params["mean_function"], x)
    n = μx.shape[0] * μx.shape[1]
    μx = flatten(μx)  # jnp.atleast_1d(μt.squeeze())
    Kxx = kernel.gram(params["kernel"], x)
    # Sigma = Kxx + identity(n) * (JITTER + obs_noise)
    Sigma = Kxx._add_diagonal(identity(n) * (JITTER + obs_noise))

    @check_shapes("x_test: [N, x_dim]")
    def predict(x_test):
        
        μt = mean_function(params["mean_function"], x_test)
        n_test = μt.shape[0] * μt.shape[1]
        Ktt = kernel.gram(params["kernel"], x_test)
        Kxt = kernel.cross_covariance(params["kernel"], x, x_test)

        # Σ⁻¹ Kxt
        Sigma_inv_Kxt = Sigma.solve(Kxt)

        # μt  +  Ktx (Kxx + Iσ²)⁻¹ (y  -  μx)
        mean = flatten(μt) + jnp.matmul(Sigma_inv_Kxt.T, flatten(y) - μx)

        # Ktt  -  Ktx (Kxx + Iσ²)⁻¹ Kxt, TODO: Take advantage of covariance structure to compute Schur complement more efficiently.
        covariance = Ktt - jnp.matmul(Kxt.T, Sigma_inv_Kxt)
        covariance = covariance._add_diagonal(identity(n_test) * (JITTER + obs_noise))

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