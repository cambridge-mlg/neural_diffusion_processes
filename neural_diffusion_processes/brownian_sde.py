from __future__ import annotations

import dataclasses
import math
from typing import Tuple

import diffrax as dfx
import jax
import jax.numpy as jnp
from check_shapes import check_shapes
from diffrax.custom_types import Bool, DenseInfo, PyTree, Scalar
from diffrax.solution import RESULTS
from diffrax.solver.euler import _ErrorEstimate, _SolverState
from diffrax.term import AbstractTerm
from jaxlinop import DenseLinearOperator, LinearOperator, identity
from jaxlinop.dense_linear_operator import _check_matrix
from jaxtyping import Array, Float, PyTree

from .sde import SDE, ScoreNetwork, div_noise, get_div_fn, sde_solve
from .utils import algebra_utils as utils
from .utils.misc import flatten, unflatten
from .utils.types import Callable, Mapping, Optional, Sequence, Tuple


class ProjectionOperator(DenseLinearOperator):
    def __init__(self, matrix: Float[Array, "N D D"]):
        """Initialize the covariance operator.

        Args:
            matrix (Float[Array, "N N"]): Dense matrix.
        """
        jax.vmap(_check_matrix)(matrix)
        self.matrix = matrix
        self.dim = matrix.shape[-1]

        # NOTE: to avoid unflattening -> flattening can do the following:
        # self.matrix = jax.scipy.linalg.block_diag(
        #     *[m.squeeze(0) for m in jnp.split(matrix, matrix.shape[0])]
        # )
        # but likely slower?
        # then also no need to oerride __mul__

    def __matmul__(self, other: Float[Array, "N M"]) -> Float[Array, "N M"]:
        """Matrix multiplication.

        Args:
            other (Float[Array, "N M"]): Matrix to multiply with.

        Returns:
            Float[Array, "N M"]: Result of matrix multiplication.
        """

        return flatten(
            jax.vmap(lambda A, v: jnp.matmul(A, v))(
                self.matrix, unflatten(other, self.dim)
            )
        )

    def __mul__(self, other: float) -> LinearOperator:
        """Multiply covariance operator by scalar.

        Args:
            other (LinearOperator): Scalar.

        Returns:
            LinearOperator: Covariance operator multiplied by a scalar.
        """

        return ProjectionOperator(matrix=self.matrix * other)


@dataclasses.dataclass
class SphericalMetric:
    dim: int = 2  # NOTE: actual dim not the dim of the ambient space

    @property
    def volume(self):
        half_dim = (self.dim + 1) / 2
        return 2 * jnp.pi**half_dim / math.gamma(half_dim)

    @property
    def log_volume(self):
        """log area of n-sphere https://en.wikipedia.org/wiki/N-sphere#Closed_forms"""
        half_dim = (self.dim + 1) / 2
        return math.log(2) + half_dim * math.log(math.pi) - math.lgamma(half_dim)

    @check_shapes("point: [N, y_dim]", "base_point: [N, y_dim]", "return: [N, y_dim]")
    def log(self, point, base_point):
        # inner_prod = self.embedding_metric.inner_product(base_point, point)
        inner_prod = jax.vmap(jnp.dot)(base_point, point)
        cos_angle = jnp.clip(inner_prod, -1.0, 1.0)
        squared_angle = jnp.arccos(cos_angle) ** 2
        coef_1_ = utils.taylor_exp_even_func(
            squared_angle, utils.inv_sinc_close_0, order=5
        )
        coef_2_ = utils.taylor_exp_even_func(
            squared_angle, utils.inv_tanc_close_0, order=5
        )
        log = jnp.einsum("...,...j->...j", coef_1_, point) - jnp.einsum(
            "...,...j->...j", coef_2_, base_point
        )
        # log = coef_1_ * point - coef_2_ * base_point
        return log

    @check_shapes(
        "tangent_vec: [N, y_dim]", "base_point: [N, y_dim]", "return: [N, y_dim]"
    )
    def exp(self, tangent_vec, base_point):
        proj_tangent_vec = self.to_tangent(tangent_vec, base_point)
        # norm2 = self.embedding_metric.squared_norm(proj_tangent_vec)
        norm2 = jnp.square(proj_tangent_vec).sum(-1)

        coef_1 = utils.taylor_exp_even_func(norm2, utils.cos_close_0, order=4)
        coef_2 = utils.taylor_exp_even_func(norm2, utils.sinc_close_0, order=4)
        exp = jnp.einsum("...,...j->...j", coef_1, base_point) + jnp.einsum(
            "...,...j->...j", coef_2, proj_tangent_vec
        )
        # exp = coef_1 * base_point + coef_2 * proj_tangent_vec
        return exp

    @check_shapes("vector: [N, y_dim]", "base_point: [N, y_dim]", "return: [N, y_dim]")
    def to_tangent(self, vector, base_point):
        sq_norm = jnp.sum(base_point**2, axis=-1)
        # inner_prod = self.metric.embedding_metric.inner_product(base_point, vector)
        inner_prod = jax.vmap(jnp.dot)(base_point, vector)
        coef = inner_prod / sq_norm
        tangent_vec = vector - jnp.einsum("...,...j->...j", coef, base_point)
        # tangent_vec = vector - coef * base_point

        return tangent_vec

    @check_shapes("point_a: [N, y_dim]", "point_b: [N, y_dim]")
    def dist(self, point_a, point_b):
        norm_a = jnp.linalg.norm(point_a, axis=-1)
        norm_b = jnp.linalg.norm(point_b, axis=-1)
        inner_prod = jnp.einsum("...i,...i->...", point_a, point_b)

        cos_angle = inner_prod / (norm_a * norm_b)
        cos_angle = jnp.clip(cos_angle, -1, 1)

        dist = jnp.arccos(cos_angle)

        return dist


metric = SphericalMetric(2)


class SphericalGRW(dfx.Euler):
    """Euler's method.

    1st order explicit Runge--Kutta method. Does not support adaptive step sizing.

    When used to solve SDEs, converges to the Itô solution.
    """

    metric = SphericalMetric(2)

    def retraction(self, v, x):
        def projection(point):
            return point / jnp.linalg.norm(point, axis=-1, keepdims=True)

        return flatten(projection(unflatten(x + v, 3)))

    def step(
        self,
        terms: AbstractTerm,
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Bool,
    ) -> Tuple[PyTree, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        control = terms.contr(t0, t1)
        # y1 = (y0**ω + terms.vf_prod(t0, y0, args, control) ** ω).ω
        # y1 = self.retraction(terms.vf_prod(t0, y0, args, control) ** ω, y0**ω).ω
        # y1 = self.retraction(terms.vf_prod(t0, y0, args, control), y0)
        point = unflatten(y0, 3)
        tv = unflatten(terms.vf_prod(t0, y0, args, control), 3)
        y1 = self.metric.exp(tv, point)
        y1 = flatten(y1)
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful


def gegenbauer_polynomials(alpha: float, l_max: int, x):
    """https://en.wikipedia.org/wiki/Gegenbauer_polynomials"""
    shape = x.shape if len(x.shape) > 0 else (1,)
    p = jnp.zeros((max(l_max + 1, 2), shape[0]))
    C_0 = jnp.ones_like(x)
    C_1 = 2 * alpha * x
    p = p.at[0].set(C_0)
    p = p.at[1].set(C_1)

    def body_fun(n, p_val):
        C_nm1 = p_val[n - 1]
        C_nm2 = p_val[n - 2]
        C_n = 1 / n * (2 * x * (n + alpha - 1) * C_nm1 - (n + 2 * alpha - 2) * C_nm2)
        p_val = p_val.at[n].set(C_n)
        return p_val

    if l_max >= 2:
        p = jax.lax.fori_loop(lower=2, upper=l_max + 1, body_fun=body_fun, init_val=p)

    return p[: l_max + 1]


# @check_shapes("x0: [N, y_dim]", "x: [N, y_dim]", "t: []", "return: [N, y_dim]")
def _log_heat_kernel(x0, x, t, n_max):
    """
    log p_t(x, y) = \sum^\infty_n e^{-t \lambda_n} \psi_n(x) \psi_n(y)
    = \sum^\infty_n e^{-n(n+1)t} \frac{2n+d-1}{d-1} \frac{1}{A_{\mathbb{S}^n}} \mathcal{C}_n^{(d-1)/2}(x \cdot y
    """

    # NOTE: Should we rely on the Russian roulette estimator even though the log would bias it?
    # if len(t.shape) == len(x.shape):
    # t = t[..., 0]
    t = t / 2  # NOTE: to match random walk
    d = metric.dim
    if d == 1:
        n = jnp.expand_dims(jnp.arange(-n_max, n_max + 1), axis=-1)
        t = jnp.expand_dims(t, axis=0)
        sigma_squared = t  # NOTE: factor 2 is needed empirically to match kernel?
        cos_theta = jnp.sum(x0 * x, axis=-1)
        theta = jnp.arccos(cos_theta)
        coeffs = jnp.exp(-jnp.power(theta + 2 * math.pi * n, 2) / 2 / sigma_squared)
        prob = jnp.sum(coeffs, axis=0)
        prob = prob / jnp.sqrt(2 * math.pi * sigma_squared[0])
    else:
        n = jnp.expand_dims(jnp.arange(0, n_max + 1), axis=-1)
        t = jnp.expand_dims(t, axis=0)
        coeffs = jnp.exp(-n * (n + 1) * t) * (2 * n + d - 1) / (d - 1) / metric.volume
        inner_prod = jnp.sum(x0 * x, axis=-1)
        cos_theta = jnp.clip(inner_prod, -1.0, 1.0)
        P_n = gegenbauer_polynomials(
            alpha=(metric.dim - 1) / 2, l_max=n_max, x=cos_theta
        )
        prob = jnp.sum(coeffs * P_n, axis=0)
    return jnp.log(prob)


@check_shapes("x0: [N, y_dim]", "x: [N, y_dim]", "t: []", "return: [N, y_dim]")
def grad_log_heat_kernel_exp(x0, x, t):
    return metric.log(x0, x) / jnp.expand_dims(t, -1)


@check_shapes("x0: [N, y_dim]", "x: [N, y_dim]", "t: []", "return: [N, y_dim]")
def grad_marginal_log_prob(x0, x, t, thresh, n_max):
    cond = jnp.expand_dims(t <= thresh, -1)
    approx = grad_log_heat_kernel_exp(x0, x, t)
    # log_heat_kernel = lambda x0, x, s: jnp.reshape(
    #     _log_heat_kernel(x0, x, s, n_max=n_max), ()
    # )
    log_heat_kernel = lambda x0, x: _log_heat_kernel(
        x0[None], x[None], t, n_max=n_max
    ).squeeze()
    logp_grad_fn = jax.grad(log_heat_kernel, argnums=1)
    logp_grad = jax.vmap(logp_grad_fn)(x0, x)
    exact = metric.to_tangent(logp_grad, x)
    return jnp.where(cond, approx, exact)


@dataclasses.dataclass
class SphericalBrownian(SDE):
    dim: int = 2  # NOTE: actual dim not the dim of the ambient space
    loss_type: str = "ism"

    # @check_shapes("t: []", "yt: [N, y_dim]", "x: [N, x_dim]", "return: [N, y_dim]")
    # def score(self, key, t: Array, yt: Array, x: Array, network: ScoreNetwork) -> Array:
    #     score = super().score(key, t, yt, x, network)
    #     return unflatten(self.projection(yt) @ flatten(score), d=yt.shape[-1])

    @check_shapes("x: [N, x_dim]", "return: [N, y_dim]")
    def sample_prior(self, key, x):
        """sample from white noise / U(S^d)^n"""
        n = x.shape[0]

        def sample_spherical_uniform(key):
            u = jax.random.normal(key, (self.dim + 1,))
            return u / jnp.linalg.norm(u, axis=-1)

        subkeys = jax.random.split(key, n)
        return jax.vmap(sample_spherical_uniform)(subkeys)

    @check_shapes("x: [N, x_dim]", "y: [N, y_dim]", "return: []")
    def log_prob_prior(self, x, y):
        """white noise prob  / U(S^d)^n"""
        n = x.shape[0]
        # https://en.wikipedia.org/wiki/N-sphere#Closed_forms
        metric = SphericalMetric(self.dim)
        return n * metric.log_volume

    @check_shapes("x: [N, x_dim]")
    def limiting_gram(self, x) -> LinearOperator:
        return identity(x.shape[0] * self.dim)

    @check_shapes("y: [N, y_dim]")
    def projection(self, y) -> LinearOperator:
        # A = I - N N^t with N orthonormal basis of normal space
        A = jax.vmap(lambda y: jnp.eye(y.shape[-1]) - y[:, None] @ y[:, None].T)(y)
        # K = SquaredLinearOperator(flatten(A))
        K = ProjectionOperator(A)
        return K

    @check_shapes("t: []", "x: [N, x_dim]", "y: [N, y_dim]", "return: [N, y_dim]")
    def sample_marginal(self, key, t: Array, x: Array, y: Array) -> Array:
        kwargs = {
            "forward": True,
            "rtol": None,
            "num_steps": 10,
            "solver": SphericalGRW(),
        }
        return sde_solve(self, None, key=key, x=x, y=y, tf=t, **kwargs).squeeze()

    @check_shapes("t: []", "yt: [N, y_dim]", "x: [N, x_dim]", "return: [N, y_dim]")
    def drift(self, t: Array, yt: Array, x: Array) -> Array:
        return jnp.zeros_like(yt)

    @check_shapes("t: []", "yt: [N, y_dim]", "x: [N, x_dim]")
    def diffusion(self, t, yt, x) -> LinearOperator:
        """projection operator"""
        P = self.projection(yt)
        beta_term = jnp.sqrt(self.beta_schedule(t))
        diffusion = beta_term * P
        return diffusion

    @check_shapes("t: []", "y: [N, y_dim]", "x: [N, x_dim]", "return: []")
    def loss(self, key, t: Array, y: Array, x: Array, network: ScoreNetwork) -> Array:
        var = 1.0 - jnp.exp(-self.beta_schedule.B(t))
        std = jnp.sqrt(var)
        beta = self.beta_schedule(t)

        ekey, nkey, dkey = jax.random.split(key, 3)
        yt = self.sample_marginal(ekey, t, x, y)
        score_net = self.score(nkey, t, yt, x, network)
        if self.loss_type == "ism":
            sq_norm_score = jnp.sum(jnp.square(score_net), -1)
            hutchinson_samples = 1
            hutchinson_type = "None"
            eps = div_noise(dkey, (hutchinson_samples, *y.shape), hutchinson_type)
            drift_fn = lambda t, yt, arg: self.score(nkey, t, yt, arg, network)
            div_fn = get_div_fn(drift_fn, hutchinson_type)

            div_score = div_fn(t, yt, x, eps)
            loss = 0.5 * sq_norm_score + div_score

            # TODO: not sure about what weighting should be used /!\
            if self.weighted:  # `likelihood' weighting
                loss = loss * beta**2

        elif self.loss_type == "dsm":
            kwargs = {"thresh": 0.5, "n_max": 20}
            s = self.beta_schedule.B(t)
            logp_grad = grad_marginal_log_prob(y, yt, s, **kwargs)
            if self.weighted:  # `std' weighting
                loss = jnp.square(std * score_net - std * logp_grad).sum(-1)
            else:  # `Likelihood' weighting
                # loss = jnp.sum(jnp.square(score_net - logp_grad), -1) # true DSM loss
                loss = jnp.square(score_net - logp_grad).sum(-1) * beta**2

        else:
            raise Exception()

        loss = jnp.mean(loss)
        return loss
