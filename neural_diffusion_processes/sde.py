from __future__ import annotations
from abc import abstractmethod
import operator

import copy
from functools import partial
import dataclasses

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import diffrax as dfx
from diffrax import AbstractStepSizeController, PIDController, ConstantStepSize
from diffrax import AbstractSolver, Dopri5, Tsit5
import jaxkern
import gpjax
from gpjax.mean_functions import AbstractMeanFunction, Zero, Constant
from jaxlinop import LinearOperator, DenseLinearOperator, DiagonalLinearOperator, identity, ZeroLinearOperator

from jaxtyping import Array, Float, PyTree
from check_shapes import check_shapes

from .types import Tuple, Callable, Mapping, Sequence
from .data import DataBatch
from .kernels import prior_gp, sample_prior_gp, log_prob_prior_gp, promote_compute_engines
from .constants import JITTER
from .misc import flatten, unflatten


class ScoreNetwork(Callable):
    @abstractmethod
    @check_shapes("t: []", "yt: [N, y_dim]", "x: [N, x_dim]", "return: [N, y_dim]")
    def __call__(self, t: Array, yt: Array, x: Array, *, key) -> Array:
        ...


@dataclasses.dataclass
class LinearBetaSchedule:
    t0: float = 1e-5
    t1: float = 1.0
    beta0: float = 0.0
    beta1: float = 20.0

    @check_shapes("t: [batch...]", "return: [batch...]")
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        interval = self.t1 - self.t0
        normed_t = (t - self.t0) / interval
        return self.beta0 + normed_t * (self.beta1 - self.beta0)

    @check_shapes("t: [batch...]", "return: [batch...]")
    def B(self, t: jnp.ndarray) -> jnp.ndarray:
        r"""
        integrates \int_{s=0}^t beta(s) ds
        """
        interval = self.t1 - self.t0
        normed_t = (t - self.t0) / interval
        # This is not done in the package.
        return interval * (
            self.beta0 * normed_t + 0.5 * (normed_t**2) * (self.beta1 - self.beta0)
        )


class SDE:
    def __init__(
        self,
        limiting_kernel: jaxkern.base.AbstractKernel,
        limiting_mean_fn: gpjax.mean_functions.AbstractMeanFunction,
        limiting_params: Mapping,
        beta_schedule: LinearBetaSchedule,
    ):
        self.limiting_kernel = limiting_kernel
        self.limiting_mean_fn = limiting_mean_fn
        assert isinstance(self.limiting_mean_fn, (Zero, Constant))
        self.limiting_params = limiting_params
        self.beta_schedule = beta_schedule
        # self.weighted = True
        self.std_trick = False
        self.residual_trick = True

    @check_shapes("x: [N, x_dim]", "return: [N, y_dim]")
    def sample_prior(self, key, x):
        return sample_prior_gp(
            key, self.limiting_mean_fn, self.limiting_kernel, self.limiting_params, x
        )

    @check_shapes("x: [N, x_dim]", "y: [N, y_dim]", "return: []")
    def log_prob_prior(self, x, y):
        return log_prob_prior_gp(
            self.limiting_mean_fn, self.limiting_kernel, self.limiting_params, x, y
        )

    def p0t(
        self,
        t,
        y0: Callable[[Float[Array, "N x_dim"]], Float[Array, "N y_dim"]] | Float[Array, "N x_dim"],
        k0: jaxkern.base.AbstractKernel = None,
        k0_params: Mapping | None = None
    ) -> Tuple[AbstractMeanFunction, jaxkern.base.AbstractKernel, dict]:
        # TODO: add equations as method description

        # # backwards compatibility...
        # if not callable(y0):
        #     y0 = lambda _: y0

        # E[Y_t|Y_0]
        mean_coef = jnp.exp(-0.5 * self.beta_schedule.B(t))

        class _Mean(AbstractMeanFunction):
            def __call__(self_, params: Mapping, x: Float[Array, "N D"]) -> Float[Array, "N Q"]:
                del params
                μT_value = self.limiting_mean_fn(self.limiting_params["mean_fn"], x)
                # return mean_coef * y0(x) + (1.0 - mean_coef) * μT_value
                return mean_coef * y0 + (1.0 - mean_coef) * μT_value
            def init_params(self, key):
                return {}

        μ0t = _Mean()

        # Cov[Y_t|Y_0]
        cov_coef = jnp.exp(-self.beta_schedule.B(t))
        if k0 is None:
            k0t = self.limiting_kernel
            k0t_params = copy.deepcopy(self.limiting_params["kernel"])
            k0t_params["variance"] = k0t_params["variance"] * (1.0 - cov_coef)
        #     k0 = self.limiting_kernel # as we set the variance to 0.0 we can pick any kernel.
        #     k0_params = {"variance": self.limiting_params["kernel"]["variance"] * 0.0}
        else:
            assert k0_params is not None
            k0_params["variance"] = k0_params["variance"] * cov_coef
            kt_param = copy.deepcopy(self.limiting_params["kernel"])
            kt_param["variance"] = kt_param["variance"] * (1. - cov_coef)
            k0t = jaxkern.SumKernel(
                [k0, self.limiting_kernel],
                compute_engine=promote_compute_engines(k0.compute_engine, self.limiting_kernel.compute_engine)
            )
            k0t_params = [k0_params, kt_param]
        params = {"mean_fn": {}, "kernel": k0t_params}
        return μ0t, k0t, params

    @check_shapes("t: []", "x: [N, x_dim]", "y: [N, y_dim]", "return: [N, y_dim]")
    def sample_marginal(self, key, t: Array, x: Array, y: Array) -> Array:
        μ0t, k0t, params = self.p0t(t, y)
        return sample_prior_gp(key, μ0t, k0t, params, x)

    @check_shapes("t: []", "yt: [N, y_dim]", "x: [N, x_dim]", "return: [N, y_dim]")
    def drift(self, t: Array, yt: Array, x: Array) -> Array:
        μT = self.limiting_mean_fn(self.limiting_params["mean_fn"], x)
        return -0.5 * self.beta_schedule(t) * (yt - μT)

    @check_shapes("t: []", "yt: [N, y_dim]", "x: [N, x_dim]")
    def diffusion(self, t, yt, x) -> LinearOperator:
        np = yt.shape[-2] * yt.shape[-1]
        del yt
        Ktt = self.limiting_kernel.gram(self.limiting_params["kernel"], x)
        Ktt = Ktt._add_diagonal(JITTER * identity(np))
        sqrt_K = Ktt.to_root()
        beta_term = jnp.sqrt(self.beta_schedule(t))
        diffusion = beta_term * sqrt_K
        return diffusion

    @check_shapes("t: []", "yt: [N, y_dim]", "x: [N, x_dim]", "return: [N, y_dim]")
    def score(self, key, t: Array, yt: Array, x: Array, network: ScoreNetwork) -> Array:
        """ This parametrise the preconditioned score K(x,x) grad log p(y_t|x) """
        score = network(t, yt, x, key=key)
        if self.std_trick:
            std = jnp.sqrt(1.0 - jnp.exp(-self.beta_schedule.B(t)))
            score = score / std
        if self.residual_trick:
            # NOTE: s.t. bwd SDE = fwd SDE
            fwd_drift = self.drift(t, yt, x)
            residual = 2 * fwd_drift / self.beta_schedule(t)
            score += residual
        return score

    @check_shapes("t: []", "yt: [N, y_dim]", "x: [N, x_dim]", "return: [N, y_dim]")
    def reverse_drift_ode(self, key, t: Array, yt: Array, x: Array, network) -> Array:
        return self.drift(t, yt, x) - 0.5 * self.beta_schedule(t) * self.score(
            key, t, yt, x, network
        )

    @check_shapes("t: []", "yt: [N, y_dim]", "x: [N, x_dim]", "return: [N, y_dim]")
    def reverse_drift_sde(self, key, t: Array, yt: Array, x: Array, network) -> Array:
        return self.drift(t, yt, x) - self.beta_schedule(t) * self.score(
            key, t, yt, x, network
        )

    @check_shapes("t: []", "y: [N, y_dim]", "x: [N, x_dim]", "return: []")
    def loss(self, key, t: Array, y: Array, x: Array, network: ScoreNetwork) -> Array:
        # TODO: this is DSM loss, refactor to enable ISM loss etc
        """ grad log p(y_t|y_0) = - \Sigma^-1 (y_t - mean) """

        factor = (1.0 - jnp.exp(-self.beta_schedule.B(t)))
        std = jnp.sqrt(factor)
        # TODO: allow for 'likelihood' weight with weight=diffusion**2?
        # if self.weighted:
        #     weight = factor
        # else:
        #     weight = 1.0

        ekey, nkey = jax.random.split(key)
        μ0t, k0t, params = self.p0t(t, y)
        dist = prior_gp(μ0t, k0t, params)(x)

        sqrt = dist.scale.to_root()
        Z = jax.random.normal(ekey, flatten(y).shape)
        affine_transformation = lambda x: dist.loc + sqrt @ x
        yt = unflatten(affine_transformation(Z), y.shape[-1])

        precond_score_net = self.score(nkey, t, yt, x, network)
        loss = jnp.mean(jnp.sum(jnp.square(std * precond_score_net + unflatten(sqrt @ Z, y.shape[-1])), -1))
        return loss

        yt = sample_prior_gp(ekey, μ0t, k0t, params, x)
        precond_score = -(yt - μ0t(params["mean_fn"], x))# / factor
        precond_score_net = self.score(nkey, t, yt, x, network)
        loss = jnp.mean(jnp.sum(jnp.square(precond_score - factor * precond_score_net), -1))
        # return factor * loss
        return loss / factor



def loss(sde: SDE, network: ScoreNetwork, batch: DataBatch, key):
    batch_size = len(batch.xs)
    t0 = sde.beta_schedule.t0
    t1 = sde.beta_schedule.t1

    key, tkey = jax.random.split(key)
    # Low-discrepancy sampling over t to reduce variance
    t = jax.random.uniform(tkey, (batch_size,), minval=t0, maxval=t1 / batch_size)
    t = t + (t1 / batch_size) * jnp.arange(batch_size)

    keys = jax.random.split(key, batch_size)
    losses = jax.vmap(sde.loss, in_axes=[0, 0, 0, 0, None])(
        keys, t, batch.ys, batch.xs, network
    )
    return jnp.mean(losses)


class MatVecControlTerm(dfx.ControlTerm):
    @staticmethod
    def prod(vf: PyTree, control: PyTree) -> PyTree:
        return jtu.tree_map(lambda a, b: a @ b, vf, control)

class LinOpControlTerm(dfx.ControlTerm):
    @staticmethod
    def prod(vf: LinearOperator, control: PyTree) -> PyTree:
        # return jtu.tree_map(lambda a, b: a @ b, vf, control)
        return vf @ control


def reverse_solve(
    sde: SDE,
    network: ScoreNetwork,
    x,
    *,
    key,
    prob_flow: bool = False,
    num_steps: int = 200,
    yT = None
):
    key, ykey = jax.random.split(key)
    yT = sde.sample_prior(ykey, x) if yT is None else yT
    y_dim = yT.shape[-1]
    yT = flatten(yT)

    t0, t1 = sde.beta_schedule.t0, sde.beta_schedule.t1
    dt = (t1 - t0) / num_steps  # TODO: dealing properly with endpoint?
    # ts = jnp.linspace(t0, t1, 9)[::-1]
    # saveat = dfx.SaveAt(ts=ts)

    if prob_flow:
        reverse_drift_ode = lambda t, yt, arg: sde.reverse_drift_ode(
            key, t, yt, arg, network
        )
        terms = dfx.ODETerm(reverse_drift_ode)
    else:
        shape = jax.ShapeDtypeStruct(yT.shape, yT.dtype)
        key, subkey = jax.random.split(key)
        bm = dfx.VirtualBrownianTree(t0=t1, t1=t0, tol=dt, shape=shape, key=subkey)
        reverse_drift_sde = lambda t, yt, arg: flatten(
            sde.reverse_drift_sde(key, t, unflatten(yt, y_dim), arg, network)
        )
        diffusion = lambda t, yt, arg: sde.diffusion(t, unflatten(yt, y_dim), arg)

        terms = dfx.MultiTerm(
            dfx.ODETerm(reverse_drift_sde), LinOpControlTerm(diffusion, bm)
        )
    # TODO: adaptive step?
    ys = dfx.diffeqsolve(
        terms,
        solver=dfx.Euler(),
        t0=t1,
        t1=t0,
        dt0=-dt,
        y0=yT,
        args=x,
        adjoint=dfx.NoAdjoint(),
    ).ys
    return unflatten(ys, y_dim)


@check_shapes(
    "x_context: [num_context, x_dim]",
    "y_context: [num_context, y_dim]",
    "x_test: [num_target, x_dim]",
    "return: [num_target, y_dim]",
)
def conditional_sample(
    sde: SDE,
    network: ScoreNetwork,
    x_context,
    y_context,
    x_test,
    *,
    key,
    num_steps: int = 100,
    num_inner_steps: int = 5
):
    # TODO: Langevin dynamics option
    num_context = len(x_context)
    num_target = len(x_test)
    y_dim = y_context.shape[-1]

    shape_augmented_state = [(num_context + num_target) * y_dim]
    t0 = sde.beta_schedule.t0
    t1 = sde.beta_schedule.t1
    ts = jnp.linspace(t1, t0, num_steps, endpoint=True)
    dt = ts[0] - ts[1]

    solver = dfx.Euler()
    # reverse ODE:
    reverse_drift_ode = lambda t, yt, arg: flatten(
        sde.reverse_drift_ode(key, t, unflatten(yt, y_dim), arg, network)
    )
    ode_terms_reverse = dfx.ODETerm(reverse_drift_ode)

    # TODO: argument for using reverse SDE vs ODE
    # reverse SDE:
    # shape = jax.ShapeDtypeStruct(shape_augmented_state, y_context.dtype)
    # key, subkey = jax.random.split(key)
    # bm = dfx.VirtualBrownianTree(t0=t1, t1=t0, tol=dt, shape=shape, key=subkey)
    # ode_terms_reverse = dfx.MultiTerm(dfx.ODETerm(reverse_drift_sde), dfx.ControlTerm(diffusion, bm))

    # forward SDE:
    shape = jax.ShapeDtypeStruct(shape_augmented_state, y_context.dtype)
    key, subkey = jax.random.split(key)
    bm = dfx.VirtualBrownianTree(t0=t0, t1=t1, tol=dt, shape=shape, key=subkey)
    diffusion = lambda t, yt, arg: sde.diffusion(t, unflatten(yt, y_dim), arg)
    drift = lambda t, yt, arg: flatten(sde.drift(t, unflatten(yt, y_dim), arg))
    sde_terms_forward = dfx.MultiTerm(
        dfx.ODETerm(drift), LinOpControlTerm(diffusion, bm)
    )

    def inner_loop(key, yt, t):
        yt_context = flatten(sde.sample_marginal(key, t, x_context, y_context))
        yt_augmented = jnp.concatenate([yt_context, yt], axis=0)
        x_augmented = jnp.concatenate([x_context, x_test], axis=0)

        # reverse step
        yt_m_dt, *_ = solver.step(
            ode_terms_reverse,
            t,
            t - dt,
            yt_augmented,
            x_augmented,
            None,
            made_jump=False,
        )

        # forward step
        yt, *_ = solver.step(
            sde_terms_forward, t - dt, t, yt_m_dt, x_augmented, None, made_jump=False
        )

        # strip context from augmented state
        return yt[num_context * y_dim :], yt_m_dt[num_context * y_dim :]

    def outer_loop(key, yt, t):
        _, yt_m_dt = jax.lax.scan(
            lambda yt, key: inner_loop(key, yt, t),
            yt,
            jax.random.split(key, num_inner_steps),
        )
        yt = yt_m_dt[-1]
        return yt, yt

    key, subkey = jax.random.split(key)
    yT = flatten(sde.sample_prior(subkey, x_test))

    xs = (ts[:-1], jax.random.split(key, len(ts) - 1))
    y0, _ = jax.lax.scan(lambda yt, x: outer_loop(x[1], yt, x[0]), yT, xs)
    return unflatten(y0, y_dim)

## Likelihood evaluation

def get_estimate_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    @check_shapes("t: []", "y: [N, y_dim]", "eps: [N, y_dim]", "return: []")
    def div_fn(t, y: jnp.ndarray, arg, eps: jnp.ndarray) -> jnp.ndarray:
        y_dim = y.shape[-1]
        flattened_fn = lambda y: flatten(fn(t, unflatten(y, y_dim), arg))
        _, vjp_fn = jax.vjp(flattened_fn, flatten(y))
        (eps_dfdy,) = vjp_fn(flatten(eps))
        return jnp.sum(eps_dfdy * flatten(eps), axis=-1)

    return div_fn


def get_exact_div_fn(fn):
    "flatten all but the last axis and compute the true divergence"

    @check_shapes("t: []", "y: [N, y_dim]", "return: []")
    def div_fn(t, y: jnp.ndarray, arg) -> jnp.ndarray:
        y_dim = y.shape[-1]
        flattened_fn = lambda t, y, arg: flatten(fn(t, unflatten(y, y_dim), arg))
        jac = jax.jacrev(flattened_fn, argnums=1)(t, flatten(y), arg)
        return jnp.trace(jac, axis1=-1, axis2=-2)

    return div_fn


def get_div_fn(drift_fn, hutchinson_type):
    """Pmapped divergence of the drift function."""
    if hutchinson_type == "None":
        return lambda y, t, context, eps: get_exact_div_fn(drift_fn)(y, t, context)
    else:
        return lambda y, t, context, eps: get_estimate_div_fn(drift_fn)(
            y, t, context, eps
        )


def div_noise(
    rng: jax.random.KeyArray, shape: Sequence[int], hutchinson_type: str
) -> jnp.ndarray:
    """Sample noise for the hutchinson estimator."""
    if hutchinson_type == "Gaussian":
        epsilon = jax.random.normal(rng, shape)
    elif hutchinson_type == "Rademacher":
        epsilon = (
            jax.random.randint(rng, shape, minval=0, maxval=2).astype(jnp.float32) * 2 - 1
        )
    elif hutchinson_type == "None":
        epsilon = None
    else:
        raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")
    return epsilon



@check_shapes("x: [N, x_dim]", "y: [N, y_dim]", "return: []")
def log_prob(
    sde: SDE,
    network: ScoreNetwork,
    x,
    y,
    *,
    key,
    num_steps: int = 100,
    solver: AbstractSolver = Dopri5(),
    stepsize_controller: AbstractStepSizeController = ConstantStepSize(),
    hutchinson_type = None
):
    reverse_drift_ode = lambda t, yt, arg: sde.reverse_drift_ode(
        key, t, yt, arg, network
    )

    div_fn = get_div_fn(reverse_drift_ode, hutchinson_type)
    def logp_wrapper(t, carry, static_args):
        yt, _ = carry
        eps, x = static_args

        drift = reverse_drift_ode(t, yt, x)
        logp = div_fn(t, yt, x, eps)

        return drift, logp


    t0 = sde.beta_schedule.t0
    t1 = sde.beta_schedule.t1
    dt = (t1 - t0) / num_steps

    term = dfx.ODETerm(logp_wrapper)
    # #NOTE: should we resample?
    eps = div_noise(key, y.shape, hutchinson_type)

    sol = dfx.diffeqsolve(
        term,
        solver,
        t1,
        t0,
        -dt,
        (y, 0.0),
        (eps, x),
        stepsize_controller=stepsize_controller,
    )
    yT, delta_logp = sol.ys
    yT, delta_logp = yT.squeeze(0), delta_logp.squeeze(0)
    logp_prior = sde.log_prob_prior(x, yT)

    return logp_prior + delta_logp
