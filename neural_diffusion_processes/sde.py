from abc import abstractmethod
from typing import Callable, Mapping

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
from jaxlinop import identity

from jaxtyping import Array, Float, PyTree
from check_shapes import check_shapes

from .types import Tuple
from .data import DataBatch
from .kernels import sample_prior_gp, log_prob_prior_gp
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
        # TODO: Notice the additional scaling by the interval t1-t0.
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
        self.weighted = True

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

    @check_shapes("t: []", "y0: [N, y_dim]")
    def p0t(self, t, y0) -> Tuple[AbstractMeanFunction, jaxkern.base.AbstractKernel, dict]:
        # TODO: add equations as method description
        mean_coef = jnp.exp(-0.5 * self.beta_schedule.B(t))
        # E[Y_t|Y_0]
        μ0t = Constant(output_dim=self.limiting_mean_fn.output_dim)
        μT_value = self.limiting_params["mean_fn"].get(
            "constant", jnp.zeros(shape=μ0t.output_dim)
        )
        μ0t_params = {"constant": mean_coef * y0 + (1.0 - mean_coef) * μT_value}
        # Cov[Y_t|Y_0]
        k0t_param = copy.copy(self.limiting_params["kernel"])
        k0t_param["variance"] = k0t_param["variance"] * (
            1.0 - jnp.exp(-self.beta_schedule.B(t))
        )
        params = {"mean_fn": μ0t_params, "kernel": k0t_param}
        return μ0t, self.limiting_kernel, params

    @check_shapes("t: []", "x: [N, x_dim]", "y: [N, y_dim]", "return: [N, y_dim]")
    def sample_marginal(self, key, t, x, y):
        μ0t, k0t, params = self.p0t(t, y)
        return sample_prior_gp(key, μ0t, k0t, params, x)

    @check_shapes("t: []", "yt: [N, y_dim]", "x: [N, x_dim]", "return: [N, y_dim]")
    def drift(self, t, yt, x):
        μT = self.limiting_mean_fn(self.limiting_params["mean_fn"], x)
        return -0.5 * self.beta_schedule(t) * (yt - μT)  # [N, 1]

    # @check_shapes("t: []", "yt: [N, y_dim]", "x: [N, x_dim]", "return: [N, N]")
    # @check_shapes(
    #     "t: []", "yt: [N, y_dim]", "x: [N, x_dim]", "return: [N * y_dim, N * y_dim]"
    # )
    def diffusion(self, t, yt, x):
        np = yt.shape[-2] * yt.shape[-1]
        # print("yt", type(yt), yt.shape)
        del yt
        Ktt = self.limiting_kernel.gram(self.limiting_params["kernel"], x)
        Ktt += JITTER * identity(np)
        # print("Ktt", type(Ktt), Ktt.shape)
        sqrt_K = Ktt.to_root()
        # print("sqrt_K", type(sqrt_K), sqrt_K.shape)
        beta_term = jnp.sqrt(self.beta_schedule(t))
        # print("beta_term", type(beta_term), beta_term.shape)
        diffusion = beta_term * sqrt_K
        # print("diffusion", type(diffusion), (diffusion).shape)
        # TODO: beta_term multiplication is loosing the linear op structure!
        return diffusion.to_dense()

    @check_shapes("t: []", "yt: [N, y_dim]", "x: [N, x_dim]", "return: [N, y_dim]")
    def score(self, key, t: Array, yt: Array, x: Array, network: ScoreNetwork) -> Array:
        factor = (1.0 - jnp.exp(self.beta_schedule.B(t))) ** -1
        score = factor * network(t, yt, x, key=key)
        return score

    @check_shapes("t: []", "yt: [N, y_dim]", "x: [N, x_dim]", "return: [N, y_dim]")
    def reverse_drift_ode(self, key, t, yt, x, network):
        return self.drift(t, yt, x) - 0.5 * self.beta_schedule(t) * self.score(
            key, t, yt, x, network
        )

    @check_shapes("t: []", "yt: [N, y_dim]", "x: [N, x_dim]", "return: [N, y_dim]")
    def reverse_drift_sde(self, key, t, yt, x, network):
        return self.drift(t, yt, x) - self.beta_schedule(t) * self.score(
            key, t, yt, x, network
        )

    @check_shapes("t: []", "y: [N, y_dim]", "x: [N, x_dim]", "return: []")
    def loss(self, key, t: Array, y: Array, x: Array, network: ScoreNetwork) -> Array:

        if self.weighted:
            weight = 1 - jnp.exp(-self.beta_schedule.B(t))
        else:
            weight = 1.0

        ekey, nkey = jax.random.split(key)
        μ0t, k0t, params = self.p0t(t, y)
        yt = sample_prior_gp(ekey, μ0t, k0t, params, x)
        objective = -(yt - μ0t(params["mean_fn"], x))

        precond_score_pt = network(t, yt, x, key=nkey)
        return weight * jnp.mean(jnp.sum((objective - precond_score_pt) ** 2, -1), -1)


def loss(sde: SDE, network: ScoreNetwork, batch: DataBatch, key):
    batch_size = len(batch.xs)
    t0 = sde.beta_schedule.t0
    t1 = sde.beta_schedule.t1

    key, tkey = jax.random.split(key)
    # Low-discrepancy sampling over t to reduce variance
    t = jax.random.uniform(tkey, (batch_size,), minval=t0, maxval=t1 / batch_size)
    t = t + (t1 / batch_size) * jnp.arange(batch_size)

    keys = jax.random.split(key, batch_size)
    error = jax.vmap(sde.loss, in_axes=[0, 0, 0, 0, None])(
        keys, t, batch.ys, batch.xs, network
    )
    return jnp.mean(error)


class MatVecControlTerm(dfx.ControlTerm):
    @staticmethod
    def prod(vf: PyTree, control: PyTree) -> PyTree:
        # TODO: use linop structure
        # print("vf", type(vf), vf.shape)
        # print("control", type(control), control.shape)
        # print("(vf @ control)", type((vf @ control)), (vf @ control).shape)
        return jtu.tree_map(lambda a, b: a @ b, vf, control)


# def reverse_solve(sde: SDE, network: ScoreNetwork, x, *, key, prob_flow: bool = True):
def reverse_solve(
    sde: SDE,
    network: ScoreNetwork,
    x,
    *,
    key,
    prob_flow: bool = False,
    num_steps: int = 200
):
    key, ykey = jax.random.split(key)
    yT = sde.sample_prior(ykey, x)
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
            dfx.ODETerm(reverse_drift_sde), MatVecControlTerm(diffusion, bm)
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
        dfx.ODETerm(drift), MatVecControlTerm(diffusion, bm)
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

# def get_div_fn(drift_fn, hutchinson_type: str = "None"):
#     """Pmapped divergence of the drift function."""
#     if hutchinson_type == "None":
#         return lambda y, t, context, eps: get_exact_div_fn(drift_fn)(y, t, context)
#     else:
#         return lambda y, t, context, eps: get_estimate_div_fn(drift_fn)(
#             y, t, context, eps
#         )


# def div_noise(
#     rng: jax.random.KeyArray, shape: Sequence[int], hutchinson_type: str
# ) -> jnp.ndarray:
#     """Sample noise for the hutchinson estimator."""
#     if hutchinson_type == "Gaussian":
#         epsilon = jax.random.normal(rng, shape)
#     elif hutchinson_type == "Rademacher":
#         epsilon = (
#             jax.random.randint(rng, shape, minval=0, maxval=2).astype(jnp.float32) * 2 - 1
#         )
#     elif hutchinson_type == "None":
#         epsilon = None
#     else:
#         raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")
#     return epsilon



# def get_estimate_div_fn(fn):
#     """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

#     @check_shapes("t: []", "y: [N, y_dim]", "return: []")
#     def div_fn(t, y: jnp.ndarray, arg, eps: jnp.ndarray):
#         y_dim = y.shape[-1]
#         eps = flatten(eps)
#         flattened_fn = lambda t, y, arg: flatten(fn(t, unflatten(y, y_dim), arg))
#         # eps = eps.reshape(eps.shape[0], -1)
#         # grad_fn = lambda y: jnp.sum(fn(y, t, arg) * eps)
#         # grad_fn_eps = jax.grad(grad_fn)(y).reshape(y.shape[0], -1)
#         # return jnp.sum(grad_fn_eps * eps, axis=tuple(range(1, len(eps.shape))))
#         grad_fn = lambda y: jnp.sum(flattened_fn(y, t, arg) * eps)
#         grad_fn_eps = jax.grad(grad_fn)(flatten(y))
#         print("y", y.shape)
#         print("eps", eps.shape)
#         print("grad_fn_eps", grad_fn_eps.shape)
#         return jnp.sum(grad_fn_eps * eps, axis=-1)

#     return div_fn


def get_exact_div_fn(fn):
    "flatten all but the last axis and compute the true divergence"

    @check_shapes("t: []", "y: [N, y_dim]", "return: []")
    def div_fn(t, y: jnp.ndarray, arg):
        y_dim = y.shape[-1]
        flattened_fn = lambda t, y, arg: flatten(fn(t, unflatten(y, y_dim), arg))
        jac = jax.jacrev(flattened_fn, argnums=1)(t, flatten(y), arg)
        return jnp.trace(jac, axis1=-1, axis2=-2)

    return div_fn


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
    stepsize_controller: AbstractStepSizeController = ConstantStepSize()
):
    reverse_drift_ode = lambda t, yt, arg: sde.reverse_drift_ode(
        key, t, yt, arg, network
    )

    # div_fn = get_estimate_div_fn(reverse_drift_ode)
    div_fn = get_exact_div_fn(reverse_drift_ode)
    def logp_wrapper(t, carry, static_args):
        yt, _ = carry
        _, x = static_args

        drift = reverse_drift_ode(t, yt, x)
        logp = div_fn(t, yt, x)

        return drift, logp


    t0 = sde.beta_schedule.t0
    t1 = sde.beta_schedule.t1
    dt = (t1 - t0) / num_steps

    # TODO: approx vs exact as argument
    # term = dfx.ODETerm(approx_logp_wrapper)
    term = dfx.ODETerm(logp_wrapper)

    eps = jax.random.normal(key, y.shape)

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
