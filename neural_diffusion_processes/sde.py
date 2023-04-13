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
from gpjax.mean_functions import Zero, Constant
from jaxlinop import LinearOperator, DenseLinearOperator, DiagonalLinearOperator, identity, ZeroLinearOperator
# import equinox as eqx
import numpy as np

from jaxtyping import Array, Float, PyTree
from check_shapes import check_shapes

from .utils.types import Tuple, Callable, Mapping, Sequence, Optional
from .data import DataBatch
from .kernels import prior_gp, sample_prior_gp, log_prob_prior_gp, promote_compute_engines, SumKernel
from .utils.misc import flatten, unflatten
from .config import get_config


class AbstractMeanFunction(gpjax.mean_functions.AbstractMeanFunction):
    def init_params(self, key):
        return {}

def scale_kernel_variance(params, coeff):
    if isinstance(params, list):
        for k in params:
            k["variance"] = k["variance"] * coeff
    else:
        params["variance"] = params["variance"] * coeff
    return params

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


@dataclasses.dataclass
class SDE:
    limiting_kernel: jaxkern.base.AbstractKernel
    limiting_mean_fn: gpjax.mean_functions.AbstractMeanFunction
    limiting_params: Mapping
    beta_schedule: LinearBetaSchedule
    std_trick: bool = True
    residual_trick: bool = True
    is_score_preconditioned: bool = True
    exact_score: bool = False

    def __post_init__(self):
        if self.exact_score:
            assert (
                not self.std_trick and
                not self.residual_trick# and
                # not self.is_score_preconditioned
            ), "Exact score. Do not apply re-parameterizations or preconditioning"

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

    @check_shapes("x: [N, x_dim]")
    def limiting_gram(self, x) -> LinearOperator:
        K = self.limiting_kernel.gram(self.limiting_params["kernel"], x)
        return K

    def p0t(
        self,
        t,
        y0: Callable[[Float[Array, "N x_dim"]], Float[Array, "N y_dim"]] | Float[Array, "N x_dim"],
    ) -> Tuple[AbstractMeanFunction, jaxkern.base.AbstractKernel, dict]:
        
        # E[Y_t|Y_0]
        mean_coef = jnp.exp(-0.5 * self.beta_schedule.B(t))
        class _Mean(AbstractMeanFunction):
            def __call__(self_, params: Mapping, x: Float[Array, "N D"]) -> Float[Array, "N Q"]:
                μT_value = self.limiting_mean_fn(self.limiting_params["mean_function"], x)
                return mean_coef * y0 + (1.0 - mean_coef) * μT_value
        μ0t = _Mean()

        # Cov[Y_t|Y_0]
        cov_coef = jnp.exp(-self.beta_schedule.B(t))
        k0t = self.limiting_kernel
        k0t_params = copy.deepcopy(self.limiting_params["kernel"])
        # k0t_params = scale_kernel_variance(k0t_params, 1.0 - cov_coef)
        if isinstance(k0t_params, list):
            for k in k0t_params:
                k["variance"] = k["variance"] * (1.0 - cov_coef)
                params = {"mean_function": {}, "kernel": [*k0t_params]}
        else:
            k0t_params["variance"] = k0t_params["variance"] * (1.0 - cov_coef)
            params = {"mean_function": {}, "kernel": k0t_params}
        return μ0t, k0t, params
    
    def pt(
        self,
        t,  
        y0: Callable[[Float[Array, "N x_dim"]], Float[Array, "N y_dim"]] | Float[Array, "N x_dim"],
        k0: jaxkern.base.AbstractKernel,
        k0_params: Mapping
    ) -> Tuple[AbstractMeanFunction, jaxkern.base.AbstractKernel, dict]:

        # E[Y_t|Y_0]
        mean_coef = jnp.exp(-0.5 * self.beta_schedule.B(t))
        class _Mean(AbstractMeanFunction):
            def __call__(self_, params: Mapping, x: Float[Array, "N D"]) -> Float[Array, "N Q"]:
                μT_value = self.limiting_mean_fn(self.limiting_params["mean_function"], x)
                # return mean_coef * y0(x) + (1.0 - mean_coef) * μT_value
                return mean_coef * y0 + (1.0 - mean_coef) * μT_value
        μ0t = _Mean()

        # Cov[Y_t|Y_0]
        cov_coef = jnp.exp(-self.beta_schedule.B(t))
        k0_params = copy.deepcopy(k0_params)
        k0_params = scale_kernel_variance(k0_params, cov_coef)
        kt_param = copy.deepcopy(self.limiting_params["kernel"])
        kt_param = scale_kernel_variance(kt_param, 1.0 - cov_coef)
        k0t = SumKernel([k0, self.limiting_kernel])
        # params = {"mean_function": {}, "kernel": [k0_params, kt_param]}
        
        k0_params = k0_params if isinstance(k0_params, list) else [k0_params]
        kt_param = kt_param if isinstance(kt_param, list) else [kt_param]
        params = {"mean_function": {}, "kernel": [*k0_params, *kt_param]}

        return μ0t, k0t, params


    @check_shapes("t: []", "x: [N, x_dim]", "y: [N, y_dim]", "return: [N, y_dim]")
    def sample_marginal(self, key, t: Array, x: Array, y: Array) -> Array:
        μ0t, k0t, params = self.p0t(t, y)
        return sample_prior_gp(key, μ0t, k0t, params, x)

    @check_shapes("t: []", "yt: [N, y_dim]", "x: [N, x_dim]", "return: [N, y_dim]")
    def drift(self, t: Array, yt: Array, x: Array) -> Array:
        μT = self.limiting_mean_fn(self.limiting_params["mean_function"], x)
        return -0.5 * self.beta_schedule(t) * (yt - μT)

    @check_shapes("t: []", "yt: [N, y_dim]", "x: [N, x_dim]")
    def diffusion(self, t, yt, x) -> LinearOperator:
        np = yt.shape[-2] * yt.shape[-1]
        del yt
        Ktt = self.limiting_gram(x)
        Ktt = Ktt._add_diagonal(get_config().jitter * identity(np))
        sqrt_K = Ktt.to_root()
        beta_term = jnp.sqrt(self.beta_schedule(t))
        diffusion = beta_term * sqrt_K
        return diffusion

    @check_shapes("t: []", "yt: [N, y_dim]", "x: [N, x_dim]", "return: [N, y_dim]")
    def score(self, key, t: Array, yt: Array, x: Array, network: ScoreNetwork) -> Array:
        """ This parametrises the (preconditioned) score K(x,x) grad log p(y_t|x) """
        score = network(t, yt, x, key=key)
        if self.std_trick:
            std = jnp.sqrt(1.0 - jnp.exp(-self.beta_schedule.B(t)))
            score = score / (std + 1e-3)
        if self.residual_trick:
            # NOTE: s.t. bwd SDE = fwd SDE
            # NOTE: wrong sign?
            # fwd_drift = self.drift(t, yt, x)
            # residual = 2 * fwd_drift / self.beta_schedule(t)
            residual = - yt
            score += residual
        return score
    
    def get_exact_score(self, mean0: AbstractMeanFunction, k0: jaxkern.base.AbstractKernel, params0: Mapping) -> ScoreNetwork:
        """Returns a ScoreNetwork which computes the returns the true score. Can only be computed for a Gaussian data dist."""

        class _ExactScoreNetwork(ScoreNetwork):
            "Exact marginal score in Gaussian setting"
            def __call__(self2, t: Array, yt: Array, x: Array, *, key) -> Array:
                y0 = mean0(params0["mean_function"], x)
                mu_t, k_t, params = self.pt(t, y0, k0, params0["kernel"])
                b = flatten(yt - mu_t(params['mean_function'], x))
                Sigma_t = k_t.gram(params['kernel'], x) + identity(np.prod(x.shape).item()) * get_config().jitter
                # Σ⁻¹ (yt - m_0(x))
                Sigma_inv_b = Sigma_t.solve(b)
                out = - Sigma_inv_b
                if self.is_score_preconditioned:
                   out = self.limiting_gram(x) @ out
                return unflatten(out, yt.shape[-1])

        return _ExactScoreNetwork()
        

    @check_shapes("t: []", "yt: [N, y_dim]", "x: [N, x_dim]", "return: [N, y_dim]")
    def reverse_drift_ode(self, key, t: Array, yt: Array, x: Array, network) -> Array:
        second_term = 0.5 * self.beta_schedule(t) * self.score(key, t, yt, x, network)
        if not self.is_score_preconditioned:
            second_term = unflatten(self.limiting_gram(x) @ flatten(second_term), yt.shape[-1])
        return self.drift(t, yt, x) - second_term


    @check_shapes("t: []", "yt: [N, y_dim]", "x: [N, x_dim]", "return: [N, y_dim]")
    def reverse_drift_sde(self, key, t: Array, yt: Array, x: Array, network) -> Array:
        second_term = self.beta_schedule(t) * self.score(key, t, yt, x, network)
        if not self.is_score_preconditioned:
            second_term = unflatten(self.limiting_gram(x) @ flatten(second_term), yt.shape[-1])
        return self.drift(t, yt, x) - second_term


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
        precond_noise = sqrt @ Z
        if not self.is_score_preconditioned:
            precond_noise = self.limiting_gram(x).solve(precond_noise)
        loss = jnp.square(std * precond_score_net + unflatten(precond_noise, y.shape[-1]))
        loss = jnp.mean(jnp.sum(loss, -1))
        return loss


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


############################
##        Sampling        ##
############################

class LinOpControlTerm(dfx.ControlTerm):
    @staticmethod
    def prod(vf: LinearOperator, control: PyTree) -> PyTree:
        # return jtu.tree_map(lambda a, b: a @ b, vf, control)
        return vf @ control


# class MatVecControlTerm(dfx.ControlTerm):
#     @staticmethod
#     def prod(vf: PyTree, control: PyTree) -> PyTree:
#         return jtu.tree_map(lambda a, b: a @ b, vf, control)


def sde_solve(
    sde: SDE,
    network: ScoreNetwork,
    x,
    *,
    key,
    prob_flow: bool = False,
    num_steps: int = 100,
    y = None,
    solver: AbstractSolver = dfx.Heun(),
    rtol: float = 1e-3,
    atol: float = 1e-4,
    forward: bool = False,
    ts = None
):
    if rtol is None or atol is None:
        stepsize_controller = ConstantStepSize()
    else:
        stepsize_controller =  dfx.PIDController(rtol=rtol, atol=atol)

    key, ykey = jax.random.split(key)
    y = sde.sample_prior(ykey, x) if y is None else y
    y_dim = y.shape[-1]
    y = flatten(y)

    #NOTE: default is time-reversal
    if forward:
        t0, t1 = sde.beta_schedule.t0, sde.beta_schedule.t1
        drift_sde = lambda t, yt, arg: flatten(
            sde.drift(t, unflatten(yt, y_dim), arg)
        )
    else:
        t1, t0 = sde.beta_schedule.t0, sde.beta_schedule.t1
        drift_sde = lambda t, yt, arg: flatten(
            sde.reverse_drift_sde(key, t, unflatten(yt, y_dim), arg, network)
        )
    dt = (t1 - t0) / num_steps  # TODO: dealing properly with endpoint?

    if prob_flow:
        reverse_drift_ode = lambda t, yt, arg: flatten(sde.reverse_drift_ode(
            key, t, unflatten(yt, y_dim), arg, network
        ))
        terms = dfx.ODETerm(reverse_drift_ode)
    else:
        shape = jax.ShapeDtypeStruct(y.shape, y.dtype)
        key, subkey = jax.random.split(key)
        bm = dfx.VirtualBrownianTree(t0=t0, t1=t1, tol=jnp.abs(dt), shape=shape, key=subkey)
        diffusion = lambda t, yt, arg: sde.diffusion(t, unflatten(yt, y_dim), arg)

        terms = dfx.MultiTerm(
            dfx.ODETerm(drift_sde), LinOpControlTerm(diffusion, bm)
        )
    
    saveat = dfx.SaveAt(t1=True) if ts is None else dfx.SaveAt(ts=ts)
    out = dfx.diffeqsolve(
        terms,
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=y,
        args=x,
        # adjoint=dfx.NoAdjoint(),
        stepsize_controller=stepsize_controller,
        saveat=saveat
    )
    ys = out.ys
    return unflatten(ys, y_dim)

# @check_shapes(
#     "x_context: [num_context, x_dim]",
#     "y_context: [num_context, y_dim]",
#     "x_test: [num_target, x_dim]",
#     "return: [num_target, y_dim]",
# )
# def conditional_sample(
#     sde: SDE,
#     network: ScoreNetwork,
#     x_context,
#     y_context,
#     x_test,
#     *,
#     key,
#     num_steps: int = 100,
#     num_inner_steps: int = 5
# ):
#     # TODO: Langevin dynamics option
#     num_context = len(x_context)
#     num_target = len(x_test)
#     y_dim = y_context.shape[-1]

#     shape_augmented_state = [(num_context + num_target) * y_dim]
#     t0 = sde.beta_schedule.t0
#     t1 = sde.beta_schedule.t1
#     ts = jnp.linspace(t1, t0, num_steps, endpoint=True)
#     dt = ts[0] - ts[1]

#     solver = dfx.Euler()
#     # reverse ODE:
#     reverse_drift_ode = lambda t, yt, arg: flatten(
#         sde.reverse_drift_ode(key, t, unflatten(yt, y_dim), arg, network)
#     )
#     ode_terms_reverse = dfx.ODETerm(reverse_drift_ode)

#     # TODO: argument for using reverse SDE vs ODE
#     # reverse SDE:
#     # shape = jax.ShapeDtypeStruct(shape_augmented_state, y_context.dtype)
#     # key, subkey = jax.random.split(key)
#     # bm = dfx.VirtualBrownianTree(t0=t1, t1=t0, tol=dt, shape=shape, key=subkey)
#     # ode_terms_reverse = dfx.MultiTerm(dfx.ODETerm(reverse_drift_sde), dfx.ControlTerm(diffusion, bm))

#     # forward SDE:
#     shape = jax.ShapeDtypeStruct(shape_augmented_state, y_context.dtype)
#     key, subkey = jax.random.split(key)
#     bm = dfx.VirtualBrownianTree(t0=t0, t1=t1, tol=dt, shape=shape, key=subkey)
#     diffusion = lambda t, yt, arg: sde.diffusion(t, unflatten(yt, y_dim), arg)
#     drift = lambda t, yt, arg: flatten(sde.drift(t, unflatten(yt, y_dim), arg))
#     sde_terms_forward = dfx.MultiTerm(
#         dfx.ODETerm(drift), LinOpControlTerm(diffusion, bm)
#     )

#     def inner_loop(key, yt, t):
#         yt_context = flatten(sde.sample_marginal(key, t, x_context, y_context))
#         yt_augmented = jnp.concatenate([yt_context, yt], axis=0)
#         x_augmented = jnp.concatenate([x_context, x_test], axis=0)

#         # reverse step
#         yt_m_dt, *_ = solver.step(
#             ode_terms_reverse,
#             t,
#             t - dt,
#             yt_augmented,
#             x_augmented,
#             None,
#             made_jump=False,
#         )

#         # forward step
#         yt, *_ = solver.step(
#             sde_terms_forward, t - dt, t, yt_m_dt, x_augmented, None, made_jump=False
#         )

#         # strip context from augmented state
#         return yt[num_context * y_dim :], yt_m_dt[num_context * y_dim :]

#     def outer_loop(key, yt, t):
#         _, yt_m_dt = jax.lax.scan(
#             lambda yt, key: inner_loop(key, yt, t),
#             yt,
#             jax.random.split(key, num_inner_steps),
#         )
#         yt = yt_m_dt[-1]
#         return yt, yt

#     key, subkey = jax.random.split(key)
#     yT = flatten(sde.sample_prior(subkey, x_test))

#     xs = (ts[:-1], jax.random.split(key, len(ts) - 1))
#     y0, _ = jax.lax.scan(lambda yt, x: outer_loop(x[1], yt, x[0]), yT, xs)
#     return unflatten(y0, y_dim)

# @eqx.filter_jit
@check_shapes(
    "x_context: [num_context, x_dim]",
    "y_context: [num_context, y_dim]",
    "x_test: [num_target, x_dim]",
    "return: [num_target, y_dim]",
)
def conditional_sample2(
    sde: SDE,
    network: ScoreNetwork,
    x_context,
    y_context,
    x_test,
    *,
    key,
    num_steps: int = 100,
    num_inner_steps: int = 5,
    prob_flow: bool = True,
    langevin_kernel = True,
    psi: float = 1.,
    lambda0: float = 1.,
    tau: float = None,
):
    # TODO: Langevin dynamics option

    num_context = len(x_context)
    num_target = len(x_test)
    y_dim = y_context.shape[-1]
    shape_augmented_state = [(num_context + num_target) * y_dim]
    x_augmented = jnp.concatenate([x_context, x_test], axis=0)

    t0 = sde.beta_schedule.t0
    t1 = sde.beta_schedule.t1
    ts = jnp.linspace(t1, t0, num_steps, endpoint=True)
    dt = ts[0] - ts[1]
    tau = tau if tau is not None else t1

    solver = dfx.Euler()

    diffusion = lambda t, yt, arg: sde.diffusion(t, unflatten(yt, y_dim), arg)
    if not prob_flow:
        # reverse SDE:
        reverse_drift_sde = lambda t, yt, arg: flatten(sde.reverse_drift_sde(key, t, unflatten(yt, y_dim), arg, network))

        shape = jax.ShapeDtypeStruct(shape_augmented_state, y_context.dtype)
        key, subkey = jax.random.split(key)
        bm = dfx.VirtualBrownianTree(t0=t1, t1=t0, tol=dt, shape=shape, key=key)
        terms_reverse = dfx.MultiTerm(
            dfx.ODETerm(reverse_drift_sde), LinOpControlTerm(diffusion, bm)
        )
    else:
        # reverse ODE:
        reverse_drift_ode = lambda t, yt, arg: flatten(
            sde.reverse_drift_ode(key, t, unflatten(yt, y_dim), arg, network)
        )
        terms_reverse = dfx.ODETerm(reverse_drift_ode)

    # langevin dynamics:
    def reverse_drift_langevin(t, yt, x) -> Array:
        yt = unflatten(yt, y_dim)
        score = flatten(sde.score(key, t, yt, x, network))
        if langevin_kernel:
            if sde.is_score_preconditioned:
                score = score
            else:
                score = sde.limiting_gram(x) @ score
        else:
            if sde.is_score_preconditioned:
                score = sde.limiting_gram(x).solve(score)
            else:
                score = score
        return 0.5 * sde.beta_schedule(t) * score
    
    def diffusion_langevin(t, yt, x) -> LinearOperator:
        if langevin_kernel:
            return diffusion(t, yt, x)
        else:
            return jnp.sqrt(sde.beta_schedule(t)) * identity(yt.shape[-1])

    key, subkey = jax.random.split(key)
    shape = jax.ShapeDtypeStruct(shape_augmented_state, y_context.dtype)
    bm = dfx.VirtualBrownianTree(t0=t0, t1=t1, tol=dt, shape=shape, key=subkey)
    # bm = dfx.UnsafeBrownianPath(shape=shape, key=subkey)
    langevin_terms = dfx.MultiTerm(
        dfx.ODETerm(reverse_drift_langevin),
        LinOpControlTerm(diffusion_langevin, bm)
    )


    def sample_marginal(key, t, x_context, y_context):
        if len(y_context) == 0:
            return y_context
        else:
            return flatten(sde.sample_marginal(key, t, x_context, y_context))


    def inner_loop(key, ys, t):
        # reverse step
        yt, yt_context = ys
        yt_context = sample_marginal(key, t, x_context, y_context) # NOTE: should resample?
        yt_augmented = jnp.concatenate([yt_context, yt], axis=0)

        # yt_m_dt, *_ = solver.step(
        #     langevin_terms,
        #     t - dt,
        #     t,
        #     # t + dt,
        #     yt_augmented,
        #     x_augmented,
        #     None,
        #     made_jump=False,
        # )

        yt_m_dt = yt_augmented
        yt_m_dt += lambda0 * psi * dt * reverse_drift_langevin(t - dt, yt_augmented, x_augmented)
        noise = jnp.sqrt(psi) * jnp.sqrt(dt) * jax.random.normal(key, shape=yt_augmented.shape)
        yt_m_dt += diffusion_langevin(t - dt, yt_augmented, x_augmented) @ noise
        # yt_m_dt += langevin_terms.contr(t, t)[0] * langevin_terms.vf(t, yt_augmented, x_augmented)[0]
        # yt_m_dt += langevin_terms.vf(t, yt_augmented, x_augmented)[1] @ noise
        
        yt = yt_m_dt[num_context * y_dim :]
        # strip context from augmented state
        return (yt, yt_context), yt_m_dt

    def outer_loop(key, yt, t):
        # jax.debug.print("time {t}", t=t)

        # yt_context = sde.sample_marginal(key, t, x_context, y_context)
        yt_context = sample_marginal(key, t, x_context, y_context)
        # yt_context = y_context #NOTE: doesn't need to be noised?
        yt_augmented = jnp.concatenate([yt_context, yt], axis=0)

        yt_m_dt, *_ = solver.step(
            terms_reverse,
            t,
            t - dt,
            yt_augmented,
            x_augmented,
            None,
            made_jump=False,
        )
        # yt_m_dt = yt_augmented 
        # yt_m_dt += -dt * reverse_drift_diffeq(t, yt_augmented, x_augmented)
        # # yt_m_dt += terms_reverse.contr(t, t-dt) * terms_reverse.vf(t, yt_augmented, x_augmented)
        # noise = jax.random.normal(key, shape=yt_augmented.shape)
        # yt_m_dt += jnp.sqrt(dt) * sde.diffusion(t, yt_augmented, x_augmented) @ noise

        def corrector(key, yt, yt_context, t):
            _, yt_m_dt = jax.lax.scan(
                lambda ys, key: inner_loop(key, ys, t),
                (yt, yt_context),
                jax.random.split(key, num_inner_steps),
            )
            yt = yt_m_dt[-1][num_context * y_dim :]
            return yt
    
        yt = jax.lax.cond(
            tau > t,
            corrector,
            lambda key, yt, yt_context, t: yt,
            key, yt, yt_context, t
        )
        return yt, yt

    key, subkey = jax.random.split(key)
    yT = flatten(sde.sample_prior(subkey, x_test))

    xs = (ts[:-1], jax.random.split(key, len(ts) - 1))
    y0, _ = jax.lax.scan(lambda yt, x: outer_loop(x[1], yt, x[0]), yT, xs)
    return unflatten(y0, y_dim)


############################
## Likelihood evaluation  ##
############################

def get_estimate_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    @check_shapes("t: []", "y: [N, y_dim]", "eps: [K, M, y_dim]", "return: []")
    def div_fn(t, y: jnp.ndarray, arg, eps: jnp.ndarray, num_context = None) -> jnp.ndarray:
        y_dim = y.shape[-1]
        if num_context:
            yc = y[:num_context * y_dim]
            y = y[num_context * y_dim :]
            flattened_fn = lambda y: flatten(fn(t, jnp.concatenate([yc, unflatten(y, y_dim)], axis=0), arg))[num_context * y_dim :]
        else:
            flattened_fn = lambda y: flatten(fn(t, unflatten(y, y_dim), arg))
        _, vjp_fn = jax.vjp(flattened_fn, flatten(y) )

        # NOTE: sequential
        # def f(carry, eps):
        #     (eps_dfdy,) = vjp_fn(flatten(eps))
        #     trace_estimate = jnp.sum(eps_dfdy * flatten(eps), axis=-1)
        #     return carry + trace_estimate, None
        
        # sums, _ = jax.lax.scan(f, 0., eps)
        # return sums / eps.shape[0]

        # NOTE: parallel
        def f(eps):
            (eps_dfdy,) = vjp_fn(flatten(eps))
            trace_estimate = jnp.sum(eps_dfdy * flatten(eps), axis=-1)
            return trace_estimate
        
        return jax.vmap(f)(eps).mean(axis=0)

    return div_fn


def get_exact_div_fn(fn):
    "flatten all but the last axis and compute the true divergence"

    @check_shapes("t: []", "y: [N, y_dim]", "return: []")
    def div_fn(t, y, arg, num_context = None) -> jnp.ndarray:
        y_dim = y.shape[-1]
        if num_context:
            yc = y[:num_context * y_dim]
            y = y[num_context * y_dim :]
            flattened_fn = lambda y: flatten(fn(t, jnp.concatenate([yc, unflatten(y, y_dim)], axis=0), arg))[num_context * y_dim :]
        else:
            flattened_fn = lambda y: flatten(fn(t, unflatten(y, y_dim), arg))
        jac = jax.jacrev(flattened_fn)(flatten(y))
        return jnp.trace(jac, axis1=-1, axis2=-2)

    return div_fn


def get_div_fn(drift_fn, hutchinson_type):
    """Divergence of the drift function."""
    if hutchinson_type == "None":
        return lambda y, t, context, eps, **kwargs: get_exact_div_fn(drift_fn)(y, t, context, **kwargs)
    else:
        return lambda y, t, context, eps, **kwargs: get_estimate_div_fn(drift_fn)(
            y, t, context, eps, **kwargs
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


@check_shapes("x: [N, x_dim]", "y: [N, y_dim]")#, "return: []")
def log_prob(
    sde: SDE,
    network: ScoreNetwork,
    x,
    y,
    *,
    key,
    # dt=1e-3/2,
    num_steps: int = 100,
    solver: AbstractSolver = Tsit5(),
    rtol: float = 1e-3,
    atol: float = 1e-4,
    hutchinson_type: str = 'None',
    hutchinson_samples: int = 1,
    forward: bool = True,
    ts = None,
    return_all: bool = False,
    x_known: Optional[Array] = None,
    y_known: Optional[Array] = None,
):

    if rtol is None or atol is None:
        stepsize_controller = ConstantStepSize()
    else:
        stepsize_controller =  dfx.PIDController(rtol=rtol, atol=atol)

    if forward:
        t0, t1 = sde.beta_schedule.t0, sde.beta_schedule.t1
    else:
        t1, t0 = sde.beta_schedule.t0, sde.beta_schedule.t1
        # dt = -1.0 * abs(dt)
    dt = (t1 - t0) / num_steps
    # dt = 1e-3/2.

    reverse_drift_ode = lambda t, yt, arg: sde.reverse_drift_ode(
        key, t, yt, arg, network
    )
    div_fn = get_div_fn(reverse_drift_ode, hutchinson_type)
    eps = div_noise(key, (hutchinson_samples, *y.shape), hutchinson_type)

    if x_known is not None and y_known is not None:
        num_context = len(x_known)
        x = jnp.concatenate([x_known, x], axis=0)
        y = jnp.concatenate([y_known, y], axis=0)
        div_fn = partial(div_fn, num_context=num_context)

    # eps = div_noise(key, y.shape, hutchinson_type)
    y_dim = y.shape[-1]
    y = flatten(y)

    @jax.jit
    def logp_wrapper(t, carry, static_args):
        yt, _ = carry
        eps, x = static_args
        yt = unflatten(yt, y_dim)

        drift = flatten(reverse_drift_ode(t, yt, x))
        logp = div_fn(t, yt, x, eps)
        return drift, logp

    terms = dfx.ODETerm(logp_wrapper)
    #NOTE: should we resample?
    saveat = dfx.SaveAt(t1=True) if ts is None else dfx.SaveAt(ts=ts)

    sol = dfx.diffeqsolve(
        terms,
        solver,
        t0=t0,
        t1=t1,#-1e-3,
        dt0=dt,
        y0=(y, 0.0),
        args=(eps, x),
        # adjoint=dfx.NoAdjoint(),
        stepsize_controller=stepsize_controller,
        saveat=saveat
    )
    yT, delta_logp = sol.ys
    yT = unflatten(yT, y_dim)
    nfe = sol.stats['num_steps']
    logp_prior = jax.vmap(lambda y: sde.log_prob_prior(x, y))(yT)

    if return_all:
        return logp_prior, delta_logp, nfe, yT
    else:
        return logp_prior + delta_logp, nfe
