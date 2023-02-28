from abc import abstractmethod
from typing import Callable, Mapping

from functools import partial
import dataclasses
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import diffrax as dfx
import numpy as np
import jaxkern
import jax.tree_util as jtu

from jaxtyping import Array, Float, PyTree
from check_shapes import check_shapes

from .data import DataBatch
from .misc import sample_mvn


class ScoreNetwork(Callable):

    @abstractmethod
    @check_shapes(
        "t: []",
        "yt: [num_points, output_dim]",
        "x: [num_points, input_dim]",
        "return: [num_points, output_dim]",
    )
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
            self.beta0 * normed_t + 0.5 * (normed_t ** 2) * (self.beta1 - self.beta0)
        )


class SDE:

    def __init__(self, limiting_kernel: jaxkern.base.AbstractKernel, limiting_kernel_params: Mapping, beta_schedule: LinearBetaSchedule):
        self.limiting_kernel = limiting_kernel
        self.limiting_kernel_params = limiting_kernel_params
        self.is_diag = isinstance(limiting_kernel, jaxkern.White)
        self.limiting_mean_fn = lambda x: jnp.zeros_like(x[..., :1])
        self.beta_schedule = beta_schedule
        self.weighted = True
    
    @check_shapes(
        "x: [num_points, 1]",
        "return: [num_points, 1]",
    )
    def sample_prior(self, key, x):
        gram = self.limiting_kernel.gram(self.limiting_kernel_params, x).to_dense()
        if self.is_diag:
            eps = jax.random.normal(key, (len(x), 1))
            yT = jnp.sqrt(jnp.diag(gram))[:, None] * eps
        else:
            yT = sample_mvn(key, jnp.zeros_like(x), gram)
        return yT

    @check_shapes(
        "t: []",
        "y0: [num_points, 1]",
        "x: [num_points, 1]",
        "return[0]: [num_points, 1]",
        "return[1]: [num_points, num_points] if full_cov",
        "return[1]: [num_points, 1] if not full_cov",
    )
    def _pt(self, t, y0, x, full_cov: bool):
        mean_coef = jnp.exp(-0.5 * self.beta_schedule.B(t))
        mean = mean_coef * y0 + (1.0 - mean_coef) * self.limiting_mean_fn(x)
        kT = self.limiting_kernel.gram(self.limiting_kernel_params, x).to_dense()
        cov = (1.0 -  jnp.exp(-self.beta_schedule.B(t))) * kT
        if full_cov:
            return mean, cov
        else:
            return mean, jnp.diag(cov)[..., None]
    
    def pt(self, t, y0, x):
        return self._pt(t, y0, x, full_cov=not self.is_diag)

    @check_shapes("t: []", "yt: [N, 1]", "x: [N, 1]", "return: [N, 1]",)
    def drift(self, t, yt, x):
        return  - 0.5 * self.beta_schedule(t) * (yt - self.limiting_mean_fn(x)) # [N, 1]

    @check_shapes("t: []", "yt: [N, 1]", "x: [N, 1]", "return: [N, N]",)
    def diffusion(self, t, yt, x):
        del yt
        kT = self.limiting_kernel.gram(self.limiting_kernel_params, x).to_dense()
        sqrt_K = jnp.linalg.cholesky(kT + 1e-12 * jnp.eye(len(x)))
        return jnp.sqrt(self.beta_schedule(t)) * sqrt_K

    @check_shapes(
        "t: []",
        "y: [num_points, output_dim]",
        "x: [num_points, input_dim]",
        "return: []"
    )
    def loss(
        self,
        key,
        t: Array,
        y: Array,
        x: Array,
        network: ScoreNetwork,
    ) -> Array:

        if self.weighted:
            weight = 1 - jnp.exp(-self.beta_schedule.B(t))
        else:
            weight = 1.

        mean, cov = self.pt(t, y, x)
        
        ekey, nkey = jax.random.split(key)
        eps = jax.random.normal(ekey, mean.shape)

        if self.is_diag:
            yt = mean + jnp.sqrt(cov) * eps
            objective = eps / jnp.sqrt(cov)
        else:
            L = jnp.linalg.cholesky(cov + 1e-12 * jnp.eye(len(mean)))
            yt = mean + L @ eps
            raise NotImplementedError
            # TODO objective L^{-1} eps
            # objective = eps / jnp.sqrt(cov)

        precond_score_pt = network(t, yt, x, key=nkey)
        return weight * jnp.mean(jnp.sum((objective - precond_score_pt)**2, -1), -1)
    
    @check_shapes(
        "t: []",
        "yt: [num_points, output_dim]",
        "x: [num_points, input_dim]",
        "return: [num_points, output_dim]"
    )
    def score(
        self,
        key,
        t: Array,
        yt: Array,
        x: Array,
        network: ScoreNetwork
    ) -> Array:
        # covariance doesn't depend on y0
        _, cov = self.pt(t, jnp.ones_like(yt) * jnp.nan, x)
        factor = (1.0 -  jnp.exp(-0.5 * self.beta_schedule.B(t))) ** -1
        if self.is_diag:
            score = - factor * cov * network(t, yt, x, key=key)
        else:
            raise NotImplementedError
        return score

    @check_shapes(
        "t: []",
        "yt: [N, 1]",
        "x: [N, 1]",
        "return: [N, 1]",
    )
    def reverse_drift_ode(self, key, t, yt, x, network):
        return self.drift(t, yt, x) - 0.5 * self.beta_schedule(t) * self.score(key, t, yt, x, network)  # [N, 1]


def loss(sde: SDE, network: ScoreNetwork, batch: DataBatch, key):
    batch_size = len(batch.function_inputs)
    t0 = sde.beta_schedule.t0
    t1 = sde.beta_schedule.t1

    key, tkey = jax.random.split(key)
    # Low-discrepancy sampling over t to reduce variance
    t = jax.random.uniform(tkey, (batch_size,), minval=t0, maxval=t1 / batch_size)
    t = t + (t1 / batch_size) * jnp.arange(batch_size)

    keys = jax.random.split(key, batch_size)
    error = jax.vmap(sde.loss, in_axes=[0,0,0,0,None])(keys, t, batch.function_outputs, batch.function_inputs, network)
    return jnp.mean(error)


def reverse_solve(sde: SDE, network: ScoreNetwork, x, *, key, prob_flow: bool = True):
    key, ykey = jax.random.split(key)
    yT = sde.sample_prior(ykey, x)

    t0, t1 = sde.beta_schedule.t0, sde.beta_schedule.t1
    # ts = jnp.linspace(t0, t1, 9)[::-1]
    # saveat = dfx.SaveAt(ts=ts)

    if prob_flow:
        reverse_drift_ode = lambda t, yt, arg: sde.reverse_drift_ode(key, t, yt, arg, network)
        terms = dfx.ODETerm(reverse_drift_ode)
    else:
        raise NotImplementedError

    return dfx.diffeqsolve(terms, solver=dfx.Euler(), t0=t1, t1=t0, dt0=-1e-3 / 2.0, y0=yT, args=x, adjoint=dfx.NoAdjoint(),).ys


class MatVecControlTerm(dfx.ControlTerm):

    @staticmethod
    def prod(vf: PyTree, control: PyTree) -> PyTree:
        return jtu.tree_map(lambda a,b: a @ b, vf, control)


def conditional_sample(sde: SDE, network: ScoreNetwork, x_context, y_context, x_test, *, key, num_steps: int = 100, num_inner_steps: int = 5):
    len_context = len(x_context)
    shape_augmented_state = [len(x_test) + len(x_context), 1]  # assume 1d output

    t0 = sde.beta_schedule.t0
    t1 = sde.beta_schedule.t1
    ts = jnp.linspace(t1, t0, num_steps, endpoint=True)
    dt = ts[0] - ts[1]

    solver = dfx.Euler()
    # reverse ODE:
    reverse_drift_ode = lambda t, yt, arg: sde.reverse_drift_ode(key, t, yt, arg, network)
    ode_terms_reverse = dfx.ODETerm(reverse_drift_ode)

    # forward SDE:
    shape = jax.ShapeDtypeStruct(shape_augmented_state, y_context.dtype)
    key, subkey = jax.random.split(key)
    bm = dfx.VirtualBrownianTree(t0=t0, t1=t1, tol=dt, shape=shape, key=subkey)
    sde_terms_forward = dfx.MultiTerm(dfx.ODETerm(sde.drift), MatVecControlTerm(sde.diffusion, bm))

    def inner_loop(key, yt, t):
        mean, cov = sde.pt(t, y_context, x_context)
        if sde.is_diag:
            yt_context = mean + jnp.sqrt(cov) * jax.random.normal(key, mean.shape)
        else:
            raise NotImplemented()
        yt_augmented = jnp.concatenate([yt_context, yt], axis=0)
        x_augmented = jnp.concatenate([x_context, x_test], axis=0)

        # reverse step
        yt_m_dt, *_ = solver.step(ode_terms_reverse, t, t - dt, yt_augmented, x_augmented, None, made_jump=False)

        # forward step
        yt, *_ = solver.step(sde_terms_forward, t - dt, t, yt_m_dt, x_augmented, None, made_jump=False)

        # strip context from augmented state
        return yt[len_context:], yt_m_dt[len_context:]

    def outer_loop(key, yt, t):
        _, yt_m_dt = jax.lax.scan(lambda yt, key: inner_loop(key, yt, t), yt, jax.random.split(key, num_inner_steps))
        yt = yt_m_dt[-1]
        return yt, yt

    key, subkey = jax.random.split(key)
    yT = sde.sample_prior(subkey, x_test)

    xs = (ts[:-1], jax.random.split(key, len(ts) - 1))
    y0, _ = jax.lax.scan(lambda yt, x: outer_loop(x[1], yt, x[0]), yT, xs)
    return y0