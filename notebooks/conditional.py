# %%
from __future__ import annotations
from typing import Mapping
import itertools as it
import dataclasses
import einops
import jaxkern
import gpjax
import jax
import einops
import jax.numpy as jnp
import diffrax as dfx

import matplotlib.pyplot as plt

from jaxtyping import Array, Float

import neural_diffusion_processes as ndp

from jax.config import config
config.update("jax_enable_x64", True)


@dataclasses.dataclass
class GpConfig:
    mean_function: str
    kernel: str
    mean_function_params: Mapping[str, Float[Array, "..."]]
    kernel_params: Mapping[str, Float[Array, "..."]]


@dataclasses.dataclass
class Config:
    seed: int = 20230313
    num_points: int = 101
    data: GpConfig = GpConfig(
        mean_function="zero",
        kernel="squared_exponential",
        mean_function_params={},
        kernel_params={
            "variance": jnp.ones((1,)),
            "lengthscale": jnp.ones((1,)) * 0.25,
        } 
    )
    limiting_process: GpConfig = GpConfig(
        mean_function="zero",
        kernel="white",
        mean_function_params={},
        kernel_params={
            "variance": jnp.ones((1,)),
        } 
    )


def solve(key, y0, x):
    t0, t1 = beta.t0, beta.t1
    shape = jax.ShapeDtypeStruct(y0.shape, y0.dtype)
    bm = dfx.VirtualBrownianTree(t0=beta.t0, t1=beta.t1, tol=1e-3 / 2., shape=shape, key=key)
    terms = dfx.MultiTerm(
        dfx.ODETerm(sde.drift),
        ndp.sde.MatVecControlTerm(sde.diffusion, bm)
    )
    solve_kwargs = dict(
        y0=y0, args=x, saveat=dfx.SaveAt(steps=True), adjoint=dfx.NoAdjoint()
    )
    return dfx.diffeqsolve(
        terms, solver=dfx.Euler(), t0=t0, t1=t1, dt0=1e-3, **solve_kwargs
    )
#%%

config = Config()

key = jax.random.PRNGKey(config.seed)
key_iter = ndp.misc.get_key_iter(key)

####### init relevant diffusion classes
beta = ndp.sde.LinearBetaSchedule()
data_kernel = ndp.kernels.get_kernel(config.data.kernel, active_dims=list(range(1)))

sde = ndp.sde.SDE(
    limiting_kernel=ndp.kernels.get_kernel(config.limiting_process.kernel, active_dims=[0]),
    limiting_mean_fn=ndp.kernels.get_mean_fn(config.limiting_process.mean_function),
    limiting_params={
        "kernel": config.limiting_process.kernel_params,
        "mean_fn": config.limiting_process.mean_function_params,
    },
    beta_schedule=beta,
)


num_samples = 3
x = jnp.linspace(-1, 1, config.num_points)[:, None]

kernel_0 = ndp.kernels.get_kernel(config.data.kernel, active_dims=[0])
mean_fn_0 = ndp.kernels.get_mean_fn(config.data.mean_function)
params_0 = {
    "mean_fn": config.data.mean_function_params,
    "kernel": config.data.kernel_params,
}


def y0_fn(key, x):
    p = mean_fn_0(params_0["mean_fn"], x).shape[-1]
    dist = ndp.kernels.prior_gp(mean_fn_0, kernel_0, params_0, 0.0)(x)
    s = dist.sample(seed=key, sample_shape=1)
    return einops.rearrange(s, "1 (n p) -> n p", p=p)

y0_keys = jax.random.split(next(key_iter), num_samples)
y0s = jax.vmap(y0_fn, in_axes=[0, None])(y0_keys, x)

# y0s = ndp.kernels.sample_prior_gp(
#     next(key_iter),
#     mean_fn_0,
#     kernel=kernel_0,
#     params=params_0,
#     x=x,
#     num_samples=num_samples
# )


subkeys = jax.random.split(next(key_iter), num=num_samples)
out = jax.vmap(solve, in_axes=[0, 0, None])(subkeys, y0s, x)
num_steps = int(out.stats['num_steps'][0])
# %%

plot_num_timesteps = 5
fig, axes = plt.subplots(plot_num_timesteps, num_samples, sharex=True, sharey=True, figsize=(num_samples * 3, plot_num_timesteps*1.5))

t_indices = ndp.misc.generate_logarithmic_sequence(num_steps, plot_num_timesteps - 1)

from functools import partial

for (i, t_index), sample_index in it.product(enumerate(t_indices), range(num_samples)):
    ax = axes[i, sample_index]
    t = out.ts[0, t_index]
    mean_t, k_t, params_t = sde.p0t(t, partial(y0_fn, y0_keys[sample_index]))
    mean = mean_t(params_t["mean_fn"], x)
    cov = k_t.gram(params_t["kernel"], x)
    print(type(cov))
    std = cov.diagonal().reshape(-1,1)**0.5
    ax.plot(x, mean, 'k--', alpha=.5)
    lo, up = (v.flatten() for v in (mean - 2 * std, mean + 2 * std))
    ax.fill_between(x.flatten(), lo, up, alpha=.1, color='k')
    ax.plot(x, out.ys[sample_index, t_index], 'o-', ms=2)
    ax.set_ylim(-2.5, 2.5)
    if sample_index == 0:
        ax.set_ylabel(f"t = {t:.2f}")

plt.tight_layout()
# %%

# %%
