import functools

import gpjax
import jax
import jax.numpy as jnp
import jaxkern
from config import Config, toy_config
from jaxtyping import Array

import neural_diffusion_processes as ndp
import neural_diffusion_processes.sde_with_mask as ndp_sde
from neural_diffusion_processes.data import regression1d
from neural_diffusion_processes.ml_tools.state import (
    TrainingState, find_latest_checkpoint_step_index, load_checkpoint)

############# Config
SEED = 1
LIMITING_KERNEL = "noisy-se"
T0 = 5e-4
NOISE_VARIANCE = 0.05**2
SCORE_PARAMETERIZATION = "preconditioned_s"
DATASET = "se"


key = jax.random.PRNGKey(SEED)

####### init relevant diffusion classes
beta = ndp.sde.LinearBetaSchedule(t0=T0)
limiting_kernel = LIMITING_KERNEL[LIMITING_KERNEL.find("-")+1:]
limiting_kernel = ndp.kernels.get_kernel(limiting_kernel, active_dims=[0])
hyps = {
    "mean_function": {},
    "kernel": limiting_kernel.init_params(None),
}
if "noisy" in LIMITING_KERNEL:
    limiting_kernel = jaxkern.SumKernel(
        [limiting_kernel, jaxkern.stationary.White(active_dims=[0])]
    )
    v = NOISE_VARIANCE
    hyps["kernel"]["variance"] = 1. - v
    hyps["kernel"] = [hyps["kernel"], {"variance": NOISE_VARIANCE}]


factory = regression1d._DATASET_FACTORIES[DATASET] 
assert isinstance(factory, regression1d.GPFunctionalDistribution)
mean0, kernel0, params0 = factory.mean, factory.kernel, factory.params

sde = ndp_sde.SDE(
    kernel0,
    gpjax.mean_functions.Zero(),
    params0,
    beta,
    score_parameterization=ndp_sde.ScoreParameterization.get(
        "y0"
    ),
    std_trick=False,
    residual_trick=False,
    weighted=None,
    exact_score=True,
)


true_score_network = sde.get_exact_score(mean0, kernel0, params0)

x = jnp.linspace(-1, 1, 7)[:, None]
mask = jnp.zeros_like(x[:, 0])


def network(t: Array, yt: Array, x: Array, mask, *, key) -> Array:
    gamma_t =  sde.beta_schedule.B(t)
    # return ((2 - jnp.exp(-gamma_t)) / jnp.exp(-gamma_t/2.)) * yt
    return yt * jnp.exp(-gamma_t/2.)

key, subkey = jax.random.split(key)
yt = jnp.ones_like(x) * 0.5
d = sde.reverse_drift_ode(key, 0.5, yt, x, mask, network)
print(d)


xx = jnp.linspace(-1, 1, 100)[:, None]

import matplotlib.pyplot as plt

# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# fig, ax = plt.subplots()
# samples_prior = jax.vmap(lambda k: sde.sample_prior(k, xx))(jax.random.split(key, 20))
# print(samples_prior.shape)

# sample = jax.vmap(lambda k: ndp_sde.sde_solve(
#     sde,
#     network,
#     xx,
#     key=k,
#     prob_flow=True,
# ))(jax.random.split(key, 20))
# print(sample.shape)

# ax.plot(xx, samples_prior[..., 0].T, color="black", alpha=0.2)
# ax.plot(xx, sample[:, 0, :, 0].T, color="C1", alpha=0.2)
# fig.savefig("zero_drift.png")


@jax.vmap
@jax.jit
def delta_logp(x, y, mask, key):
    return ndp_sde.log_prob(sde, network, x, y, mask, key=key, rtol=None)

def log_prob(x, y, mask, key):
    dlp, yT = delta_logp(x, y, mask, key)
    logp_prior = jax.vmap(sde.log_prob_prior)(
        x[:, ~mask[0].astype(jnp.bool_)], yT[:, ~mask[0].astype(jnp.bool_)]
    )
    return logp_prior + dlp



def get_dataset(key):
    task = "interpolation"
    total_num_samples = int(2**14)
    batch_size = 32
    return regression1d.get_dataset(DATASET, task, key=key, batch_size=batch_size, samples_per_epoch=total_num_samples, num_epochs=1)


key, dkey = jax.random.split(key)
ds = get_dataset(dkey)
batch = next(ds)

from tqdm.contrib import tenumerate

logliks = []

for i, batch in tenumerate(ds):
    # if i >= 124: break
    x = jnp.concatenate([batch.xc, batch.xs], axis=1)
    y = jnp.concatenate([batch.yc, batch.ys], axis=1)
    mask = jnp.concatenate([batch.mask_context, batch.mask], axis=1)
    # mask = jnp.zeros_like(mask)
    mask_context = batch.mask_context

    key, jkey, ckey = jax.random.split(key, num=3)
    logp_joint = log_prob(x, y, mask, jax.random.split(jkey, len(x)))
    logp_context = log_prob(batch.xc, batch.yc, mask_context, jax.random.split(ckey, len(x)))
    logp_ndp_cond = logp_joint - logp_context
    logliks.append(logp_ndp_cond / 50.)

vals = jnp.concatenate(logliks)
err = lambda v: 1.96 * jnp.std(v) / jnp.sqrt(len(v))
print(jnp.mean(vals))
print(jnp.std(vals))