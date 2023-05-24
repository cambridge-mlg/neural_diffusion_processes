from dataclasses import dataclass
from typing import Mapping

import diffrax as dfx
import gpjax
import jax
import jax.numpy as jnp
import jaxkern

import neural_diffusion_processes as ndp
import neural_diffusion_processes.sde_with_mask as ndp_sde
from neural_diffusion_processes import config
from neural_diffusion_processes.data import regression1d

DATASET = "weaklyperiodic"  # squared exponential


def get_dataset(key):
    task = "interpolation"
    total_num_samples = int(2**14)
    batch_size = 32
    return regression1d.get_dataset(DATASET, task, key=key, batch_size=batch_size, samples_per_epoch=total_num_samples, num_epochs=1)


@dataclass
class GP:
    mean: gpjax.mean_functions.AbstractMeanFunction
    kernel: gpjax.kernels.AbstractKernel
    params: Mapping

    def log_prob(self, x, y, data=None):
        noise = self.params["noise_variance"]
        if data is None:
            return ndp.kernels.log_prob_prior_gp(
                self.mean, self.kernel, self.params, x, y, obs_noise=noise
            )
        else:
            post = ndp.kernels.posterior_gp(
                self.mean, self.kernel, self.params, data[0], data[1], obs_noise=noise
            )
            return post(x).log_prob(jnp.reshape(y, (-1,)))



key = jax.random.PRNGKey(0)
import jaxkern

############ Limiting process
beta = ndp.sde.LinearBetaSchedule(t0=5e-4)
meanT = gpjax.mean_functions.Zero()
kernelT = ndp.kernels.get_kernel("matern52", active_dims=[0], noisy=False)
paramsT = {
    "mean_function": {},
    "kernel": kernelT.init_params(None),
    "noise_variance": 0.0
}
kernelT = jaxkern.SumKernel(
    [kernelT, jaxkern.stationary.White(active_dims=[0])]
)
paramsT["kernel"] = [paramsT["kernel"], {"variance": 0.05}]
p_ref = GP(meanT, kernelT, paramsT)

sde = ndp_sde.SDE(
    kernelT,
    meanT,
    paramsT,
    beta,
    is_score_preconditioned=False,
    std_trick=False,
    residual_trick=False,
    exact_score=True
)


factory = regression1d._DATASET_FACTORIES[DATASET] 
assert isinstance(factory, regression1d.GPFunctionalDistribution)
mean0, kernel0, params0 = factory.mean, factory.kernel, factory.params
print(params0)
p_data = GP(mean0, kernel0, params0)
true_score_network = sde.get_exact_score(mean0, kernel0, params0)


@jax.vmap
@jax.jit
def delta_logp(x, y, mask, key):
    return ndp_sde.log_prob(sde, true_score_network, x, y, mask, key=key)

def log_prob(x, y, mask, key):
    dlp, yT = delta_logp(x, y, mask, key)
    logp_prior = jax.vmap(sde.log_prob_prior)(
        x[:, ~mask[0].astype(jnp.bool_)], yT[:, ~mask[0].astype(jnp.bool_)]
    )
    return logp_prior + dlp


key, dkey = jax.random.split(key)
ds = get_dataset(dkey)
batch = next(ds)


def m2i(mask):
    """mask to indices"""
    return ~mask.astype(jnp.bool_)


values = []

logliks = []


ji = 1e-6
config.set_config(config.Config(jitter=ji))

from tqdm.contrib import tenumerate

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

    # logp_gp_joint = p_data.log_prob(x[m2i(mask)], y[m2i(mask)])
    # D = (
    #     batch.xc[:, m2i(mask_context)],
    #     batch.yc[:, m2i(mask_context)]
    # )
    # logp_gp_cond = p_data.log_prob(
    #     batch.xs[:, m2i(batch.mask[0])],
    #     batch.ys[m2i(batch.mask[0])],
    #     # batch.xs[0],
    #     # batch.ys[0],
    #     data=D
    # )
    # print(logp_joint.shape)
    logliks.append(logp_ndp_cond / 50.)

    values.append({
        "jitter": ji,
        "logp_ndp_prior": jnp.mean(logp_joint),
        # "logp_gp_prior": logp_gp_joint,
        "logp_ndp_cond": jnp.mean(logp_ndp_cond),
        # "logp_gp_cond": logp_gp_cond,
        "num_target": m2i(batch.mask[0]).sum(),
        "num_context": m2i(batch.mask_context[0]).sum()
    })
    # print(values[-1])

import pandas as pd

df = pd.DataFrame(values)
print(df)

vals = jnp.concatenate(logliks)
err = lambda v: 1.96 * jnp.std(v) / jnp.sqrt(len(v))

print(jnp.mean(vals))
print(jnp.var(vals))
print(err(vals))


