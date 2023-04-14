from typing import Mapping
from dataclasses import dataclass

import diffrax as dfx
import jax
import jax.numpy as jnp
import gpjax
import jaxkern

import neural_diffusion_processes as ndp
from neural_diffusion_processes.data import regression1d

from neural_diffusion_processes import config

DATASET = "se"  # squared exponential


def get_dataset(key):
    task = "interpolation"
    total_num_samples = int(2**14)
    batch_size = 2
    return regression1d.get_dataset(DATASET, task, key=key, batch_size=batch_size, samples_per_epoch=total_num_samples, num_epochs=None)


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

############ Limiting process
beta = ndp.sde.LinearBetaSchedule(t0=5e-4)
meanT = gpjax.mean_functions.Zero()
kernelT = ndp.kernels.get_kernel("white", active_dims=[0])
paramsT = {
    "mean_function": {},
    "kernel": kernelT.init_params(None),
    "noise_variance": 0.0
}
p_ref = GP(meanT, kernelT, paramsT)

sde = ndp.sde.SDE(
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

def log_prob(x, y, mask, key):
    return ndp.sde.log_prob(sde, true_score_network, x, y, mask, key=key, rtol=None)[0][0]


key, dkey = jax.random.split(key)
ds = get_dataset(dkey)
batch = next(ds)


def m2i(mask):
    """mask to indices"""
    return ~mask.astype(jnp.bool_)


values = []


for ji in [1e-6]:
    config.set_config(config.Config(jitter=ji))
    x = jnp.concatenate([batch.xc, batch.xs], axis=1)[0]
    y = jnp.concatenate([batch.yc, batch.ys], axis=1)[0]
    mask = jnp.concatenate([batch.mask_context, batch.mask], axis=1)[0]
    mask = jnp.zeros_like(mask)
    mask_context = jnp.zeros_like(batch.mask_context)

    key, jkey, ckey = jax.random.split(key, num=3)
    logp_joint = log_prob(x, y, mask, jkey)
    logp_context = log_prob(batch.xc[0], batch.yc[0], mask_context[0], ckey)
    logp_ndp_cond = logp_joint - logp_context

    logp_gp_joint = p_data.log_prob(x[m2i(mask)], y[m2i(mask)])
    D = (
        batch.xc[0][m2i(mask_context[0])],
        batch.yc[0][m2i(mask_context[0])]
    )
    logp_gp_cond = p_data.log_prob(
        # batch.xs[0][m2i(batch.mask[0])],
        # batch.ys[0][m2i(batch.mask[0])],
        batch.xs[0],
        batch.ys[0],
        data=D
    )

    values.append({
        "jitter": ji,
        "logp_ndp_prior": logp_joint,
        "logp_gp_prior": logp_gp_joint,
        "logp_ndp_cond": logp_ndp_cond,
        "logp_gp_cond": logp_gp_cond,
    })
    print(values[-1])

import pandas as pd

df = pd.DataFrame(values)
print(df)
