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


DATASET = "se"  # squared exponential

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
paramsT["kernel"]["variance"] = 1.
p_ref = GP(meanT, kernelT, paramsT)

sde = ndp.sde.SDE(
    kernelT,
    meanT,
    paramsT,
    beta,
    is_score_preconditioned=False,
    std_trick=False,
    residual_trick=False,
)


factory = regression1d._DATASET_FACTORIES[DATASET] 
assert isinstance(factory, regression1d.GPFunctionalDistribution)
mean0, kernel0, params0 = factory.mean, factory.kernel, factory.params
print(params0)
p_data = GP(mean0, kernel0, params0)
true_score_network = sde.get_exact_score(mean0, kernel0, params0)

key, bkey = jax.random.split(key)


def log_prob(x, y, key):
    return ndp.sde.log_prob(sde, true_score_network, x, y, key=key, rtol=1e-6, atol=1e-6)[0][0]


def log_prob_cond(x, y, xc, yc, key):
    return ndp.sde.log_prob(sde, true_score_network, x, y, x_known=xc, y_known=yc, key=key, rtol=1e-6, atol=1e-6)[0][0]
    # return ndp.sde.log_prob(sde, true_score_network, x, y, x_known=xc, y_known=yc, key=key, rtol=1e-6, atol=1e-6, hutchinson_type="Gaussian")[0][0]


values = []


for ji in [1e-6, 1e-8, 1e-12]:
    config.set_config(config.Config(jitter=ji))
    batch = regression1d.get_batch(bkey, 4, DATASET, "training")
    x = jnp.concatenate([batch.xc, batch.xs], axis=1)[0]
    y = jnp.concatenate([batch.yc, batch.ys], axis=1)[0]

    key, jkey, ckey = jax.random.split(key, num=3)
    logp_joint = log_prob(x, y, jkey)
    logp_context = log_prob(batch.xc[0], batch.yc[0], ckey)
    logp_ndp_cond = logp_joint - logp_context
    logp_ndp_cond2 = log_prob_cond(x, y, batch.xc[0]+1e-8, batch.yc[0], jkey)

    logp_gp_prior = p_data.log_prob(x, y)
    D = (batch.xc[0], batch.yc[0])
    logp_gp_cond = p_data.log_prob(batch.xs[0], batch.ys[0], data=D)

    values.append({
        "jitter": ji,
        "logp_ndp_prior": logp_joint,
        "logp_gp_prior": logp_gp_prior,
        "logp_ndp_cond (diff)": logp_ndp_cond,
        "logp_ndp_cond (direct)": logp_ndp_cond2,
        "logp_gp_cond": logp_gp_cond,
    })
    print(values[-1])

import pandas as pd

df = pd.DataFrame(values)
print(df)
