#%%
from __future__ import annotations
from typing import Tuple, List, Callable, Mapping

from dataclasses import dataclass
import jaxkern
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
import distrax
import matplotlib

import jax
jax.config.update("jax_enable_x64", True)

from neural_diffusion_processes.misc import sample_mvn


@dataclass
class UniformDiscrete:
    lower: int
    upper: int

    def sample(self, key, shape):
        if self.lower == self.upper:
            return jnp.ones(shape, dtype=jnp.int32) * self.lower
        return jax.random.randint(key, shape, minval=self.lower, maxval=self.upper + 1)


_DATASET = [
    "se",
    "matern",
    "weaklyperiodic",
    "sawtooth",
    "mixture",
]

_TASKS = [
    "training",
    "interpolation",
    "extrapolation",
    "generalization",
]

@dataclass
class TaskConfig:
    x_context_dist: distrax.Distribution
    x_target_dist: distrax.Distribution

@dataclass
class DatasetConfig:
    train_num_context: UniformDiscrete
    train_num_target: UniformDiscrete
    eval_num_context: UniformDiscrete
    eval_num_target: UniformDiscrete
    lengthscale: float | None
    noise_variance: float

_NOISE_VAR = 1e-8
_KERNEL_VAR = 1.0
_LENGTHSCALE = .7

_DATASET_CONFIGS = {
    "se": DatasetConfig(
        train_num_context=UniformDiscrete(0, 50),
        train_num_target=UniformDiscrete(50, 50),
        eval_num_context=UniformDiscrete(10, 10),
        eval_num_target=UniformDiscrete(50, 50),
        lengthscale=0.25,
        noise_variance=_NOISE_VAR
    ),
    "matern": DatasetConfig(
        train_num_context=UniformDiscrete(0, 50),
        train_num_target=UniformDiscrete(50, 50),
        eval_num_context=UniformDiscrete(10, 10),
        eval_num_target=UniformDiscrete(50, 50),
        lengthscale=0.25,
        noise_variance=_NOISE_VAR
    ),
    "weaklyperiodic": DatasetConfig(
        train_num_context=UniformDiscrete(0, 50),
        train_num_target=UniformDiscrete(50, 50),
        eval_num_context=UniformDiscrete(10, 10),
        eval_num_target=UniformDiscrete(50, 50),
        lengthscale=None,  # different lengthscale for SE and Periodic SE
        noise_variance=_NOISE_VAR
    ),
    "sawtooth": DatasetConfig(
        train_num_context=UniformDiscrete(0, 100),
        train_num_target=UniformDiscrete(100, 50),
        eval_num_context=UniformDiscrete(10, 10),
        eval_num_target=UniformDiscrete(100, 100),
        lengthscale=None,  # no lengthscale
        noise_variance=_NOISE_VAR
    ),
    "mixture": DatasetConfig(
        train_num_context=UniformDiscrete(0, 100),
        train_num_target=UniformDiscrete(100, 50),
        eval_num_context=UniformDiscrete(10, 10),
        eval_num_target=UniformDiscrete(100, 100),
        lengthscale=None,  # see above
        noise_variance=_NOISE_VAR
    ),
}

_TASK_CONFIGS = {
    "training": TaskConfig(
        x_context_dist=distrax.Uniform(-2, 2),
        x_target_dist=distrax.Uniform(-2, 2),
    ),
    "interpolation": TaskConfig(
        x_context_dist=distrax.Uniform(-2, 2),
        x_target_dist=distrax.Uniform(-2, 2),
    ),
    "extrapolation": TaskConfig(
        x_context_dist=distrax.Uniform(-2, 2),
        x_target_dist=distrax.MixtureOfTwo(
            prob_a=0.5,
            component_a=distrax.Uniform(-4, -2),
            component_b=distrax.Uniform(2, 4),
        ) 
    ),
    "generalization": TaskConfig(
        x_context_dist=distrax.Uniform(2, 6),
        x_target_dist=distrax.Uniform(2, 6),
    ),
}


class FuntionalDistribution:
    
    def sample(self, key, x: Float[Array, "N D"]) -> Float[Array, "N P"]:
        pass

class GPFunctionalDistribution(FuntionalDistribution):

    def __init__(self, kernel: jaxkern.base.AbstractKernel, params: Mapping):
        self._kernel = kernel
        self._params = params
    
    def sample(self, key, x: Float[Array, "N D"]) -> Float[Array, "N P"]:
        noise_var = self._params["noise_variance"]
        gram = self._kernel.gram(self._params, x).to_dense()
        return sample_mvn(key, jnp.zeros_like(x), gram, noise_var=noise_var)


DatasetFactory: Callable[[], FuntionalDistribution]

_DATASET_FACTORIES = {}

def register_dataset_factory(name: str):

    def wrap(f: DatasetFactory):
        _DATASET_FACTORIES[name] = f()
    
    return wrap
    
@register_dataset_factory("se")
def _se_dataset_factory():
    kernel = jaxkern.stationary.RBF(active_dims=[0])
    params = {"lengthscale": _LENGTHSCALE, "variance": _KERNEL_VAR, "noise_variance": _NOISE_VAR}
    return GPFunctionalDistribution(kernel, params)


@register_dataset_factory("matern")
def _matern_dataset_factory():
    kernel = jaxkern.stationary.Matern52(active_dims=[0])
    params = {"lengthscale": _LENGTHSCALE, "variance": _KERNEL_VAR, "noise_variance": _NOISE_VAR}
    return GPFunctionalDistribution(kernel, params)


def get_batch(key, batch_size: int, name: str, task: str):
    if name not in _DATASET:
        raise NotImplementedError("Unknown dataset: %s." % name)
    if task not in _TASKS:
        raise NotImplementedError("Unknown task: %s." % task)
    

    key, nckey, ntkey = jax.random.split(key, 3)
    if task == "training":
        n_context = _DATASET_CONFIGS[name].train_num_context.sample(nckey, ())
        n_target = _DATASET_CONFIGS[name].train_num_target.sample(ntkey, ())
    else:
        n_context = _DATASET_CONFIGS[name].eval_num_context.sample(nckey, ())
        n_target = _DATASET_CONFIGS[name].eval_num_target.sample(ntkey, ())

    key, ckey, tkey = jax.random.split(key, 3)
    task = _TASK_CONFIGS[task]
    x_context = task.x_context_dist.sample(seed=ckey, sample_shape=(batch_size, n_context, 1))

    x_target = task.x_target_dist.sample(seed=tkey, sample_shape=(batch_size, n_target, 1))
    x = jnp.concatenate([x_context, x_target], axis=1)

    keys = jax.random.split(key, batch_size)
    y = jax.vmap(_DATASET_FACTORIES[name].sample)(keys, x)
    return x_context, y[:, :n_context, :], x_target, y[:, n_context:, :]


#%%
import numpy
import matplotlib.pyplot as plt

def info(a, name):
    print(name)
    print(a.shape)
    print(jnp.min(a))
    print(jnp.max(a))
    print("="*10)

def plot_data(xc, yc, xt, yt, ax, legend=True, ns=1):
    ax.plot(xc[:ns, :, 0].T, yc[:ns, :, 0].T, "C0x", label="context")
    ax.plot(xt[:ns, :, 0].T, yt[:ns, :, 0].T, "C1x", label="target")
    handles, labels = ax.get_legend_handles_labels()
    labels, ids = numpy.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    if legend:
        ax.legend(handles, labels, loc='best')


key = jax.random.PRNGKey(0)

fig, axes = plt.subplots(len(_DATASET_FACTORIES), len(_TASK_CONFIGS), figsize=(10, 5), sharex=True, sharey=True, tight_layout=True)

import itertools

for (i, dataset), (j, task) in itertools.product(enumerate(_DATASET_FACTORIES.keys()), enumerate(_TASK_CONFIGS.keys())):
    print(dataset, task)
    ax.set_xlim(-4, 6)
    ax = axes[i,j]
    data = get_batch(key, 16, dataset, task)
    plot_data(*data, ax, legend=(i==0) and (j==0))
    if i == 0:
        ax.set_title(task)
    if j == 0:
        ax.ylabel(dataset)

# %%
