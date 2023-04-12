#%%
from __future__ import annotations
from typing import Tuple, List, Callable, Mapping, Optional

import abc
from dataclasses import dataclass
import jaxkern
import jax
import gpjax
import jax.numpy as jnp
from jaxtyping import Float, Array
import distrax

import jax
jax.config.update("jax_enable_x64", True)

from neural_diffusion_processes.kernels import sample_prior_gp
from neural_diffusion_processes.data.data import DataBatch


MAX_NUM_CONTEXT = 10
MAX_NUM_TARGET = 100


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
    "noisymixture",
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
    train_num_target: UniformDiscrete
    eval_num_context: UniformDiscrete
    eval_num_target: UniformDiscrete


_NOISE_VAR = 1e-8
_KERNEL_VAR = 1.0
_LENGTHSCALE = .25

_DATASET_CONFIGS = {
    "se": DatasetConfig(
        train_num_target=UniformDiscrete(1, 60),
        eval_num_target=UniformDiscrete(50, 50),
        eval_num_context=UniformDiscrete(1, 10),
    ),
    "matern": DatasetConfig(
        train_num_target=UniformDiscrete(1, 60),
        eval_num_target=UniformDiscrete(50, 50),
        eval_num_context=UniformDiscrete(1, 10),
    ),
    "weaklyperiodic": DatasetConfig(
        train_num_target=UniformDiscrete(1, 60),
        eval_num_target=UniformDiscrete(50, 50),
        eval_num_context=UniformDiscrete(1, 10),
    ),
    "noisymixture": DatasetConfig(
        train_num_target=UniformDiscrete(1, 60),
        eval_num_target=UniformDiscrete(50, 50),
        eval_num_context=UniformDiscrete(1, 10),
    ),
    "sawtooth": DatasetConfig(
        train_num_target=UniformDiscrete(1, 100),
        eval_num_target=UniformDiscrete(90, 90),
        eval_num_context=UniformDiscrete(1, 10),
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


class FuntionalDistribution(abc.ABC):
    
    @abc.abstractmethod
    def sample(self, key, x: Float[Array, "N 1"]) -> Float[Array, "N 1"]:
        raise NotImplementedError()
        

class GPFunctionalDistribution(FuntionalDistribution):

    def __init__(self, kernel: jaxkern.base.AbstractKernel, params: Mapping):
        self.kernel = kernel
        self.params = params
        self.mean = gpjax.mean_functions.Zero()
    
    def sample(self, key, x: Float[Array, "N 1"]) -> Float[Array, "N 1"]:
        return sample_prior_gp(
            key, self.mean, self.kernel, params=self.params, x=x, obs_noise=self.params["noise_variance"]
        )

DatasetFactory: Callable[[], FuntionalDistribution]

_DATASET_FACTORIES = {}

def register_dataset_factory(name: str):

    def wrap(f: DatasetFactory):
        _DATASET_FACTORIES[name] = f()
    
    return wrap
    
@register_dataset_factory("se")
def _se_dataset_factory():
    kernel = jaxkern.stationary.RBF(active_dims=[0])
    params = {
        "mean_function": {},
        "kernel": {"lengthscale": _LENGTHSCALE, "variance": _KERNEL_VAR,},
        "noise_variance": _NOISE_VAR
    }
    return GPFunctionalDistribution(kernel, params)


@register_dataset_factory("matern")
def _matern_dataset_factory():
    kernel = jaxkern.stationary.Matern52(active_dims=[0])
    params = {
        "mean_function": {},
        "kernel": {"lengthscale": _LENGTHSCALE, "variance": _KERNEL_VAR,},
        "noise_variance": _NOISE_VAR
    }
    return GPFunctionalDistribution(kernel, params)


@register_dataset_factory("weaklyperiodic")
def _weaklyper_dataset_factory():
    rbf = jaxkern.stationary.RBF(active_dims=[0])
    per = jaxkern.stationary.Periodic(active_dims=[0])
    kernel = jaxkern.ProductKernel([rbf, per])
    params = {
        "mean_function": {},
        "kernel": [
            {"lengthscale": 0.5, "variance": _KERNEL_VAR},
            {"lengthscale": _LENGTHSCALE, "variance": _KERNEL_VAR, "period": 1.},
        ],
        "noise_variance": _NOISE_VAR,
    }

    return GPFunctionalDistribution(kernel, params)


@register_dataset_factory("noisymixture")
def _noisymix_dataset_factory():
    rbf1 = jaxkern.stationary.RBF(active_dims=[0])
    rbf2 = jaxkern.stationary.RBF(active_dims=[0])
    white = jaxkern.stationary.White(active_dims=[0])
    kernel = jaxkern.SumKernel([rbf1, rbf2, white])
    params = {
        "mean_function": {},
        "kernel": [
            {"lengthscale": _LENGTHSCALE, "variance": 1.0},
            {"lengthscale": 1.0, "variance": 1.0},
            {"variance": 1e-3},
        ],
        "noise_variance": _NOISE_VAR,
    }

    return GPFunctionalDistribution(kernel, params)


class Sawtooth(FuntionalDistribution):
    """ See appendix H: https://arxiv.org/pdf/2007.01332.pdf"""
    def sample(self, key, x: Float[Array, "N 1"]) -> Float[Array, "N 1"]:
        fkey, skey, kkey = jax.random.split(key, 3)
        A = 1.
        K_max = 20
        f = jax.random.uniform(fkey, (), minval=3., maxval=5.)
        s = jax.random.uniform(skey, (), minval=-5., maxval=5.)
        ks = jnp.arange(1, K_max + 1, dtype=x.dtype)[None, :]
        vals = (-1.) ** ks * jnp.sin(2. * jnp.pi * ks * f * (x - s)) / ks
        k = jax.random.randint(kkey, (), minval=10, maxval=K_max + 1)
        mask = jnp.where(ks < k, jnp.ones_like(ks), jnp.zeros_like(ks))
        return A/2. - A/jnp.pi * jnp.sum(vals * mask, axis=1, keepdims=True)


@register_dataset_factory("sawtooth")
def _sawtooth_dataset_factory():
    return Sawtooth()


def get_batch(key, batch_size: int, name: str, task: str):
    if name not in _DATASET:
        raise NotImplementedError("Unknown dataset: %s." % name)
    if task not in _TASKS:
        raise NotImplementedError("Unknown task: %s." % task)
    

    n_context = MAX_NUM_CONTEXT #_DATASET_CONFIGS[name].train_num_context.sample(nckey, ())
    n_target = MAX_NUM_TARGET # _DATASET_CONFIGS[name].train_num_target.sample(ntkey, ())

    key, ckey, tkey = jax.random.split(key, 3)
    task = _TASK_CONFIGS[task]
    x_context = task.x_context_dist.sample(seed=ckey, sample_shape=(batch_size, n_context, 1))

    x_target = task.x_target_dist.sample(seed=tkey, sample_shape=(batch_size, n_target, 1))
    x = jnp.concatenate([x_context, x_target], axis=1)

    keys = jax.random.split(key, batch_size)
    y = jax.vmap(_DATASET_FACTORIES[name].sample)(keys, x)
    return DataBatch(
        xs=x_target,
        ys=y[:, n_context:, :],
        xc=x_context,
        yc=y[:, :n_context, :]
    )


class DatasetFromGenerator:
    def __init__(self, generator, key):
        self._key = key
        self._generator  = generator
        self._preprocess = []
    
    def map(self, function):
        self._preprocess.append(function)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        batch = next(self._generator)
        for func in self._preprocess:
            self._key, key = jax.random.split(self._key)
            batch = func(batch, key=key)
        return batch


def data_generator(key, dataset, task, total_num_samples, batch_size, num_epochs: Optional[int] = None):
    """
    :param num_epochs: if `None` generator runs forever
    """
    assert total_num_samples % batch_size == 0

    @jax.jit
    def batch(key) -> DataBatch:
        return get_batch(key, batch_size, dataset, task)

    print("run once...")
    _ = batch(key)

    if num_epochs is None:
        num_epochs = jnp.inf
    
    count_epochs = 0
    while count_epochs < num_epochs:
        count_epochs += 1
        for _ in range(total_num_samples // batch_size):
            key, bkey = jax.random.split(key)
            yield batch(bkey)



def get_padding_function(dataset: str, task: str):
    if task == "training":
        target_num_data_sampler = _DATASET_CONFIGS[dataset].train_num_target
        context_num_data_sampler = None
    else:
        target_num_data_sampler = _DATASET_CONFIGS[dataset].eval_num_target
        context_num_data_sampler = _DATASET_CONFIGS[dataset].eval_num_context

    @jax.jit
    def padding(batch: DataBatch, key):
        num_data_total = batch.xs.shape[1]
        num_data = target_num_data_sampler.sample(key, shape=())
        mask = jnp.where(
            jnp.arange(num_data_total)[None, :, None] < num_data,
            jnp.ones_like(batch.xs),
            jnp.zeros_like(batch.xs)
        )

        # repeat for context
        if context_num_data_sampler is not None:
            num_data_total = batch.xc.shape[1]
            num_data = context_num_data_sampler.sample(key, shape=())
            mask_context = jnp.where(
                jnp.arange(num_data_total)[None, :, None] < num_data,
                jnp.ones_like(batch.xc),
                jnp.zeros_like(batch.xc)
            )
        else:
            mask_context = None

        return DataBatch(
            xs=batch.xs,
            ys=batch.ys,
            mask=mask,
            xc=batch.xc,
            yc=batch.yc,
            mask_context=mask_context
        )

    return padding


def get_dataset(dataset: str, task: str, *, key, batch_size: int, samples_per_epoch: int, num_epochs: Optional[int] = None) -> DatasetFromGenerator:
    gkey, dskey = jax.random.split(key)
    gen = data_generator(gkey, dataset, task, samples_per_epoch, batch_size, num_epochs)
    ds = DatasetFromGenerator(gen, dskey)
    ds.map(get_padding_function(dataset, task))
    return ds


#%%
if __name__ == "__main__":
    import matplotlib

    def plot_data():
        import numpy
        import matplotlib.pyplot as plt
        import itertools

        def info(a, name):
            print(name)
            print(a.shape)
            print(jnp.min(a))
            print(jnp.max(a))
            print("="*10)

        def plot_data(xc, yc, xt, yt, ax, legend=True, ns=1):
            ax.plot(xc[:ns, :, 0].T, yc[:ns, :, 0].T, "C0.", label="context")
            ax.plot(xt[:ns, :, 0].T, yt[:ns, :, 0].T, "C1.", label="target")
            handles, labels = ax.get_legend_handles_labels()
            labels, ids = numpy.unique(labels, return_index=True)
            handles = [handles[i] for i in ids]
            if legend:
                ax.legend(handles, labels, loc='best')


        key = jax.random.PRNGKey(0)

        fig, axes = plt.subplots(len(_DATASET_FACTORIES), len(_TASK_CONFIGS), figsize=(15, 5), sharex=True, tight_layout=True)


        for (i, dataset), (j, task) in itertools.product(enumerate(_DATASET_FACTORIES.keys()), enumerate(_TASK_CONFIGS.keys())):
            print(dataset, task)
            ax = axes[i,j]
            ax.set_xlim(-4, 6)
            data = get_batch(key, 16, dataset, task)
            plot_data(*data, ax, legend=(i==0) and (j==0))
            if i == 0:
                ax.set_title(task)
            if j == 0:
                ax.set_ylabel(dataset)


        nrows = len(_DATASET_FACTORIES)
        fig, axes = plt.subplots(nrows, 1, figsize=(15, 3 * nrows), sharex=True)
        for i, name in enumerate(_DATASET_FACTORIES.keys()):
            ax = axes[i]
            keys = jax.random.split(key, 16)
            x = jnp.linspace(-2, 3, 500)[:, None]
            y = jax.vmap(_DATASET_FACTORIES[name].sample, in_axes=[0, None])(keys, x)
            ax.set_title(name)
            ax.plot(x, y[:3, :, 0].T)
