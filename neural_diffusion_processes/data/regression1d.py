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
    "periodic",
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
    train_num_target: UniformDiscrete
    eval_num_context: UniformDiscrete
    eval_num_target: UniformDiscrete


_NOISE_VAR = 0.05**2
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
    "periodic": DatasetConfig(
        train_num_target=UniformDiscrete(1, 60),
        eval_num_target=UniformDiscrete(50, 50),
        eval_num_context=UniformDiscrete(1, 10),
    ),
    "sawtooth": DatasetConfig(
        train_num_target=UniformDiscrete(1, 110),
        eval_num_target=UniformDiscrete(100, 100),
        eval_num_context=UniformDiscrete(1, 10),
    ),
    "mixture": DatasetConfig(
        train_num_target=UniformDiscrete(1, 110),
        eval_num_target=UniformDiscrete(100, 100),
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


@dataclass
class FuntionalDistribution(abc.ABC):
    # All GP datasets are naturally normalized so do not need additional normalization.
    # Sawtooth is not normalized so we need to normalize it in the Mixture but not when used
    # in isolation.
    is_data_naturally_normalized: bool = True
    normalize: bool = False

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
    rbf = jaxkern.stationary.RBF(active_dims=[0])
    white = jaxkern.White(active_dims=[0])
    kernel = jaxkern.SumKernel([rbf, white])
    params = {
        "mean_function": {},
        "kernel": [
            {"lengthscale": _LENGTHSCALE, "variance": _KERNEL_VAR,},
            {"variance": _NOISE_VAR,},
        ],
        "noise_variance": 0.0
    }
    return GPFunctionalDistribution(kernel, params)


@register_dataset_factory("matern")
def _matern_dataset_factory():
    mat = jaxkern.stationary.Matern52(active_dims=[0])
    white = jaxkern.White(active_dims=[0])
    kernel = jaxkern.SumKernel([mat, white])
    params = {
        "mean_function": {},
        "kernel": [
            {"lengthscale": _LENGTHSCALE, "variance": _KERNEL_VAR,},
            {"variance": _NOISE_VAR,},
        ],
        "noise_variance": 0.0
    }
    return GPFunctionalDistribution(kernel, params)


class MyWeaklyPeriodic(jaxkern.base.AbstractKernel):
    """The Radial Basis Function (RBF) kernel."""

    def __init__(
        self,
        compute_engine: jaxkern.computations.AbstractKernelComputation = jaxkern.computations.DenseKernelComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "Radial basis function kernel",
    ) -> None:
        super().__init__(compute_engine, active_dims, stationary, spectral, name)
        self.rbf = jaxkern.stationary.RBF(active_dims=active_dims)
        self.per = jaxkern.stationary.Periodic(active_dims=active_dims)

    def __call__(self, params, x, y):
        params_rbf = {
            "variance": 1.0,
            "lengthscale": params["rbf_lengthscale"],
        }
        params_per = {
            "variance": 1.0,
            "lengthscale": params["per_lengthscale"],
            "period": params["period"]
        }
        return (
            params["variance"] * self.rbf(params_rbf, x, y) * self.per(params_per, x, y)
        )

    def init_params(self, key):
        params = {
            "variance": jnp.array([1.0]),
            "per_lengthscale": jnp.array([1.0] * self.ndims),
            "rbf_lengthscale": jnp.array([1.0] * self.ndims),
            "period": jnp.array([1.0] * self.ndims),
        }
        return jax.tree_util.tree_map(lambda x: jnp.atleast_1d(x), params)


@register_dataset_factory("weaklyperiodic")
def _weaklyper_dataset_factory():
    per = MyWeaklyPeriodic(active_dims=[0])
    white = jaxkern.White(active_dims=[0])
    kernel = jaxkern.SumKernel([per, white])
    params = {
        "mean_function": {},
        "kernel": [
            {
            "variance": _KERNEL_VAR,
            "per_lengthscale": _LENGTHSCALE,
            "rbf_lengthscale": 0.5,
            "period": 1.0,
            },
            {"variance": _NOISE_VAR,},
        ],
        "noise_variance": 0.0,
    }

    return GPFunctionalDistribution(kernel, params)


@register_dataset_factory("periodic")
def _weaklyper_dataset_factory():
    per = jaxkern.stationary.Periodic(active_dims=[0])
    white = jaxkern.White(active_dims=[0])
    kernel = jaxkern.SumKernel([per, white])
    params = {
        "mean_function": {},
        "kernel": [
            {
            "variance": _KERNEL_VAR,
            "lengthscale": 0.5,
            "period": 1.0,
            },
            {"variance": _NOISE_VAR,},
        ],
        "noise_variance": 0.0,
    }

    return GPFunctionalDistribution(kernel, params)


class Sawtooth(FuntionalDistribution):

    A = 1.
    K_max = 20
    mean = 0.5
    variance = 0.07965

    """ See appendix H: https://arxiv.org/pdf/2007.01332.pdf"""
    def sample(self, key, x: Float[Array, "N 1"]) -> Float[Array, "N 1"]:
        fkey, skey, kkey = jax.random.split(key, 3)
        f = jax.random.uniform(fkey, (), minval=3., maxval=5.)
        s = jax.random.uniform(skey, (), minval=-5., maxval=5.)
        ks = jnp.arange(1, self.K_max + 1, dtype=x.dtype)[None, :]
        vals = (-1.) ** ks * jnp.sin(2. * jnp.pi * ks * f * (x - s)) / ks
        k = jax.random.randint(kkey, (), minval=10, maxval=self.K_max + 1)
        mask = jnp.where(ks < k, jnp.ones_like(ks), jnp.zeros_like(ks))
        # we substract the mean A/2
        fs = self.A/2 + self.A/jnp.pi * jnp.sum(vals * mask, axis=1, keepdims=True)
        fs = fs - self.mean
        if self.normalize:
            fs = fs / jnp.sqrt(self.variance)
        return fs


@register_dataset_factory("sawtooth")
def _sawtooth_dataset_factory():
    return Sawtooth(is_data_naturally_normalized=False, normalize=False)


class Mixture(FuntionalDistribution):
    def __init__(self, generators: List[FuntionalDistribution]):
        assert len(generators) == 4  # see tests/test_mixture.py for a more general impl.
        self._generators = generators

    def sample(self, key, x: Float[Array, "N 1"]) -> Float[Array, "N 1"]:
        key, skey = jax.random.split(key)
        rand1, rand2, rand3 = jax.random.uniform(skey, shape=(3,))
        return jax.lax.cond(
            rand1 < 0.25,
            lambda: self._generators[0].sample(key, x),
            lambda: jax.lax.cond(
                rand2 < 1./3,
                lambda: self._generators[1].sample(key, x),
                lambda: jax.lax.cond(
                    rand3 < 0.5,
                    lambda: self._generators[2].sample(key, x),
                    lambda: self._generators[3].sample(key, x),
                )
            )
        )


@register_dataset_factory("mixture")
def _mixture_dataset_factory():
    sawtooth = Sawtooth(is_data_naturally_normalized=False, normalize=True)
    return Mixture(
        generators=[
            _DATASET_FACTORIES["se"],
            _DATASET_FACTORIES["matern"],
            _DATASET_FACTORIES["weaklyperiodic"],
            sawtooth,
        ]
    )


def get_batch(key, batch_size: int, name: str, task: str):
    if name not in _DATASET:
        raise NotImplementedError("Unknown dataset: %s." % name)
    if task not in _TASKS:
        raise NotImplementedError("Unknown task: %s." % task)
    

    if task == "training":
        max_n_target = _DATASET_CONFIGS[name].train_num_target.upper
        max_n_context = 0
    else:
        max_n_target = _DATASET_CONFIGS[name].eval_num_target.upper
        max_n_context = _DATASET_CONFIGS[name].eval_num_context.upper

    key, ckey, tkey = jax.random.split(key, 3)
    task = _TASK_CONFIGS[task]
    x_context = task.x_context_dist.sample(seed=ckey, sample_shape=(batch_size, max_n_context, 1))

    x_target = task.x_target_dist.sample(seed=tkey, sample_shape=(batch_size, max_n_target, 1))
    x = jnp.concatenate([x_context, x_target], axis=1)

    keys = jax.random.split(key, batch_size)
    y = jax.vmap(_DATASET_FACTORIES[name].sample)(keys, x)
    return DataBatch(
        xs=x_target,
        ys=y[:, max_n_context:, :],
        xc=x_context,
        yc=y[:, :max_n_context, :]
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
            jnp.zeros_like(batch.xs),  # keep
            jnp.ones_like(batch.xs)  # ignore
        )[..., 0]

        # repeat for context
        if context_num_data_sampler is not None:
            num_data_total = batch.xc.shape[1]
            num_data = context_num_data_sampler.sample(key, shape=())
            mask_context = jnp.where(
                jnp.arange(num_data_total)[None, :, None] < num_data,
                jnp.zeros_like(batch.xc),  # keep
                jnp.ones_like(batch.xc),  # ignore
            )[..., 0]
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
            print("="*10)

        def plot_data(xc, yc, xt, yt, ax, legend=True, ns=1):
            info(xc, "context")
            info(xt, "target")
            ax.plot(xt[:ns, :, 0].T, yt[:ns, :, 0].T, "C1.", label="target")
            ax.plot(xc[:ns, :, 0].T, yc[:ns, :, 0].T, "C0.", label="context")
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
            plot_data(data.xc, data.yc, data.xs, data.ys, ax, legend=(i==0) and (j==0))
            if i == 0:
                ax.set_title(task)
            if j == 0:
                ax.set_ylabel(dataset)
        
        plt.savefig("fig1.png")


        nrows = len(_DATASET_FACTORIES)
        fig, axes = plt.subplots(nrows, 1, figsize=(15, 3 * nrows), sharex=True)
        for i, name in enumerate(_DATASET_FACTORIES.keys()):
            ax = axes[i]
            keys = jax.random.split(key, 16)
            x = jnp.linspace(-2, 3, 500)[:, None]
            y = jax.vmap(_DATASET_FACTORIES[name].sample, in_axes=[0, None])(keys, x)
            ax.set_title(name)
            ax.plot(x, y[:3, :, 0].T)

        plt.savefig("fig2.png")
    
    plot_data()