#%%
from __future__ import annotations
import itertools
from typing import Callable, Iterable, Mapping, Iterator
from jaxtyping import Float, Array

import datetime
import json
import pathlib
import dataclasses
import pandas as pd

import gpjax
from jaxlinop import DiagonalLinearOperator
from simple_pytree import Pytree
import tqdm


import seaborn as sns
sns.despine()

import jax
import jax.numpy as jnp

from jax.config import config
config.update("jax_enable_x64", True)

from neural_diffusion_processes.data import regression1d
from neural_diffusion_processes.kernels import posterior_gp
from neural_diffusion_processes.misc import flatten


_DATETIME = datetime.datetime.now().strftime("%b%d_%H%M%S")
_HERE = pathlib.Path(__file__).parent
_LOG_DIR = 'logs'


@dataclasses.dataclass
class Data(Pytree):
    x_context: Float[Array, "..."]
    y_context: Float[Array, "..."]
    x_target: Float[Array, "..."]
    y_target: Float[Array, "..."]



@dataclasses.dataclass
class Config:
    seed: int = 20230328
    num_samples_per_epoch: int = int(2^14),
    epochs: int = 20
    num_eval_samples: int = 4096
    batch_size: int = 128


def data_generator(key, dataset, task, total_num_samples, batch_size):
    assert total_num_samples % batch_size == 0

    def batch(key) -> Data:
        xc, yc, xt, yt = regression1d.get_batch(key, batch_size, dataset, task)
        return Data(
            x_context=xc,
            y_context=yc,
            x_target=xt,
            y_target=yt,
        )

    for _ in range(total_num_samples // batch_size):
        key, bkey = jax.random.split(key)
        yield batch(bkey)


def concatenate_arrays(dict_list):
    keys = dict_list[0].keys()
    concatenated_dict = {}
    for key in keys:
        arrays = [d[key] for d in dict_list]
        concatenated_array = jnp.concatenate(arrays, axis=0)
        concatenated_dict[key] = concatenated_array
    return concatenated_dict


import click

@click.command()
@click.option(
    '--model',
    default="gpdiag",
    type=click.Choice(["gpdiag", "gpfull"]),
)
@click.option(
    '--dataset',
    default="se",
    type=click.Choice(regression1d._DATASET),
)
def run(model: str, dataset: str):
    assert dataset != "sawtooth", (
        "GP baseline not available for Sawtooth dataset."
    )
    print(model)
    print(dataset)
    experimet_dir = _HERE / _LOG_DIR / dataset / model
    experimet_dir.mkdir(parents=True, exist_ok=True)
    config=Config()
    key = jax.random.PRNGKey(config.seed)
    
    kernel = regression1d._DATASET_FACTORIES[dataset].kernel
    params = regression1d._DATASET_FACTORIES[dataset].params
    mean_fn = gpjax.mean_functions.Zero()
    params["mean_function"] = {}
    print(params)

    @jax.vmap
    def compute_metrics(x_context, y_context, x_target, y_target) -> Mapping:
        post = posterior_gp(mean_fn, kernel, params, x_context, y_context, obs_noise=params["noise_variance"])
        post_x = post(x_target)
        log_prob_full = post_x.log_prob(flatten(y_target)).squeeze() / len(y_target) 
        post_x.scale = DiagonalLinearOperator(post_x.stddev())  # full_cov = False
        log_prob_diag = post_x.log_prob(flatten(y_target)) / len(y_target) 
        mse_ = jnp.mean((post_x.mean().flatten() - y_target.flatten()) ** 2)
        return {
            "log_prob": log_prob_full if "full" in model else log_prob_diag,
            "log_prob_diag": log_prob_diag,
            "log_prob_full": log_prob_full,
            "mse": mse_
        }

    metrics = {}
    err = lambda v: 1.96 * jnp.std(v) / jnp.sqrt(len(v))
    summary_stats = [("mean", jnp.mean), ("var", jnp.var), ("err", err)]

    for task in regression1d._TASKS:
        if task == "training": continue

        key, tkey = jax.random.split(key)
        data_iter = data_generator(tkey, dataset, task, config.num_eval_samples, config.batch_size)

        task_metrics = []
        for i in tqdm.trange(config.num_eval_samples // config.batch_size):
            batch = next(data_iter)
            out = compute_metrics(
                batch.x_context, batch.y_context, batch.x_target, batch.y_target
            )
            task_metrics.append(out)

        df = pd.DataFrame(concatenate_arrays(task_metrics))
        df.to_csv(str(experimet_dir / task) + ".csv")

        task_metrics_summary = {f"{task}_{stat[0]}_{metric}": float(stat[1](df[metric].values)) for stat, metric in itertools.product(summary_stats, df.columns)}

        metrics = {
            **metrics, 
            **task_metrics_summary
        }

    output = {
        "date": _DATETIME,
        "model": model,
        "dataset": dataset,
        "seed": config.seed,
        "num_samples_per_epoch": config.num_samples_per_epoch,
        "num_eval_samples": config.num_eval_samples,
        "num_epochs": config.epochs,
        **metrics
    }

    with open(str(experimet_dir / ("results" + ".json")), 'w') as file:
        json.dump(output, file, indent=2)


if __name__ == "__main__":
    run()