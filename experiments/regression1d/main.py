from typing import Tuple, Mapping, Iterator, List, Optional
from jaxtyping import Float, Array

import os
import dataclasses
from absl import app

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


import numpy as np
import functools
import tqdm
import pathlib
import haiku as hk
import jaxkern
import gpjax
import jax
import jax.numpy as jnp
import pandas as pd
import optax
import datetime
import matplotlib.pyplot as plt

from ml_collections import config_dict, config_flags
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)

import neural_diffusion_processes as ndp
from neural_diffusion_processes.ml_tools.state import TrainingState
from neural_diffusion_processes.ml_tools import config_utils
from neural_diffusion_processes import ml_tools
from neural_diffusion_processes.data import regression1d


try:
    from .config import Config, toy_config
except:
    from config import Config, toy_config


_DATETIME = datetime.datetime.now().strftime("%b%d_%H%M%S")
_HERE = pathlib.Path(__file__).parent
_LOG_DIR = 'logs'


_CONFIG = config_flags.DEFINE_config_dict("config", config_utils.to_configdict(Config()))


def data_generator(key, dataset, task, total_num_samples, batch_size, num_epochs: Optional[int] = None):
    assert total_num_samples % batch_size == 0

    def batch(key) -> ndp.data.DataBatch:
        return regression1d.get_batch(key, batch_size, dataset, task)

    if num_epochs is None:
        num_epochs = np.inf
    
    count_epochs = 0
    while count_epochs < num_epochs:
        count_epochs += 1
        for _ in range(total_num_samples // batch_size):
            key, bkey = jax.random.split(key)
            yield batch(bkey)


def get_experiment_name(config: Config):
    return f"{_DATETIME}_{config_utils.get_id(config)}"


def get_experiment_dir(config: Config, output: str = "root", exist_ok: bool = True) -> pathlib.Path:
    experiment_name = get_experiment_name(config)

    if output == "root":
        dir_ = _HERE / _LOG_DIR / experiment_name
    elif output == "plots":
        dir_ = _HERE / _LOG_DIR / experiment_name / "plots"
    elif output == "tensorboard":
        dir_ = _HERE / _LOG_DIR / "tensorboard" / experiment_name
    else:
        raise ValueError("Unknown output: %s" % output)

    dir_.mkdir(parents=True, exist_ok=exist_ok)
    return dir_


def _get_key_iter(init_key) -> Iterator["jax.random.PRNGKey"]:
    while True:
        init_key, next_key = jax.random.split(init_key)
        yield next_key


def main(_):
    config = config_utils.to_dataclass(Config, _CONFIG.value)

    path = get_experiment_dir(config, 'root') / 'config.yaml'
    with open(str(path), 'w') as f:
        f.write(config_utils.to_yaml(config))

    key = jax.random.PRNGKey(config.seed)
    key_iter = _get_key_iter(key)

    ####### init relevant diffusion classes
    beta = ndp.sde.LinearBetaSchedule()
    limiting_kernel = ndp.kernels.get_kernel(config.sde.limiting_kernel, active_dims=[0])
    hyps = {
        "mean_function": {},
        "kernel": limiting_kernel.init_params(None),
    }
    sde = ndp.sde.SDE(
        limiting_kernel,
        gpjax.mean_functions.Zero(),
        hyps,
        beta
    )

    ##### Plot a training databatch
    batch0 = regression1d.get_batch(next(key_iter), 2, config.data.dataset, "training")
    _, ax = plt.subplots()
    ax.plot(batch0.xc[..., 0].T, batch0.yc[..., 0].T, "C0.", label="context")
    ax.plot(batch0.xs[..., 0].T, batch0.ys[..., 0].T, "C1.", label="target")
    handles, labels = ax.get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    ax.legend(handles, labels, loc='best')
    plt.savefig(str(get_experiment_dir(config, "plots") / "data.png"))


    ####### Forward haiku model
    def network(t, y, x):
        model = ndp.models.attention.BiDimensionalAttentionModel(
            n_layers=config.network.num_bidim_attention_layers,
            hidden_dim=config.network.hidden_dim,
            num_heads=config.network.num_heads,
        )
        return model(x, y, t)


    @hk.transform
    def loss_fn(batch: ndp.data.DataBatch):
        # Network awkwardly requires a batch dimension for the inputs
        network_ = lambda t, yt, x, *, key: network(t[None], yt[None], x[None])[0]
        key = hk.next_rng_key()
        return ndp.sde.loss(sde, network_, batch, key)

    num_steps_per_epoch = config.data.num_samples_in_epoch // config.optimization.batch_size
    num_steps = num_steps_per_epoch * config.optimization.num_epochs
    learning_rate_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-4, peak_value=1e-3, warmup_steps=1000, decay_steps=num_steps, end_value=1e-4
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1.0),
    )
    
    @jax.jit
    def init(batch: ndp.data.DataBatch, key) -> TrainingState:
        key, init_rng = jax.random.split(key)
        initial_params = loss_fn.init(init_rng, batch)
        initial_opt_state = optimizer.init(initial_params)
        return TrainingState(
            params=initial_params,
            params_ema=initial_params,
            opt_state=initial_opt_state,
            key=key,
            step=0,
        )


    @jax.jit
    def update_step(state: TrainingState, batch: ndp.data.DataBatch) -> Tuple[TrainingState, Mapping]:
        new_key, loss_key = jax.random.split(state.key)
        loss_and_grad_fn = jax.value_and_grad(loss_fn.apply)
        loss_value, grads = loss_and_grad_fn(state.params, loss_key, batch)
        updates, new_opt_state = optimizer.update(grads, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        new_params_ema = jax.tree_util.tree_map(
            lambda p_ema, p: p_ema * config.optimization.ema_rate
            + p * (1.0 - config.optimization.ema_rate),
            state.params_ema,
            new_params,
        )
        new_state = TrainingState(
            params=new_params,
            params_ema=new_params_ema,
            opt_state=new_opt_state,
            key=new_key,
            step=state.step + 1
        )
        metrics = {
            'loss': loss_value,
            'step': state.step
        }
        return new_state, metrics

    state = init(batch0, jax.random.PRNGKey(config.seed))

    # state = restore_if_exists(state, path)

    exp_root_dir = get_experiment_dir(config)
    
    ########## Plotting
    net_ = hk.without_apply_rng(hk.transform(network))
    net = lambda params, t, yt, x, *, key: net_.apply(params, t[None], yt[None], x[None])[0]
    # x_context = jnp.array([-0.5, 0.2, 0.4]).reshape(-1, 1) + 1.6e-6
    # y_context = x_context * 0.0

    test_batch = regression1d.get_batch(next(key_iter), 16, config.data.dataset, "interpolation")
    x_plt = jnp.linspace(-2, 2, 50)[:, None]
    x_context_plt = test_batch.xc[0]
    y_context_plt = test_batch.yc[0]

    @jax.jit
    def plot_prior(key, params):
        net_ = functools.partial(net, params)
        return ndp.sde.reverse_solve(sde, net_, x_plt, key=key)

    @jax.jit
    def plot_cond(key, params):
        net_ = functools.partial(net, params)
        return ndp.sde.conditional_sample2(sde, net_, x_context_plt, y_context_plt, x_plt, key=key)

    def plots(state: TrainingState, key) -> Mapping[str, plt.Figure]:
        fig_reverse, ax = plt.subplots()
        out = jax.vmap(plot_prior, in_axes=[0, None])(jax.random.split(key, 100), state.params_ema)
        ax.plot(x_plt, out[:, -1, :, 0].T, "C0", alpha=.3)

        fig_cond, ax = plt.subplots()
        key = jax.random.PRNGKey(0)
        samples = jax.vmap(plot_cond, in_axes=[0, None])(jax.random.split(key, 100), state.params_ema)
        ax.plot(x_plt, samples[..., 0].T, "C0", alpha=.3)
        ax.plot(x_context_plt, y_context_plt, "ko")
        ax.set_ylim(-3, 3)
        
        return {
            "reverse": fig_reverse,
            "conditional": fig_cond,
        }

    #############
    @functools.partial(jax.jit)
    def eval(state: TrainingState, key) -> Mapping[str, float]:
        num_samples = 32
        
        @functools.partial(jax.vmap, in_axes=[None, None, None, 0])
        @functools.partial(jax.vmap, in_axes=[0, 0, 0, None])
        def f(x_context, y_context, x_target, key):
            net_ = functools.partial(net, state.params)
            return ndp.sde.conditional_sample(sde, net_, x_context, y_context, x_target, key=key)
        
        metrics = {"mse": [], "mse_median": []}
        
        for i, batch in enumerate(data_test):
            key, *keys = jax.random.split(key, num_samples+1)
            samples = f(batch.context_inputs, batch.context_outputs, batch.function_inputs, jnp.stack(keys))
            f_pred = jnp.mean(samples, axis=0)
            mse_mean_pred = jnp.mean((batch.function_outputs - f_pred) ** 2)
            metrics["mse"].append(mse_mean_pred)
            # ignore outliers
            f_pred = jnp.median(samples, axis=0)
            mse_med_pred = jnp.mean((batch.function_outputs - f_pred) ** 2)
            metrics["mse_median"].append(mse_med_pred)
        
        v = {k: jnp.mean(jnp.stack(v)) for k, v in metrics.items()}
        return v


    local_writer = ml_tools.writers.LocalWriter(exp_root_dir, flush_every_n=100)
    tb_writer = ml_tools.writers.TensorBoardWriter(get_experiment_dir(config, "tensorboard"))
    writer = ml_tools.writers.MultiWriter([tb_writer, local_writer])

    actions = [
        ml_tools.actions.PeriodicCallback(
            every_steps=1,
            callback_fn=lambda step, t, **kwargs: writer.write_scalars(step, kwargs["metrics"])
        ),
        # ml_tools.actions.PeriodicCallback(
        #     every_steps=2_000,
        #     callback_fn=lambda step, t, **kwargs: writer.write_scalars(
        #         step, eval(kwargs["state"], kwargs["key"])
        #     )
        # ),
        ml_tools.actions.PeriodicCallback(
            every_steps=1_000,
            callback_fn=lambda step, t, **kwargs: writer.write_figures(step, plots(kwargs["state"], kwargs["key"]))
        ),
        ml_tools.actions.PeriodicCallback(
            every_steps=500,
            callback_fn=lambda step, t, **kwargs: ml_tools.state.save_checkpoint(kwargs["state"], exp_root_dir, step)
        )
    ]
    train_dataloader = data_generator(
        next(key_iter),
        config.data.dataset,
        "training",
        total_num_samples=config.data.num_samples_in_epoch,
        batch_size=config.optimization.batch_size,
        num_epochs=config.optimization.num_epochs,
    )

    progress_bar = tqdm.tqdm(list(range(1, num_steps + 1)), miniters=1)

    for step, batch, key in zip(progress_bar, train_dataloader, key_iter):
        state, metrics = update_step(state, batch)
        metrics["lr"] = learning_rate_schedule(step)

        for action in actions:
            action(step, t=None, metrics=metrics, state=state, key=key)

        if step % 100 == 0:
            progress_bar.set_description(f"loss {metrics['loss']:.2f}")



if __name__ == "__main__":
    app.run(main)
