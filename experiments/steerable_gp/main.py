from typing import Tuple, Mapping, Iterator, List
import os
import socket
import logging
from collections import defaultdict
import functools
import tqdm

import haiku as hk
import jax
import jax.numpy as jnp
import optax

import matplotlib.pyplot as plt

from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate, call
from jax.config import config as jax_config

jax_config.update("jax_enable_x64", True)

import neural_diffusion_processes as ndp
from neural_diffusion_processes import ml_tools
from neural_diffusion_processes.ml_tools.state import TrainingState
from neural_diffusion_processes.utils.loggers_pl import LoggerCollection
from neural_diffusion_processes.utils.vis import plot_vector_field
from neural_diffusion_processes.data import radial_grid_2d, get_vec_gp_log_prob
# from .utils.cfg import *


def _get_key_iter(init_key) -> Iterator["jax.random.PRNGKey"]:
    while True:
        init_key, next_key = jax.random.split(init_key)
        yield next_key


log = logging.getLogger(__name__)


def run(config):
    log.info("Stage : Startup")
    log.info(f"Jax devices: {jax.devices()}")
    run_path = os.getcwd()
    log.info(f"run_path: {run_path}")
    log.info(f"hostname: {socket.gethostname()}")
    ckpt_path = os.path.join(run_path, config.ckpt_dir)
    os.makedirs(ckpt_path, exist_ok=True)
    loggers = [instantiate(logger_config) for logger_config in config.logger.values()]
    logger = LoggerCollection(loggers)
    logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))

    key = jax.random.PRNGKey(config.seed)
    key_iter = _get_key_iter(key)

    ####### init relevant diffusion classes
    beta_schedule = instantiate(config.beta_schedule)
    limiting_kernel = instantiate(config.kernel.cls)
    # limiting_kernel = instantiate(config.sde.limiting_kernel)
    limiting_mean_fn = instantiate(config.sde.limiting_mean_fn)
    limiting_params = {
        # "kernel": OmegaConf.to_container(config.kernel.hyps, resolve=True),
        "kernel": limiting_kernel.init_params(key),
        "mean_fn": limiting_mean_fn.init_params(key),
    }
    # TODO: merge kernel parameters with config ones
    sde = ndp.sde.SDE(limiting_kernel, limiting_mean_fn, limiting_params, beta_schedule)

    ####### prepare data
    data = call(
        config.data,
        key=jax.random.PRNGKey(config.data.seed),
        num_samples=config.data.num_samples_train,
    )
    dataloader = ndp.data.dataloader(
        data, batch_size=config.optimization.batch_size, key=next(key_iter)
    )
    batch0 = next(dataloader)
    x_dim = batch0.xs.shape[-1]
    y_dim = batch0.ys.shape[-1]

    data_test = call(
        config.data,
        key=jax.random.PRNGKey(config.data.seed_test),
        num_samples=config.data.num_samples_test,
    )
    dataloader_test = ndp.data.dataloader(
        data_test,
        batch_size=config.optimization.batch_size,
        key=next(key_iter),
        run_forever=False,  # only run once
    )
    data_test: List[ndp.data.DataBatch] = [
        ndp.data.split_dataset_in_context_and_target(batch, next(key_iter))
        for batch in dataloader_test
    ]

    ####### Forward haiku model
    def network(t, y, x):
        model = instantiate(config.net)
        return model(x, y, t)

    @hk.transform
    def loss_fn(batch: ndp.data.DataBatch):
        # Network awkwardly requires a batch dimension for the inputs
        network_ = lambda t, yt, x, *, key: network(t[None], yt[None], x[None])[0]
        key = hk.next_rng_key()
        return ndp.sde.loss(sde, network_, batch, key)

    learning_rate_schedule = instantiate(config.lr_schedule)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1.0),
    )

    @jax.jit
    def init(batch: ndp.data.DataBatch, key) -> TrainingState:
        key, init_rng = jax.random.split(key)
        # kernel_params = limiting_kernel.init_params(key)
        # print(type(kernel_params), kernel_params)
        initial_params = loss_fn.init(init_rng, batch)
        initial_opt_state = optimizer.init(initial_params)
        return TrainingState(
            params=initial_params,
            # kernel_params=kernel_params,
            opt_state=initial_opt_state,
            key=key,
            step=0,
        )

    @jax.jit
    def update_step(
        state: TrainingState, batch: ndp.data.DataBatch
    ) -> Tuple[TrainingState, Mapping]:
        new_key, loss_key = jax.random.split(state.key)
        loss_and_grad_fn = jax.value_and_grad(loss_fn.apply)
        loss_value, grads = loss_and_grad_fn(state.params, loss_key, batch)
        updates, new_opt_state = optimizer.update(grads, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        new_state = TrainingState(
            params=new_params, opt_state=new_opt_state, key=new_key, step=state.step + 1
        )
        metrics = {"loss": loss_value, "step": state.step}
        return new_state, metrics

    state = init(batch0, jax.random.PRNGKey(config.seed))

    nb_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    log.info(f"Number of parameters: {nb_params}")
    logger.log_hyperparams({"nb_params": nb_params})

    # state = restore_if_exists(state, path)

    progress_bar = tqdm.tqdm(
        list(range(1, config.optimization.num_steps + 1)), miniters=1
    )
    # exp_root_dir = get_experiment_dir(config)

    # ########## Plotting
    net_ = hk.without_apply_rng(hk.transform(network))
    net = lambda params, t, yt, x, *, key: net_.apply(
        params, t[None], yt[None], x[None]
    )[0]

    @jax.jit
    def plot_reverse(key, x_plt, params):
        net_ = functools.partial(net, params)
        return ndp.sde.reverse_solve(sde, net_, x_plt, key=key)

    @jax.jit
    def plot_cond(key, x_plt, params):
        net_ = functools.partial(net, params)
        if x_dim == 1:
            x_known = jnp.reshape(jnp.asarray([[-0.2, 0.2, 0.6]]), (-1, 1)) + 1.0e-2
        elif x_dim == 2:
            x_known = jnp.zeros((1, 2)) + 1.0e-2

        if x_dim == 1 and y_dim == 1:
            y_known = jnp.reshape(
                jnp.asarray([[0.0, -1.0, 0.0]]), (len(x_known), y_dim)
            )
        elif x_dim == 1 and y_dim == 2:
            y_known = jnp.reshape(
                jnp.asarray([[0.0, -1.0, 3.0, 0.2, 1.1, 0.0]]),
                (len(x_known), y_dim),
            )
        elif x_dim == 2 and y_dim == 2:
            x_known = jnp.array([[0.25, 0.5], [0.5, 0.25], [-0.25, -0.25]])
            x_known = x_known.astype(float) + 1.0e-2
            y_known = jnp.array([[1, 1], [1, -2], [-4, 3]]).astype(float)
        return ndp.sde.conditional_sample(sde, net_, x_known, y_known, x_plt, key=key)

    def plots(state: TrainingState, key) -> Mapping[str, plt.Figure]:
        # TODO: refactor properly plot depending on dataset and move in utils/vis.py
        keys = jax.random.split(key, 6)

        # plot data process
        fig_data, ax = plt.subplots(figsize=(8, 8))
        batch = next(iter(data_test))
        idx = jax.random.randint(keys[0], (), minval=0, maxval=len(batch.xs))

        if y_dim == 1:
            pass
        elif y_dim == 2:
            fig_data.set_size_inches(8, 8)
            plot_vector_field(ax, batch.xs[idx], batch.ys[idx])

        # define grid over x space
        if x_dim == 1:
            x_plt = jnp.linspace(-1, 1, 100)[:, None]
        elif x_dim == 2:
            x_plt = radial_grid_2d(20, 30)  # TODO: change grid
        else:
            return []

        # plot forward noising process
        fig_forward, ax = plt.subplots()

        if y_dim == 1:
            pass
        elif y_dim == 2:
            fig_forward.set_size_inches(8, 8)
            t = 0.5 * jnp.ones(())
            out = sde.sample_marginal(keys[1], t, batch.xs[idx], batch.ys[idx])
            plot_vector_field(ax, batch.xs[idx], out)

        # plot limiting process
        fig_limiting, ax = plt.subplots()

        if y_dim == 1:
            pass
        elif y_dim == 2:
            fig_limiting.set_size_inches(8, 8)
            out = sde.sample_prior(keys[2], x_plt)
            plot_vector_field(ax, x_plt, out)

        # plot generated samples
        fig_reverse, ax = plt.subplots()
        if x_dim == 2:
            x_plt = radial_grid_2d(20, 10)  # TODO: change grid

        if out.shape[-1] == 1:
            out = jax.vmap(plot_reverse, in_axes=[0, None, None])(
                jax.random.split(keys[3], 10), x_plt, state.params
            )
            ax.plot(x_plt, out[:, -1, :, 0].T, "C0", alpha=0.3)
        elif out.shape[-1] == 2:
            fig_reverse.set_size_inches(8, 8)
            out = plot_reverse(keys[4], x_plt, state.params).squeeze()
            plot_vector_field(ax, x_plt, out)
        else:
            return []

        # plot conditional samples
        fig_cond, ax = plt.subplots()

        if out.shape[-1] == 1:
            out = jax.vmap(plot_cond, in_axes=[0, None, None])(
                jax.random.split(keys[3], 10), x_plt, state.params
            )
            ax.plot(x_plt, out[:, -1, :, 0].T, "C0", alpha=0.3)
        elif out.shape[-1] == 2:
            fig_cond.set_size_inches(8, 8)
            out = plot_cond(keys[5], x_plt, state.params).squeeze()
            plot_vector_field(ax, x_plt, out)
        else:
            return []

        return {
            "data": fig_data,
            "noised": fig_forward,
            "limiting": fig_limiting,
            "reverse": fig_reverse,
            "conditional": fig_cond,
        }

    # #############
    @functools.partial(jax.jit)
    def eval(state: TrainingState, key) -> Mapping[str, float]:
        num_samples = 32

        @functools.partial(jax.vmap, in_axes=[None, None, None, 0])
        @functools.partial(jax.vmap, in_axes=[0, 0, 0, None])
        def conditional_sample(x_context, y_context, x_target, key):
            net_ = functools.partial(net, state.params)
            return ndp.sde.conditional_sample(
                sde, net_, x_context, y_context, x_target, key=key
            )

        @functools.partial(jax.vmap, in_axes=[0, 0, None])
        def prior_log_prob(x, y, key):
            net_ = functools.partial(net, state.params)
            return ndp.sde.log_prob(sde, net_, x, y, key=key)

        @functools.partial(jax.vmap, in_axes=[None, None, None, 0])
        @functools.partial(jax.vmap)
        def cond_log_prob(xc, yc, xs, ys):
            config.data._target_ = "neural_diffusion_processes.data.get_vec_gp_log_prob"
            posterior_log_prob = call(config.data)
            return posterior_log_prob(xc, yc, xs, ys)

        metrics = defaultdict(list)

        for i, batch in enumerate(data_test):
            key, *keys = jax.random.split(key, num_samples + 1)
            samples = conditional_sample(batch.xc, batch.yc, batch.xs, jnp.stack(keys))
            f_pred = jnp.mean(samples, axis=0)
            mse_mean_pred = jnp.sum((batch.ys - f_pred) ** 2)
            metrics["cond_mse"].append(mse_mean_pred)
            # ignore outliers
            f_pred = jnp.median(samples, axis=0)
            mse_med_pred = jnp.sum((batch.ys - f_pred) ** 2)
            metrics["cond_mse_median"].append(mse_med_pred)
            
            #TODO: clean
            logp = cond_log_prob(batch.xc, batch.yc, batch.xs, samples)
            metrics["cond_log_prob"].append(jnp.sum(jnp.mean(logp, axis=-1)))
        
            x_augmented = jnp.concatenate([batch.xs, batch.xc], axis=1)
            y_augmented = jnp.concatenate([batch.ys, batch.yc], axis=1)
            augmented_logp = prior_log_prob(x_augmented, y_augmented, key)
            context_logp = prior_log_prob(batch.xc, batch.yc, key)
            metrics["cond_log_prob2"].append(jnp.sum(augmented_logp - context_logp))

            logp = prior_log_prob(batch.xs, batch.ys, key)
            metrics["prior_log_prob"].append(jnp.sum(logp))

        v = {k: jnp.sum(jnp.stack(v)) / len(data_test) for k, v in metrics.items()}
        return v

    actions = [
        ml_tools.actions.PeriodicCallback(
            every_steps=1,
            callback_fn=lambda step, t, **kwargs: logger.log_metrics(
                kwargs["metrics"], step
            ),
        ),
        ml_tools.actions.PeriodicCallback(
            # every_steps=2_000,
            every_steps=1,
            callback_fn=lambda step, t, **kwargs: logger.log_metrics(
                eval(kwargs["state"], kwargs["key"]), step
            ),
        ),
        ml_tools.actions.PeriodicCallback(
            every_steps=2_000,
            # every_steps=1,
            callback_fn=lambda step, t, **kwargs: logger.log_plot(
                "process", plots(kwargs["state"], kwargs["key"]), step
            ),
        ),
        ml_tools.actions.PeriodicCallback(
            every_steps=500,
            callback_fn=lambda step, t, **kwargs: ml_tools.state.save_checkpoint(
                kwargs["state"], ckpt_path, step
            ),
        ),
    ]

    for step, batch, key in zip(progress_bar, dataloader, key_iter):
        state, metrics = update_step(state, batch)
        metrics["lr"] = learning_rate_schedule(step)

        for action in actions:
            action(step, t=None, metrics=metrics, state=state, key=key)

        if step % 100 == 0:
            progress_bar.set_description(f"loss {metrics['loss']:.2f}")


@hydra.main(config_path="config", config_name="main", version_base="1.3.2")
def main(cfg):
    os.environ["GEOMSTATS_BACKEND"] = "jax"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["WANDB_START_METHOD"] = "thread"
    # os.environ["JAX_ENABLE_X64"] = "True"

    # from run import run

    return run(cfg)


if __name__ == "__main__":
    main()
