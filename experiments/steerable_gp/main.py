import math
from typing import Tuple, Mapping, Iterator, List
import os
import socket
import logging
from collections import defaultdict
from functools import partial
import tqdm

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import haiku as hk
import jax
import jax.numpy as jnp
import optax

import matplotlib.pyplot as plt

from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate, call
from jax.config import config as jax_config

import neural_diffusion_processes as ndp
from neural_diffusion_processes import ml_tools
from neural_diffusion_processes.ml_tools.state import TrainingState
from neural_diffusion_processes.utils.loggers_pl import LoggerCollection
from neural_diffusion_processes.utils.vis import plot_scalar_field, plot_vector_field, plot_covariances
from neural_diffusion_processes.data import radial_grid_2d, DataBatch


def _get_key_iter(init_key) -> Iterator["jax.random.PRNGKey"]:
    while True:
        init_key, next_key = jax.random.split(init_key)
        yield next_key


log = logging.getLogger(__name__)

def run(config):
    # jax.config.update("jax_enable_x64", True)
    log.info("Stage : Startup")
    log.info(f"Jax devices: {jax.devices()}")
    run_path = os.getcwd()
    log.info(f"run_path: {run_path}")
    log.info(f"hostname: {socket.gethostname()}")
    ckpt_path = os.path.join(run_path, config.paths.ckpt_dir)
    os.makedirs(ckpt_path, exist_ok=True)
    loggers = [instantiate(logger_config) for logger_config in config.logger.values()]
    logger = LoggerCollection(loggers)
    logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))

    key = jax.random.PRNGKey(config.seed)
    key_iter = _get_key_iter(key)

    ####### init relevant diffusion classes
    limiting_kernel = instantiate(config.kernel.cls)
    limiting_mean_fn = instantiate(config.sde.limiting_mean_fn)
    limiting_params = {
        "kernel": limiting_kernel.init_params(key),
        "mean_function": limiting_mean_fn.init_params(key),
    }
    limiting_params["kernel"].update(OmegaConf.to_container(config.kernel.params, resolve=True)) # NOTE: breaks RFF?
    log.info(f"limiting GP: {type(limiting_kernel)} params={limiting_params['kernel']}")
    # sde = ndp.sde.SDE(limiting_kernel, limiting_mean_fn, limiting_params, beta_schedule)
    sde = instantiate(config.sde, limiting_params=limiting_params, beta_schedule=config.beta_schedule)

    ####### prepare data
    data = call(
        config.data,
        key=jax.random.PRNGKey(config.data.seed),
        num_samples=config.data.num_samples_train,
    )
    dataloader = ndp.data.dataloader(
        data, batch_size=config.optim.batch_size, key=next(key_iter), n_points=config.data.n_points
    )
    batch0 = next(dataloader)
    x_dim = batch0.xs.shape[-1]
    y_dim = batch0.ys.shape[-1]
    log.info(f"num elements: {batch0.xs.shape[-2]} & x_dim: {x_dim} & y_dim: {y_dim}")

    data_test = call(
        config.data,
        key=jax.random.PRNGKey(config.data.seed_test),
        num_samples=config.data.num_samples_test,
    )
    plot_batch = DataBatch(xs=data_test[0][:32], ys=data_test[1][:32])
    plot_batch = ndp.data.split_dataset_in_context_and_target(plot_batch, next(key_iter), config.data.min_context, config.data.max_context)
    
    dataloader_test = ndp.data.dataloader(
        data_test,
        batch_size=config.optim.batch_size,
        key=next(key_iter),
        run_forever=False,  # only run once
        n_points=config.data.n_points
    )
    data_test: List[ndp.data.DataBatch] = [
        ndp.data.split_dataset_in_context_and_target(batch, next(key_iter), config.data.min_context, config.data.max_context)
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
        # initial_params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), initial_params)
        initial_opt_state = optimizer.init(initial_params)
        return TrainingState(
            params=initial_params,
            params_ema=initial_params,
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
        new_params_ema = jax.tree_util.tree_map(
                lambda p_ema, p: p_ema * config.optim.ema_rate
                + p * (1.0 - config.optim.ema_rate),
                state.params_ema,
                new_params,
        )
        new_state = TrainingState(
            params=new_params, params_ema=new_params_ema,opt_state=new_opt_state, key=new_key, step=state.step + 1
        )
        metrics = {"loss": loss_value, "step": state.step}
        return new_state, metrics

    state = init(batch0, jax.random.PRNGKey(config.seed))

    nb_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    log.info(f"Number of parameters: {nb_params}")
    logger.log_hyperparams({"nb_params": nb_params})

    # state = restore_if_exists(state, path)

    progress_bar = tqdm.tqdm(
        list(range(1, config.optim.num_steps + 1)), miniters=1
        # list(range(0, config.optim.num_steps)), miniters=1
    )
    # exp_root_dir = get_experiment_dir(config)

    # ########## Plotting
    net_ = hk.without_apply_rng(hk.transform(network))
    net = lambda params, t, yt, x, *, key: net_.apply(
        params, t[None], yt[None], x[None]
    )[0]

    @jax.jit
    def plot_reverse(key, x_plt, params, yT=None):
        net_ = partial(net, params)
        return ndp.sde.reverse_solve(sde, net_, x_plt, key=key, yT=yT)

    @jax.jit
    def plot_cond(key, x_plt, params, x_context, y_context):
        net_ = partial(net, params)
        x_context = x_context.astype(float) + 1.0e-2 #NOTE: to avoid context overlapping with grid
        return ndp.sde.conditional_sample2(sde, net_, x_context, y_context, x_plt, key=key)

    def plots(state: TrainingState, key, t) -> Mapping[str, plt.Figure]:
        # TODO: refactor properly plot depending on dataset and move in utils/vis.py
        n_samples = 30
        keys = jax.random.split(key, 6)
        plot_vf = partial(plot_vector_field, scale=50*math.sqrt(config.data.variance), width=0.005)
        plot_cov = partial(plot_covariances, scale=0.1, zorder=-1)
        def plot_vf_and_cov(x, ys, axes, title="", cov=True):
            # print(title, x.shape, ys.shape)
            plot_vf(x, ys[0], ax=axes[0])
            plot_vf(x, jnp.mean(ys, 0), ax=axes[1])
            covariances = jax.vmap(partial(jax.numpy.cov, rowvar=False), in_axes=[1])(ys)
            plot_cov(x, covariances, ax=axes[1])
            axes[0].set_title(title)

        # batch = next(iter(data_test))
        batch = next(iter(plot_batch))
        idx = jax.random.randint(keys[0], (), minval=0, maxval=len(batch.xs))

        # define grid over x space
        x_plt = jnp.linspace(-1, 1, 100)[:, None]
        x_plt = radial_grid_2d(config.data.x_radius, config.data.num_points)

        fig_backward, axes = plt.subplots(2, 5, figsize=(8*5, 8*2), sharex=True, sharey=True)
        fig_backward.subplots_adjust(wspace=0, hspace=0.)
        
        plot_vf_and_cov(batch.xs[idx], batch.ys, axes[:, 0], rf"$p_{{data}}$")

        # plot limiting
        y_ref = jax.vmap(sde.sample_prior, in_axes=[0, None])(jax.random.split(keys[3], n_samples), x_plt).squeeze()
        plot_vf_and_cov(x_plt, y_ref, axes[:, 1], rf"$p_{{ref}}$")

        # plot generated samples
        # TODO: start from same limiting samples as above with yT argument
        y_model = jax.vmap(plot_reverse, in_axes=[0, None, None, 0])(
            jax.random.split(keys[3], n_samples), x_plt, state.params_ema, y_ref
        ).squeeze()
        # ax.plot(x_plt, y_model[:, -1, :, 0].T, "C0", alpha=0.3)
        plot_vf_and_cov(x_plt, y_model, axes[:, 2], rf"$p_{{model}}$")
        
        # plot conditional data
        plot_vf(batch.xs[idx], batch.ys[idx], ax=axes[0][3])
        plot_vf(batch.xc[idx], batch.yc[idx], color="red", ax=axes[0][3])
        plot_vf(batch.xc[idx], batch.yc[idx], color="red", ax=axes[1][3])
        axes[0][3].set_title(rf"$p_{{data}}$")
        
        # plot conditional samples
        xc = batch.xc[idx]
        y_cond = jax.vmap(lambda key: plot_cond(key, x_plt, state.params_ema, xc, batch.yc[idx]))(jax.random.split(keys[3], n_samples)).squeeze()
            # ax.plot(x_plt, y_cond[:, -1, :, 0].T, "C0", alpha=0.3)
        plot_vf_and_cov(x_plt, y_cond, axes[:, 4], rf"$p_{{model}}$")
        plot_vf(batch.xc[idx], batch.yc[idx], color="red", ax=axes[0][4])
        plot_vf(batch.xc[idx], batch.yc[idx], color="red", ax=axes[1][4])

        dict_plots = {"backward": fig_backward}
        
        if t == 0: # NOTE: Only plot fwd at the beggining
            # ts = [0.2, 0.5, 0.8, sde.beta_schedule.t1]
            ts = [0.8, sde.beta_schedule.t1]
            nb_cols = len(ts) + 2
            fig_forward, axes = plt.subplots(2, nb_cols, figsize=(8*nb_cols, 8*2), sharex=True, sharey=True)
            fig_forward.subplots_adjust(wspace=0, hspace=0)

            # plot data process
            plot_vf_and_cov(batch.xs[0], batch.ys, axes[:, 0], rf"$p_{{data}}$")

            # TODO: only solve once and return different timesaves
            for k, t in enumerate(ts):
                yt = jax.vmap(lambda key: sde.sample_marginal(key,  t * jnp.ones(()), batch.xs[idx], batch.ys[idx]))(jax.random.split(keys[1], n_samples)).squeeze()
                plot_vf_and_cov(batch.xs[idx], yt, axes[:, k+1], rf"$p_{{t={t}}}$")

            plot_vf_and_cov(x_plt, y_ref, axes[:, -1], rf"$p_{{ref}}$")

            dict_plots["forward"] = fig_forward
        
        plt.close()
        return dict_plots
    

    # #############
    @partial(jax.jit)
    def eval(state: TrainingState, key, t) -> Mapping[str, float]:
        num_samples = 32

        @partial(jax.vmap, in_axes=[None, 0])
        @partial(jax.vmap, in_axes=[0, None])
        def model_sample(x_target, key):
            net_ = partial(net, state.params_ema)
            return ndp.sde.reverse_solve(
                sde, net_, x_target, key=key
            )

        @partial(jax.vmap, in_axes=[None, None, None, 0])
        @partial(jax.vmap, in_axes=[0, 0, 0, None])
        def conditional_sample(x_context, y_context, x_target, key):
            net_ = partial(net, state.params_ema)
            return ndp.sde.conditional_sample(
                sde, net_, x_context, y_context, x_target, key=key
            )

        @partial(jax.vmap, in_axes=[0, 0, None])
        def prior_log_prob(x, y, key):
            net_ = partial(net, state.params_ema)
            return ndp.sde.log_prob(sde, net_, x, y, key=key, hutchinson_type="Gaussian")

        @partial(jax.vmap, in_axes=[None, None, None, 0])
        @partial(jax.vmap)
        def cond_log_prob(xc, yc, xs, ys):
            config.data._target_ = "neural_diffusion_processes.data.get_vec_gp_cond_log_prob"
            posterior_log_prob = call(config.data)
            return posterior_log_prob(xc, yc, xs, ys)

        @partial(jax.vmap, in_axes=[None, 0])
        @partial(jax.vmap)
        def data_log_prob(xs, ys):
            config.data._target_ = "neural_diffusion_processes.data.get_vec_gp_prior_log_prob"
            prior_log_prob = call(config.data)
            return prior_log_prob(xs, ys)

        metrics = defaultdict(list)

        for i, batch in enumerate(data_test):
            key, *keys = jax.random.split(key, num_samples + 1)
            # samples = conditional_sample(batch.xc, batch.yc, batch.xs, jnp.stack(keys))
            # f_pred = jnp.mean(samples, axis=0)
            # mse_mean_pred = jnp.sum((batch.ys - f_pred) ** 2)
            # metrics["cond_mse"].append(mse_mean_pred)
            # # ignore outliers
            # f_pred = jnp.median(samples, axis=0)
            # mse_med_pred = jnp.sum((batch.ys - f_pred) ** 2)
            # metrics["cond_mse_median"].append(mse_med_pred)
            
            # #TODO: clean
            # logp = cond_log_prob(batch.xc, batch.yc, batch.xs, samples)
            # metrics["cond_log_prob"].append(jnp.mean(jnp.mean(logp, axis=-1)))
        
            # x_augmented = jnp.concatenate([batch.xs, batch.xc], axis=1)
            # y_augmented = jnp.concatenate([batch.ys, batch.yc], axis=1)
            # augmented_logp = prior_log_prob(x_augmented, y_augmented, key)
            # context_logp = prior_log_prob(batch.xc, batch.yc, key)
            # metrics["cond_log_prob2"].append(jnp.mean(augmented_logp - context_logp))
            logp = prior_log_prob(batch.xs, batch.ys, key)
            metrics["log_prob"].append(jnp.mean(logp))


        # NOTE: currently assuming same batch size, should use sum and / len(data_test) instead?
        v = {k: jnp.mean(jnp.stack(v)) for k, v in metrics.items()}
        # samples = model_sample(batch.xs, jnp.stack(keys)).squeeze()
        # logp = data_log_prob(batch.xs, samples)
        # v["data_log_prob"] = jnp.mean(jnp.mean(logp, axis=-1))
        
        return v

    actions = [
        ml_tools.actions.PeriodicCallback(
            every_steps=1,
            callback_fn=lambda step, t, **kwargs: logger.log_metrics(
                kwargs["metrics"], step
            ),
        ),
        ml_tools.actions.PeriodicCallback(
            every_steps=config.optim.num_steps // 10,
            callback_fn=lambda step, t, **kwargs: logger.log_metrics(
                eval(kwargs["state"], kwargs["key"], step), step
            ),
        ),
        ml_tools.actions.PeriodicCallback(
            every_steps=config.optim.num_steps // 10,
            callback_fn=lambda step, t, **kwargs: logger.log_plot(
                "process", plots(kwargs["state"], kwargs["key"], step), step
            ),
        ),
        ml_tools.actions.PeriodicCallback(
            every_steps=config.optim.num_steps // 100,
            callback_fn=lambda step, t, **kwargs: ml_tools.state.save_checkpoint(
                kwargs["state"], ckpt_path, step
            ),
        ),
    ]

    # net(state.params, 0.5 * jnp.ones(()), radial_grid_2d(20, 30), )
    # out = plot_reverse(key, radial_grid_2d(20, 30), state.params)
    
    # logger.log_plot("process", plots(state, key, 0), 0)
    # logger.log_metrics(eval(state, key, 0), 0)
    # logger.save()

    for step, batch, key in zip(progress_bar, dataloader, key_iter):
        state, metrics = update_step(state, batch)
        if jnp.isnan(metrics['loss']).any():
            log.warning("Loss is nan")
            break
        metrics["lr"] = learning_rate_schedule(step)

        for action in actions:
            action(step, t=None, metrics=metrics, state=state, key=key)

        if step % 100 == 0:
            progress_bar.set_description(f"loss {metrics['loss']:.2f}")


@hydra.main(config_path="config", config_name="main", version_base="1.3.2")
def main(cfg):
    # os.environ["GEOMSTATS_BACKEND"] = "jax"
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["WANDB_START_METHOD"] = "thread"
    # from run import run

    return run(cfg)


if __name__ == "__main__":
    main()
