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
from jax import jit, vmap
import jax.numpy as jnp
import optax
import jmp
import numpy as np
from einops import rearrange
from equinox import filter_jit

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate, call
from jax.config import config as jax_config

import neural_diffusion_processes as ndp
from neural_diffusion_processes import ml_tools
from neural_diffusion_processes.ml_tools.state import (
    TrainingState,
    load_checkpoint,
    save_checkpoint,
)
from neural_diffusion_processes.utils.loggers_pl import LoggerCollection
from neural_diffusion_processes.utils.vis import (
    plot_scalar_field,
    plot_vector_field,
    plot_covariances,
)
from neural_diffusion_processes.utils import flatten, unflatten
from neural_diffusion_processes.data import radial_grid_2d, DataBatch


def _get_key_iter(init_key) -> Iterator["jax.random.PRNGKey"]:
    while True:
        init_key, next_key = jax.random.split(init_key)
        yield next_key


log = logging.getLogger(__name__)


def run(cfg):
    # jax.config.update("jax_enable_x64", True)
    policy = jmp.get_policy("params=float32,compute=float32,output=float32")

    log.info("Stage : Startup")
    log.info(f"Jax devices: {jax.devices()}")
    run_path = os.getcwd()
    log.info(f"run_path: {run_path}")
    log.info(f"hostname: {socket.gethostname()}")
    ckpt_path = os.path.join(run_path, cfg.paths.ckpt_dir)
    os.makedirs(ckpt_path, exist_ok=True)
    loggers = [instantiate(logger_config) for logger_config in cfg.logger.values()]
    logger = LoggerCollection(loggers)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    key = jax.random.PRNGKey(cfg.seed)
    key_iter = _get_key_iter(key)

    ####### init relevant diffusion classes
    limiting_kernel = instantiate(cfg.kernel.cls)
    limiting_mean_fn = instantiate(cfg.sde.limiting_mean_fn)
    limiting_params = {
        "kernel": limiting_kernel.init_params(key),
        "mean_function": limiting_mean_fn.init_params(key),
    }
    limiting_params["kernel"].update(
        OmegaConf.to_container(cfg.kernel.params, resolve=True)
    )  # NOTE: breaks RFF?
    log.info(f"limiting GP: {type(limiting_kernel)} params={limiting_params['kernel']}")
    # sde = ndp.sde.SDE(limiting_kernel, limiting_mean_fn, limiting_params, beta_schedule)
    sde = instantiate(
        cfg.sde, limiting_params=limiting_params, beta_schedule=cfg.beta_schedule
    )
    log.info(f"beta_schedule: {sde.beta_schedule}")

    ####### prepare data
    data = call(
        cfg.data,
    )
    data = (data[0], data[1][..., 0:1])  # only lat
    dataloader = ndp.data.dataloader(
        data,
        batch_size=cfg.optim.batch_size,
        key=next(key_iter),
        n_points=cfg.data.n_points,
    )
    batch0 = next(dataloader)
    x_dim = batch0.xs.shape[-1]
    y_dim = batch0.ys.shape[-1]
    log.info(f"num elements: {batch0.xs.shape[-2]} & x_dim: {x_dim} & y_dim: {y_dim}")

    plot_batch = next(
        ndp.data.dataloader(
            data,
            batch_size=100,
            key=next(key_iter),
            n_points=cfg.data.n_points,
        )
    )
    # plot_batch = DataBatch(xs=data[0][:100], ys=data[1][:100])
    # data_test = call(
    #     cfg.data,
    #     key=jax.random.PRNGKey(cfg.data.seed_test),
    #     num_samples=cfg.data.num_samples_test,
    # )
    # plot_batch = DataBatch(xs=data_test[0][:32], ys=data_test[1][:32])
    # plot_batch = ndp.data.split_dataset_in_context_and_target(
    #     plot_batch, next(key_iter), cfg.data.min_context, cfg.data.max_context
    # )

    # dataloader_test = ndp.data.dataloader(
    #     data_test,
    #     batch_size=cfg.optim.eval_batch_size,
    #     key=next(key_iter),
    #     run_forever=False,  # only run once
    #     n_points=cfg.data.n_points,
    # )
    # data_test: List[ndp.data.DataBatch] = [
    #     ndp.data.split_dataset_in_context_and_target(
    #         batch, next(key_iter), cfg.data.min_context, cfg.data.max_context
    #     )
    #     for batch in dataloader_test
    # ]

    ####### Forward haiku model

    @hk.without_apply_rng
    @hk.transform
    def network(t, y, x):
        t, y, x = policy.cast_to_compute((t, y, x))
        model = instantiate(cfg.net)
        log.info(f"network: {model} | shape={y.shape}")
        return model(x, y, t)

    @jit
    def net(params, t, yt, x, *, key):
        # NOTE: Network awkwardly requires a batch dimension for the inputs
        return network.apply(params, t[None], yt[None], x[None])[0]

    def loss_fn(params, batch: ndp.data.DataBatch, key):
        network_ = partial(net, params)
        return ndp.sde.loss(sde, network_, batch, key)

    learning_rate_schedule = instantiate(cfg.lr_schedule)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1.0),
    )

    @jit
    def init(batch: ndp.data.DataBatch, key) -> TrainingState:
        key, init_rng = jax.random.split(key)
        t = 1.0 * jnp.zeros((batch.ys.shape[0]))
        initial_params = network.init(init_rng, t=t, y=batch.ys, x=batch.xs)
        initial_params = policy.cast_to_param((initial_params))
        initial_opt_state = optimizer.init(initial_params)
        return TrainingState(
            params=initial_params,
            params_ema=initial_params,
            opt_state=initial_opt_state,
            key=key,
            step=0,
        )

    @jit
    def update_step(
        state: TrainingState, batch: ndp.data.DataBatch
    ) -> Tuple[TrainingState, Mapping]:
        new_key, loss_key = jax.random.split(state.key)
        loss_and_grad_fn = jax.value_and_grad(loss_fn)
        loss_value, grads = loss_and_grad_fn(state.params, batch, loss_key)
        updates, new_opt_state = optimizer.update(grads, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        new_params_ema = jax.tree_util.tree_map(
            lambda p_ema, p: p_ema * cfg.optim.ema_rate
            + p * (1.0 - cfg.optim.ema_rate),
            state.params_ema,
            new_params,
        )
        new_state = TrainingState(
            params=new_params,
            params_ema=new_params_ema,
            opt_state=new_opt_state,
            key=new_key,
            step=state.step + 1,
        )
        metrics = {"loss": loss_value, "step": state.step}
        return new_state, metrics

    state = init(batch0, jax.random.PRNGKey(cfg.seed))
    if cfg.mode == "eval":  # if resume or evaluate
        state = load_checkpoint(state, ckpt_path, cfg.optim.num_steps)

    nb_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    log.info(f"Number of parameters: {nb_params}")
    logger.log_hyperparams({"nb_params": nb_params})

    progress_bar = tqdm.tqdm(list(range(1, cfg.optim.num_steps + 1)), miniters=1)
    # exp_root_dir = get_experiment_dir(cfg)

    # ########## Plotting
    @jit
    def reverse_sample(key, x_grid, params, yT=None):
        print("reverse_sample", x_grid.shape)
        net_ = partial(net, params)
        return ndp.sde.sde_solve(sde, net_, x_grid, key=key, y=yT, prob_flow=False)

    @filter_jit
    def reverse_sample_times(key, x_grid, params, yT=None, ts=None):
        print("reverse_sample", x_grid.shape)
        net_ = partial(net, params)
        return ndp.sde.sde_solve(
            sde, net_, x_grid, key=key, y=yT, prob_flow=False, ts=ts
        )

    @jit
    def cond_sample(key, x_grid, x_context, y_context, params):
        net_ = partial(net, params)
        x_context += 1.0e-5  # NOTE: to avoid context overlapping with grid
        # return ndp.sde.conditional_sample2(sde, net_, x_context, y_context, x_grid, key=key, num_inner_steps=50)
        return ndp.sde.conditional_sample_independant_context_noise(
            sde,
            net_,
            x_context,
            y_context,
            x_grid,
            key=key,
            num_steps=50,
            num_inner_steps=100,
            tau=0.5,
            psi=2.0,
            lambda0=1.5,
            prob_flow=False,
        )

    def plots(state: TrainingState, key, t) -> Mapping[str, plt.Figure]:
        print("plots", t)
        # TODO: refactor properly plot depending on dataset and move in utils/vis.py
        n_samples = 20
        keys = jax.random.split(key, 6)

        batch = plot_batch  # batch = next(iter(data_test))
        # idx = jax.random.randint(keys[0], (), minval=0, maxval=len(batch.xs))
        idx = 0

        x_grid = batch.xs[0]  # jnp.linspace(0, plot_batch.xs.max(), 100)[..., None]

        TWOPI = 2 * jnp.pi
        RADDEG = TWOPI / 360

        ts_fwd = [0.1, 0.2, 0.5, 0.8, float(sde.beta_schedule.t1)]
        ts_bwd = [0.8, 0.5, 0.2, 0.1, float(sde.beta_schedule.t0)]

        def plot_tracks(xs, ys, axes, title=""):
            # ys = jnp.stack(
            #     (
            #         ((ys[..., 0] + jnp.pi) / 2 % jnp.pi) - jnp.pi / 2,
            #         ((ys[..., 1] + jnp.pi) % (2 * jnp.pi)) - jnp.pi,
            #     ),
            #     axis=-1,
            # )
            if len(xs.shape) == 2:
                for y in ys:
                    # m = Basemap(ax=axes, fix_aspect=True)
                    axes.plot(
                        xs[..., 0],
                        y[..., 0],
                        # latlon=True,
                        linewidth=0.3,
                    )
            elif len(xs.shape) == 3:
                for x, y in zip(xs, ys):
                    # m = Basemap(ax=axes, fix_aspect=True)
                    axes.plot(
                        x[..., 0],
                        y[..., 0],
                        # latlon=True,
                        linewidth=0.3,
                    )
            axes.set_title(title)
            # axes.set_aspect("equal")
            # axes.set_xlim([-180, 180])
            # axes.set_ylim([-90, 90])

        nb_cols = len(ts_fwd) + 2
        fig_backward, axes = plt.subplots(
            1, nb_cols, figsize=(2 * nb_cols * 2, 2), sharex=True, sharey=True
        )
        fig_backward.subplots_adjust(wspace=0, hspace=0.0)

        # plot_vf_and_cov(batch.xs[idx], batch.ys, axes[:, 0], rf"$p_{{data}}$")
        plot_tracks(batch.xs, batch.ys, axes[0], rf"$p_{{data}}$")

        # plot limiting
        y_ref = vmap(sde.sample_prior, in_axes=[0, None])(
            jax.random.split(keys[3], n_samples), x_grid
        )
        # y_ref =
        # plot_vf_and_cov(x_grid, y_ref, axes[:, 1], rf"$p_{{ref}}$")t
        plot_tracks(x_grid, y_ref, axes[-1], rf"$p_{{ref}}$")

        # plot generated samples
        # TODO: start from same limiting samples as above with yT argument
        # y_model = vmap(reverse_sample, in_axes=[0, None, None, 0])(
        #     jax.random.split(keys[3], n_samples), x_grid, state.params_ema, y_ref
        # ).squeeze()
        y_model = vmap(reverse_sample_times, in_axes=[0, None, None, 0, None])(
            jax.random.split(keys[3], n_samples),
            x_grid,
            state.params_ema,
            y_ref,
            ts_bwd,
        )

        # ax.plot(x_grid, y_model[:, -1, :, 0].T, "C0", alpha=0.3)
        # plot_vf_and_cov(x_grid, y_model, axes[:, 2], rf"$p_{{model}}$")
        for i in range(nb_cols - 2):
            plot_tracks(
                x_grid,
                y_model[:, i],
                axes[i + 1],
                rf"$p_{{model}} t={ts_bwd[i]}$",
            )

        # plot conditional data
        # x_grid_w_contxt = jnp.array([x for x in set(tuple(x) for x in x_grid.tolist()).difference(set(tuple(x) for x in batch.xc[idx].tolist()))])
        x_grid_w_contxt = x_grid

        # posterior_gp = cond_log_prob(batch.xc[idx], batch.yc[idx], x_grid_w_contxt)
        # # y_cond = posterior_gp.sample(seed=keys[3], sample_shape=(n_samples))
        # # y_cond = rearrange(y_cond, "k (n d) -> k n d", d=y_dim)
        # # plot_vf_and_cov(x_grid_w_contxt, y_cond, axes[:, 3], rf"$p_{{model}}$")
        # y_cond = posterior_gp.sample(seed=keys[3], sample_shape=(1))
        # y_cond = rearrange(y_cond, "k (n d) -> k n d", d=y_dim)
        # plot_vf(x_grid_w_contxt, y_cond.squeeze(), ax=axes[0, 3])
        # plot_vf(x_grid_w_contxt, unflatten(posterior_gp.mean(), y_dim), ax=axes[1, 3])
        # ktt = rearrange(posterior_gp.covariance(), '(n1 p1) (n2 p2) -> n1 n2 p1 p2', p1=y_dim, p2=y_dim)
        # covariances = ktt[jnp.diag_indices(ktt.shape[0])]
        # plot_cov(x_grid_w_contxt, covariances, ax=axes[1, 3])
        # axes[0, 3].set_title(rf"$p_{{model}}$")

        # # plot_vf(batch.xs[idx], batch.ys[idx], ax=axes[0][3])
        # plot_vf(batch.xc[idx], batch.yc[idx], color="red", ax=axes[0][3])
        # plot_vf(batch.xc[idx], batch.yc[idx], color="red", ax=axes[1][3])
        # axes[0][3].set_title(rf"$p_{{data}}$")
        # axes[1][3].set_aspect('equal')

        # # plot conditional samples
        # y_cond = vmap(lambda key: cond_sample(key, x_grid_w_contxt, batch.xc[idx], batch.yc[idx], state.params_ema))(jax.random.split(keys[3], n_samples)).squeeze()
        #     # ax.plot(x_grid, y_cond[:, -1, :, 0].T, "C0", alpha=0.3)
        # plot_vf_and_cov(x_grid_w_contxt, y_cond, axes[:, 4], rf"$p_{{model}}$")
        # plot_vf(batch.xc[idx], batch.yc[idx], color="red", ax=axes[0][4])
        # plot_vf(batch.xc[idx], batch.yc[idx], color="red", ax=axes[1][4])

        dict_plots = {"backward": fig_backward}

        fig_comp, axes = plt.subplots(
            4,
            4,
            figsize=(4 * 2, 4 * 2),
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        for i in range(4):
            plot_tracks(x_grid, y_model[i : (i + 1), -1], axes[0][i], "Model")
        for i in range(4):
            plot_tracks(x_grid, batch.ys[i : (i + 1)], axes[1][i], "Data")
        for i in range(4):
            plot_tracks(x_grid, y_model[(i + 4) : (i + 5), -1], axes[2][i], "Model")
        for i in range(4):
            plot_tracks(x_grid, batch.ys[(i + 4) : (i + 5)], axes[3][i], "Data")

        dict_plots["comparison"] = fig_comp

        if t == 0:  # NOTE: Only plot fwd at the beggining
            # ts = [0.8, sde.beta_schedule.t1]
            nb_cols = len(ts_fwd) + 2
            nb_rows = 4
            fig_forward, axes = plt.subplots(
                nb_rows,
                nb_cols,
                figsize=(2 * nb_cols * 2, nb_rows * 2),
                sharex=True,
                sharey=True,
                squeeze=False,
            )
            fig_forward.subplots_adjust(wspace=0, hspace=0)

            for i in range(nb_rows):
                # plot data process
                # plot_vf_and_cov(batch.xs[0], batch.ys, axes[:, 0], rf"$p_{{data}}$")
                plot_tracks(batch.xs, batch.ys, axes[i, 0], rf"$p_{{data}}$")

                # TODO: only solve once and return different timesaves
                for k, t in enumerate(ts_fwd):
                    yt = vmap(
                        lambda key: sde.sample_marginal(
                            key, t * jnp.ones(()), batch.xs[idx + i], batch.ys[idx + i]
                        )
                    )(jax.random.split(keys[1], n_samples))
                    # plot_vf_and_cov(batch.xs[idx], yt, axes[:, k+1], rf"$p_{{t={t}}}$")
                    plot_tracks(batch.xs, yt, axes[i, k + 1], rf"$p_{{t={t}}}$")

                # plot_vf_and_cov(x_grid, y_ref, axes[:, -1], rf"$p_{{ref}}$")
                plot_tracks(x_grid, y_ref, axes[i, -1], rf"$p_{{ref}}$")

            dict_plots["forward"] = fig_forward

            fig_data, axes = plt.subplots(
                4,
                4,
                figsize=(4 * 2, 4 * 2),
                sharex=True,
                sharey=True,
                squeeze=False,
            )
            axes = [item for sublist in axes for item in sublist]
            for i in range(len(axes)):
                plot_tracks(batch.xs[0:1], batch.ys[i : (i + 1)], axes[i])

            dict_plots["data"] = fig_data

        plt.close()
        return dict_plots

    # # #############
    # def prior_log_prob(key, x, y, params):
    #     print("log_prob")
    #     net_ = partial(net, params)
    #     # return ndp.sde.log_prob(sde, net_, x, y, key=key, hutchinson_type="Gaussian")
    #     # return ndp.sde.log_prob(sde, net_, x, y, key=key, hutchinson_type="None", rtol=1e-6, atol=1e-6)
    #     # return ndp.sde.log_prob(sde, net_, x, y, key=key, hutchinson_type="Gaussian", rtol=1e-6, atol=1e-6, hutchinson_samples=1)
    #     return ndp.sde.log_prob(sde, net_, x, y, key=key, hutchinson_type="Gaussian", rtol=None, atol=None, hutchinson_samples=1)
    #     # return ndp.sde.log_prob(sde, net_, x, y, key=key, hutchinson_type="Gaussian", rtol=None, atol=None)
    #     return ndp.sde.log_prob(sde, net_, x, y, key=key, hutchinson_type="None")
    # prior_log_prob = jit(vmap(partial(prior_log_prob, params=state.params_ema)))

    # # @partial(jax.vmap, in_axes=[None, None, None, 0])
    # # @partial(jax.vmap)
    # # def cond_log_prob(xc, yc, xs, ys):
    # #     cfg.data._target_ = "neural_diffusion_processes.data.get_vec_gp_cond"
    # #     posterior_log_prob = call(cfg.data)
    # #     return posterior_log_prob(xc, yc, xs, ys)

    # cfg.data._target_ = "neural_diffusion_processes.data.get_vec_gp_prior"
    # prior = call(cfg.data)

    # # @jit
    # def eval(state: TrainingState, key, step) -> Mapping[str, float]:
    #     num_samples = 32
    #     # num_samples = 20
    #     metrics = defaultdict(list)

    #     for i, batch in enumerate(data_test):
    #         key, *keys = jax.random.split(key, num_samples + 1)
    #         # print(step, i, batch.xs.shape, batch.ys.shape, key.shape)

    #         if step != cfg.optim.num_steps and i < 1:
    #             # true_mean = batch.ys
    #             true_mean = vmap(lambda x: unflatten(prior(x).mean(), y_dim))(batch.xs)
    #             def f(x):
    #                 ktt = prior(x).covariance()
    #                 ktt = rearrange(ktt, '(n1 p1) (n2 p2) -> n1 n2 p1 p2', p1=y_dim, p2=y_dim)
    #                 return ktt[jnp.diag_indices(ktt.shape[0])]
    #             true_cov = vmap(f)(batch.xs)

    #             ys = jit(vmap(lambda key: vmap(lambda xs, xc, yc: cond_sample(key, xs, xc, yc, state.params_ema))(batch.xs, batch.xc, batch.yc)))(jnp.stack(keys)).squeeze()
    #             f_pred = jnp.mean(ys, axis=0)
    #             mse_mean_pred = jnp.sum((true_mean - f_pred) ** 2, -1).mean(1).mean(0)
    #             metrics["cond_mse"].append(mse_mean_pred)
    #             f_pred = jnp.median(ys, axis=0)
    #             mse_med_pred = jnp.sum((true_mean - f_pred) ** 2, -1).mean(1).mean(0)
    #             metrics["cond_mse_median"].append(mse_med_pred)

    #             f_pred = vmap(vmap(partial(jax.numpy.cov, rowvar=False), in_axes=[1]), in_axes=[2])(ys).transpose(1,0,2,3)
    #             mse_cov_pred = jnp.sum((true_cov - f_pred).reshape(*true_cov.shape[:-2], -1) ** 2, -1)
    #             mse_cov_pred = mse_cov_pred.mean(1).mean(0)
    #             metrics["mse_cov_pred"].append(mse_cov_pred)

    #         # #TODO: clean
    #         # logp = cond_log_prob(batch.xc, batch.yc, batch.xs, samples)
    #         # metrics["cond_log_prob"].append(jnp.mean(jnp.mean(logp, axis=-1)))

    #         # x_augmented = jnp.concatenate([batch.xs, batch.xc], axis=1)
    #         # y_augmented = jnp.concatenate([batch.ys, batch.yc], axis=1)
    #         # augmented_logp = prior_log_prob(x_augmented, y_augmented, key)
    #         # context_logp = prior_log_prob(batch.xc, batch.yc, key)
    #         # metrics["cond_log_prob2"].append(jnp.mean(augmented_logp - context_logp))

    #         # if i > 0:
    #             # continue
    #         # true_logp = jax.vmap(dist.log_prob)(flatten(batch.ys))
    #         # # metrics["true_bpd"].append(jnp.mean(true_logp) * np.log2(np.exp(1)) / np.prod(batch.ys.shape[-2:]))
    #         # metrics["true_logp"].append(jnp.mean(true_logp))
    #         # # logp, nfe = prior_log_prob(key, batch.xs, batch.ys)
    #         # subkeys = jax.random.split(key, num=batch.ys.shape[0])
    #         # # logp_prior, delta_logp, nfe, yT = prior_log_prob(subkeys, batch.xs, batch.ys)
    #         # # logp = logp_prior + delta_logp
    #         # # print("logp_prior, delta_logp", logp_prior.shape, delta_logp.shape)
    #         # # print("true_logp", true_logp)
    #         # # print(logp_prior.squeeze())
    #         # # print(delta_logp.squeeze())
    #         # # print(logp.squeeze())
    #         # # metrics["bpd"].append(jnp.mean(logp) * np.log2(np.exp(1)) / np.prod(batch.ys.shape[-2:]))
    #         # logp, nfe = prior_log_prob(subkeys, batch.xs, batch.ys)
    #         # metrics["logp"].append(jnp.mean(logp))
    #         # metrics["nfe"].append(jnp.mean(nfe))
    #         # print(metrics["logp"][-1], metrics["nfe"][-1])

    #     # NOTE: currently assuming same batch size, should use sum and / len(data_test) instead?
    #     v = {k: jnp.mean(jnp.stack(v)) for k, v in metrics.items()}
    #     return v

    actions = [
        ml_tools.actions.PeriodicCallback(
            every_steps=1,
            callback_fn=lambda step, t, **kwargs: logger.log_metrics(
                kwargs["metrics"], step
            ),
        ),
        ml_tools.actions.PeriodicCallback(
            every_steps=cfg.optim.num_steps // 10,
            callback_fn=lambda step, t, **kwargs: save_checkpoint(
                kwargs["state"], ckpt_path, step
            ),
        ),
        # ml_tools.actions.PeriodicCallback(
        #     # every_steps=cfg.optim.num_steps // 10,
        #     every_steps=cfg.optim.num_steps // 10,
        #     callback_fn=lambda step, t, **kwargs: logger.log_metrics(
        #         eval(kwargs["state"], kwargs["key"], step), step
        #     ),
        # ),
        ml_tools.actions.PeriodicCallback(
            every_steps=cfg.optim.num_steps // 10,
            callback_fn=lambda step, t, **kwargs: logger.log_plot(
                "process", plots(kwargs["state"], kwargs["key"], step), step
            ),
        ),
    ]

    if cfg.mode == "train":
        logger.log_plot("process", plots(state, jax.random.PRNGKey(cfg.seed), 0), 0)
        for step, batch, key in zip(progress_bar, dataloader, key_iter):
            state, metrics = update_step(state, batch)
            if jnp.isnan(metrics["loss"]).any():
                log.warning("Loss is nan")
                break
            metrics["lr"] = learning_rate_schedule(step)

            for action in actions:
                action(step, t=None, metrics=metrics, state=state, key=key)

            if step % 100 == 0:
                progress_bar.set_description(f"loss {metrics['loss']:.2f}")
    else:
        # for action in actions[3:]:
        action = actions[2]
        action._cb_fn(cfg.optim.num_steps + 1, t=None, state=state, key=key)
        logger.save()


@hydra.main(config_path="config", config_name="main", version_base="1.3.2")
def main(cfg):
    # os.environ["GEOMSTATS_BACKEND"] = "jax"
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["WANDB_START_METHOD"] = "thread"
    # from run import run

    return run(cfg)


if __name__ == "__main__":
    main()
