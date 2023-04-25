import math
from typing import Tuple, Mapping, Iterator, List
import os
import socket
import logging
from collections import defaultdict
from functools import partial
import tqdm
import yaml

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import haiku as hk
import jax
from jax import jit, vmap
import jax.numpy as jnp
import optax
import jmp
import numpy as np
from einops import rearrange

import matplotlib.pyplot as plt

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
from neural_diffusion_processes.kernels import SumKernel, WhiteVec


def _get_key_iter(init_key) -> Iterator["jax.random.PRNGKey"]:
    while True:
        init_key, next_key = jax.random.split(init_key)
        yield next_key


log = logging.getLogger(__name__)


def run(cfg):
    jax.config.update("jax_enable_x64", True)
    policy = jmp.get_policy("params=float32,compute=float32,output=float32")

    log.info("Stage : Startup")
    log.info(f"Jax devices: {jax.devices()}")
    run_path = os.getcwd()
    log.info(f"run_path: {run_path}")
    log.info(f"hostname: {socket.gethostname()}")
    ckpt_path = os.path.join(run_path, cfg.paths.ckpt_dir)
    wandb_cfg_path = os.path.join(run_path, "wandb", "config.yaml")
    os.makedirs(os.path.dirname(wandb_cfg_path), exist_ok=True)

    os.makedirs(ckpt_path, exist_ok=True)
    if "wandb" in cfg.logger:
        if cfg.mode == "eval":
            with open(wandb_cfg_path, "r") as file:
                cfg_yaml = yaml.safe_load(file)
            cfg.logger.wandb.id = cfg_yaml["wandb_id"]
        else:
            import wandb

            cfg.logger.wandb.id = wandb.util.generate_id()
            with open(wandb_cfg_path, "w+") as file:
                yaml.safe_dump({"wandb_id": cfg.logger.wandb.id}, file)
    loggers = [instantiate(logger_config) for logger_config in cfg.logger.values()]
    logger = LoggerCollection(loggers)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    key = jax.random.PRNGKey(cfg.seed)
    key_iter = _get_key_iter(key)

    ####### prepare data
    data = call(
        cfg.data,
        key=jax.random.PRNGKey(cfg.data.seed),
        num_samples=cfg.data.num_samples_train,
        dataset="train",
    )
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

    data_test = call(
        cfg.data,
        key=jax.random.PRNGKey(cfg.data.seed_test),
        num_samples=cfg.data.num_samples_test,
        dataset="test",
    )
    plot_batch = DataBatch(xs=data_test[0][:32], ys=data_test[1][:32])
    plot_batch = ndp.data.split_dataset_in_context_and_target(
        plot_batch, next(key_iter), cfg.data.min_context, cfg.data.max_context
    )

    dataloader_test = ndp.data.dataloader(
        data_test,
        batch_size=cfg.eval.batch_size,
        key=next(key_iter),
        run_forever=False,  # only run once
        n_points=cfg.data.n_points,
    )
    data_test: List[ndp.data.DataBatch] = [
        ndp.data.split_dataset_in_context_and_target(
            batch, next(key_iter), cfg.data.min_context, cfg.data.max_context
        )
        for batch in dataloader_test
    ]

    ####### init relevant diffusion classes
    limiting_kernel = instantiate(cfg.kernel.cls)
    kernel_params = limiting_kernel.init_params(key)
    kernel_params.update(
        OmegaConf.to_container(cfg.kernel.params, resolve=True)
    )  # NOTE: breaks RFF?
    if not isinstance(limiting_kernel, WhiteVec):
        limiting_kernel = SumKernel([limiting_kernel, WhiteVec(y_dim)])
        kernel_params = [kernel_params, {"variance": cfg.kernel.noise}]
    limiting_mean_fn = instantiate(cfg.sde.limiting_mean_fn)
    limiting_params = {
        "kernel": kernel_params,
        "mean_function": limiting_mean_fn.init_params(key),
    }
    # limiting_params["kernel"].update(OmegaConf.to_container(cfg.kernel.params, resolve=True)) # NOTE: breaks RFF?
    log.info(f"limiting GP: {type(limiting_kernel)} params={limiting_params['kernel']}")
    # sde = ndp.sde.SDE(limiting_kernel, limiting_mean_fn, limiting_params, beta_schedule)
    # sde = instantiate(cfg.sde, limiting_params=limiting_params, beta_schedule=cfg.beta_schedule)
    sde = instantiate(
        cfg.sde,
        limiting_kernel=limiting_kernel,
        limiting_params=limiting_params,
        beta_schedule=cfg.beta_schedule,
    )
    log.info(f"limiting_params: {sde.limiting_params['kernel']}")
    log.info(f"beta_schedule: {sde.beta_schedule}")

    ####### Forward haiku model
    if cfg.sde.exact_score:
        mean0 = instantiate(cfg.sde.limiting_mean_fn)
        kernel0 = instantiate(cfg.data.kernel)
        kernel0 = SumKernel([kernel0, WhiteVec(y_dim)])
        params0 = OmegaConf.to_container(cfg.data.params, resolve=True)
        params0["kernel"] = [params0["kernel"], {"variance": cfg.data.obs_noise}]
        exact_score = sde.get_exact_score(mean0, kernel0, params0)
        sde.std_trick = False
        sde.residual_trick = False

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

    # exp_root_dir = get_experiment_dir(config)

    # ########## Plotting
    @partial(jit, static_argnums=(5))
    def sde_sample(key, x_grid, params, yT=None, ts=None, forward=False):
        print("sde_sample", x_grid.shape)
        net_ = partial(net, params) if not cfg.sde.exact_score else exact_score
        config = cfg.eval.prior
        return ndp.sde.sde_solve(
            sde,
            net_,
            x_grid,
            key=key,
            y=yT,
            num_steps=config.n_steps,
            rtol=config.rtol,
            atol=config.atol,
            prob_flow=config.prob_flow,
            forward=forward,
            ts=ts,
        )

    @jit
    def cond_sample(key, x_grid, x_context, y_context, params):
        net_ = partial(net, params) if not cfg.sde.exact_score else exact_score
        config = cfg.eval.cond
        x_context += 1.0e-8  # NOTE: to avoid context overlapping with grid
        return ndp.sde.conditional_sample2(
            sde,
            net_,
            x_context,
            y_context,
            x_grid,
            key=key,
            num_steps=config.n_steps,
            num_inner_steps=config.n_inner_steps,
            tau=config.tau,
            psi=config.psi,
            lambda0=config.lambda0,
            prob_flow=config.prob_flow,
            langevin_kernel=False,
        )

    # TODO: data as a class with such methods?
    # @jit
    def cond_log_prob(xc, yc, xs):
        cfg.data._target_ = "neural_diffusion_processes.data.get_vec_gp_cond"
        posterior_log_prob = call(cfg.data)
        return posterior_log_prob(xc, yc, xs)

    def plots(state: TrainingState, key, t) -> Mapping[str, plt.Figure]:
        print("plots", t)
        dict_plots = {}
        # TODO: refactor properly plot depending on dataset and move in utils/vis.py
        n_samples = 20
        keys = jax.random.split(key, 6)
        plot_vf = partial(
            plot_vector_field, scale=50 * math.sqrt(cfg.data.variance), width=0.005
        )
        plot_cov = partial(
            plot_covariances, scale=0.3 / math.sqrt(cfg.data.variance), zorder=-1
        )

        def plot_vf_and_cov(x, ys, axes, title="", cov=True, idx=0):
            plot_vf(x, ys[idx], ax=axes[0])
            plot_vf(x, jnp.mean(ys, 0), ax=axes[1])
            covariances = vmap(partial(jax.numpy.cov, rowvar=False), in_axes=[1])(ys)
            plot_cov(x, covariances, ax=axes[1])
            axes[0].set_title(title)

        batch = plot_batch  # batch = next(iter(data_test))
        idx = jax.random.randint(keys[0], (), minval=0, maxval=len(batch.xs))

        # define grid over x space
        x_grid = radial_grid_2d(cfg.data.x_radius, cfg.data.num_points)

        fig_backward, axes = plt.subplots(
            2, 5, figsize=(8 * 5, 8 * 2), sharex=True, sharey=True
        )
        fig_backward.subplots_adjust(wspace=0, hspace=0.0)

        plot_vf_and_cov(batch.xs[idx], batch.ys, axes[:, 0], rf"$p_{{data}}$", idx=idx)

        # plot limiting
        y_ref = vmap(sde.sample_prior, in_axes=[0, None])(
            jax.random.split(keys[3], n_samples), x_grid
        ).squeeze()
        plot_vf_and_cov(x_grid, y_ref, axes[:, 1], rf"$p_{{ref}}$")

        # plot generated samples
        if cfg.eval.prior.n_steps > 0:
            y_model = vmap(sde_sample, in_axes=[0, None, None, 0])(
                jax.random.split(keys[3], n_samples), x_grid, state.params_ema, y_ref
            ).squeeze()

            plot_vf_and_cov(x_grid, y_model, axes[:, 2], rf"$p_{{model}}$")

        # plot conditional data
        # x_grid_w_contxt = x_grid
        x_grid_w_contxt = batch.xs[idx]

        posterior_gp = cond_log_prob(batch.xc[idx], batch.yc[idx], x_grid_w_contxt)
        # y_cond = posterior_gp.sample(seed=keys[3], sample_shape=(n_samples))
        # y_cond = rearrange(y_cond, "k (n d) -> k n d", d=y_dim)
        # plot_vf_and_cov(x_grid_w_contxt, y_cond, axes[:, 3], rf"$p_{{model}}$")
        y_cond = posterior_gp.sample(seed=keys[3], sample_shape=(1))
        y_cond = rearrange(y_cond, "k (n d) -> k n d", d=y_dim)
        plot_vf(x_grid_w_contxt, y_cond.squeeze(), ax=axes[0, 3])
        plot_vf(x_grid_w_contxt, unflatten(posterior_gp.mean(), y_dim), ax=axes[1, 3])
        ktt = rearrange(
            posterior_gp.covariance(),
            "(n1 p1) (n2 p2) -> n1 n2 p1 p2",
            p1=y_dim,
            p2=y_dim,
        )
        covariances = ktt[jnp.diag_indices(ktt.shape[0])]
        plot_cov(x_grid_w_contxt, covariances, ax=axes[1, 3])
        axes[0, 3].set_title(rf"$p_{{model}}$")

        # plot_vf(batch.xs[idx], batch.ys[idx], ax=axes[0][3])
        plot_vf(batch.xc[idx], batch.yc[idx], color="red", ax=axes[0][3])
        plot_vf(batch.xc[idx], batch.yc[idx], color="red", ax=axes[1][3])
        axes[0][3].set_title(rf"$p_{{data}}$")
        axes[1][3].set_aspect("equal")

        # plot conditional samples
        if cfg.eval.cond.n_steps > 0:
            y_cond = vmap(
                lambda key: cond_sample(
                    key, x_grid_w_contxt, batch.xc[idx], batch.yc[idx], state.params_ema
                )
            )(jax.random.split(keys[3], n_samples)).squeeze()
            print("y_cond", (jnp.isinf(y_cond) | jnp.isnan(y_cond)).sum())
            # ax.plot(x_grid, y_cond[:, -1, :, 0].T, "C0", alpha=0.3)
            plot_vf_and_cov(x_grid_w_contxt, y_cond, axes[:, 4], rf"$p_{{model}}$")
            plot_vf(batch.xc[idx], batch.yc[idx], color="red", ax=axes[0][4])
            plot_vf(batch.xc[idx], batch.yc[idx], color="red", ax=axes[1][4])

        dict_plots["backward"] = fig_backward

        if t == 0:  # NOTE: Only plot fwd at the beggining
            ts = jnp.array([0.1, 0.2, 0.5, 0.8, sde.beta_schedule.t1])
            # ts = jnp.array([0.8, sde.beta_schedule.t1])
            nb_cols = len(ts) + 2
            fig_forward, axes = plt.subplots(
                2, nb_cols, figsize=(8 * nb_cols, 8 * 2), sharex=True, sharey=True
            )
            fig_forward.subplots_adjust(wspace=0, hspace=0)

            # plot data process
            plot_vf_and_cov(
                batch.xs[idx], batch.ys, axes[:, 0], rf"$p_{{data}}$", idx=idx
            )

            subkeys = jax.random.split(keys[3], n_samples)
            y_model = vmap(sde_sample, in_axes=[0, None, None, None, None, None])(
                subkeys, batch.xs[idx], state.params_ema, batch.ys[idx], ts, True
            ).squeeze()

            for k, t in enumerate(ts):
                yt = y_model[:, k]
                plot_vf_and_cov(
                    batch.xs[idx], yt, axes[:, k + 1], rf"$p_{{t={t}|0}}$", idx=idx
                )

            plot_vf_and_cov(x_grid, y_ref, axes[:, -1], rf"$p_{{ref}}$")

            dict_plots["forward"] = fig_forward

        plt.close()
        return dict_plots

    # #############
    def log_prob(key, x, y, params, xc=None, yc=None):
        log.info("log_prob")
        net_ = partial(net, params) if not cfg.sde.exact_score else exact_score
        config = cfg.eval.like
        return ndp.sde.log_prob(
            sde,
            net_,
            x,
            y,
            x_known=xc,
            y_known=yc,
            key=key,
            num_steps=config.n_steps,
            rtol=config.rtol,
            atol=config.atol,
            hutchinson_type=config.hutchinson_type,
            hutchinson_samples=config.hutchinson_samples,
        )

    cfg.data._target_ = "neural_diffusion_processes.data.get_vec_gp_prior"
    true_prior = call(cfg.data)
    cfg.data._target_ = "neural_diffusion_processes.data.get_vec_gp_cond"
    true_posterior = call(cfg.data)

    # @jit
    def eval(state: TrainingState, key, step) -> Mapping[str, float]:
        num_samples = 32
        metrics = defaultdict(list)
        eval_log_prob = jit(vmap(partial(log_prob, params=state.params_ema)))

        if step == cfg.optim.num_steps:
            log.info("Evaluate ground truth")
            # TODO: evaluate diagonal cov GP
            for i, batch in enumerate(data_test):
                log.info(f"${step=}, ${i=}, ${batch.xs.shape=}, ${batch.ys.shape=}")
                n_test = batch.ys.shape[-2]
                true_cond_logp = jax.vmap(
                    lambda xc, yc, x, y: true_posterior(xc, yc, x).log_prob(flatten(y))
                )(batch.xc, batch.yc, batch.xs, batch.ys)
                metrics["true_cond_logp"].append(jnp.mean(true_cond_logp / n_test))

                x = jnp.concatenate([batch.xs, batch.xc], axis=1)
                y = jnp.concatenate([batch.ys, batch.yc], axis=1)
                true_logp = jax.vmap(lambda x, y: true_prior(x).log_prob(flatten(y)))(
                    x, y
                )
                n = y.shape[-2]
                metrics["true_logp"].append(jnp.mean(true_logp / n))

        log.info("Evaluate model")
        for i, batch in enumerate(data_test):
            key, *keys = jax.random.split(key, num_samples + 1)

            # NOTE: only eval on 2 batches appart from the last iteration
            if step >= cfg.optim.num_steps or i < 2:
                # if True:
                log.info(f"${step=}, ${i=}, ${batch.xs.shape=}, ${batch.ys.shape=}")
                subkeys = jax.random.split(key, num=batch.ys.shape[0])

                if cfg.eval.like.n_steps > 0:
                    # predictive log-likelihood
                    n_test = batch.ys.shape[-2]

                    # cond_logp2, nfe = eval_log_prob(
                    #     subkeys, batch.xs, batch.ys, xc=batch.xc, yc=batch.yc
                    # )
                    # metrics["cond_logp2"].append(jnp.mean(cond_logp2 / n_test))
                    # metrics["cond_nfe"].append(jnp.mean(nfe))
                    # print("cond_logp2", cond_logp2.shape)
                    # print("true_cond_logp", true_cond_logp.shape)
                    # print("cond logp", metrics["cond_logp2"][-1])

                    logp_context, _ = eval_log_prob(subkeys, batch.xc, batch.yc)
                    x = jnp.concatenate([batch.xs, batch.xc], axis=1)
                    y = jnp.concatenate([batch.ys, batch.yc], axis=1)
                    logp_joint, nfe = eval_log_prob(subkeys, x, y)
                    cond_logp = logp_joint - logp_context
                    metrics["cond_logp"].append(jnp.mean(cond_logp / n_test))
                    print("cond logp", metrics["cond_logp"][-1])
                    # raise

                    # prior likelihood
                    n = y.shape[-2]
                    metrics["prior_logp"].append(jnp.mean(logp_joint / n))
                    metrics["prior_nfe"].append(jnp.mean(nfe))
                    print("prior logp", metrics["prior_logp"][-1])

                # # predictive mean and covariance mse
                # # TODO: depends on dataset if dist is avail or not
                # if cfg.eval.cond.n_steps > 0:
                #     # true_mean = batch.ys
                #     true_mean = vmap(lambda x: unflatten(true_prior(x).mean(), y_dim))(batch.xs)
                #     def f(x):
                #         ktt = true_prior(x).covariance()
                #         ktt = rearrange(ktt, '(n1 p1) (n2 p2) -> n1 n2 p1 p2', p1=y_dim, p2=y_dim)
                #         return ktt[jnp.diag_indices(ktt.shape[0])]
                #     true_cov = vmap(f)(batch.xs)

                #     ys = jit(vmap(lambda key: vmap(lambda xs, xc, yc: cond_sample(key, xs, xc, yc, state.params_ema))(batch.xs, batch.xc, batch.yc)))(jnp.stack(keys)).squeeze()
                #     f_pred = jnp.mean(ys, axis=0)
                #     mse_mean_pred = jnp.sum((true_mean - f_pred) ** 2, -1).mean(1).mean(0)
                #     metrics["cond_mse"].append(mse_mean_pred)
                #     f_pred = jnp.median(ys, axis=0)
                #     mse_med_pred = jnp.sum((true_mean - f_pred) ** 2, -1).mean(1).mean(0)
                #     metrics["cond_mse_median"].append(mse_med_pred)

                #     f_pred = vmap(vmap(partial(jax.numpy.cov, rowvar=False), in_axes=[1]), in_axes=[2])(ys).transpose(1,0,2,3)
                #     mse_cov_pred = jnp.sum((true_cov - f_pred).reshape(*true_cov.shape[:-2], -1) ** 2, -1)
                #     mse_cov_pred = mse_cov_pred.mean(1).mean(0)
                #     metrics["mse_cov_pred"].append(mse_cov_pred)

        # NOTE: currently assuming same batch size, should use sum and / len(data_test) instead?
        v = {k: jnp.mean(jnp.stack(v)) for k, v in metrics.items()}
        return v

    actions = [
        ml_tools.actions.PeriodicCallback(
            every_steps=10,
            callback_fn=lambda step, t, **kwargs: logger.log_metrics(
                kwargs["metrics"], step
            ),
        ),
        ml_tools.actions.PeriodicCallback(
            every_steps=cfg.optim.num_steps // 1,
            callback_fn=lambda step, t, **kwargs: save_checkpoint(
                kwargs["state"], ckpt_path, step
            ),
        ),
        ml_tools.actions.PeriodicCallback(
            every_steps=cfg.optim.num_steps // 5,
            # every_steps=cfg.optim.num_steps // 20,
            callback_fn=lambda step, t, **kwargs: logger.log_metrics(
                eval(kwargs["state"], kwargs["key"], step), step
            ),
        ),
        ml_tools.actions.PeriodicCallback(
            every_steps=cfg.optim.num_steps // 5,
            # every_steps=cfg.optim.num_steps // 100,
            callback_fn=lambda step, t, **kwargs: logger.log_plot(
                "process", plots(kwargs["state"], kwargs["key"], step), step
            ),
        ),
    ]

    # net(state.params, 0.5 * jnp.ones(()), radial_grid_2d(20, 30), )
    # out = plot_reverse(key, radial_grid_2d(20, 30), state.params)

    logger.log_plot("process", plots(state, key, 0), 0)
    logger.log_metrics(eval(state, key, 0), 0)
    # logger.log_metrics(eval(state, key, 1), 1)
    # logger.log_plot("process", plots(state, key, 1), 1)
    # logger.save()

    if cfg.mode == "train":
        miniters = 50
        progress_bar = tqdm.tqdm(
            list(range(1, cfg.optim.num_steps + 1)),
            mininterval=5.0,
        )
        for step, batch, key in zip(progress_bar, dataloader, key_iter):
            state, metrics = update_step(state, batch)
            if jnp.isnan(metrics["loss"]).any():
                log.warning("Loss is nan")
                break
            metrics["lr"] = learning_rate_schedule(step)

            for action in actions:
                action(step, t=None, metrics=metrics, state=state, key=key)

            if step == 1 or step % miniters == 0:
                progress_bar.set_description(
                    f"loss {metrics['loss']:.2f}", refresh=False
                )
    else:
        for action in actions[2:]:
            # action = actions[2]
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
