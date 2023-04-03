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

import matplotlib.pyplot as plt

from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate, call
from jax.config import config as jax_config

import neural_diffusion_processes as ndp
from neural_diffusion_processes import ml_tools
from neural_diffusion_processes.ml_tools.state import TrainingState, load_checkpoint, save_checkpoint
from neural_diffusion_processes.utils.loggers_pl import LoggerCollection
from neural_diffusion_processes.utils.vis import plot_scalar_field, plot_vector_field, plot_covariances
from neural_diffusion_processes.utils import flatten
from neural_diffusion_processes.data import radial_grid_2d, DataBatch


def _get_key_iter(init_key) -> Iterator["jax.random.PRNGKey"]:
    while True:
        init_key, next_key = jax.random.split(init_key)
        yield next_key


log = logging.getLogger(__name__)

def run(config):
    jax.config.update("jax_enable_x64", True)
    policy = jmp.get_policy('params=float32,compute=float32,output=float32')

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
    log.info(f"beta_schedule: {sde.beta_schedule}")

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
        batch_size=config.optim.eval_batch_size,
        key=next(key_iter),
        run_forever=False,  # only run once
        n_points=config.data.n_points
    )
    data_test: List[ndp.data.DataBatch] = [
        ndp.data.split_dataset_in_context_and_target(batch, next(key_iter), config.data.min_context, config.data.max_context)
        for batch in dataloader_test
    ]

    ####### Forward haiku model

    @hk.without_apply_rng
    @hk.transform
    def network(t, y, x):
        t, y, x = policy.cast_to_compute((t, y, x))
        model = instantiate(config.net)
        log.info(f"network: {model} | shape={y.shape}")
        return model(x, y, t)
    
    @jit
    def net(params, t, yt, x, *, key):
        #NOTE: Network awkwardly requires a batch dimension for the inputs
        return network.apply(params, t[None], yt[None], x[None])[0]


    def loss_fn(params, batch: ndp.data.DataBatch, key):
        network_ = partial(net, params)
        return ndp.sde.loss(sde, network_, batch, key)

    learning_rate_schedule = instantiate(config.lr_schedule)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1.0),
    )

    @jit
    def init(batch: ndp.data.DataBatch, key) -> TrainingState:
        key, init_rng = jax.random.split(key)
        t = 1. * jnp.zeros((batch.ys.shape[0]))
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
    if config.mode == "eval":  # if resume or evaluate
        state = load_checkpoint(state, ckpt_path, config.optim.num_steps)

    nb_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    log.info(f"Number of parameters: {nb_params}")
    logger.log_hyperparams({"nb_params": nb_params})


    progress_bar = tqdm.tqdm(
        list(range(1, config.optim.num_steps + 1)), miniters=1
    )
    # exp_root_dir = get_experiment_dir(config)

    # ########## Plotting
    @jit
    def reverse_sample(key, x_grid, params, yT=None):
        print("reverse_sample", x_grid.shape)
        net_ = partial(net, params)
        return ndp.sde.sde_solve(sde, net_, x_grid, key=key, yT=yT, prob_flow=False)

    @jit
    def cond_sample(key, x_grid, x_context, y_context, params):
        net_ = partial(net, params)
        x_context += 1.0e-2 #NOTE: to avoid context overlapping with grid
        return ndp.sde.conditional_sample2(sde, net_, x_context, y_context, x_grid, key=key, num_inner_steps=50)
    
    #TODO: data as a class with such methods?
    # @jit
    def cond_log_prob(xc, yc, xs):
        config.data._target_ = "neural_diffusion_processes.data.get_vec_gp_cond"
        posterior_log_prob = call(config.data)
        return posterior_log_prob(xc, yc, xs)

    def plots(state: TrainingState, key, t) -> Mapping[str, plt.Figure]:
        print("plots", t)
        # TODO: refactor properly plot depending on dataset and move in utils/vis.py
        n_samples = 20
        keys = jax.random.split(key, 6)
        plot_vf = partial(plot_vector_field, scale=50*math.sqrt(config.data.variance), width=0.005)
        plot_cov = partial(plot_covariances, scale=0.3/math.sqrt(config.data.variance), zorder=-1)
        def plot_vf_and_cov(x, ys, axes, title="", cov=True):
            plot_vf(x, ys[0], ax=axes[0])
            plot_vf(x, jnp.mean(ys, 0), ax=axes[1])
            covariances = vmap(partial(jax.numpy.cov, rowvar=False), in_axes=[1])(ys)
            plot_cov(x, covariances, ax=axes[1])
            axes[0].set_title(title)

        batch = plot_batch # batch = next(iter(data_test))
        idx = jax.random.randint(keys[0], (), minval=0, maxval=len(batch.xs))

        # define grid over x space
        x_grid = radial_grid_2d(config.data.x_radius, config.data.num_points)

        fig_backward, axes = plt.subplots(2, 5, figsize=(8*5, 8*2), sharex=True, sharey=True)
        fig_backward.subplots_adjust(wspace=0, hspace=0.)
        
        plot_vf_and_cov(batch.xs[idx], batch.ys, axes[:, 0], rf"$p_{{data}}$")

        # plot limiting
        y_ref = vmap(sde.sample_prior, in_axes=[0, None])(jax.random.split(keys[3], n_samples), x_grid).squeeze()
        plot_vf_and_cov(x_grid, y_ref, axes[:, 1], rf"$p_{{ref}}$")

        # plot generated samples
        # TODO: start from same limiting samples as above with yT argument
        y_model = vmap(reverse_sample, in_axes=[0, None, None, 0])(
            jax.random.split(keys[3], n_samples), x_grid, state.params_ema, y_ref
        ).squeeze()
        # def curl(vf):
        #     vf = rearrange(vf, "(n n2) d -> n n2 d", n=math.sqrt(vf.shape[0]))
        #     x_grid = rearrange(x_grid, "(n n2) d -> n n2 d", n=math.sqrt(vf.shape[0]))
        #     u = vf[..., 0]
        #     v = vf[..., 1]
        #     delta_x = delta_y = x_grid[0, -1, 0] - x_grid[0, -2, 0]
        #     print("delta_x", delta_x)
        #     return (v[1:, :] - v[:-1, :]) / delta_x - (u[:, 1:] - u[:, :-1]) / delta_y
        # print("curl", vmap(curl)(y_model).shape)

        # ax.plot(x_grid, y_model[:, -1, :, 0].T, "C0", alpha=0.3)
        plot_vf_and_cov(x_grid, y_model, axes[:, 2], rf"$p_{{model}}$")
        
        # plot conditional data
        # x_grid_w_contxt = jnp.array([x for x in set(tuple(x) for x in x_grid.tolist()).difference(set(tuple(x) for x in batch.xc[idx].tolist()))])
        x_grid_w_contxt = x_grid
        
        posterior_gp = cond_log_prob(batch.xc[idx], batch.yc[idx], x_grid_w_contxt)
        y_cond = posterior_gp.sample(seed=keys[3], sample_shape=(n_samples))
        y_cond = rearrange(y_cond, "k (n d) -> k n d", d=y_dim)
        plot_vf_and_cov(x_grid_w_contxt, y_cond, axes[:, 3], rf"$p_{{model}}$")
        # plot_vf(batch.xs[idx], batch.ys[idx], ax=axes[0][3])
        plot_vf(batch.xc[idx], batch.yc[idx], color="red", ax=axes[0][3])
        plot_vf(batch.xc[idx], batch.yc[idx], color="red", ax=axes[1][3])
        axes[0][3].set_title(rf"$p_{{data}}$")
        axes[1][3].set_aspect('equal')
        
        # plot conditional samples
        y_cond = vmap(lambda key: cond_sample(key, x_grid_w_contxt, batch.xc[idx], batch.yc[idx], state.params_ema))(jax.random.split(keys[3], n_samples)).squeeze()
            # ax.plot(x_grid, y_cond[:, -1, :, 0].T, "C0", alpha=0.3)
        plot_vf_and_cov(x_grid_w_contxt, y_cond, axes[:, 4], rf"$p_{{model}}$")
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
                yt = vmap(lambda key: sde.sample_marginal(key,  t * jnp.ones(()), batch.xs[idx], batch.ys[idx]))(jax.random.split(keys[1], n_samples)).squeeze()
                plot_vf_and_cov(batch.xs[idx], yt, axes[:, k+1], rf"$p_{{t={t}}}$")

            plot_vf_and_cov(x_grid, y_ref, axes[:, -1], rf"$p_{{ref}}$")

            dict_plots["forward"] = fig_forward
        
        plt.close()
        return dict_plots
    

    # #############
    def prior_log_prob(key, x, y, params):
        print("log_prob")
        net_ = partial(net, params)
        # return ndp.sde.log_prob(sde, net_, x, y, key=key, hutchinson_type="Gaussian")
        return ndp.sde.log_prob(sde, net_, x, y, key=key, hutchinson_type="None")
    prior_log_prob = jit(vmap(partial(prior_log_prob, params=state.params_ema), in_axes=[None, 0, 0]))

    # @partial(jax.vmap, in_axes=[None, None, None, 0])
    # @partial(jax.vmap)
    # def cond_log_prob(xc, yc, xs, ys):
    #     config.data._target_ = "neural_diffusion_processes.data.get_vec_gp_cond"
    #     posterior_log_prob = call(config.data)
    #     return posterior_log_prob(xc, yc, xs, ys)

    # @partial(jax.vmap,    in_axes=[None, 0])
    @partial(jax.vmap)
    @jit
    def data_log_prob(xs, ys):
        config.data._target_ = "neural_diffusion_processes.data.get_vec_gp_prior"
        prior = call(config.data)
        # return vmap(lambda x, y: prior(x).log_prob(y))(xs, ys)
        return prior(xs).log_prob(flatten(ys))

    # @jit
    def eval(state: TrainingState, key, t) -> Mapping[str, float]:
        # num_samples = 32
        num_samples = 10
        metrics = defaultdict(list)
        
        for i, batch in enumerate(data_test):
            key, *keys = jax.random.split(key, num_samples + 1)

            # if i <= 1:
            #     samples = vmap(lambda key: vmap(lambda xs, xc, yc: cond_sample(key, xs, xc, yc, state.params_ema))(batch.xs, batch.xc, batch.yc))(jnp.stack(keys)).squeeze()
            #     f_pred = jnp.mean(samples, axis=0)
            #     mse_mean_pred = jnp.sum((batch.ys - f_pred) ** 2, -1).mean(1).mean(0)
            #     metrics["cond_mse"].append(mse_mean_pred)
            #     # ignore outliers
            #     f_pred = jnp.median(samples, axis=0)
            #     mse_med_pred = jnp.sum((batch.ys - f_pred) ** 2, -1).mean(1).mean(0)
            #     metrics["cond_mse_median"].append(mse_med_pred)
            
            # #TODO: clean
            # logp = cond_log_prob(batch.xc, batch.yc, batch.xs, samples)
            # metrics["cond_log_prob"].append(jnp.mean(jnp.mean(logp, axis=-1)))
        
            # x_augmented = jnp.concatenate([batch.xs, batch.xc], axis=1)
            # y_augmented = jnp.concatenate([batch.ys, batch.yc], axis=1)
            # augmented_logp = prior_log_prob(x_augmented, y_augmented, key)
            # context_logp = prior_log_prob(batch.xc, batch.yc, key)
            # metrics["cond_log_prob2"].append(jnp.mean(augmented_logp - context_logp))

            if i > 0:
                continue
            print(i, batch.xs.shape, batch.ys.shape, key.shape)
            from neural_diffusion_processes.kernels import prior_gp
            k0 = ndp.kernels.RBFCurlFree()
            k0_params = {"variance": 10, "lengthscale": 2.23606797749979}
            dist = prior_gp(lambda params, x: jnp.zeros_like(x), k0, {"kernel": k0_params, "mean_function": {}}, obs_noise=0.02)(batch.xs[0])
            # y0s = dist.sample(seed=key, sample_shape=(batch.xs.shape[0]))
            # true_logp = jax.vmap(dist.log_prob)(y0s)
            true_logp = jax.vmap(dist.log_prob)(flatten(batch.ys))
            # true_logp = data_log_prob(batch.xs, batch.ys)
            # metrics["true_bpd"].append(jnp.mean(true_logp) * np.log2(np.exp(1)) / np.prod(batch.ys.shape[-2:]))
            metrics["true_logp"].append(jnp.mean(true_logp))
            # logp, nfe = prior_log_prob(key, batch.xs, batch.ys)
            logp_prior, delta_logp, nfe, yT = prior_log_prob(key, batch.xs, batch.ys)
            print("yT", yT.shape)
            print("mean var yT", (jnp.std(yT, 0) ** 2).mean())
            print("logp_prior bis", sde.log_prob_prior(batch))
            print("logp_prior, delta_logp", logp_prior.shape, delta_logp.shape)
            logp = logp_prior + delta_logp
            print(jnp.mean(logp_prior), jnp.mean(delta_logp), jnp.mean(logp_prior + delta_logp))
            # metrics["bpd"].append(jnp.mean(logp) * np.log2(np.exp(1)) / np.prod(batch.ys.shape[-2:]))
            metrics["logp"].append(jnp.mean(logp))
            metrics["nfe"].append(jnp.mean(nfe))
            # print(i, batch.ys.shape, metrics["true_bpd"][-1], metrics["bpd"][-1], metrics["nfe"][-1])
            print(i, batch.ys.shape, metrics["true_logp"][-1], metrics["logp"][-1], metrics["nfe"][-1])


        # NOTE: currently assuming same batch size, should use sum and / len(data_test) instead?
        v = {k: jnp.mean(jnp.stack(v)) for k, v in metrics.items()}
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
            callback_fn=lambda step, t, **kwargs: save_checkpoint(
                kwargs["state"], ckpt_path, step
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
    ]

    # net(state.params, 0.5 * jnp.ones(()), radial_grid_2d(20, 30), )
    # out = plot_reverse(key, radial_grid_2d(20, 30), state.params)
    
    # logger.log_plot("process", plots(state, key, 0), 0)
    # logger.log_plot("process", plots(state, key, 1), 1)
    # # logger.log_plot("process", plots(state, key, 1), 1)
    # logger.log_metrics(eval(state, key, 0), 0)
    # logger.log_metrics(eval(state, key, 1), 1)
    # logger.save()
    # raise

    if config.mode == "train":
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
    else:
        # for action in actions[3:]:
        action = actions[2]
        action._cb_fn(config.optim.num_steps + 1, t=None, state=state, key=key)
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
