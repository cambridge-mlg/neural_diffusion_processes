# %%
%load_ext autoreload
%autoreload 2

# %%
import os

os.environ["GEOMSTATS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import jax
import numpy as np
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.collections import LineCollection

# %%
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

from neural_diffusion_processes.data.storm import LONOFFSET, RADDEG, LONSTART, LONSTOP


def _get_key_iter(init_key) -> Iterator["jax.random.PRNGKey"]:
    while True:
        init_key, next_key = jax.random.split(init_key)
        yield next_key


log = logging.getLogger(__name__)
# %%
run_path = "/data/ziz/not-backed-up/mhutchin/score-sde-sp/results/storm_data/net.hidden_dim=128,optim.batch_size=256,optim.num_steps=500000/0"
cfg = OmegaConf.load(run_path + "/.hydra/config.yaml")
# %%
    # jax.config.update("jax_enable_x64", True)
policy = jmp.get_policy("params=float32,compute=float32,output=float32")

ckpt_path = os.path.join(run_path, cfg.paths.ckpt_dir)

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
data, transform = call(
    cfg.data,
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

# plot_batch = DataBatch(xs=data[0][:100], ys=data[1][:100])
plot_batch = next(
    ndp.data.dataloader(
        data,
        batch_size=100,
        key=next(key_iter),
        n_points=cfg.data.n_points,
        shuffle_xs=False,
    )
)
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
    return ndp.sde.conditional_sample2(
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
    log.info(f"Plotting step {t}")
    # TODO: refactor properly plot depending on dataset and move in utils/vis.py
    n_samples = 100
    keys = jax.random.split(key, 6)

    batch = plot_batch  # batch = next(iter(data_test))
    # idx = jax.random.randint(keys[0], (), minval=0, maxval=len(batch.xs))
    idx = 0

    x_grid = batch.xs[0]  # jnp.linspace(0, plot_batch.xs.max(), 100)[..., None]

    ts_fwd = [0.1, 0.2, 0.5, 0.8, float(sde.beta_schedule.t1)]
    ts_bwd = [0.8, 0.5, 0.2, 0.1, float(sde.beta_schedule.t0)]

    def plot_tracks(xs, ys, axes, title=""):
        m = Basemap(
            projection="mill",
            llcrnrlat=-80,
            urcrnrlat=80,
            llcrnrlon=LONSTART,
            urcrnrlon=LONSTOP,
            lat_ts=20,
            ax=axes,
        )
        # m = Basemap(projection='npstere', lon_0=0, boundinglat=-30)
        # m = Basemap(projection='ortho',lat_0=45,lon_0=-100,resolution='l')

        m.drawmapboundary(fill_color="#ffffff")
        m.fillcontinents(color="#cccccc", lake_color="#ffffff")
        m.drawcoastlines(color="#000000", linewidth=0.2)
        m.drawparallels(
            np.linspace(-60, 60, 7, endpoint=True, dtype=int),
            linewidth=0.1,
            # labels=[True, False, False, False],
        )
        m.drawmeridians(
            np.linspace(-160, 160, 9, endpoint=True, dtype=int),
            linewidth=0.1,
            # labels=[False, False, False, True],
        )

        ys = ys * transform[1] + transform[0]
        dists = jnp.linalg.norm((ys[:, :-1] - ys[:, 1:]), axis=-1)
        nan_index = dists > ((50 * RADDEG) ** 2)
        nan_index = jnp.repeat(
            jnp.concatenate([nan_index, nan_index[:, -2:-1]], axis=-1)[..., None],
            2,
            axis=-1,
        )

        lons = ((ys[..., 1] / RADDEG) + LONOFFSET) % 360  # remove lon offset
        lons = ((lons + LONSTART) % 360) - LONSTART  # Put into plot frame
        lats = ys[..., 0] / RADDEG

        m_coords = jnp.stack(
            m(
                lons,
                lats,
            )[::-1],
            axis=-1,
        )
        m_coords = m_coords.at[nan_index].set(jnp.nan)

        for row in m_coords:
            m.plot(row[..., 1], row[..., 0], linewidth=0.3, latlon=False)
        axes.set_title(title)

    nb_cols = len(ts_fwd) + 2
    fig_backward, axes = plt.subplots(
        1,
        nb_cols,
        figsize=(2 * 2 * nb_cols, 2 + 1),
        sharex=True,
        sharey=True,
    )
    fig_backward.subplots_adjust(wspace=0, hspace=0.0)

    # plot_vf_and_cov(batch.xs[idx], batch.ys, axes[:, 0], rf"$p_{{data}}$")
    plot_tracks(batch.xs, batch.ys, axes[-1], rf"$p_{{data}}$")

    # plot limiting
    y_ref = vmap(sde.sample_prior, in_axes=[0, None])(
        jax.random.split(keys[3], n_samples), x_grid
    ).squeeze()
    # plot_vf_and_cov(x_grid, y_ref, axes[:, 1], rf"$p_{{ref}}$")
    plot_tracks(x_grid, y_ref, axes[0], rf"$p_{{ref}}$")

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
    ).squeeze()

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

    # fig_comp, axes = plt.subplots(
    #     4,
    #     4,
    #     figsize=(2 * 2 * 4, 2 * 4 + 2),
    #     sharex=True,
    #     sharey=True,
    #     squeeze=False,
    # )

    # for i in range(4):
    #     plot_tracks(x_grid, y_model[i : (i + 1), -1], axes[0][i], "Model")
    # for i in range(4):
    #     plot_tracks(x_grid, batch.ys[i : (i + 1)], axes[1][i], "Data")
    # for i in range(4):
    #     plot_tracks(x_grid, y_model[(i + 4) : (i + 5), -1], axes[2][i], "Model")
    # for i in range(4):
    #     plot_tracks(x_grid, batch.ys[(i + 4) : (i + 5)], axes[3][i], "Data")

    # dict_plots["comparison"] = fig_comp

    fig_comp, axes = plt.subplots(
        2,
        1,
        figsize=(2 * 2 * 1, 2 * 2 + 1),
        sharex=True,
        sharey=True,
        squeeze=True,
    )

    n_samples = 100
    N = 10
    k = jax.random.split(keys[3], N)
    y_model = jnp.concatenate(
        [
            vmap(reverse_sample, in_axes=[0, None, None, 0])(
                jax.random.split(k[i], n_samples),
                x_grid,
                state.params_ema,
                vmap(sde.sample_prior, in_axes=[0, None])(
                    jax.random.split(k[i], n_samples), x_grid
                ).squeeze(),
                # ts_bwd,
            ).squeeze()
            for i in range(N)
        ],
        axis=0,
    )
    # x_model = jnp.repeat(x_grid[None, :], y_model.shape[0], axis=0)

    plot_tracks(data[0], data[1], axes[0], "Full data")
    plot_tracks(x_grid, y_model, axes[1], "Model samples")

    dict_plots["comparison"] = fig_comp

    if t == 0:  # NOTE: Only plot fwd at the beggining
        # ts = [0.8, sde.beta_schedule.t1]
        nb_cols = len(ts_fwd) + 2
        nb_rows = 4
        fig_forward, axes = plt.subplots(
            nb_rows,
            nb_cols,
            figsize=(2 * 2 * nb_cols, 2 * nb_rows + 1),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        fig_forward.subplots_adjust(wspace=0, hspace=0)

        for i in range(nb_rows):
            # plot data process
            # plot_vf_and_cov(batch.xs[0], batch.ys, axes[:, 0], rf"$p_{{data}}$")
            plot_tracks(
                batch.xs, batch.ys, axes[i, 0], rf"$p_{{data}}$" if i == 0 else None
            )

            # TODO: only solve once and return different timesaves
            for k, t in enumerate(ts_fwd):
                yt = vmap(
                    lambda key: sde.sample_marginal(
                        key, t * jnp.ones(()), batch.xs[idx + i], batch.ys[idx + i]
                    )
                )(jax.random.split(keys[1], n_samples)).squeeze()
                # plot_vf_and_cov(batch.xs[idx], yt, axes[:, k+1], rf"$p_{{t={t}}}$")
                plot_tracks(
                    batch.xs,
                    yt,
                    axes[i, k + 1],
                    rf"$p_{{t=}}$" if i == 0 else None,
                )

            # plot_vf_and_cov(x_grid, y_ref, axes[:, -1], rf"$p_{{ref}}$")
            plot_tracks(
                x_grid, y_ref, axes[i, -1], rf"$p_{{data}}$" if i == 0 else None
            )

        dict_plots["forward"] = fig_forward

        # fig_data, axes = plt.subplots(
        #     4,
        #     4,
        #     figsize=(4 * 2, 4 * 1),
        #     sharex=True,
        #     sharey=True,
        #     squeeze=False,
        # )
        # axes = [item for sublist in axes for item in sublist]
        # for i in range(len(axes)):
        #     plot_tracks(batch.xs[0:1], batch.ys[i : (i + 1)], axes[i])

        # dict_plots["data"] = fig_data

        fig_data, ax = plt.subplots(1, 1)
        plot_tracks(data[0], data[1], ax, "Full data")
        dict_plots["data"] = fig_data

    plt.close()
    return dict_plots

# %%
state = init(batch0, jax.random.PRNGKey(cfg.seed))
state = load_checkpoint(state, ckpt_path, cfg.optim.num_steps)
# %%
# plots(state, next(key_iter), "post_train")
# %%
def plot_tracks(xs, ys, axes, title="", color=None, lw=0.3):
    m = Basemap(
        projection="mill",
        llcrnrlat=-80,
        urcrnrlat=80,
        llcrnrlon=LONSTART,
        urcrnrlon=LONSTOP,
        lat_ts=20,
        ax=axes,
    )
    # m = Basemap(projection='npstere', lon_0=0, boundinglat=-30)
    # m = Basemap(projection='ortho',lat_0=45,lon_0=-100,resolution='l')

    m.drawmapboundary(fill_color="#ffffff")
    m.fillcontinents(color="#cccccc", lake_color="#ffffff")
    m.drawcoastlines(color="#000000", linewidth=0.2)
    m.drawparallels(
        np.linspace(-60, 60, 7, endpoint=True, dtype=int),
        linewidth=0.1,
        # labels=[True, False, False, False],
    )
    m.drawmeridians(
        np.linspace(-160, 160, 9, endpoint=True, dtype=int),
        linewidth=0.1,
        # labels=[False, False, False, True],
    )

    ys = ys * transform[1] + transform[0]
    dists = jnp.linalg.norm((ys[:, :-1] - ys[:, 1:]), axis=-1)
    nan_index = dists > ((50 * RADDEG) ** 2)
    nan_index = jnp.repeat(
        jnp.concatenate([nan_index, nan_index[:, -2:-1]], axis=-1)[..., None],
        2,
        axis=-1,
    )

    lons = ((ys[..., 1] / RADDEG) + LONOFFSET) % 360  # remove lon offset
    lons = ((lons + LONSTART) % 360) - LONSTART  # Put into plot frame
    lats = ys[..., 0] / RADDEG

    m_coords = jnp.stack(
        m(
            lons,
            lats,
        )[::-1],
        axis=-1,
    )
    m_coords = m_coords.at[nan_index].set(jnp.nan)

    # for row in m_coords:
    axes.plot(m_coords[..., 1].T, m_coords[..., 0].T, linewidth=lw, color=color)
    # lc = LineCollection(jnp.flip(m_coords, axis=-1), array=jnp.linspace(0,1, m_coords.shape[1])[None, :], linewidth=0.3, cmap='viridis')

    axes.set_title(title)

fig_comp, axes = plt.subplots(
    1,
    2,
    figsize=(2*4 * 2, 2* 1 * (2 + 0.5)),
    sharex=True,
    sharey=True,
    squeeze=True,
)


plot_tracks(x_grid, y_model_heun[:100], axes[0], "Model samples (heun)")
plot_tracks(x_grid, y_model[:100], axes[1], "Model samples (euler)")
# %%
import diffrax as dfx
@jit
def reverse_sample2(key, x_grid, params, yT=None):
    print("reverse_sample", x_grid.shape)
    net_ = partial(net, params)
    return ndp.sde.sde_solve(sde, net_, x_grid, key=key, y=yT, prob_flow=False, solver=dfx.Euler(), rtol=None, atol=None)


x_grid = plot_batch.xs[0]

n_samples = 100
N = 100
k = jax.random.split(next(key_iter), N)
y_model = jnp.concatenate(
    [
        vmap(reverse_sample2, in_axes=[0, None, None, 0])(
            jax.random.split(k[i], n_samples),
            x_grid,
            state.params_ema,
            vmap(sde.sample_prior, in_axes=[0, None])(
                jax.random.split(k[i], n_samples), x_grid
            ).squeeze(),
            # ts_bwd,
        ).squeeze()
        for i in tqdm.trange(N)
    ],
    axis=0,
)

# %%
y_model_heun = jnp.concatenate(
    [
        vmap(reverse_sample, in_axes=[0, None, None, 0])(
            jax.random.split(k[i], n_samples),
            x_grid,
            state.params_ema,
            vmap(sde.sample_prior, in_axes=[0, None])(
                jax.random.split(k[i], n_samples), x_grid
            ).squeeze(),
            # ts_bwd,
        ).squeeze()
        for i in range(N)
    ],
    axis=0,
)
# %%

fig_comp, axes = plt.subplots(
    1,
    2,
    figsize=(2*4 * 2, 2* 1 * (2 + 0.5)),
    sharex=True,
    sharey=True,
    squeeze=True,
)


plot_tracks(data[0], data[1], axes[0], "Full data")
plot_tracks(x_grid, y_model, axes[1], "Model samples")
# %%

fig_comp, axes = plt.subplots(
    1,
    2,
    figsize=(2*4 * 2, 2* 1 * (2 + 0.5)),
    sharex=True,
    sharey=True,
    squeeze=True,
)


plot_tracks(x_grid, y_model_heun[:100], axes[0], "Model samples (heun)")
plot_tracks(x_grid, y_model[:100], axes[1], "Model samples (euler)")
# %%

def plot_tracks2(xs, ys, axes, title="", lon0=-105, lat0=40):
    m = Basemap(projection='ortho',lon_0=lon0,lat_0=lat0,resolution='l', ax=axes)

    # m = Basemap(projection='npstere', lon_0=0, boundinglat=-30)
    # m = Basemap(projection='ortho',lat_0=45,lon_0=-100,resolution='l')

    m.drawmapboundary(fill_color="#ffffff")
    m.fillcontinents(color="#cccccc", lake_color="#ffffff")
    m.drawcoastlines(color="#000000", linewidth=0.2)
    m.drawparallels(
        np.linspace(-60, 60, 7, endpoint=True, dtype=int),
        linewidth=0.1,
        # labels=[True, False, False, False],
    )
    m.drawmeridians(
        np.linspace(-160, 160, 9, endpoint=True, dtype=int),
        linewidth=0.1,
        # labels=[False, False, False, True],
    )

    ys = ys * transform[1] + transform[0]
    dists = jnp.linalg.norm((ys[:, :-1] - ys[:, 1:]), axis=-1)
    nan_index = dists > ((50 * RADDEG) ** 2)
    nan_index = jnp.repeat(
        jnp.concatenate([nan_index, nan_index[:, -2:-1]], axis=-1)[..., None],
        2,
        axis=-1,
    )

    lons = ((ys[..., 1] / RADDEG) + LONOFFSET) % 360  # remove lon offset
    lons = ((lons + LONSTART) % 360) - LONSTART  # Put into plot frame
    lats = ys[..., 0] / RADDEG

    m_coords = jnp.stack(
        m(
            lons,
            lats,
        )[::-1],
        axis=-1,
    )
    m_coords = m_coords.at[nan_index].set(jnp.nan)

    for row in m_coords:
        m.plot(row[..., 1], row[..., 0], linewidth=0.3, latlon=False)
    axes.set_title(title)

x_grid = plot_batch.xs[0]

fig_comp, axes = plt.subplots(
    1,
    4,
    figsize=(4 * 2 * 1, 4 * 2 * (1 + 0.25)),
    sharex=True,
    sharey=True,
    squeeze=True,
)

lat0=0
lon1=-90
lon2=140
plot_tracks2(data[0], data[1], axes[0], "Full data", lon0=lon1, lat0=lat0)
plot_tracks2(data[0], data[1], axes[1], "Full data", lon0=lon2, lat0=lat0)
plot_tracks2(x_grid, y_model, axes[2], "Model samples", lon0=lon1, lat0=lat0)
plot_tracks2(x_grid, y_model, axes[3], "Model samples", lon0=lon2, lat0=lat0)

# %%

n_cond = 25

x_grid = data[0][:10]
x_context = data[0][:10, :n_cond]
y_context = data[1][:10, :n_cond]

@jit
def cond_sample(key, x_grid, x_context, y_context, params):
    net_ = partial(net, params)
    x_context += 1.0e-5  # NOTE: to avoid context overlapping with grid
    # return ndp.sde.conditional_sample2(sde, net_, x_context, y_context, x_grid, key=key, num_inner_steps=50)
    return ndp.sde.conditional_sample2(
        sde,
        net_,
        x_context,
        y_context,
        x_grid,
        key=key,
        num_steps=1000,
        num_inner_steps=100,
        tau=1.0,
        psi=1.0,
        lambda0=1.0,
        prob_flow=False,
    )

samples = jax.vmap(cond_sample, in_axes=[0,0,0,0,None])(jax.random.split(next(key_iter), len(x_grid)), x_grid, x_context, y_context, state.params_ema)

fig_comp, axes = plt.subplots(
    1,
    1,
    figsize=(4 * 2 * 1, 4 * 2 * (1 + 0.25)),
    sharex=True,
    sharey=True,
    squeeze=True,
)


plot_tracks(x_context, y_context, axes, "Full data", color='tab:blue', lw=1.0)
plot_tracks(x_grid[:, n_cond:], samples[:, n_cond:], axes, "Full data", color='tab:orange', lw=1.0)
# %%
