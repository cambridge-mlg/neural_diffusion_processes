# %%
%load_ext autoreload
%autoreload 2

# %%
import os
import setGPU
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
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate, call
from jax.config import config as jax_config

import neural_diffusion_processes as ndp
from neural_diffusion_processes import ml_tools
from neural_diffusion_processes.utils.ml_tools.state import (
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
# run_path = "/data/ziz/not-backed-up/mhutchin/score-sde-sp/results/storm_data/net.hidden_dim=128,optim.batch_size=256,optim.num_steps=500000/0"
# run_path = "/data/ziz/not-backed-up/mhutchin/score-sde-sp/results/storm_data/data.normalise=true,kernel.params.variance=2.0,net.hidden_dim=128,optim.batch_size=256,optim.num_steps=500000/0"
run_path = "/data/ziz/not-backed-up/mhutchin/score-sde-sp/results/storm_data/data.basin=na,data.normalise=True,optim.num_steps=500000/0"
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
def plot_tracks(xs, ys, axes, title="", lw=0.3, **kwargs):
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
    axes.plot(m_coords[..., 1].T, m_coords[..., 0].T, lw=lw, **kwargs)
    # lc = LineCollection(jnp.flip(m_coords, axis=-1), array=jnp.linspace(0,1, m_coords.shape[1])[None, :], linewidth=0.3, cmap='viridis')

    axes.set_title(title)

def plot_tracks_latlon(xs, ys, axes, title="", lw=0.3, **kwargs):
    # m = Basemap(
    #     projection="mill",
    #     llcrnrlat=-80,
    #     urcrnrlat=80,
    #     llcrnrlon=LONSTART,
    #     urcrnrlon=LONSTOP,
    #     lat_ts=20,
    #     ax=axes,
    # )
    # m = Basemap(projection='npstere', lon_0=0, boundinglat=-30)
    # m = Basemap(projection='ortho',lat_0=45,lon_0=-100,resolution='l')

    # m.drawmapboundary(fill_color="#ffffff")
    # m.fillcontinents(color="#cccccc", lake_color="#ffffff")
    # m.drawcoastlines(color="#000000", linewidth=0.2)
    # m.drawparallels(
    #     np.linspace(-60, 60, 7, endpoint=True, dtype=int),
    #     linewidth=0.1,
    #     # labels=[True, False, False, False],
    # )
    # m.drawmeridians(
    #     np.linspace(-160, 160, 9, endpoint=True, dtype=int),
    #     linewidth=0.1,
    #     # labels=[False, False, False, True],
    # )

    # ys = ys * transform[1] + transform[0]
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

    # m_coords = jnp.stack(
    #     m(
    #         lons,
    #         lats,
    #     )[::-1],
    #     axis=-1,
    # )
    # m_coords = m_coords.at[nan_index].set(jnp.nan)

    # for row in m_coords:
    # axes.plot(m_coords[..., 1].T, m_coords[..., 0].T, lw=lw, **kwargs)
    axes[0].plot(xs[..., 0].T, ys[..., 0].T, lw=lw, **kwargs)
    axes[1].plot(xs[..., 0].T, ys[..., 1].T, lw=lw, **kwargs)
    # lc = LineCollection(jnp.flip(m_coords, axis=-1), array=jnp.linspace(0,1, m_coords.shape[1])[None, :], linewidth=0.3, cmap='viridis')

    axes[0].set_title("lat")
    axes[1].set_title("lon")

    # axes[0].set_ylim([-jnp.pi/2, jnp.pi/2])
    # axes[1].set_ylim([-jnp.pi, jnp.pi])


fig_comp, axes = plt.subplots(
    1,
    2,
    figsize=(2*4 * 2, 2* 1 * (2 + 0.5)),
    sharex=True,
    sharey=True,
    squeeze=True,
)


# plot_tracks(x_grid, y_model_heun[:100], axes[0], "Model samples (heun)")
# plot_tracks(x_grid, y_model[:100], axes[1], "Model samples (euler)")
# %%
import diffrax as dfx

@jit
def reverse_sample_euler(key, x_grid, params, yT=None, n_steps=1000):
    print("reverse_sample", x_grid.shape)
    net_ = partial(net, params)
    return ndp.sde.sde_solve(sde, net_, x_grid, key=key, y=yT, prob_flow=False, solver=dfx.Euler(), rtol=None, atol=None, num_steps=n_steps)

@jit
def reverse_sample_heun(key, x_grid, params, yT=None, n_steps=1000):
    print("reverse_sample", x_grid.shape)
    net_ = partial(net, params)
    return ndp.sde.sde_solve(sde, net_, x_grid, key=key, y=yT, prob_flow=False)

@jit
def reverse_ode_euler(key, x_grid, params, yT=None, n_steps=1000):
    print("reverse_sample", x_grid.shape)
    net_ = partial(net, params)
    return ndp.sde.sde_solve(sde, net_, x_grid, key=key, y=yT, prob_flow=True, solver=dfx.Euler(), rtol=None, atol=None, num_steps=n_steps)

@jit
def reverse_ode_heun(key, x_grid, params, yT=None, n_steps=1000):
    print("reverse_sample", x_grid.shape)
    net_ = partial(net, params)
    return ndp.sde.sde_solve(sde, net_, x_grid, key=key, y=yT, prob_flow=True)

x_grid = plot_batch.xs[0]


def batched_sample(key, x_grid, sample_fn, prior_sample_fn, n_batches, n_samples):
    k = jax.random.split(key, n_batches)
    return jnp.concatenate(
        [
            vmap(sample_fn, in_axes=[0, None, None, 0])(
                jax.random.split(k[i], n_samples),
                x_grid,
                state.params_ema,
                vmap(prior_sample_fn, in_axes=[0, None])(
                    jax.random.split(k[i], n_samples), x_grid
                ).squeeze(),
                # ts_bwd,
            ).squeeze()
            for i in tqdm.trange(n_batches)
        ],
        axis=0,
    )

y_euler = batched_sample(
    next(key_iter),
    x_grid,
    reverse_sample_euler,
    sde.sample_prior,
    1,
    100
)

y_heun = batched_sample(
    next(key_iter),
    x_grid,
    reverse_sample_heun,
    sde.sample_prior,
    1,
    100
)

y_euler_ode = batched_sample(
    next(key_iter),
    x_grid,
    reverse_ode_euler,
    sde.sample_prior,
    1,
    100
)

y_heun_ode = batched_sample(
    next(key_iter),
    x_grid,
    reverse_ode_heun,
    sde.sample_prior,
    1,
    100
)

fig, axes = plt.subplots(
    2,2,
    figsize=(2*8, 2*4), sharex=True, sharey=True,
)

plot_tracks(None, y_euler, axes[0][0], "Euler SDE")
plot_tracks(None, y_heun, axes[0][1], "Heun SDE")
plot_tracks(None, y_euler_ode, axes[1][0], "Euler ODE")
plot_tracks(None, y_heun_ode, axes[1][1], "Heun ODE")
# %%
from neural_diffusion_processes.sde import SDE, ScoreNetwork, LinOpControlTerm, Array, LinearOperator, identity

def langevin_correct(
    sde: SDE,
    network: ScoreNetwork,
    t,
    x_context,
    y_context,
    x_test,
    y_test,
    key,
    num_steps: int = 100,
    num_inner_steps: int = 5,
    prob_flow: bool = True,
    langevin_kernel=True,
    psi: float = 1.0,
    lambda0: float = 1.0,
    tau: float = None,
):
    # TODO: Langevin dynamics option

    num_context = len(x_context)
    num_target = len(x_test)
    y_dim = y_context.shape[-1]
    shape_augmented_state = [(num_context + num_target) * y_dim]
    x_augmented = jnp.concatenate([x_context, x_test], axis=0)

    t0 = sde.beta_schedule.t0
    t1 = sde.beta_schedule.t1
    ts = jnp.linspace(t1, t0, num_steps, endpoint=True)
    dt = ts[0] - ts[1]
    tau = tau if tau is not None else t1

    solver = dfx.Euler()

    diffusion = lambda t, yt, arg: sde.diffusion(t, unflatten(yt, y_dim), arg)
    if not prob_flow:
        # reverse SDE:
        reverse_drift_sde = lambda t, yt, arg: flatten(
            sde.reverse_drift_sde(key, t, unflatten(yt, y_dim), arg, network)
        )

        shape = jax.ShapeDtypeStruct(shape_augmented_state, y_context.dtype)
        key, subkey = jax.random.split(key)
        bm = dfx.VirtualBrownianTree(t0=t1, t1=t0, tol=dt, shape=shape, key=key)
        terms_reverse = dfx.MultiTerm(
            dfx.ODETerm(reverse_drift_sde), LinOpControlTerm(diffusion, bm)
        )
    else:
        # reverse ODE:
        reverse_drift_ode = lambda t, yt, arg: flatten(
            sde.reverse_drift_ode(key, t, unflatten(yt, y_dim), arg, network)
        )
        terms_reverse = dfx.ODETerm(reverse_drift_ode)

    # langevin dynamics:
    def reverse_drift_langevin(t, yt, x) -> Array:
        yt = unflatten(yt, y_dim)
        score = flatten(sde.score(key, t, yt, x, network))
        if langevin_kernel:
            if sde.is_score_preconditioned:
                score = score
            else:
                score = sde.limiting_gram(x) @ score
        else:
            if sde.is_score_preconditioned:
                score = sde.limiting_gram(x).solve(score)
            else:
                score = score
        return 0.5 * sde.beta_schedule(t) * score

    def diffusion_langevin(t, yt, x) -> LinearOperator:
        if langevin_kernel:
            return diffusion(t, yt, x)
        else:
            return jnp.sqrt(sde.beta_schedule(t)) * identity(yt.shape[-1])

    key, subkey = jax.random.split(key)
    shape = jax.ShapeDtypeStruct(shape_augmented_state, y_context.dtype)
    # bm = dfx.VirtualBrownianTree(t0=t0, t1=t1, tol=dt, shape=shape, key=subkey)
    # bm = dfx.UnsafeBrownianPath(shape=shape, key=subkey)
    # langevin_terms = dfx.MultiTerm(
    #     dfx.ODETerm(reverse_drift_langevin), LinOpControlTerm(diffusion_langevin, bm)
    # )

    def sample_marginal(key, t, x_context, y_context):
        if len(y_context) == 0:
            return y_context
        else:
            return flatten(sde.sample_marginal(key, t, x_context, y_context))

    def inner_loop(key, ys, t):
        # reverse step
        yt, yt_context = ys
        yt_context = sample_marginal(
            key, t, x_context, y_context
        )  # NOTE: should resample?
        yt_augmented = jnp.concatenate([yt_context, yt], axis=0)

        # yt_m_dt, *_ = solver.step(
        #     langevin_terms,
        #     t - dt,
        #     t,
        #     # t + dt,
        #     yt_augmented,
        #     x_augmented,
        #     None,
        #     made_jump=False,
        # )

        yt_m_dt = yt_augmented
        yt_m_dt += (
            lambda0
            * psi
            * dt
            * reverse_drift_langevin(t, yt_augmented, x_augmented)
        )
        noise = (
            jnp.sqrt(psi)
            * jnp.sqrt(dt)
            * jax.random.normal(key, shape=yt_augmented.shape)
        )
        yt_m_dt += diffusion_langevin(t, yt_augmented, x_augmented) @ noise
        # yt_m_dt += langevin_terms.contr(t, t)[0] * langevin_terms.vf(t, yt_augmented, x_augmented)[0]
        # yt_m_dt += langevin_terms.vf(t, yt_augmented, x_augmented)[1] @ noise

        yt = yt_m_dt[num_context * y_dim :]
        # strip context from augmented state
        return (yt, yt_context), yt_m_dt

    # def outer_loop(key, yt, t):
    #     # jax.debug.print("time {t}", t=t)

    #     # yt_context = sde.sample_marginal(key, t, x_context, y_context)
    #     yt_context = sample_marginal(key, t, x_context, y_context)
    #     # yt_context = y_context #NOTE: doesn't need to be noised?
    #     yt_augmented = jnp.concatenate([yt_context, yt], axis=0)

    #     yt_m_dt, *_ = solver.step(
    #         terms_reverse,
    #         t,
    #         t - dt,
    #         yt_augmented,
    #         x_augmented,
    #         None,
    #         made_jump=False,
    #     )
    #     # yt = yt_m_dt[num_context * y_dim :]
    #     # yt_m_dt = yt_augmented
    #     # yt_m_dt += -dt * reverse_drift_diffeq(t, yt_augmented, x_augmented)
    #     # # yt_m_dt += terms_reverse.contr(t, t-dt) * terms_reverse.vf(t, yt_augmented, x_augmented)
    #     # noise = jax.random.normal(key, shape=yt_augmented.shape)
    #     # yt_m_dt += jnp.sqrt(dt) * sde.diffusion(t, yt_augmented, x_augmented) @ noise

    #     def corrector(key, yt, yt_context, t):
    #         _, yt_m_dt = jax.lax.scan(
    #             lambda ys, key: inner_loop(key, ys, t),
    #             (yt, yt_context),
    #             jax.random.split(key, num_inner_steps),
    #         )
    #         yt = yt_m_dt[-1][num_context * y_dim :]
    #         return yt

    #     yt = jax.lax.cond(
    #         tau > t,
    #         corrector,
    #         lambda key, yt, yt_context, t: yt,
    #         key,
    #         yt,
    #         yt_context,
    #         t,
    #     )
    #     return yt, yt


    def corrector(key, yt, yt_context, t):
        _, yt_m_dt = jax.lax.scan(
            lambda ys, key: inner_loop(key, ys, t),
            (yt, yt_context),
            jax.random.split(key, num_inner_steps),
        )

        

        yt = yt_m_dt[-1][num_context * y_dim :]
        return yt
    
    y0 = corrector(key, flatten(y_test), flatten(y_context), t)

    # key, subkey = jax.random.split(key)
    # yT = flatten(sde.sample_prior(subkey, x_test))

    # xs = (ts[:-1], jax.random.split(key, len(ts) - 1))
    # y0, _ = jax.lax.scan(lambda yt, x: outer_loop(x[1], yt, x[0]), yT, xs)
    return unflatten(y0, y_dim)



# %%

n_cond = 25
n_samples = 5
n_tracks = 5
off=15

x_grid = data[0][:n_tracks]
x_context = jnp.concatenate([data[0][(off):(n_tracks+off), :n_cond//2], data[0][(off):(n_tracks+off), (50 - n_cond//2):]], axis=1)
y_context = jnp.concatenate([data[1][(off):(n_tracks+off), :n_cond//2], data[1][(off):(n_tracks+off), (50 - n_cond//2):]], axis=1)
x_missing = data[0][(off):(n_tracks+off), n_cond//2:(50-n_cond//2)]
y_missing = data[1][(off):(n_tracks+off), n_cond//2:(50-n_cond//2)]

@jit
def cond_sample(key, x_grid, x_context, y_context, params):
    net_ = partial(net, params)
    # x_context += 1.0e-5  # NOTE: to avoid context overlapping with grid
    # return conditional_sample3(
    return ndp.sde.conditional_sample2(
        sde, 
        net_, 
        x_context, 
        y_context, 
        x_grid, 
        key=key, 
        num_steps=1000,
        num_inner_steps=100,
        tau=0.5,
        psi=0.1,
        lambda0=1.5,
        prob_flow=False,
    )


# def langevin_correct(
#     sde: SDE,
#     network: ScoreNetwork,
#     t,
#     x_context,
#     y_context,
#     x_test,
#     y_test,
#     key,
#     num_steps: int = 100,
#     num_inner_steps: int = 5,
#     prob_flow: bool = True,
#     langevin_kernel=True,
#     psi: float = 1.0,
#     lambda0: float = 1.0,
#     tau: float = None,
# ):
@jit
def langevin_sample(key, x_grid, y_samples, x_context, y_context, params):
    net_ = partial(net, params)

    return langevin_correct(
        sde, 
        partial(net, state.params_ema),
        sde.beta_schedule.t0,
        x_context, 
        y_context, 
        x_grid,
        y_samples,
        key, 
        num_steps=1000,
        num_inner_steps=1000,
        tau=0.5,
        psi=0.1,
        lambda0=1.5,
        prob_flow=False,
    )
# %%

interp_samples = jax.vmap(
    jax.vmap(
        cond_sample, 
        in_axes=[0,0,0,0,None]
    ), 
    in_axes=[0,None,None,None,None]
)(
    jax.random.split(next(key_iter), len(x_grid)*n_samples).reshape((n_samples, n_tracks, 2)), 
    x_grid, 
    x_context, 
    y_context, 
    state.params_ema
)
# %%
langevin_samples = jax.vmap(
    jax.vmap(
        langevin_sample,
        in_axes=(0,0,0,0,0,None)
    ),
    in_axes=(0, None, 0, None, None, None),
)(
    jax.random.split(next(key_iter), len(x_grid)*n_samples).reshape((n_samples, n_tracks, 2)), 
    x_grid,
    interp_samples,
    x_context, 
    y_context, 
    state.params_ema
)

# %%

fig_comp, axes = plt.subplots(
    1,
    2,
    figsize=(4 * 2 * 2, 4 * 2 * (1 + 0.25)),
    sharex=True,
    sharey=True,
    squeeze=True,
)


plot_tracks(x_context[:,:(n_cond//2)], y_context[:,:(n_cond//2)], axes[0], "Full data", color='tab:blue', lw=1.0)
plot_tracks(x_context[:,(n_cond//2):], y_context[:,(n_cond//2):], axes[0], "Full data", color='tab:blue', lw=1.0)
plot_tracks(x_missing, y_missing, axes[0], "Full data", color='tab:green', lw=1.0)

# for sample in interp_samples:
for sample in langevin_samples:
    plot_tracks(x_grid[:, (n_cond//2):(50 - n_cond//2)], sample[:, (n_cond//2):(50 - n_cond//2)], axes[0], "Full data", color='tab:orange', lw=1.0, alpha=0.2)


plot_tracks(x_context[:,:(n_cond//2)], y_context[:,:(n_cond//2)], axes[1], "Full data", color='tab:blue', lw=1.0)
plot_tracks(x_context[:,(n_cond//2):], y_context[:,(n_cond//2):], axes[1], "Full data", color='tab:blue', lw=1.0)
plot_tracks(x_missing, y_missing, axes[1], "Full data", color='tab:green', lw=1.0)

# for sample in interp_samples:
for sample in langevin_samples:
    plot_tracks(x_grid[:, (n_cond//2):(50 - n_cond//2)], sample[:, (n_cond//2):(50 - n_cond//2)], axes[1], "Full data", color='tab:orange', lw=1.0, alpha=0.2)
# %%

fig_comp, axes = plt.subplots(
    2,
    1,
    figsize=(4 * 2 * 1, 2 * 2 * (1 + 0.5)),
    sharex=True,
    sharey=False,
    squeeze=True,
)

plot_tracks_latlon(x_context[:,:(n_cond//2)], y_context[:,:(n_cond//2)], axes, "Full data", color='tab:blue', lw=1.0)
plot_tracks_latlon(x_context[:,(n_cond//2):], y_context[:,(n_cond//2):], axes, "Full data", color='tab:blue', lw=1.0)
plot_tracks_latlon(x_missing, y_missing, axes, "Full data", color='tab:green', lw=1.0)

for sample in interp_samples:
    plot_tracks_latlon(x_grid[:, (n_cond//2):(50 - n_cond//2)], sample[:, (n_cond//2):(50 - n_cond//2)], axes, "Full data", color='tab:orange', lw=1.0, alpha=0.2)
# %%
fig_comp, axes = plt.subplots(
    2,
    5,
    figsize=(4 * 2 * 5, 2 * 2 * (1 + 0.5)),
    sharex=True,
    sharey=False,
    squeeze=True,
)

for idx, i in enumerate([0,100,500,800,999]):
    plot_tracks_latlon(x_context[:,:(n_cond//2)], y_context[:,:(n_cond//2)], [axes[0][idx], axes[1][idx]], "Full data", color='tab:blue', lw=1.0)
    plot_tracks_latlon(x_context[:,(n_cond//2):], y_context[:,(n_cond//2):], [axes[0][idx], axes[1][idx]], "Full data", color='tab:blue', lw=1.0)
    plot_tracks_latlon(x_missing, y_missing, [axes[0][idx], axes[1][idx]], "Full data", color='tab:green', lw=1.0)
    for sample in interp_tracks[:, :, i, :]:
        plot_tracks_latlon(x_grid, sample, [axes[0][idx], axes[1][idx]], "Full data", color='tab:orange', lw=1.0, alpha=0.2)

# %%
n_cond = 25
n_samples = 20
n_tracks = 2
off=17

x_grid = data[0][:n_tracks]
x_context = data[0][(off):(n_tracks+off), :n_cond]
y_context = data[1][(off):(n_tracks+off), :n_cond]
x_missing = data[0][(off):(n_tracks+off), n_cond:]
y_missing = data[1][(off):(n_tracks+off), n_cond:]

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
        num_inner_steps=10,
        tau=0.5,
        psi=1.0,
        lambda0=1.0,
        prob_flow=False,
    )

extrap_samples = jax.vmap(jax.vmap(cond_sample, in_axes=[0,0,0,0,None]), in_axes=[0,None,None,None,None])(jax.random.split(next(key_iter), len(x_grid)*n_samples).reshape((n_samples, n_tracks, 2)), x_grid, x_context, y_context, state.params_ema)

# %%

fig_comp, axes = plt.subplots(
    1,
    1,
    figsize=(4 * 2 * 1, 4 * 2 * (1 + 0.25)),
    sharex=True,
    sharey=True,
    squeeze=True,
)


plot_tracks(x_context, y_context, axes, "Full data", color='tab:blue', lw=1.0)
plot_tracks(x_missing, y_missing, axes, "Full data", color='tab:green', lw=1.0)

for sample in extrap_samples:
    plot_tracks(x_grid[:, n_cond:], sample[:, n_cond:], axes, "Full data", color='tab:orange', lw=1.0, alpha=0.2)
# %%

fig_comp, axes = plt.subplots(
    2,
    1,
    figsize=(4 * 2 * 1, 2 * 2 * (1 + 0.5)),
    sharex=True,
    sharey=False,
    squeeze=True,
)

plot_tracks_latlon(x_context, y_context, axes, "Full data", color='tab:blue', lw=1.0)
plot_tracks_latlon(x_missing, y_missing, axes, "Full data", color='tab:green', lw=1.0)

for sample in extrap_samples:
    plot_tracks_latlon(x_grid[:, n_cond:], sample[:, n_cond:], axes, "Full data", color='tab:orange', lw=1.0, alpha=0.2)
#

# %%
