# %%
%load_ext autoreload
%autoreload 2
# %%
import neural_diffusion_processes as ndp

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
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600
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
import diffrax as dfx

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate, call, get_method
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
from neural_diffusion_processes.data import shuffle_data, split_data, DataBatch

from neural_diffusion_processes.data.storm import LONOFFSET, RADDEG, LONSTART, LONSTOP
from neural_diffusion_processes.brownian_sde import SphericalGRW


def _get_key_iter(init_key) -> Iterator["jax.random.PRNGKey"]:
    while True:
        init_key, next_key = jax.random.split(init_key)
        yield next_key


log = logging.getLogger(__name__)
# %%
# run_path = "/data/ziz/not-backed-up/mhutchin/score-sde-sp/results/storm_data/net.hidden_dim=128,optim.batch_size=256,optim.num_steps=500000/0"
# run_path = "/data/ziz/not-backed-up/mhutchin/score-sde-sp/results/storm_data/data.normalise=true,kernel.params.variance=2.0,net.hidden_dim=128,optim.batch_size=256,optim.num_steps=500000/0"
# run_path = "/data/ziz/not-backed-up/mhutchin/score-sde-sp/results/storm_data/data.basin=na,data.normalise=True,optim.num_steps=500000/0"
# run_path = "/data/ziz/not-backed-up/mhutchin/score-sde-sp/results/storm_data/beta_schedule.beta1=5.0,optim.batch_size=256,sde=sphere,solver=spherical_euler,transform=3d/0"
run_path = "/data/ziz/not-backed-up/mhutchin/score-sde-sp/results/storm_data/beta_schedule.beta1=10.0,optim.lr=0.0003,optim.num_steps=250000,sde=sphere,solver=spherical_euler,transform=3d/0"
cfg = OmegaConf.load(run_path + "/.hydra/config.yaml")
# %%
policy = jmp.get_policy("params=float32,compute=float32,output=float32")

log.info("Stage : Startup")
log.info(f"Jax devices: {jax.devices()}")
log.info(f"run_path: {run_path}")
log.info(f"hostname: {socket.gethostname()}")
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
data, data_normalisation = call(
    cfg.data,
)
transform = get_method(cfg.transform._target_)

def model_to_latlon_coords(ys):
    ys = transform(ys, reverse=True)
    ys = ys * data_normalisation[1] + data_normalisation[0]
    return ys

data = (
    data[0],
    transform(data[1]),
)
data = shuffle_data(cfg.data.seed, data)
train_data, validation_data, test_data = split_data(
    data, cfg.data.split_proportions
)

train_dataloader = ndp.data.dataloader(
    train_data,
    batch_size=cfg.optim.batch_size,
    key=next(key_iter),
    n_points=cfg.data.n_points,
)
test_dataloader = ndp.data.dataloader(
    train_data,
    batch_size=cfg.optim.batch_size,
    key=next(key_iter),
    n_points=cfg.data.n_points,
    run_forever=False,
)
batch0 = next(train_dataloader)
x_dim = batch0.xs.shape[-1]
y_dim = batch0.ys.shape[-1]
log.info(f"num elements: {batch0.xs.shape[-2]} & x_dim: {x_dim} & y_dim: {y_dim}")

solver = instantiate(cfg.solver.solver)

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

# ########## Plotting
@jit
def reverse_sample(key, x_grid, params, yT=None):
    print("reverse_sample", x_grid.shape)
    net_ = partial(net, params)
    return ndp.sde.sde_solve(
        sde,
        net_,
        x_grid,
        key=key,
        y=yT,
        prob_flow=False,
        solver=solver,
        **cfg.solver.kwargs,
    )

@filter_jit
def reverse_sample_times(key, x_grid, params, yT=None, ts=None):
    print("reverse_sample", x_grid.shape)
    net_ = partial(net, params)
    return ndp.sde.sde_solve(
        sde,
        net_,
        x_grid,
        key=key,
        y=yT,
        prob_flow=False,
        ts=ts,
        solver=solver,
        **cfg.solver.kwargs,
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

ts_fwd = [0.1, 0.2, 0.5, 0.8, float(sde.beta_schedule.t1)]
ts_bwd = [0.8, 0.5, 0.2, 0.1, float(sde.beta_schedule.t0)]

def plot_tracks(xs, ys, axes, title="", lw=0.3, boundary_lw=1.0, latmin=-80, latmax=80, lonmin=LONSTART, lonmax=LONSTOP, fix_aspect=True, **kwargs):
    m = Basemap(
        projection="mill",
        llcrnrlat=latmin,
        urcrnrlat=latmax,
        llcrnrlon=lonmin,
        urcrnrlon=lonmax,
        # lat_ts=latmin,
        ax=axes,
        fix_aspect=fix_aspect,
        anchor='C'
    )

    m.drawmapboundary(fill_color="#ffffff", linewidth=boundary_lw)
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

    ys = model_to_latlon_coords(ys)

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

    axes.plot(m_coords[..., 1].T, m_coords[..., 0].T, lw=lw, **kwargs)
    axes.set_title(title, )



state = init(batch0, jax.random.PRNGKey(cfg.seed))
state = load_checkpoint(state, ckpt_path, cfg.optim.num_steps)
# %%

def plot_tracks_latlon(xs, ys, axes, title="", lw=0.3, **kwargs):
    ys = model_to_latlon_coords(ys)

    axes[0].plot(xs[..., 0].T, ys[..., 0].T, lw=lw, **kwargs)
    axes[1].plot(xs[..., 0].T, ys[..., 1].T, lw=lw, **kwargs)

    axes[0].set_title("lat")
    axes[1].set_title("lon")


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

cfg.solver.kwargs.num_steps=1000

@jit
def reverse_sample_sde(key, x_grid, params, yT=None):
    print("reverse_sample", x_grid.shape)
    net_ = partial(net, params)
    return ndp.sde.sde_solve(sde, net_, x_grid, key=key, y=yT, prob_flow=False, solver=solver,**cfg.solver.kwargs,)

@jit
def reverse_sample_ode(key, x_grid, params, yT=None):
    print("reverse_sample", x_grid.shape)
    net_ = partial(net, params)
    return ndp.sde.sde_solve(sde, net_, x_grid, key=key, y=yT, prob_flow=True, solver=solver,**cfg.solver.kwargs,)


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

# y_sde= batched_sample(
#     next(key_iter),
#     x_grid,
#     reverse_sample_sde,
#     sde.sample_prior,
#     1,
#     100
# )

y_ode = batched_sample(
    next(key_iter),
    x_grid,
    reverse_sample_ode,
    sde.sample_prior,
    1,
    1000
)
# %%


import matplotlib
from matplotlib import font_manager
fonts = font_manager.findSystemFonts("/data/ziz/not-backed-up/mhutchin/score-sde-sp/scripts/fonts")
for font in fonts:
    font_manager.fontManager.addfont(font)

font = {'family' : 'serif',
        'serif'  : ['Times New Roman'],
        # 'weight' : 'bold',
        'size'   : 9
        }

mathtext = {"rm"  : "serif",
            "it"  : "serif:italic",
            "bf"  : "serif:bold",
            "fontset": "custom",
}

matplotlib.rc('font', **font)
# mpl.rc('mathtext', **mathtext)
matplotlib.rc('text', usetex='false')
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amsfonts}')




tw = 5.5
fw = tw

thin_lw = 0.4 * (fw/5.5) / 2
thick_lw = 0.9 * (fw/5.5) / 2

fig, axes = plt.subplots(
    1,2,
    figsize=(fw, fw/3)
)

# plt.rc('axes', titlesize=10.95)     # fontsize of the axes title
# plt.rc('axes', labelsize=10.95)    # fontsize of the x and y labels

# plot_tracks(None, y_sde, axes[0], "SDE")
# plot_tracks(None, train_data[1][:100], axes[0], "Data", lw=thin_lw, boundary_lw=thin_lw)
# plot_tracks(None, y_ode[:100], axes[1], "Model Samples", lw=thin_lw, boundary_lw=thin_lw)
# plot_tracks(None, y_euler_ode, axes[1][0], "Euler ODE")
# plot_tracks(None, y_heun_ode, axes[1][1], "Heun ODE")
fig.subplots_adjust(top=1, bottom=0, left=0,right=1)
# fig.tight_layout()
fig.savefig("data_vs_model.pdf")

# %%

def split_interp(data, context_points):
    x_context = jnp.concatenate([data.xs[:, :context_points//2], data.xs[:, (data.xs.shape[1] - n_cond//2):]], axis=1)
    y_context = jnp.concatenate([data.ys[:, :context_points//2], data.ys[:, (data.ys.shape[1] - n_cond//2):]], axis=1)
    x_missing = data.xs[:, n_cond//2:(data.xs.shape[1]-n_cond//2)]
    y_missing = data.ys[:, n_cond//2:(data.ys.shape[1]-n_cond//2)]

    return DataBatch(x_missing, y_missing, x_context, y_context)

def split_extrap(data, context_points):
    x_context = data.xs[:, :context_points]
    y_context = data.ys[:, :context_points]
    x_missing = data.xs[:, context_points:]
    y_missing = data.ys[:, context_points:]

    return DataBatch(x_missing, y_missing, x_context, y_context)

import dataclasses
from typing import Tuple, Iterator, Optional, Mapping, Union
from jaxtyping import Array
from simple_pytree import Pytree

@dataclasses.dataclass
class CondSampleBatch(Pytree):
    xs: Array
    ys: Array
    xc: Array
    yc: Array
    y_samples: Array

    def __len__(self) -> int:
        return len(self.xs)

    @property
    def num_points(self) -> int:
        return self.xs.shape[1]


# %%

def get_batched_cond_sampler(sample_fn, **kwargs):
    @jit
    def cond_sample(key, x_context, y_context, x_missing, params):
        net_ = partial(net, params)
        # x_context += 1.0e-5  # NOTE: to avoid context overlapping with grid
        # return conditional_sample3(
        return sample_fn(
            sde, 
            net_, 
            x_context, 
            y_context, 
            x_missing, 
            key=key, 
            **kwargs,
        )
    def batched_cond_sample(key, samples, data_batch, params):
        n_tracks = len(data_batch.xs)
        y_samples = jax.vmap(
            jax.vmap(
                cond_sample, 
                in_axes=[0,0,0,0,None]
            ), 
            in_axes=[0,None,None,None,None]
        )(
            jax.random.split(key, n_tracks*samples).reshape((samples, n_tracks, 2)), 
            data_batch.xc,
            data_batch.yc,
            data_batch.xs, 
            params,
        )
        return CondSampleBatch(
            data_batch.xs,
            data_batch.ys,
            data_batch.xc,
            data_batch.yc,
            y_samples,
        )
    return batched_cond_sample

def get_batched_langevin_sampler(langevin_sample_fn, **kwargs):
    @jit
    def langevin_sample(key, t, x_context, y_context, x_target, y_target, params):
        net_ = partial(net, params)

        return langevin_sample_fn(
            sde, 
            partial(net, state.params_ema),
            t,
            x_context, 
            y_context, 
            x_target,
            y_target,
            key, 
            **kwargs,
        )
    def batched_langevin_sample(key, t, cond_batch, params):
        n_tracks = len(cond_batch.xs)
        samples = cond_batch.y_samples.shape[0]
        y_samples = jax.vmap(
            jax.vmap(
                langevin_sample, 
                in_axes=[0,None,0,0,0,0,None]
            ), 
            in_axes=[0,None,None,None,None,0,None]
        )(
            jax.random.split(key, n_tracks*samples).reshape((samples, n_tracks, 2)), 
            t,
            cond_batch.xc,
            cond_batch.yc,
            cond_batch.xs, 
            cond_batch.y_samples,
            params,
        )
        return CondSampleBatch(
            cond_batch.xs,
            cond_batch.ys,
            cond_batch.xc,
            cond_batch.yc,
            y_samples,
        )
    return batched_langevin_sample
# %%
tw = 5.5
fw = 5.5

thin_lw = 0.4 * (fw/5.5)
thick_lw = 0.9 * (fw/5.5)

def mill_coords(points):
    lat = points[..., 0]
    lon = points[..., 1]
    return jnp.stack(
        [
            5/4 *jnp.arcsinh(jnp.tan(4/5 * lat)),
            lon,
        ], axis=-1
    )

def latlon_coords(points):
    y = points[..., 0]
    x = points[..., 1]
    return jnp.stack([
        5/4 * jnp.arctan(jnp.sinh(4/5 * y)),
        x,
    ], axis=-1)

def plot_interp_sample_i(i, cond_sample_batch, ax, buffer=0.2, n_closest=10, fix_aspect=True, sample_color='tab:orange'):
    n_cond = cond_sample_batch.xc.shape[1]

    all_ys = jnp.concatenate([
        model_to_latlon_coords(cond_sample_batch.ys[i]),
        model_to_latlon_coords(cond_sample_batch.yc[i]),
    ], axis=0)

    lons = ((all_ys[..., 1] / RADDEG) + LONOFFSET) % 360  # remove lon offset
    lons = ((lons + LONSTART) % 360) - LONSTART  # Put into plot frame
    lats = all_ys[..., 0] / RADDEG

    all_ys = jnp.stack([lats, lons], axis=-1)

    mins = mill_coords(all_ys.min(axis=(0)) * RADDEG)
    maxs = mill_coords(all_ys.max(axis=(0)) * RADDEG)
    centers = (maxs+mins)/2
    range = jnp.abs(maxs-mins).max()
    mins = centers - range/2 - range * buffer
    maxs = centers + range/2 + range * buffer
    # mins -= buffer
    # maxs += buffer
    mins = latlon_coords(mins) / RADDEG
    maxs = latlon_coords(maxs) / RADDEG

    # data_context = jnp.concatenate([data[1][:, :n_cond//2], data[1][:, (50 - n_cond//2):]], axis=1)
    # track = jnp.concatenate(
    #     [cond_sample_batch.yc[i, :n_cond//2], cond_sample_batch.ys[i], cond_sample_batch.yc[i, n_cond//2:]], axis=0
    # )
    # # dists = jax.vmap(metric.dist, in_axes=[0, None])(model_to_latlon_coords(data_context), model_to_latlon_coords(y_context[i]))
    # dists = data[1] - track[None, ...] # cond_sample_batch.yc[i][None, ...]
    # # dists = data_context - cond_sample_batch.yc[i][None, ...]
    # dists = jnp.nan_to_num(dists)
    # dists = jnp.sqrt((dists**2).mean(axis=-1)).sum(axis=1)
    # sort_idx = jnp.argsort(dists)
    # y_closest = data[1][sort_idx[:n_closest]]
    # x_closest = data[0][sort_idx[:n_closest]]

    kwargs = {
        "lonmin":mins[1],
        "lonmax":maxs[1],
        "latmin":mins[0],
        "latmax":maxs[0],
        "fix_aspect": fix_aspect,
        "boundary_lw": thin_lw,
        # "title": f"{i}"
    }

    # cmap = mpl.cm.get_cmap('Purples')
    # for j, (x, y) in enumerate(zip(x_closest, y_closest)):
    #     plot_tracks(x[None, :], y[None, :], ax, color=cmap((j+1)/(n_closest+1)), lw=thin_lw, **kwargs)

    plot_tracks(cond_sample_batch.xs[i:(i+1)], cond_sample_batch.y_samples[:, i], ax, color=sample_color, lw=thin_lw, alpha=1.0, **kwargs)
    plot_tracks(cond_sample_batch.xc[i:(i+1),:(n_cond//2)], cond_sample_batch.yc[i:(i+1),:(n_cond//2)], ax, color='tab:blue', lw=thick_lw, **kwargs)
    plot_tracks(cond_sample_batch.xc[i:(i+1),(n_cond//2):], cond_sample_batch.yc[i:(i+1),(n_cond//2):], ax, color='tab:blue', lw=thick_lw, **kwargs)
    plot_tracks(cond_sample_batch.xs[i:(i+1)], cond_sample_batch.ys[i:(i+1)], ax, color='tab:green', lw=thick_lw, **kwargs)

def test_sampler(batch, sample_fn, langevin_fn, n_samples):
    samples = sample_fn(jax.random.PRNGKey(0), n_samples, batch, state.params_ema)
    langevin_samples = langevin_fn(jax.random.PRNGKey(0), sde.beta_schedule.t0, samples, state.params_ema)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(fw, fw*2/3),
        sharex=False,
        sharey=False,
        squeeze=True,
    )

    for i, axis in enumerate([ax for axs in axes for ax in axs]):
        plot_interp_sample_i(i, samples, axis, fix_aspect=True)
    for i, axis in enumerate([ax for axs in axes for ax in axs]):
        plot_interp_sample_i(i, langevin_samples, axis, fix_aspect=True, sample_color="tab:red")

    fig.subplots_adjust(wspace=0.05, hspace=0.05, top=1, bottom=0, left=0, right=1)
# %%
n_cond = 20
n_samples = 10
n_tracks = 9
off=17

interp_batch = split_interp(
    DataBatch(
        test_data[0][(off):(n_tracks+off)],
        test_data[1][(off):(n_tracks+off)],
    ), n_cond
)

langevin_sample_kwargs = {
    "num_steps": 1000,
    "num_inner_steps": 150,
    "tau": 0.5,
    "psi_per_inner": 150.0,
    "lambda0": 2.5,
    "prob_flow": False,
    "solver": SphericalGRW(),
}
batched_langevin_sample = get_batched_langevin_sampler(
    ndp.sde.langevin_correct, 
    **langevin_sample_kwargs
)
# %%
cond_sample_kwargs = {
    "num_steps": 1000,
    "num_inner_steps": 25,
    "tau": 0.5,
    "psi_per_inner": 25.0,
    "lambda0": 2.5,
    "prob_flow": False,
    "solver": SphericalGRW(),
    "resample_inner_context": False
}
batched_cond_sample = get_batched_cond_sampler(
    ndp.sde.conditional_sample_independant_context_noise, 
    **cond_sample_kwargs
)
test_sampler(interp_batch, batched_cond_sample, batched_langevin_sample, 10)
# %%
cond_path_sample_kwargs = {
    "num_steps": 1000,
    "num_inner_steps": 25,
    "tau": 0.5,
    "psi_per_inner": 25.0,
    "lambda0": 2.5,
    "prob_flow": False,
    "solver": SphericalGRW(),
}
batched_cond_path_sample = get_batched_cond_sampler(
    ndp.sde.conditional_sample_path_context_noise, 
    **cond_path_sample_kwargs
)
test_sampler(interp_batch, batched_cond_path_sample, batched_langevin_sample, 10)
# %%
cond_hybrid_sample_kwargs = {
    "num_steps": 3000,
    "psi": 0.5,
    "lambda0": 2.5,
    "solver": SphericalGRW(),
}
batched_cond_hybrid_sample = get_batched_cond_sampler(
    ndp.sde.conditional_sample_hybrid_langevin, 
    **cond_hybrid_sample_kwargs
)
test_sampler(interp_batch, batched_cond_hybrid_sample, batched_langevin_sample, 10)

# %%
interp_samples = batched_cond_sample(jax.random.PRNGKey(0), n_samples, interp_batch, state.params_ema)
interp_langevin_samples = batched_langevin_sample(jax.random.PRNGKey(0), sde.beta_schedule.t0, interp_samples, state.params_ema)
# %%
interp_path_samples = batched_cond_path_sample(jax.random.PRNGKey(0), n_samples, interp_batch, state.params_ema)
interp_path_langevin_samples = batched_langevin_sample(jax.random.PRNGKey(0), sde.beta_schedule.t0, interp_path_samples, state.params_ema)
# %%
interp_hybrid_samples = batched_cond_hybrid_sample(jax.random.PRNGKey(0), n_samples, interp_batch, state.params_ema)
interp_hybrid_langevin_samples = batched_langevin_sample(jax.random.PRNGKey(0), sde.beta_schedule.t0, interp_hybrid_samples, state.params_ema)
# %%
tw = 5.5
fw = 5.5

thin_lw = 0.4 * (fw/5.5)
thick_lw = 0.9 * (fw/5.5)

fig, axes = plt.subplots(
    2,
    3,
    figsize=(fw, fw*2/3),
    sharex=False,
    sharey=False,
    squeeze=True,
)



for i, axis in enumerate([ax for axs in axes for ax in axs]):
    plot_interp_sample_i(i, interp_samples, axis, fix_aspect=True)
    break
for i, axis in enumerate([ax for axs in axes for ax in axs]):
    plot_interp_sample_i(i, interp_langevin_samples, axis, fix_aspect=True, sample_color="tab:red")
    break
# for i, axis in enumerate([ax for axs in axes for ax in axs]):
#     plot_interp_sample_i(i, interp_path_samples, axis, fix_aspect=True)
# for i, axis in enumerate([ax for axs in axes for ax in axs]):
#     plot_interp_sample_i(i, interp_path_langevin_samples, axis, fix_aspect=True, sample_color="tab:red")

# for i, axis in enumerate([ax for axs in axes for ax in axs]):
#     plot_interp_sample_i(i, interp_hybrid_samples, axis, fix_aspect=True, sample_color="tab:orange")
# for i, axis in enumerate([ax for axs in axes for ax in axs]):
#     plot_interp_sample_i(i, interp_hybrid_langevin_samples, axis, fix_aspect=True, sample_color="tab:red")



fig.subplots_adjust(wspace=0.05, hspace=0.05, top=1, bottom=0, left=0, right=1)
# fig.savefig("interpolation_examples.pdf")

# %%

n_cond = 20
n_samples = 10
n_tracks = 9
off=17

# extrap_batch = split_extrap(
#     DataBatch(
#         data[0][(off):(n_tracks+off)],
#         data[1][(off):(n_tracks+off)],
#     ), n_cond
# )

extrap_batch = split_extrap(
    DataBatch(
        test_data[0][(off):(n_tracks+off)],
        test_data[1][(off):(n_tracks+off)],
    ), n_cond
)

extrap_samples = batched_cond_sample(jax.random.PRNGKey(0), n_samples, extrap_batch, state.params_ema)
extrap_langevin_samples = batched_langevin_sample(jax.random.PRNGKey(0), sde.beta_schedule.t0, extrap_samples, state.params_ema)
# %%
tw = 5.5
fw = 5.5 / 2.05

thin_lw = 0.4 * (fw/5.5)
thick_lw = 0.9 * (fw/5.5)


fig, axes = plt.subplots(
    3,
    3,
    figsize=(fw, fw),
    sharex=False,
    sharey=False,
    squeeze=True,
)

def plot_extrap_sample_i(
        i,
        cond_sample_batch, 
        ax, 
        buffer=0.5, 
        n_closest=10, 
        fix_aspect=True, 
        sample_color='tab:orange'
    ):
    n_cond = cond_sample_batch.xc.shape[1]

    all_ys = jnp.concatenate([
        model_to_latlon_coords(cond_sample_batch.ys[i]),
        model_to_latlon_coords(cond_sample_batch.yc[i]),
    ], axis=0)

    lons = ((all_ys[..., 1] / RADDEG) + LONOFFSET) % 360  # remove lon offset
    lons = ((lons + LONSTART) % 360) - LONSTART  # Put into plot frame
    lats = all_ys[..., 0] / RADDEG

    all_ys = jnp.stack([lats, lons], axis=-1)

    mins = mill_coords(all_ys.min(axis=(0)) * RADDEG)
    maxs = mill_coords(all_ys.max(axis=(0)) * RADDEG)
    centers = (maxs+mins)/2
    range = jnp.abs(maxs-mins).max()
    mins = centers - range/2 - range * buffer
    maxs = centers + range/2 + range * buffer
    # mins -= buffer
    # maxs += buffer
    mins = latlon_coords(mins) / RADDEG
    maxs = latlon_coords(maxs) / RADDEG

    data_context = data[1][:, :n_cond]
    track = jnp.concatenate(
        [cond_sample_batch.yc[i, :n_cond], cond_sample_batch.ys[i]], axis=0
    )
    # dists = jax.vmap(metric.dist, in_axes=[0, None])(model_to_latlon_coords(data_context), model_to_latlon_coords(y_context[i]))
    # dists = data[1] - track[None, ...] # cond_sample_batch.yc[i][None, ...]
    dists = data_context - cond_sample_batch.yc[i][None, ...]
    dists = jnp.nan_to_num(dists)
    dists = jnp.sqrt((dists**2).mean(axis=-1)).sum(axis=1)
    sort_idx = jnp.argsort(dists)
    y_closest = data[1][sort_idx[:n_closest]]
    x_closest = data[0][sort_idx[:n_closest]]

    kwargs = {
        "lonmin":mins[1],
        "lonmax":maxs[1],
        "latmin":mins[0],
        "latmax":maxs[0],
        "fix_aspect": fix_aspect,
        "boundary_lw": thin_lw,
        # "title": f"{i}"
    }

    cmap = mpl.cm.get_cmap('Purples')
    for j, (x, y) in enumerate(zip(x_closest, y_closest)):
        plot_tracks(x[None, :], y[None, :], ax, color=cmap((j+1)/(n_closest+1)), lw=thin_lw, **kwargs)

    plot_tracks(cond_sample_batch.xs[i:(i+1)], cond_sample_batch.y_samples[:, i], ax, color=sample_color, lw=thin_lw, alpha=1.0, **kwargs)
    # plot_tracks(cond_sample_batch.xs[i:(i+1)], cond_sample_batch.y_samples[:, i], ax, color=sample_color, lw=3.0, alpha=0.1, **kwargs)
    plot_tracks(cond_sample_batch.xc[i:(i+1)], cond_sample_batch.yc[i:(i+1)], ax, color='tab:blue', lw=thick_lw, **kwargs)
    plot_tracks(cond_sample_batch.xs[i:(i+1)], cond_sample_batch.ys[i:(i+1)], ax, color='tab:green', lw=thick_lw, **kwargs)

for i, axis in enumerate([ax for axs in axes for ax in axs]):
    plot_extrap_sample_i(i, extrap_samples, axis, fix_aspect=True)
for i, axis in enumerate([ax for axs in axes for ax in axs]):
    plot_extrap_sample_i(i, extrap_langevin_samples, axis, fix_aspect=True, sample_color="tab:red")

fig.subplots_adjust(wspace=0.05, hspace=0.05, top=1, bottom=0, left=0, right=1)
fig.savefig("extrapolation_examples.pdf")
# %%
from neural_diffusion_processes.brownian_sde import SphericalMetric

metric = SphericalMetric(2)

cond_sampler = jax.vmap(
    jax.vmap(
        cond_sample, 
        in_axes=[0,0,0,0,None]
    ), 
    in_axes=[0,None,None,None,None]
)

test_dataloader = ndp.data.dataloader(
    (data[0][:10], data[1][:10]),
    batch_size=5,
    key=next(key_iter),
    n_points=cfg.data.n_points,
    run_forever=False,
)

interp_context_points = 20




lat_lon_mse = jnp.array([0,0])
r2_mse = jnp.array(0.0)
s2_mse = jnp.array(0.0)
N=0

for i, batch in enumerate(test_dataloader):
    batch = split_interp(batch, interp_context_points)

    cond_samples = cond_sampler(
        jax.random.split(next(key_iter), len(batch.xs)*n_samples).reshape((n_samples, n_tracks, 2)), 
        batch.xs, 
        batch.xc, 
        batch.yc, 
        state.params_ema
    )

    cond_samples = model_to_latlon_coords(cond_samples)
    ys = model_to_latlon_coords(batch.ys)

    euclidean_diff = cond_samples - ys
    sphere_diff = jax.vmap(jax.vmap(metric.dist), in_axes=[0, None])(cond_samples, ys)

    lat_lon_mse += (euclidean_diff**2).mean(axis=(0,1,2))
    r2_mse += (euclidean_diff**2).mean()
    s2_mse += (sphere_diff**2).mean()
    N=i+1

lat_lon_mse /= N
r2_mse /= N
s2_mse /= N


# %%
cond_samples = model_to_latlon_coords(cond_samples)
ys = model_to_latlon_coords(batch.ys)

euclidean_diff = cond_samples - ys
sphere_diff = jax.vmap(jax.vmap(metric.dist), in_axes=[0, None])(cond_samples, ys)

lat_lon_mse = (euclidean_diff**2).mean(axis=(0,1,2))
r2_mse = (euclidean_diff**2).mean()
s2_mse = (sphere_diff**2).mean()
# %%
n_cond = 2
n_samples = 5
n_tracks = 5
off=15

# x_grid = data[0][:n_tracks]
x_context = jnp.concatenate([data[0][(off):(n_tracks+off), :n_cond//2], data[0][(off):(n_tracks+off), (50 - n_cond//2):]], axis=1)
y_context = jnp.concatenate([data[1][(off):(n_tracks+off), :n_cond//2], data[1][(off):(n_tracks+off), (50 - n_cond//2):]], axis=1)
x_missing = data[0][(off):(n_tracks+off), n_cond//2:(50-n_cond//2)]
y_missing = data[1][(off):(n_tracks+off), n_cond//2:(50-n_cond//2)]
# %%
interp_samples = jax.vmap(
    jax.vmap(
        cond_sample, 
        in_axes=[0,0,0,0,None]
    ), 
    in_axes=[0,None,None,None,None]
)(
    jax.random.split(next(key_iter), len(x_missing)*n_samples).reshape((n_samples, n_tracks, 2)), 
    x_missing, 
    x_context, 
    y_context, 
    state.params_ema
)
# %%

n_cond = 20
n_samples = 5
n_tracks = 5
off=15

# x_grid = data[0][:n_tracks]
x_context = jnp.concatenate([data[0][(off):(n_tracks+off), :n_cond//2], data[0][(off):(n_tracks+off), (50 - n_cond//2):]], axis=1)
y_context = jnp.concatenate([data[1][(off):(n_tracks+off), :n_cond//2], data[1][(off):(n_tracks+off), (50 - n_cond//2):]], axis=1)
x_missing = data[0][(off):(n_tracks+off), n_cond//2:(50-n_cond//2)]
y_missing = data[1][(off):(n_tracks+off), n_cond//2:(50-n_cond//2)]

x_context = x_context[0]
y_context = y_context[0]
x_test = x_missing[0]

num_steps = 1000

tau = 1.0
langevin_kernel=True
psi = 0.1
lambda0 = 1.5


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

key = jax.random.PRNGKey(0)

solver = dfx.Euler()

diffusion = lambda t, yt, arg: sde.diffusion(t, unflatten(yt, y_dim), arg)

def sample_marginal(key, t, x_context, y_context):
    if len(y_context) == 0:
        return y_context
    else:
        return flatten(sde.sample_marginal(key, t, x_context, y_context))
    

def reverse_drift_langevin(t, yt, x) -> Array:
    yt = unflatten(yt, y_dim)
    score = flatten(sde.score(key, t, yt, x, partial(net, state.params_ema)))
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
    

reverse_drift_sde = lambda t, yt, arg: flatten(
    sde.reverse_drift_sde(key, t, unflatten(yt, y_dim), arg, partial(net, state.params_ema))
)

shape = jax.ShapeDtypeStruct(shape_augmented_state, y_context.dtype)
key, subkey = jax.random.split(key)
# bm = dfx.VirtualBrownianTree(t0=t1, t1=t0, tol=dt, shape=shape, key=key)
bm = dfx.UnsafeBrownianPath(shape=shape, key=key)
terms_reverse = dfx.MultiTerm(
    dfx.ODETerm(reverse_drift_sde), LinOpControlTerm(diffusion, bm)
)
# %%
t = ts[0]

yt = yT = flatten(sde.sample_prior(subkey, x_test))
yt_context = sample_marginal(next(key_iter), t, x_context, y_context)
yt_augmented = jnp.concatenate([yt_context, yt], axis=0)

yt_m_dt, *_ = solver.step(
            terms_reverse,
            t,
            t - dt,
            yt_augmented,
            x_augmented,
            None,
            made_jump=False,
        )
# %%

bm = dfx.UnsafeBrownianPath(shape=shape, key=subkey)
langevin_terms = dfx.MultiTerm(
    dfx.ODETerm(reverse_drift_langevin), LinOpControlTerm(diffusion_langevin, bm)
)
noise = jnp.sqrt(dt) * jax.random.normal(next(key_iter), shape=yt_augmented.shape)
diffusion_langevin(t - dt, yt_augmented, x_augmented) @ noise
# %%
t=ts[200]
noise = (
    # jnp.sqrt(psi)
    jnp.sqrt(t-2*dt - t-dt)
    * jax.random.normal(next(key_iter), shape=yt_augmented.shape)
)

n = 1000
noise = jnp.concatenate([    jnp.sqrt(dt) * jax.random.normal(next(key_iter), shape=yt_augmented.shape) for i in range(n)], axis=0)

bm = dfx.UnsafeBrownianPath(shape=shape, key=subkey)
langevin_terms = dfx.MultiTerm(
    dfx.ODETerm(reverse_drift_langevin), LinOpControlTerm(diffusion_langevin, bm)
)

langevin_noise = jnp.concatenate([langevin_terms.contr(t-2*dt, t-dt)[1] for i in range(n)])

from typing import Tuple, Union

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from diffrax.misc import split_by_tree, force_bitcast_convert_type

# %%
t0 = t-2*dt
t1 = t-dt


@jit
def get_noise(t0, t1):
    key = jax.random.PRNGKey(0)
    t0 = eqxi.nondifferentiable(t0, name="t0")
    t1 = eqxi.nondifferentiable(t1, name="t1")
    t0_ = force_bitcast_convert_type(t0, jnp.int32)
    t1_ = force_bitcast_convert_type(t1, jnp.int32)
    key = jrandom.fold_in(key, t0_)
    key = jrandom.fold_in(key, t1_)
    # key = split_by_tree(key, shape)
    return jrandom.normal(key, shape=shape.shape, dtype=shape.dtype) * jnp.sqrt(t1 - t0).astype(
        jnp.float32
    )

noise2 = jnp.stack([get_noise(t0+t0*(i * jnp.finfo(jnp.float32).eps), t1+t0*(i * jnp.finfo(jnp.float32).eps)) for i in range(n)])
# noise2 = [get_noise(t0, t1) for i in range(n)]

print(langevin_noise.mean(), langevin_noise.std())
print(noise.mean(), noise.std())
print(noise2.mean(), noise2.std())
# %%
drifts = [
    reverse_drift_langevin(t1+t0*(i * jnp.finfo(jnp.float32).eps), yt_augmented, x_augmented)
    for i in range(1000)
]
drifts_delta = [
    (drifts[i] - drifts[0]) / drifts[0] for i in range(1000)
]
delta_per_step = jnp.array(
    [jnp.mean(jnp.abs(d))for d in drifts_delta]
)
plt.plot(delta_per_step*100)
plt.xlabel("step")
plt.ylabel(r"avg % error")
# %%
drift_ratio = [
    (drifts[i] / drifts[0]) - 1 for i in range(1000)
]

# %%
lambda0 = 1.5
def lambdat(t):
    alphat = jnp.exp(-sde.beta_schedule.B(t))
    return lambda0 / (alphat + (1-alphat)*lambda0)

plt.plot(lambdat(jnp.linspace(0,1,100)))
# %%
from __future__ import annotations
from abc import abstractmethod
import operator
import math

import copy
from functools import partial
import dataclasses

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import diffrax as dfx
from diffrax import AbstractStepSizeController, PIDController, ConstantStepSize
from diffrax import AbstractSolver, Dopri5, Tsit5
import jaxkern
import gpjax
from jaxlinop import LinearOperator, identity

# import equinox as eqx
import numpy as np

from jaxtyping import Array, Float, PyTree
from check_shapes import check_shapes
from einops import rearrange

from neural_diffusion_processes.utils.types import Tuple, Callable, Mapping, Sequence, Optional
from neural_diffusion_processes.data import DataBatch
from neural_diffusion_processes.kernels import (
    prior_gp,
    sample_prior_gp,
    log_prob_prior_gp,
    promote_compute_engines,
    SumKernel,
)
from neural_diffusion_processes.utils.misc import flatten, unflatten
from neural_diffusion_processes.config import get_config
from neural_diffusion_processes.sde import LinOpControlTerm, sde_solve


x_context = interp_batch.xc[0]
y_context = interp_batch.yc[0]
x_test = interp_batch.xs[0]

num_steps = 25000
lambda0 = 1.5
psi=5.0
langevin_kernel = True
key = jax.random.PRNGKey(0)

net_ = partial(net, state.params_ema)

num_context = len(x_context)
num_target = len(x_test)
y_dim = y_context.shape[-1]
shape_augmented_state = [(num_context + num_target) * y_dim]
x_augmented = jnp.concatenate([x_context, x_test], axis=0)

t0 = sde.beta_schedule.t0
t1 = sde.beta_schedule.t1
ts = jnp.linspace(t1, t0, num_steps, endpoint=True)
dt = ts[0] - ts[1]

def lambdat(t):
    alphat = jnp.exp(-sde.beta_schedule.B(t))
    return lambda0 / (alphat + (1-alphat)*lambda0)

# langevin dynamics:
diffusion = lambda t, yt, arg: sde.diffusion(t, unflatten(yt, y_dim), arg)
def reverse_drift_hybrid_langevin(t, yt, x) -> Array:
    yt = unflatten(yt, y_dim)
    score = flatten(sde.score(key, t, yt, x, net_))
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
    return (lambdat(t) + 0.5 * lambda0 * psi) * sde.beta_schedule(t) * score

def diffusion_langevin(t, yt, x) -> LinearOperator:
    if langevin_kernel:
        return jnp.sqrt(1+psi) * diffusion(t, yt, x)
    else:
        return jnp.sqrt((1 + psi) * sde.beta_schedule(t)) * identity(yt.shape[-1])

key, subkey = jax.random.split(key)
shape = jax.ShapeDtypeStruct(shape_augmented_state, y_context.dtype)
# bm = dfx.VirtualBrownianTree(t0=t0, t1=t1, tol=dt, shape=shape, key=subkey)
bm = dfx.UnsafeBrownianPath(shape=shape, key=subkey)
hybrid_terms = dfx.MultiTerm(
    dfx.ODETerm(reverse_drift_hybrid_langevin), LinOpControlTerm(diffusion_langevin, bm)
)

key, subkey = jax.random.split(key)
context_trajectory = jnp.flip(flatten(sde_solve(
    sde,
    x_context,
    None,
    y=y_context,
    key=subkey,
    num_steps=num_steps,
    solver=solver,
    rtol=None,
    atol=None,
    forward=True,
    ts=[]
)[:num_steps]), axis=0)

def loop(yt, t, yct):
    yt_augmented = jnp.concatenate([yt, yct], axis=0)
    yt_m_dt, *_ = solver.step(
        hybrid_terms,
        t,
        t + dt,
        yt_augmented,
        x_augmented,
        None,
        made_jump=False,
    )
    yt = yt_m_dt[num_context * y_dim :]
    return yt, yt

key, subkey = jax.random.split(key)
yT = flatten(sde.sample_prior(subkey, x_test))
# %%
# loop(yT, ts[0], context_trajectory[0])
control = hybrid_terms.contr(ts[0], ts[0] + dt)
# y1 = (y0**ω + terms.vf_prod(t0, y0, args, control) ** ω).ω
# %%

fig, ax = plt.subplots(1,1)
plot_tracks(None, unflatten(context_trajectory[0], 3), ax)