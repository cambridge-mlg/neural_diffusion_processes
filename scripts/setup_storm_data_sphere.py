# %%
%load_ext autoreload
%autoreload 2

# %%
import os
import setGPU

os.environ["GEOMSTATS_BACKEND"] = "jax"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# %%
# from neural_diffusion_processes.data import StormDataset
# %%
import os

import jax
import jax.numpy as jnp
import pandas as pd

from einops import rearrange

TWOPI = 2 * jnp.pi
RADDEG = TWOPI/360
from neural_diffusion_processes.data.storm import *
from neural_diffusion_processes.sde import SDE, LinearBetaSchedule, sde_solve
from neural_diffusion_processes.brownian_sde import SphericalBrownian, SphericalGRW
from neural_diffusion_processes.kernels import WhiteVec
from gpjax import Zero

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
import numpy as np
mpl.rcParams['figure.dpi'] = 300


(x, y), data_normalisation = storm_data(
    "/data/ziz/not-backed-up/mhutchin/score-sde-sp/data",
    50,
    limit=True,
    normalise=False,
    # limit_and_normalise=True,
    # max_data_points=1
)

transform = proj_3d
y_sphere = transform(y)

sde = SphericalBrownian(
    None,
    None,
    None,
    beta_schedule=LinearBetaSchedule(
        t0 = 5e-4,
        t1 = 1.0,
        beta0 = 1e-4,
        beta1 = 2.0
    ),
    std_trick=False,
    residual_trick=False,
    is_score_preconditioned=False,    
)

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

    ys = transform(ys, reverse=True)
    ys = ys * data_normalisation[1] + data_normalisation[0]
    dists = jnp.linalg.norm((ys[:, :-1] - ys[:, 1:]), axis=-1)
    # nan_index = dists > ((50 * RADDEG) ** 2)
    # nan_index = jnp.repeat(
    #     jnp.concatenate([nan_index, nan_index[:, -2:-1]], axis=-1)[..., None],
    #     2,
    #     axis=-1,
    # )

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
    # m_coords = m_coords.at[nan_index].set(jnp.nan)

    # for row in m_coords:
    axes.plot(m_coords[..., 1].T, m_coords[..., 0].T, lw=lw, **kwargs)
    # lc = LineCollection(jnp.flip(m_coords, axis=-1), array=jnp.linspace(0,1, m_coords.shape[1])[None, :], linewidth=0.3, cmap='viridis')

    axes.set_title(title)

ts = [0.1,0.5,0.8,0.9,1.0]
nb_cols = len(ts) + 2
nb_rows = 4
fig_forward, axes = plt.subplots(
    nb_rows, nb_cols, figsize=(2 * nb_cols * 0.7, 2 * nb_rows), sharex=True, sharey=True, squeeze=False
)
fig_forward.subplots_adjust(wspace=0, hspace=0)

kwargs = {"forward": True, "rtol": None, "num_steps": 100, "solver": SphericalGRW()}

xs = x[:300]
ys = y[:300]
for i in range(nb_rows):
    idx = 10+i
    key = jax.random.PRNGKey(i)
    n_samples = 10
    # plot data process
    # plot_vf_and_cov(batch.xs[0], batch.ys, axes[:, 0], rf"$p_{{data}}$")
    plot_tracks(xs[idx:idx+1], ys[idx:idx+1], axes[i, 0], rf"$p_{{data}}$")


    x_grid = xs[0]

    y_ref = jax.vmap(sde.sample_prior, in_axes=[0, None])(
        jax.random.split(key, n_samples), x_grid
    ).squeeze()

    # TODO: only solve once and return different timesaves
    # yt = jax.vmap(
    #     lambda key: sde.sample_marginal(
    #         key, t * jnp.ones(()), xs[idx], ys[idx]
    #     )
    # )(jax.random.split(key, n_samples)).squeeze()
    yt = jax.vmap(lambda key: sde_solve(sde, None, key=key, x=x[idx+i], y=y_sphere[idx+i], tf=1.0, ts=ts, **kwargs))(jax.random.split(key, n_samples))
    for k, t in enumerate(ts):

        # plot_vf_and_cov(batch.xs[idx], yt, axes[:, k+1], rf"$p_{{t={t}}}$")
        plot_tracks(xs, yt[:, k], axes[i, k + 1], rf"$p_{{t={t}}}$")

    plot_tracks(x_grid, y_ref, axes[i, -1], rf"$p_{{ref}}$")
    # if i == 0:
    #     plot_tracks(x_grid, y, axes[i, -1], rf"$p_{{data}}$")

plt.tight_layout()

mean_factor = jnp.exp(-0.5*sde.beta_schedule.B(1))
var_factor = jnp.exp(-sde.beta_schedule.B(1))
print(f"{mean_factor=}, {var_factor=}")

decayed_points = ys * mean_factor
decayed_spread = decayed_points.max(axis=(0,1)) - decayed_points.min(axis=(0,1))
data_spread = ys.max(axis=(0,1)) - ys.min(axis=(0,1))
# std = sde.limiting_params["kernel"]["variance"]**0.5
# print(f"Max deviataion {decayed_spread} on range {data_spread} ({decayed_spread/std * 100}% of prior std)")

# %%
