# %%
%load_ext autoreload
%autoreload 2

# %%
import os

os.environ["GEOMSTATS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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


x, y = storm_data(
    "/data/ziz/not-backed-up/mhutchin/score-sde-sp/data",
    50
)

points = y[:100]
((points - proj_stereo(proj_stereo(points), reverse=True)) ** 2)
# %%
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
# %%
m = Basemap()
# m = Basemap(projection='npstere', lon_0=0, boundinglat=-30)
# m = Basemap(projection='ortho',lat_0=45,lon_0=-100,resolution='l')

m.drawmapboundary(fill_color='#99ffff')
m.fillcontinents(color='#cc9966',lake_color='#99ffff')
for row in y[:100]:
    m.plot(
        row[...,1]/RADDEG, row[..., 0]/RADDEG, linewidth=.3, latlon=True
    )
# %%
m = Basemap()
# m = Basemap(projection='npstere', lon_0=0, boundinglat=-30)
# m = Basemap(projection='ortho',lat_0=45,lon_0=-100,resolution='l')

m.drawmapboundary(fill_color='#99ffff')
m.fillcontinents(color='#cc9966',lake_color='#99ffff')
for row in y[:100]:
    m.plot(
        row[...,1]/RADDEG, row[..., 0]/RADDEG, linewidth=.3, latlon=True, c='blue'
    )
for row in proj_stereo(proj_stereo(y[:100]), reverse=True):
    m.plot(
        row[...,1]/RADDEG, row[..., 0]/RADDEG, linewidth=.3, latlon=True, c='orange'
    )
# %%
ax = plt.figure().add_subplot(projection='3d')
for row in proj_3d(latlons[:100]):
    # print(row)
    # m.plot(
    #     row[...,1]/RADDEG, row[..., 0]/RADDEG - 180, linewidth=.3
    # )
    ax.plot(row[..., 0], row[..., 1], -row[..., 2])

import numpy as np
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
r = 0.98
x = r * np.outer(np.cos(u), np.sin(v))
y = r * np.outer(np.sin(u), np.sin(v))
z = r * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the surface
# ax.plot_surface(x, y, z, zsort='min')

ax.view_init(elev=0, azim=0)
ax.set_aspect('equal')
# %%
points = ds.latlons[:100]

points - ds.proj_3d(ds.proj_3d(points), reverse=True)
# %%
ax = plt.figure().add_subplot()

for row in ds.proj_stereo(ds.latlons[:100]):
    # print(row)
    # m.plot(
    #     row[...,1]/RADDEG, row[..., 0]/RADDEG - 180, linewidth=.3
    # )
    ax.plot(row[..., 1], row[..., 0],)

ax.set_aspect('equal')
# %%
