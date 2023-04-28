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
import numpy as np

from einops import rearrange

TWOPI = 2 * jnp.pi
RADDEG = TWOPI/360
from neural_diffusion_processes.data.storm import *

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


(x, y), _ = storm_data(
    "/data/ziz/not-backed-up/mhutchin/score-sde-sp/data",
    50,
    limit=False,
    normalise=False
)
# %%
m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=0,urcrnrlon=360,lat_ts=20)
# m = Basemap(projection='npstere', lon_0=0, boundinglat=-30)
# m = Basemap(projection='ortho',lat_0=45,lon_0=-100,resolution='l')

m.drawmapboundary(fill_color='#ffffff')
m.fillcontinents(color='#cccccc',lake_color='#ffffff')
m.drawcoastlines(color="#000000", linewidth=0.2)
m.drawparallels(np.linspace(-60,60,7, endpoint=True, dtype=int), linewidth=0.1, labels=[True, False, False, False])
m.drawmeridians(np.linspace(-160,160,9, endpoint=True, dtype=int), linewidth=0.1, labels=[False, False, False, True])


coords = jnp.stack(
    m(((y[..., 1]/RADDEG) + 180) % 360, y[..., 0]/RADDEG)[::-1], axis=-1
)
for i, row in enumerate(coords):
    m.plot(
        row[...,1], row[..., 0], linewidth=.3, latlon=False
    )
# %%    
m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=0,urcrnrlon=360,lat_ts=20)
# m = Basemap(projection='npstere', lon_0=0, boundinglat=-30)
# m = Basemap(projection='ortho',lat_0=45,lon_0=-100,resolution='l')

m.drawmapboundary(fill_color='#ffffff')
m.fillcontinents(color='#cccccc',lake_color='#ffffff')
m.drawcoastlines(color="#000000", linewidth=0.2)
m.drawparallels(np.linspace(-60,60,7, endpoint=True, dtype=int), linewidth=0.1, labels=[True, False, False, False])
m.drawmeridians(np.linspace(-160,160,9, endpoint=True, dtype=int), linewidth=0.1, labels=[False, False, False, True])

coords = y/RADDEG
# coords = jnp.stack(
#     m(y[..., 1]/RADDEG, y[..., 0]/RADDEG)[::-1], axis=-1
# )
for i, row in enumerate(coords):
    m.plot(
        row[...,1], row[..., 0], linewidth=.3, latlon=True
    )